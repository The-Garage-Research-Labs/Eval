
import polars as pl
import os
import math
import threading
from typing import Any, Dict, List, Optional, Iterable

import torch
from html_eval.core.experiment import Experiment
from html_eval.configs.pipeline_config import RerankerExtractorConfig
from html_eval.util.html_util import merge_html_chunks, extract_visible_xpaths_leaves, merge_xpaths_to_html, clean_html, SmartHTMLProcessor
from html_eval.util.json_util import is_schema
import ast
import concurrent.futures
import time
import re

class AIExtractor:

    def __init__(self, config: RerankerExtractorConfig):
        
        self.config = config

        self.llm_client = self.config.llm_config.create_llm_client()
        if self.config.same_llm_config:
            self.llm_pruner_client = self.llm_client
        else:
            self.llm_pruner_client = self.config.llm_pruner_config.create_llm_client()

        self.schema_prompt_template = self.config.schema_generation_prompt_template
        self.query_prompt_template = self.config.query_generation_prompt_template

        self.model_name: str = self.config.reranker_huggingface_model
        self.max_length = self.config.reranker_max_prompt_length
        self.default_top_k = self.config.reranker_default_top_k 
        
        self.vllm_kwargs = {
            "tensor_parallel_size": self.config.reranker_tensor_parallel_size if self.config.reranker_tensor_parallel_size is not None else torch.cuda.device_count(),
            "quantization": self.config.reranker_quantization,
            "gpu_memory_utilization": self.config.reranker_gpu_memory_utilization,
            "max_model_len": self.config.reranker_max_total_length,
            "enable_prefix_caching": self.config.reranker_enable_prefix_caching,
        }

        self.classification_prompt_template = self.config.classification_prompt_template
        self.reranker_classification_threshold = self.config.reranker_classification_threshold



        # placeholders to be set by _load_reranker
        self.tok = None
        self.llm = None
        self.suffix_ids = None
        self.yes_id = None
        self.no_id = None
        self.sampling = None
        self.llm_lock = threading.Lock()

        # load the reranker model/tokenizer into memory
        if not self.config.disable_reranker:
            self._load_reranker()

        self.html_processor = SmartHTMLProcessor()

    def set_experiment(self, experiment: Experiment ):
        self.experiment = experiment
        

    def _load_reranker(self):
        """
        Load tokenizer, vLLM LLM and sampling params into this instance.
        """
        # ensure environment flags similar to your script
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        os.environ.setdefault("VLLM_USE_V1", "1")

        # local imports (fail early if not installed)
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        MODEL_NAME = self.model_name

        # Tokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        tok.padding_side = "left"
        tok.pad_token = tok.eos_token

        # LLM
        # pass through any kwargs set in self.vllm_kwargs
        llm = LLM(model=MODEL_NAME, **self.vllm_kwargs)

        # Suffix and yes/no token ids (keeps same behavior as your file)
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        suffix_ids = tok.encode(suffix, add_special_tokens=False)

        yes_id = tok("yes", add_special_tokens=False).input_ids[0]
        no_id = tok("no", add_special_tokens=False).input_ids[0]

        sampling = SamplingParams(
            # seed=self.experiment._config.seed,
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[yes_id, no_id],
        )

        # assign to instance
        self.tok = tok
        self.llm = llm
        self.suffix_ids = suffix_ids
        self.yes_id = yes_id
        self.no_id = no_id
        self.sampling = sampling
        self.llm_lock = threading.Lock()

    def _format_templates(self, query: str, passages: List[str]) -> List[List[Dict[str, str]]]:
        """
        Build the chat-style templates for each passage (list-of-templates).
        Returns list of templates matching the vLLM chat API shape used in your server.
        """
        INST = self.classification_prompt_template

        def _format(q: str, d: str):
            return [
                {"role": "system", "content": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Answer only "yes" or "no".'},
                {"role": "user",   "content": f"<Instruct>: {INST}\n\n<Query>: {q}\n\n<Document>: {d}"},
            ]

        templates = [_format(query, p) for p in passages]
        return templates

    def _classify(self, processed_batch: List) -> List[float]:
        if self.config.disable_reranker:
            return [1.0] * len(processed_batch)
        
        if not processed_batch:
            return []

        # ensure model loaded
        if self.llm is None or self.tok is None:
            raise RuntimeError("Reranker model/tokenizer not loaded")

        # Tokenize using tokenizer's chat helper
        # apply_chat_template returns token ids lists when tokenize=True
        tokenized = self.tok.apply_chat_template(processed_batch, tokenize=True, add_generation_prompt=False, enable_thinking=False)
        # cap + append suffix ids
        tokenized = [ids[: self.max_length] + self.suffix_ids for ids in tokenized]

        # Prepare TokensPrompt objects
        from vllm.inputs.data import TokensPrompt
        msgs = [TokensPrompt(prompt_token_ids=ids) for ids in tokenized]

        # Call llm.generate (serialize with lock to be safe)
        def _call_generate():
            with self.llm_lock:
                return self.llm.generate(msgs, self.sampling, use_tqdm=False)

        outs = _call_generate()

        # Compute probabilities (softmax over yes/no logits) per passage
        scores: List[float] = []
        for o in outs:
            # defensive access to last token logprobs
            lp = o.outputs[0].logprobs[-1]
            true_logits = lp.get(self.yes_id, type("L", (), {"logprob": -10})).logprob
            false_logits = lp.get(self.no_id,  type("L", (), {"logprob": -10})).logprob

            # convert to probabilities (numerical stable enough for just two tokens)
            y = math.exp(true_logits)
            n = math.exp(false_logits)
            prob_yes = y / (y + n) if (y + n) != 0 else 0.0
            scores.append(prob_yes)
        
        return scores
    


    def _longest_common_xpath_prefix(self, xpaths: Iterable[str]) -> str:
        """
        Compute the longest common xpath prefix across the provided xpaths.
        We treat '/' as separator and only cut at path boundaries (i.e., between steps).
        Returns '/' when there is no non-empty common prefix.
        """
        # keep only non-empty strings
        parts_list = []
        for xp in xpaths:
            if not xp:
                continue
            # normalize: ensure starts with '/'
            s = xp if xp.startswith("/") else "/" + xp
            # split keeps leading '' for the initial slash; that's fine
            parts_list.append(s.split("/"))

        if not parts_list:
            return "/"

        # find common prefix of lists of path segments
        common = []
        for segs in zip(*parts_list):
            # segs contains the next segment from each path
            if all(seg == segs[0] for seg in segs):
                common.append(segs[0])
            else:
                break

        # join back; if common is only [''] (only leading slash) -> return '/'
        if not common or (len(common) == 1 and common[0] == ""):
            return "/"
        prefix = "/".join(common)
        # ensure it begins with '/'
        if not prefix.startswith("/"):
            prefix = "/" + prefix
        return prefix


    def _escape_single_quotes(self, s: str) -> str:
        """Escape single quotes for insertion inside single-quoted string in the prompt."""
        if s is None:
            return ""
        return s.replace("'", "\\'")


    def _remove_prefix_from_xpath(self, xpath: str, prefix: str) -> str:
        """
        Remove prefix from xpath following rules:
        - If xpath == prefix -> return '/'
        - If xpath startswith prefix -> remove prefix and ensure resulting path starts with '/'
        - Otherwise return xpath unchanged (but ensure it starts with '/')
        """
        if xpath is None or xpath == "":
            return "/"
        if not xpath.startswith("/"):
            xpath = "/" + xpath
        if prefix == "/":
            # nothing to remove
            return xpath
        if xpath == prefix:
            return "/"
        if xpath.startswith(prefix):
            rel = xpath[len(prefix):]
            # if removal yields empty or not starting with '/', ensure leading slash
            if rel == "" or not rel.startswith("/"):
                rel = "/" + rel.lstrip("/")
            return rel
        # doesn't start with prefix: return as-is (ensure leading slash)
        return xpath


    def _promp_gen(self, xpath_content_pair_ls: List, query: str) -> str:
        """
        Build the prompt string for the LLM pruner.

        - xpath_content_pair_ls is expected to be an iterable of pairs/tuples like:
            (xpath, content_text)
        but this function tolerates several shapes (tuples, lists, dicts with keys 'xpath'/'content').
        - query is inserted into the template at {query} and content at {content}.
        """
        # Normalize input into list of (xpath, text) tuples while preserving original index order
        normalized = []
        for pair in xpath_content_pair_ls:
            if pair is None:
                normalized.append(("", ""))
                continue
            # tuple/list-like: (xpath, text)
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                xpath, text = pair[0] or "", pair[1] or ""
            elif isinstance(pair, dict):
                xpath = pair.get("xpath", "") or pair.get("0", "") or ""
                text = pair.get("content", "") or pair.get("1", "") or ""
            else:
                # fallback: try to coerce to str
                try:
                    # if pair is something like a Series
                    xpath = str(pair[0]) if getattr(pair, "__len__", None) and len(pair) >= 1 else ""
                    text = str(pair[1]) if getattr(pair, "__len__", None) and len(pair) >= 2 else ""
                except Exception:
                    xpath, text = "", str(pair)
            normalized.append((xpath, text))

        # collect xpaths for prefix computation
        xpaths_for_prefix = [xp for xp, _ in normalized if xp]
        prefix = self._longest_common_xpath_prefix(xpaths_for_prefix)

        # Build the content block lines with prefix removed and single quotes escaped
        lines = []
        # top line
        lines.append(f"The entire chunk is under: '{self._escape_single_quotes(prefix)}'")

        for idx, (xp, txt) in enumerate(normalized):
            rel = self._remove_prefix_from_xpath(xp, prefix)
            # ensure strings and escape single quotes
            rel_escaped = self._escape_single_quotes(rel)
            txt_escaped = self._escape_single_quotes(txt)
            # ensure the relative xpath string begins with a slash (or is '/')
            if not rel_escaped.startswith("/"):
                rel_escaped = "/" + rel_escaped
            lines.append(f"{idx} ('{rel_escaped}', '{txt_escaped}')")

        full_content = "\n".join(lines)
        # Insert into your configured template (keeps original behaviour)
        prompt = self.config.llm_pruner_prompt.format(query=query, content=full_content)
        return prompt

    
    # def _llm_filter(self, chunk_content: str, query: str ) -> str:
    #     chunk_xpaths = extract_visible_xpaths_leaves(chunk_content)
    #     prompt = self._promp_gen(chunk_xpaths, query)
    #     response = self.llm_pruner_client.call_api(prompt)
    #     # for i , x in chunk_xpaths:
    #     #     print(f"{i} : {x}")

    #     # print("-"*80)
    #     # print("prompt: ", prompt)
    #     # print("response: ", response)
    #     # print("-"*80)

    #     match = re.search(r'\[(.*?)\]', response, re.DOTALL)
    #     if match:
    #         inside = "[" + match.group(1).strip() + "]"  # rebuild valid list string
    #         try:
    #             chosen = ast.literal_eval(inside)
    #         except Exception as e:
    #             print("Error evaluating list:", e)
    #             chosen = []
    #     else:
    #         chosen = []

    #     final_list = []
    #     for idx in chosen:
    #         if 0 <= idx < len(chunk_xpaths):
    #             final_list.append(chunk_xpaths[idx])
    #     # print("THIS IS THE FINAL LIST: ", final_list )
    #     # final_content = merge_xpaths_to_html(final_list)
    #     # final_content = clean_html(final_content)
    #     final_content = final_list
    #     # print("final_content: ", final_content)
    #     return final_content




    def _filter(self, batch: pl.DataFrame, threshold: float = 0.5) -> pl.DataFrame:
        """
        Filter the batch based on a threshold and LLM pruning.
        FIXED: Preserves context order by storing xpaths for every row.
        """
        if 'score_norm' not in batch.columns:
            raise ValueError("Batch must contain 'score_norm' column for filtering.")

        # 1. Basic numeric filter
        filtered_batch = batch.filter(pl.col('score_norm') >= threshold)

        if filtered_batch.height == 0:
            return filtered_batch

        if not self.config.use_llm_pruner:
            return filtered_batch
        
        max_workers = getattr(self.config, "llm_pruner_workers", None) or min(32, (os.cpu_count() or 1) * 5)

        rows: List[Dict[str, Any]] = filtered_batch.select(["chunkcontent", "query"]).to_dicts()
        
        prompts = []
        # NEW: We must save the specific xpaths for each row to use during extraction later
        all_rows_xpaths = [] 

        # 2. Prepare Prompts and Context
        for row in rows:
            chunk_content = row['chunkcontent']
            query = row['query']
            
            # Extract structure
            #### OLD
            # chunk_xpaths = extract_visible_xpaths_leaves(chunk_content)
            # xpath_pairs = chunk_xpaths
            #### NEW
            chunk_xpaths = self.html_processor.extract_chunks(chunk_content)
            xpath_pairs = [(item['xpath'], item['content']) for item in chunk_xpaths]
            
            # Store it! Crucial for fixing the order bug.
            all_rows_xpaths.append(chunk_xpaths) 
            prompt = self._promp_gen(xpath_pairs, query)
            prompts.append(prompt)
        
        # 3. Batch Inference
        # Your LLMClient correctly preserves list index order, so prompts[0] matches llm_results[0]
        llm_results = self.llm_pruner_client.call_batch(prompts, max_workers=max_workers,  adapter_name="pruner")
        
        # 4. Extract Results mapping strictly to the correct row
        final_pruned_contents = []

        # zip ensures we match the Response with the specific XPaths from that specific row
        for response, row_xpaths in zip(llm_results, all_rows_xpaths):
            if not response: 
                # Handle failed LLM call: return original or empty? 
                # Usually safer to return original list or empty list depending on logic.
                final_pruned_contents.append(row_xpaths) 
                continue
            # print("res: ",response)
            match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            chosen = []
            if match:
                inside = "[" + match.group(1).strip() + "]"
                try:
                    chosen = ast.literal_eval(inside)
                except Exception as e:
                    print("Error evaluating list:", e)
                    chosen = []
            
            # Filter the specific xpaths for THIS row based on indices
            row_final_list = []
            for idx in chosen:
                # Ensure index is valid for THIS specific row's content
                if isinstance(idx, int) and 0 <= idx < len(row_xpaths):
                    row_final_list.append(row_xpaths[idx])
            
            # If you want to merge back to HTML, uncomment:
            # final_content = merge_xpaths_to_html(row_final_list)
            # final_content = clean_html(final_content)
            
            # Currently returning list of tuples as per your snippet
            final_pruned_contents.append(row_final_list)
        # print("final_content: ", final_pruned_contents)
        # 5. Update the DataFrame
        # We replace the 'chunkcontent' (or create a new col) with the pruned version
        # Use pl.Series to maintain alignment with the filtered_batch
        return filtered_batch.with_columns(
            pl.Series(name="chunkcontent", values=final_pruned_contents)
        )




    def _generate_output(self, batch: pl.DataFrame) -> pl.DataFrame:
        """
        Group by doc_id, merge HTML chunks using BeautifulSoup, remove newline chars,
        build prompts, call LLM in batch, and attach responses.
        """
        # print(batch)
        excluded = {"chunkcontent", "chunkid", "score", "score_norm", "doc_id"}
        # Collect other columns with first() and collect chunkcontent into a list
        agg_exprs = [
            pl.col(col).first()
            for col in batch.columns
            if col not in excluded
        ] + [
            # Correct: just reference the column - it automatically aggregates into a list
            pl.col("chunkcontent").alias("chunks")
        ]
        # Group and aggregate
        df_grouped = batch.group_by("doc_id", maintain_order=True).agg(agg_exprs)

        # STRUCT_DTYPE = pl.Struct([
        #     pl.Field("id", pl.Int64),
        #     pl.Field("xpath", pl.Utf8),
        #     pl.Field("content", pl.Utf8),
        #     pl.Field("type", pl.Utf8),
        # ])

        # df_grouped = df_grouped.with_columns(
        #     pl.col("chunks")
        #     .map_elements(lambda lol: [item for inner in (lol or []) for item in inner],return_dtype=pl.List(STRUCT_DTYPE))
        #     .alias("chunks")
        # )
        # print("Grouped DF: ", df_grouped)
        # Merge chunks using your Python function per-row, then remove newlines
        # NOTE: adding fallback
        df_grouped = df_grouped.with_columns(
            # pl.col("chunks")
            pl.struct(["content", "chunks"])
            .map_elements(lambda s: merge_html_chunks(s["chunks"], s["content"]), return_dtype=pl.Utf8)
            # .map_elements(lambda s: self.html_processor.reconstruct_skeleton(s["content"], s["chunks"]), return_dtype=pl.Utf8)
            .alias("full_content")
        ).drop("chunks")
        
        # Remove newline characters
        df_grouped = df_grouped.with_columns(
            pl.col("full_content").str.replace_all("\n", "").alias("full_content")
        )
        
        # Build prompt column
        # df_prompt = df_grouped.with_columns(
        #     pl.struct(["query", "full_content"]).map_elements(
        #         lambda s: self.schema_prompt_template.format(query=s["query"], content=s["full_content"]),
        #         return_dtype=pl.Utf8
        #     ).alias("prompt")
        # )
        # Build a prompt column based on the query type (schema vs non-schema)
        def build_prompt(row):
            query = row["query"]
            content = row["full_content"]
            if is_schema(query):
                return self.schema_prompt_template.format(query=query, content=content)
            else:
                return self.query_prompt_template.format(query=query, content=content)
        df_prompt = df_grouped.with_columns(
            pl.struct(["query", "full_content"]).map_elements(
                build_prompt,
                return_dtype=pl.Utf8
            ).alias("prompt")
        )

        # Call LLM and attach responses
        prompts = df_prompt["prompt"].to_list()
        responses = self.llm_client.call_batch(prompts , adapter_name="extractor")
        # responses = self.llm_client.call_batch(prompts , adapter_name="extractor", thinking=True)
        
        df_response = df_prompt.with_columns(
            pl.Series("response", responses, dtype=pl.Utf8)
        )
        return df_response
  
    
    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        
        # print(f"Shape before: format templates {df.shape} ")
        # Format the input Dataframe
        processed = []
        for row in df.iter_rows(named=True):
            for chunk in row['chunks']:
                processed += self._format_templates(row['query'], [chunk['chunkcontent']])
        # print(f"Shape before: classify {df.shape} ")
        # Score the passages

        scores = self._classify(processed)
        # print(f"Shape before: new DF {df.shape} ")


        # Step 2: explode the list into multiple rows
        expanded_df = df.explode("chunks")
        # print(f"Shape before: unnest {expanded_df.shape} ")

        # Step 3: unnest the struct inside the list
        expanded_df = expanded_df.unnest("chunks")
        # print(f"Shape before: float {expanded_df.shape} ")

        scores_df = expanded_df.with_columns(
            pl.Series('score', scores, dtype=pl.Float64)
        )

        # print(f"Shape before: norm {scores_df.shape} ")

        # Max normalization of scores for each docid TODO: need to add it do the config
        norm_df = scores_df.with_columns(
            (pl.col("score") / pl.col("score").max().over("doc_id")).alias("score_norm")
        )
        # print(f"Shape before: filter {norm_df.shape} ")

        # Filter the DataFrame based on the score threshold
        filtered_df = self._filter(norm_df, threshold=self.reranker_classification_threshold)
        # print(f"Shape before: generates {filtered_df.shape} ")
        # print(filtered_df)
        generated_df = self._generate_output(filtered_df)
        # print(f"Shape before: extract exact {filtered_df.shape} ")

        final_df = generated_df
        # print(final_df)
        return final_df 

        

 
        

