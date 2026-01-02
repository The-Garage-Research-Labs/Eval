import polars as pl
import os
import math
import threading
from typing import Any, Dict, List, Optional, Iterable, Tuple, Union
import torch
from html_eval.core.experiment import Experiment
from html_eval.configs.pipeline_config import RerankerExtractorConfig
from html_eval.util.html_util import merge_html_chunks, extract_visible_xpaths_leaves, merge_xpaths_to_html, clean_html, SmartHTMLProcessor
from html_eval.util.json_util import is_schema
import ast
import concurrent.futures
import time
import re
from concurrent.futures import ProcessPoolExecutor

# ==============================================================================
# 1. STANDALONE HELPER FUNCTIONS (Moved out of class to allow Pickling)
# ==============================================================================

def _longest_common_xpath_prefix(xpaths: Iterable[str]) -> str:
    """Compute longest common xpath prefix."""
    parts_list = []
    for xp in xpaths:
        if not xp: continue
        s = xp if xp.startswith("/") else "/" + xp
        parts_list.append(s.split("/"))

    if not parts_list: return "/"

    common = []
    for segs in zip(*parts_list):
        if all(seg == segs[0] for seg in segs):
            common.append(segs[0])
        else:
            break

    if not common or (len(common) == 1 and common[0] == ""):
        return "/"
    prefix = "/".join(common)
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    return prefix

def _escape_single_quotes(s: str) -> str:
    if s is None: return ""
    return s.replace("'", "\\'")

def _remove_prefix_from_xpath(xpath: str, prefix: str) -> str:
    if xpath is None or xpath == "": return "/"
    if not xpath.startswith("/"): xpath = "/" + xpath
    if prefix == "/": return xpath
    if xpath == prefix: return "/"
    if xpath.startswith(prefix):
        rel = xpath[len(prefix):]
        if rel == "" or not rel.startswith("/"):
            rel = "/" + rel.lstrip("/")
        return rel
    return xpath

def generate_pruner_prompt(xpath_content_pair_ls: List, query: str, prompt_template: str) -> str:
    """
    Standalone version of _promp_gen. 
    Accepts prompt_template string directly instead of config object.
    """
    normalized = []
    for pair in xpath_content_pair_ls:
        if pair is None:
            normalized.append(("", ""))
            continue
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            xpath, text = pair[0] or "", pair[1] or ""
        elif isinstance(pair, dict):
            xpath = pair.get("xpath", "") or pair.get("0", "") or ""
            text = pair.get("content", "") or pair.get("1", "") or ""
        else:
            try:
                xpath = str(pair[0]) if getattr(pair, "__len__", None) and len(pair) >= 1 else ""
                text = str(pair[1]) if getattr(pair, "__len__", None) and len(pair) >= 2 else ""
            except Exception:
                xpath, text = "", str(pair)
        normalized.append((xpath, text))

    xpaths_for_prefix = [xp for xp, _ in normalized if xp]
    prefix = _longest_common_xpath_prefix(xpaths_for_prefix)

    lines = []
    lines.append(f"The entire chunk is under: '{_escape_single_quotes(prefix)}'")

    for idx, (xp, txt) in enumerate(normalized):
        rel = _remove_prefix_from_xpath(xp, prefix)
        rel_escaped = _escape_single_quotes(rel)
        txt_escaped = _escape_single_quotes(txt)
        if not rel_escaped.startswith("/"):
            rel_escaped = "/" + rel_escaped
        lines.append(f"{idx} ('{rel_escaped}', '{txt_escaped}')")

    full_content = "\n".join(lines)
    prompt = prompt_template.format(query=query, content=full_content)
    return prompt

# ==============================================================================
# 2. WORKER FUNCTIONS (Must be top-level for Multiprocessing)
# ==============================================================================

def _worker_filter_prep(args):
    """
    Worker for _filter.
    Receives: (chunk_content, query, template_string)
    Returns: (chunk_xpaths_object, prompt_string)
    """
    row_content, row_query, prompt_template = args
    
    # 1. Instantiate Processor inside the worker (avoid pickling the object)
    processor = SmartHTMLProcessor() 
    
    # 2. Heavy CPU: Parse HTML
    chunk_xpaths = processor.extract_chunks(row_content)
    
    # 3. Prepare data for prompt generation
    xpath_pairs = [(item['xpath'], item['content']) for item in chunk_xpaths]
    
    # 4. Generate Prompt using STANDALONE function
    prompt = generate_pruner_prompt(xpath_pairs, row_query, prompt_template)
    
    return chunk_xpaths, prompt

def _worker_merge_html(args):
    """
    Worker for _generate_output.
    Receives: (chunks_list, content_fallback)
    Returns: string
    """
    chunks, content = args
    # Import locally to be safe, though util imports are usually fine
    from html_eval.util.html_util import merge_html_chunks
    
    # Heavy CPU: Merge and clean HTML
    merged = merge_html_chunks(chunks, content)
    
    # Optimization: remove newlines here in the worker
    return merged.replace("\n", "")

# ==============================================================================
# 3. CLASS DEFINITION
# ==============================================================================

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

        self.tok = None
        self.llm = None
        self.suffix_ids = None
        self.yes_id = None
        self.no_id = None
        self.sampling = None
        self.llm_lock = threading.Lock()

        if not self.config.disable_reranker:
            self._load_reranker()

        self.html_processor = SmartHTMLProcessor()

    def set_experiment(self, experiment: Experiment ):
        self.experiment = experiment
        
    def _load_reranker(self):
        # ... (Same as before) ...
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        os.environ.setdefault("VLLM_USE_V1", "1")
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        MODEL_NAME = self.model_name
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        tok.padding_side = "left"
        tok.pad_token = tok.eos_token
        llm = LLM(model=MODEL_NAME, **self.vllm_kwargs)
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        suffix_ids = tok.encode(suffix, add_special_tokens=False)
        yes_id = tok("yes", add_special_tokens=False).input_ids[0]
        no_id = tok("no", add_special_tokens=False).input_ids[0]
        sampling = SamplingParams(
            temperature=0, max_tokens=1, logprobs=20, allowed_token_ids=[yes_id, no_id],
        )
        self.tok = tok
        self.llm = llm
        self.suffix_ids = suffix_ids
        self.yes_id = yes_id
        self.no_id = no_id
        self.sampling = sampling
        self.llm_lock = threading.Lock()

    def _format_templates(self, query: str, passages: List[str]) -> List[List[Dict[str, str]]]:
        INST = self.classification_prompt_template
        def _format(q: str, d: str):
            return [
                {"role": "system", "content": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Answer only "yes" or "no".'},
                {"role": "user",   "content": f"<Instruct>: {INST}\n\n<Query>: {q}\n\n<Document>: {d}"},
            ]
        templates = [_format(query, p) for p in passages]
        return templates

    def _classify(self, processed_batch: List) -> List[float]:
        # ... (Same as before) ...
        if self.config.disable_reranker:
            return [1.0] * len(processed_batch)
        if not processed_batch:
            return []
        if self.llm is None or self.tok is None:
            raise RuntimeError("Reranker model/tokenizer not loaded")

        tokenized = self.tok.apply_chat_template(processed_batch, tokenize=True, add_generation_prompt=False, enable_thinking=False)
        tokenized = [ids[: self.max_length] + self.suffix_ids for ids in tokenized]
        from vllm.inputs.data import TokensPrompt
        msgs = [TokensPrompt(prompt_token_ids=ids) for ids in tokenized]

        def _call_generate():
            with self.llm_lock:
                return self.llm.generate(msgs, self.sampling, use_tqdm=False)
        outs = _call_generate()
        scores: List[float] = []
        for o in outs:
            lp = o.outputs[0].logprobs[-1]
            true_logits = lp.get(self.yes_id, type("L", (), {"logprob": -10})).logprob
            false_logits = lp.get(self.no_id,  type("L", (), {"logprob": -10})).logprob
            y = math.exp(true_logits)
            n = math.exp(false_logits)
            prob_yes = y / (y + n) if (y + n) != 0 else 0.0
            scores.append(prob_yes)
        return scores

    # ---------------------------------------------------------
    # OPTIMIZED _filter
    # ---------------------------------------------------------
    def _filter(self, batch: pl.DataFrame, threshold: float = 0.5) -> pl.DataFrame:
        if 'score_norm' not in batch.columns:
            raise ValueError("Batch must contain 'score_norm' column for filtering.")

        filtered_batch = batch.filter(pl.col('score_norm') >= threshold)

        if filtered_batch.height == 0:
            return filtered_batch
        
        if not self.config.use_llm_pruner:
            return filtered_batch

        # 1. Prepare Data for Parallel Processing
        # Extract rows as tuples: (chunkcontent, query)
        rows_data = filtered_batch.select(["chunkcontent", "query"]).rows()
        
        # Prepare arguments: (content, query, template_string)
        # Note: We pass the template STRING, not self or config.
        template_str = self.config.llm_pruner_prompt
        worker_args = [(r[0], r[1], template_str) for r in rows_data]

        max_workers = getattr(self.config, "llm_pruner_workers", None) or min(32, (os.cpu_count() or 1) * 4)

        # 2. Parallel CPU Execution (HTML Parsing + Prompt Gen)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Map returns an iterator, list() consumes it
            results = list(executor.map(_worker_filter_prep, worker_args))

        # Unpack results: results is list of (chunk_xpaths, prompt)
        all_rows_xpaths, prompts = zip(*results)
        all_rows_xpaths = list(all_rows_xpaths)
        prompts = list(prompts)
        
        # 3. Batch GPU Inference (Fast)
        llm_results = self.llm_pruner_client.call_batch(prompts, max_workers=max_workers, adapter_name="pruner")
        
        # 4. Process Results (Light CPU work)
        final_pruned_contents = []

        for response, row_xpaths in zip(llm_results, all_rows_xpaths):
            if not response: 
                final_pruned_contents.append(row_xpaths) 
                continue
            
            match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            chosen = []
            if match:
                inside = "[" + match.group(1).strip() + "]"
                try:
                    chosen = ast.literal_eval(inside)
                except Exception as e:
                    chosen = []
            
            row_final_list = []
            for idx in chosen:
                if isinstance(idx, int) and 0 <= idx < len(row_xpaths):
                    row_final_list.append(row_xpaths[idx])
            
            final_pruned_contents.append(row_final_list)

        return filtered_batch.with_columns(
            pl.Series(name="chunkcontent", values=final_pruned_contents)
        )

    # ---------------------------------------------------------
    # OPTIMIZED _generate_output
    # ---------------------------------------------------------
    def _generate_output(self, batch: pl.DataFrame) -> pl.DataFrame:
        
        excluded = {"chunkcontent", "chunkid", "score", "score_norm", "doc_id"}
        
        # Aggregation Logic
        agg_exprs = [
            pl.col(col).first() for col in batch.columns if col not in excluded
        ] + [
            pl.col("chunkcontent").alias("chunks")
        ]
        df_grouped = batch.group_by("doc_id", maintain_order=True).agg(agg_exprs)

        # 1. Prepare Data for Parallel Processing
        # Get columns as Python lists
        chunks_list = df_grouped["chunks"].to_list()
        content_list = df_grouped["content"].to_list()
        
        # Zip them for the worker
        worker_args = zip(chunks_list, content_list)

        # 2. Parallel CPU Execution (Merge HTML)
        max_workers = min(32, (os.cpu_count() or 1) * 2)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            full_contents = list(executor.map(_worker_merge_html, worker_args))

        # 3. Assign back to Polars (Avoids map_elements)
        df_grouped = df_grouped.with_columns(
            pl.Series("full_content", full_contents, dtype=pl.Utf8)
        ).drop("chunks")
        
        # (Newlines already removed in worker)
        
        # 4. Prompt Generation (Fast enough in Polars usually, or could be parallelized similarly)
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

        # =========================================================
        # SPLIT BATCH LOGIC (QA vs SCHEMA)
        # =========================================================
        
        prompts = df_prompt["prompt"].to_list()
        queries = df_prompt["query"].to_list()
        
        # Storage for split batches
        qa_indices = []
        qa_prompts = []
        
        schema_indices = []
        schema_prompts = []

        # 1. Split based on Query Type
        for idx, (q, p) in enumerate(zip(queries, prompts)):
            if is_schema(q):
                schema_indices.append(idx)
                schema_prompts.append(p)
            else:
                qa_indices.append(idx)
                qa_prompts.append(p)

        # Holder for final results in original order
        final_responses = [None] * len(prompts)

        # 2. Run QA Batch (Adapter: "qa")
        if qa_prompts:
            # print(f"Processing {len(qa_prompts)} QA queries...")
            qa_responses = self.llm_client.call_batch(qa_prompts, adapter_name="qa")
            
            # Map back to original indices
            for original_idx, response in zip(qa_indices, qa_responses):
                final_responses[original_idx] = response

        # 3. Run Schema Batch (Adapter: "schema")
        if schema_prompts:
            # print(f"Processing {len(schema_prompts)} Schema queries...")
            schema_responses = self.llm_client.call_batch(schema_prompts, adapter_name="schema")
            
            # Map back to original indices
            for original_idx, response in zip(schema_indices, schema_responses):
                final_responses[original_idx] = response

        # =========================================================

        df_response = df_prompt.with_columns(
            pl.Series("response", final_responses, dtype=pl.Utf8)
        )
        return df_response
  
    def extract(self, df: pl.DataFrame) -> pl.DataFrame:
        
        # Step 1: Format & Classify
        processed = []
        for row in df.iter_rows(named=True):
            for chunk in row['chunks']:
                processed += self._format_templates(row['query'], [chunk['chunkcontent']])
        
        scores = self._classify(processed)

        expanded_df = df.explode("chunks")
        expanded_df = expanded_df.unnest("chunks")

        scores_df = expanded_df.with_columns(
            pl.Series('score', scores, dtype=pl.Float64)
        )

        norm_df = scores_df.with_columns(
            (pl.col("score") / pl.col("score").max().over("doc_id")).alias("score_norm")
        )

        # Step 2: Filter (Optimized Parallel)
        filtered_df = self._filter(norm_df, threshold=self.reranker_classification_threshold)
        
        # Step 3: Generate (Optimized Parallel)
        generated_df = self._generate_output(filtered_df)

        final_df = generated_df
        return final_df