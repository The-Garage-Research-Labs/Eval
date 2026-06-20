from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Iterable, List, Optional, Dict, Any, Union
import os
import polars as pl

from html_eval.util.json_util import extract_and_repair_json, is_schema
from html_eval.util.html_util import find_closest_html_node, normalize_html_text
from html_eval.core.types import SamplePrediction
from html_eval.configs.pipeline_config import RerankerPostprocessorConfig

# ==============================================================================
# WORKER FUNCTION (Top-level for Pickling)
# ==============================================================================

def _safe_extract_worker(response: str, content: str, query: str, extract_exact: bool) -> Dict[str, Any]:
    """
    Optimized worker that takes raw strings and returns parsed prediction + step logs.
    """
    import copy
    exact_match_log = {}
    error_msg = None
    parsed_raw = None
    
    try:
        # 1. Parse JSON
        is_schema_query = is_schema(query) if query else False
        parsed_response = extract_and_repair_json(response, not is_schema_query)

        parsed_raw = parsed_response

        if isinstance(parsed_response, str):
            return {
                "prediction": parsed_response,
                "postprocessor_log": {
                    "raw_response": response,
                    "error": None,
                    "exact_match_log": {}
                }
            }
            
        # Validate dict
        if not isinstance(parsed_response, dict):
            error_msg = "[PARSE_ERROR] expected JSON object (dict) from extract_and_repair_json"
            return {
                "prediction": {"__error__": error_msg},
                "postprocessor_log": {
                    "raw_response": response,
                    "error": error_msg,
                    "exact_match_log": {}
                }
            }
        
        # 2. Exact Extraction (Heavy CPU part)
        if extract_exact:
            if not content:
                error_msg = "[PARSE_ERROR] exact_extraction requested but no content provided"
                return {
                    "prediction": {"__error__": error_msg},
                    "postprocessor_log": {
                        "raw_response": response,
                        "error": error_msg,
                        "exact_match_log": {}
                    }
                }

            # create deep copy for log before editing
            parsed_raw = copy.deepcopy(parsed_response) if hasattr(parsed_response, "copy") else parsed_response

            # Iterate over keys and perform fuzzy matching
            for attribute, value in list(parsed_response.items()):
                if value is None:
                    parsed_response[attribute] = None
                    exact_match_log[attribute] = {
                        "value": None,
                        "original_extracted": None,
                        "status": "null"
                    }
                    continue
                try:
                    val_str = str(value)
                    
                    # Heavy CPU call -> find_closest_html_node
                    best_match = find_closest_html_node(html_text=content, search_text=val_str)
                    
                    if best_match and 'text' in best_match:
                        matched_text = normalize_html_text(best_match['text'])
                        parsed_response[attribute] = matched_text
                        exact_match_log[attribute] = {
                            "value": matched_text,
                            "original_extracted": value,
                            "xpath": best_match.get("xpath"),
                            "score": best_match.get("score"),
                            "status": "success"
                        }
                    else:
                        parsed_response[attribute] = None
                        exact_match_log[attribute] = {
                            "value": None,
                            "original_extracted": value,
                            "status": "not_found"
                        }
                except Exception as e:
                    err_msg = f"[MATCH_ERROR] {e}"
                    parsed_response[attribute] = {"__error__": err_msg, "original": value}
                    exact_match_log[attribute] = {
                        "value": None,
                        "original_extracted": value,
                        "error": err_msg,
                        "status": "error"
                    }

        return {
            "prediction": parsed_response,
            "postprocessor_log": {
                "raw_response": response,
                "error": error_msg,
                "exact_match_log": exact_match_log
            }
        }

    except Exception as e:
        error_msg = f"[PARSE_ERROR] {e}"
        return {
            "prediction": {"__error__": error_msg},
            "postprocessor_log": {
                "raw_response": response,
                "error": error_msg,
                "exact_match_log": {}
            }
        }


# ==============================================================================
# CLASS DEFINITION
# ==============================================================================

class PostProcessor:
    """
    Optimized PostProcessor for high-throughput batch processing.
    """
    def __init__(self, config: RerankerPostprocessorConfig):
        self._config = config
        self._exact_extraction = bool(config.exact_extraction)

    def process(self, response: str, **meta: Any) -> SamplePrediction:
        """Process one response (Legacy/Single item support)."""
        content = meta.get('content')
        query = meta.get('query')
        
        parsed_dict = _safe_extract_worker(
            response=response, 
            content=content, 
            query=query, 
            extract_exact=self._exact_extraction
        )
        prediction = parsed_dict["prediction"]
        postprocessor_log = parsed_dict["postprocessor_log"]
        
        step_logs = {
            **(meta.get("step_logs") or {}),
            "postprocessor": postprocessor_log
        }
        
        return SamplePrediction(
            prediction=prediction,
            step_logs=step_logs,
            preprocessed_content=meta.get("preprocessed_content"),
            **{k: v for k, v in meta.items() if k not in ("step_logs", "preprocessed_content")}
        )

    def process_responses(
        self,
        responses: List[str],
        contents: List[str],
        queries: List[str],
        metas: List[Dict[str, Any]],
        n_workers: Optional[int] = None,
        use_process: bool = True, # Default to True for speed
    ) -> List[SamplePrediction]:
        """
        Parallel processing core.
        """
        n_items = len(responses)
        if n_workers is None:
            # Leave one core free for system stability
            n_workers = max(1, (os.cpu_count() or 2) - 1)

        Executor = ProcessPoolExecutor if use_process else ThreadPoolExecutor

        # Prepare flags generator
        extract_flags = [self._exact_extraction] * n_items

        # If contents or queries are None/Empty lists, handle gracefully
        if not contents: contents = [None] * n_items
        if not queries: queries = [None] * n_items

        # EXECUTE IN PARALLEL
        with Executor(max_workers=n_workers) as ex:
            # We only map the necessary data. We do NOT map 'metas'.
            # Mapping 'metas' pickles huge objects unnecessarily.
            parsed_results = list(ex.map(
                _safe_extract_worker, 
                responses, 
                contents, 
                queries, 
                extract_flags
            ))

        # Re-assemble results in the main process
        # This is fast because no heavy computation happens here
        final_predictions = []
        for parsed_dict, meta in zip(parsed_results, metas):
            prediction = parsed_dict["prediction"]
            postprocessor_log = parsed_dict["postprocessor_log"]
            
            step_logs = {
                **(meta.get("step_logs") or {}),
                "postprocessor": postprocessor_log
            }
            
            final_predictions.append(SamplePrediction(
                prediction=prediction,
                step_logs=step_logs,
                preprocessed_content=meta.get("preprocessed_content"),
                **{k: v for k, v in meta.items() if k not in ("step_logs", "preprocessed_content")}
            ))
            
        return final_predictions

    def process_dataframe(
        self,
        df: pl.DataFrame,
        response_col: str = "response",
        id_col: str = "id",
        query_col: str = "query",
        gt_col: str = "ground_truth",
        content_col: str = "content",
        filtered_html_col: str = "full_content",
        n_workers: Optional[int] = None,
        use_process: bool = True, # Default to True for speed
        return_polars: bool = False,
    ) -> Union[List[SamplePrediction], pl.DataFrame]:
        """
        Optimized DataFrame processor.
        Extracts columns as lists (Fast) instead of iterating rows (Slow).
        """
        if response_col not in df.columns:
            raise KeyError(f"response_col '{response_col}' not present")

        # 1. Fast Column Extraction (Avoids df.to_dicts overhead)
        responses = df[response_col].to_list()
        
        # Handle optional columns gracefully
        queries = df[query_col].to_list() if query_col in df.columns else [None] * len(df)
        
        # For exact extraction, we prefer 'cleaned_content' if present, else fallback
        contents = df["cleaned_content"].to_list() if "cleaned_content" in df.columns else (df[content_col].to_list() if content_col in df.columns else [None] * len(df))
        
        # 2. Build Metadata Lightweights (pointers, not deep copies)
        ids = df[id_col].to_list() if id_col in df.columns else [None] * len(df)
        gts = df[gt_col].to_list() if gt_col in df.columns else [None] * len(df)
        f_htmls = df[filtered_html_col].to_list() if filtered_html_col in df.columns else [None] * len(df)
        
        # Capture step logs columns if present
        preprocessor_logs = df["preprocessor_log"].to_list() if "preprocessor_log" in df.columns else [None] * len(df)
        cleaned_contents = df["cleaned_content"].to_list() if "cleaned_content" in df.columns else [None] * len(df)
        reranker_chunks = df["reranker_chunks"].to_list() if "reranker_chunks" in df.columns else [None] * len(df)
        pruner_logs = df["pruner_logs"].to_list() if "pruner_logs" in df.columns else [None] * len(df)
        prompts = df["prompt"].to_list() if "prompt" in df.columns else [None] * len(df)
        
        # Reconstruct metas list for the final SamplePrediction objects
        metas = []
        for i, q, g, f, c, prep_log, clean_c, rer_chk, prn_log, p, r in zip(
            ids, queries, gts, f_htmls, contents, preprocessor_logs, cleaned_contents, reranker_chunks, pruner_logs, prompts, responses
        ):
            # Format reranker logs if present
            rer_log = None
            if rer_chk is not None:
                # Convert list of structs (Polars/Python dicts) to lists of dicts
                if hasattr(rer_chk, "to_list"):
                    rer_chk_list = rer_chk.to_list()
                else:
                    rer_chk_list = list(rer_chk)
                rer_log = {"chunks": rer_chk_list}

            # Format pruner logs if present
            prn_log_val = None
            if prn_log is not None:
                if hasattr(prn_log, "to_list"):
                    prn_log_val = prn_log.to_list()
                else:
                    prn_log_val = list(prn_log)

            step_logs = {
                "preprocessor": prep_log,
                "reranker": rer_log,
                "pruner": prn_log_val,
                "extractor": {
                    "prompt": p,
                    "raw_response": r
                }
            }
            
            metas.append({
                "id": i,
                "query": q,
                "ground_truth": g,
                "filtered_html": f,
                "content": c,
                "preprocessed_content": clean_c,
                "step_logs": step_logs
            })

        # 3. Process
        preds = self.process_responses(
            responses=responses,
            contents=contents,
            queries=queries,
            metas=metas,
            n_workers=n_workers,
            use_process=use_process,
        )

        if not return_polars:
            return preds

        pred_values = [p.prediction for p in preds]
        df_out = df.with_columns([
            pl.Series("prediction", pred_values)
        ])
        if "preprocessed_content" not in df_out.columns:
            df_out = df_out.with_columns(pl.Series("preprocessed_content", [p.preprocessed_content for p in preds]))
        if "step_logs" not in df_out.columns:
            df_out = df_out.with_columns(pl.Series("step_logs", [p.step_logs for p in preds]))
        return df_out