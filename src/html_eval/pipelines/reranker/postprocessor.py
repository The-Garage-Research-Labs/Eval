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
    Optimized worker that takes raw strings instead of a meta dict.
    """
    try:
        # 1. Parse JSON
        is_schema_query = is_schema(query) if query else False
        parsed_response = extract_and_repair_json(response, not is_schema_query)

        if isinstance(parsed_response, str):
            return parsed_response
            
        # Validate dict
        if not isinstance(parsed_response, dict):
            return {"__error__": "[PARSE_ERROR] expected JSON object (dict) from extract_and_repair_json"}
        
        # 2. Exact Extraction (Heavy CPU part)
        if extract_exact:
            if not content:
                return {"__error__": "[PARSE_ERROR] exact_extraction requested but no content provided"}

            # Iterate over keys and perform fuzzy matching
            for attribute, value in list(parsed_response.items()):
                if value is None:
                    parsed_response[attribute] = None
                    continue
                try:
                    val_str = str(value)
                    
                    # Heavy CPU call -> find_closest_html_node
                    best_match = find_closest_html_node(html_text=content, search_text=val_str)
                    
                    parsed_response[attribute] = (
                        normalize_html_text(best_match['text']) 
                        if best_match and 'text' in best_match 
                        else None
                    )
                except Exception as e:
                    parsed_response[attribute] = {"__error__": f"[MATCH_ERROR] {e}", "original": value}

        return parsed_response

    except Exception as e:
        return {"__error__": f"[PARSE_ERROR] {e}"}


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
        
        parsed = _safe_extract_worker(
            response=response, 
            content=content, 
            query=query, 
            extract_exact=self._exact_extraction
        )
        return SamplePrediction(prediction=parsed, **meta)

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
        for parsed, meta in zip(parsed_results, metas):
            final_predictions.append(SamplePrediction(prediction=parsed, **meta))
            
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
        
        # For exact extraction, we prefer 'content', but fallback to 'filtered_html' if needed
        # depending on your logic. Usually 'content' is the raw HTML.
        contents = df[content_col].to_list() if content_col in df.columns else [None] * len(df)
        
        # 2. Build Metadata Lightweights (pointers, not deep copies)
        # We construct this list to zip back later.
        # Note: Using get_column(name).to_list() is faster than row iteration
        ids = df[id_col].to_list() if id_col in df.columns else [None] * len(df)
        gts = df[gt_col].to_list() if gt_col in df.columns else [None] * len(df)
        f_htmls = df[filtered_html_col].to_list() if filtered_html_col in df.columns else [None] * len(df)
        
        # Reconstruct metas list for the final SamplePrediction objects
        metas = [
            {
                "id": i,
                "query": q,
                "ground_truth": g,
                "filtered_html": f,
                "content": c
            }
            for i, q, g, f, c in zip(ids, queries, gts, f_htmls, contents)
        ]

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
        return df.with_columns(pl.Series("prediction", pred_values))