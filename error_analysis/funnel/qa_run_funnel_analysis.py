import os
import re
import json
import html
import unicodedata
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import polars as pl

# ---------- Optional fuzzy deps ----------
try:
    from rapidfuzz import fuzz
except ImportError:  # fall back to stdlib
    from difflib import SequenceMatcher

    class _Fuzz:
        @staticmethod
        def ratio(a, b): return SequenceMatcher(None, a, b).ratio() * 100
        @staticmethod
        def partial_ratio(a, b):
            if not a or not b: return 0.0
            short, long = (a, b) if len(a) <= len(b) else (b, a)
            return SequenceMatcher(None, short, long).find_longest_match(
                0, len(short), 0, len(long)
            ).size / len(short) * 100
    fuzz = _Fuzz()


# ---------- Regex caches ----------
_HTML_TAG_RE  = re.compile(r'<[^>]+>')
_WS_RE        = re.compile(r'\s+')
_PUNCT_RE     = re.compile(r'[^\w\s]')


# ---------- Config ----------
@dataclass
class MatchingConfig:
    strip_html: bool = True
    decode_entities: bool = True
    unicode_normalize: bool = True
    lowercase: bool = True
    collapse_whitespace: bool = True
    strip_punct: bool = False
    use_substring: bool = True
    use_token_subset: bool = True
    use_prefix_match: bool = True
    use_fuzzy: bool = False
    fuzzy_threshold: float = 99.0
    postprocessor_fuzzy_threshold: float = 99.0
    min_candidate_len: int = 1
    require_word_boundary: bool = True


# ---------- Helpers ----------
def is_not_null(val: Any) -> bool:
    if val is None: return False
    if isinstance(val, str) and val.strip().lower() in ("", "none", "null", "<null>"): return False
    return True

def _is_yes_no(val: Any) -> bool:
    """Check if a value is strictly 'yes' or 'no' (case-insensitive, ignoring trailing punctuation)."""
    if not is_not_null(val):
        return False
    s = str(val).strip().lower()
    s = re.sub(r'[.!?]+$', '', s).strip()
    return s in ("yes", "no")

def strip_html_tags(text: str) -> str:
    return _HTML_TAG_RE.sub(' ', text) if text else ""

def decode_html_entities(text: str) -> str:
    return html.unescape(text) if text else ""

def normalize_text(text: Any, cfg: MatchingConfig, strip_html: Optional[bool] = None) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if cfg.unicode_normalize:
        text = unicodedata.normalize("NFKC", text)
    _strip_html = cfg.strip_html if strip_html is None else strip_html
    if _strip_html:
        text = strip_html_tags(text)
    if cfg.decode_entities:
        text = decode_html_entities(text)
    if cfg.lowercase:
        text = text.lower()
    if cfg.strip_punct:
        text = _PUNCT_RE.sub(' ', text)
    if cfg.collapse_whitespace:
        text = _WS_RE.sub(' ', text).strip()
    return text

def normalize_candidates(gt: Any) -> List[str]:
    if gt is None: return []
    s = str(gt).strip()
    return [s] if s and s.lower() != "<null>" else []

def fuzzy_ratio(a: str, b: str, partial: bool = False) -> float:
    if not a or not b: return 0.0
    return (fuzz.partial_ratio(a, b) if partial else fuzz.ratio(a, b))

def _bounded_contains(candidate: str, text: str) -> bool:
    if not candidate or not text: return False
    pattern = r'(?<![0-9A-Za-z])' + re.escape(candidate) + r'(?![0-9A-Za-z])'
    return re.search(pattern, text) is not None

def match_in_text(gt: Any, text_content: Optional[str], cfg: MatchingConfig) -> Tuple[bool, str, float]:
    if not text_content: return False, "no_text", 0.0
    candidates = normalize_candidates(gt)
    if not candidates: return False, "no_candidates", 0.0

    text_raw  = str(text_content)
    text_norm = normalize_text(text_raw, cfg)
    text_tokens = set(text_norm.split()) if cfg.use_token_subset else set()

    best_score = 0.0
    best_type  = "no_match"

    for cand in candidates:
        cand_norm = normalize_text(cand, cfg, strip_html=False)
        if not cand_norm: continue

        if cfg.use_substring:
            if _bounded_contains(cand, text_raw) if cfg.require_word_boundary else cand in text_raw:
                return True, "exact_raw", 100.0
            if _bounded_contains(cand_norm, text_norm) if cfg.require_word_boundary else cand_norm in text_norm:
                return True, "normalized_substring", 100.0

        if cfg.use_token_subset:
            cand_tokens = cand_norm.split()
            if cand_tokens and all(t in text_tokens for t in cand_tokens):
                if 100.0 > best_score:
                    best_score, best_type = 100.0, "token_subset"

        if cfg.use_prefix_match:
            for chunk in re.split(r'[|•·\n\r;]+', text_norm):
                chunk = chunk.strip()
                if not chunk: continue
                if len(cand_norm) >= cfg.min_candidate_len and (chunk.startswith(cand_norm) or cand_norm.startswith(chunk)):
                    overlap = min(len(chunk), len(cand_norm)) / max(len(chunk), len(cand_norm)) * 100
                    if overlap > best_score:
                        best_score, best_type = overlap, "prefix_match"

        if cfg.use_fuzzy and len(cand_norm) >= cfg.min_candidate_len and text_norm:
            score = fuzzy_ratio(cand_norm, text_norm, partial=True)
            if score > best_score:
                best_score, best_type = score, f"fuzzy_{score:.0f}"
            if score >= cfg.fuzzy_threshold:
                return True, f"fuzzy_{score:.0f}", score

    matched = best_score >= cfg.fuzzy_threshold or best_type in {"exact_raw", "normalized_substring", "token_subset", "prefix_match"}
    return matched, best_type, best_score

def values_match(a: Any, b_candidates: List[str], cfg: MatchingConfig, threshold: Optional[float] = None) -> bool:
    if not a: return False
    a_str = str(a).strip()
    if not a_str or a_str == "<NULL>": return False
    
    thr = threshold if threshold is not None else cfg.postprocessor_fuzzy_threshold
    a_norm = normalize_text(a_str, cfg, strip_html=False)
    
    for cand in b_candidates:
        cand_norm = normalize_text(cand, cfg, strip_html=False)
        if not cand_norm: continue
        if a_str == cand or a_norm == cand_norm:
            return True
        if len(a_norm) >= cfg.min_candidate_len:
            score = fuzzy_ratio(a_norm, cand_norm, partial=False)
            if score >= thr:
                return True
    return False

def _parse_raw_response(raw_resp: Any) -> Optional[str]:
    """Extract 'answer' from raw JSON response, handling markdown code blocks."""
    if not raw_resp: return None
    s = str(raw_resp).strip()
    if s.startswith("```json"): s = s[7:]
    elif s.startswith("```"): s = s[3:]
    if s.endswith("```"): s = s[:-3]
    try:
        parsed = json.loads(s.strip())
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed["answer"]
    except Exception:
        pass
    return None


# ---------- Classification ----------
def classify_error(record: Dict[str, Any], cfg: MatchingConfig) -> Tuple[str, Dict[str, Any]]:
    debug: Dict[str, Any] = {"id": record.get("id", "unknown")}

    gt_raw = record.get("ground_truth", "")
    gt_is_null = not is_not_null(gt_raw) or str(gt_raw).strip() == "" or str(gt_raw).strip().lower() == "<null>"

    step_logs = record.get("step_logs", {}) if isinstance(record.get("step_logs"), dict) else {}
    extractor_log = step_logs.get("extractor", {}) if isinstance(step_logs.get("extractor"), dict) else {}
    postprocessor_log = step_logs.get("postprocessor", {}) if isinstance(step_logs.get("postprocessor"), dict) else {}

    original_extracted = _parse_raw_response(extractor_log.get("raw_response"))
    postprocessed_value = _parse_raw_response(postprocessor_log.get("raw_response"))
    
    # Fallback to prediction field if postprocessor didn't yield anything
    if not postprocessed_value:
        postprocessed_value = record.get("prediction")

    debug["original_extracted"] = str(original_extracted) if original_extracted else ""
    debug["postprocessed_value"] = str(postprocessed_value) if postprocessed_value else ""
    debug["gt"] = str(gt_raw) if gt_raw else ""

    # --- NEW: Skip Yes/No boolean answers ---
    if _is_yes_no(gt_raw) or _is_yes_no(original_extracted) or _is_yes_no(postprocessed_value):
        return "Skipped (Yes/No)", debug

    if gt_is_null:
        if original_extracted or postprocessed_value:
            return "Hallucination", debug
        return "Investigate", debug

    candidates = normalize_candidates(gt_raw)

    # 1. GXR Error: extractor matched, but postprocessor/prediction broke it
    if values_match(original_extracted, candidates, cfg):
        if not values_match(postprocessed_value, candidates, cfg):
            return "GXR Error", debug

    # 2. Check filtered_html (Extractor Error)
    filtered_html = record.get("filtered_html")
    found_filtered, mt_filtered, sc_filtered = match_in_text(gt_raw, filtered_html, cfg)
    debug["found_in_filtered"] = found_filtered
    debug["match_type_filtered"] = mt_filtered
    debug["score_filtered"] = round(sc_filtered, 1)

    if found_filtered:
        debug["extractor_had_wrong_value"] = bool(
            original_extracted and not values_match(original_extracted, candidates, cfg)
        )
        return "Extractor Error", debug

    # 3. Check preprocessed_content (Pruner Error)
    pre_content = record.get("preprocessed_content")
    found_pre, mt_pre, sc_pre = match_in_text(gt_raw, pre_content, cfg)
    debug["found_in_preprocessed"] = found_pre
    debug["match_type_preprocessed"] = mt_pre
    debug["score_preprocessed"] = round(sc_pre, 1)

    if found_pre:
        return "Pruner Error", debug

    return "Investigate", debug


# ---------- Main entry ----------
def run_funnel_analysis(ndjson_path: str, cfg: Optional[MatchingConfig] = None, verbose: bool = True) -> pl.DataFrame:
    cfg = cfg or MatchingConfig()
    print(f"Reading and analyzing errors in: {ndjson_path}...")
    print(f"Matching config: {asdict(cfg)}")

    results: List[Dict[str, Any]] = []

    with open(ndjson_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str: continue
            record = json.loads(line_str)
            record_id = record.get("id", "unknown")
            evaluation = record.get("evaluation", {})
            
            score = evaluation.get("f1")
            if score is None: score = evaluation.get("recall")
            if score is None: score = evaluation.get("score")

            if score is None or score >= 1.0:
                continue

            error_type, debug = classify_error(record, cfg)

            results.append({
                "id": record_id,
                "score": score,
                "ground_truth": debug.get("gt", record.get("ground_truth", "")),
                "original_extracted": debug.get("original_extracted", ""),
                "postprocessed_value": debug.get("postprocessed_value", record.get("prediction", "")),
                "error_classification": error_type,
                "extractor_had_wrong_value": debug.get("extractor_had_wrong_value", False),
                "found_in_filtered": debug.get("found_in_filtered", False),
                "match_type_filtered": debug.get("match_type_filtered", ""),
                "score_filtered": debug.get("score_filtered", 0.0),
                "found_in_preprocessed": debug.get("found_in_preprocessed", False),
                "match_type_preprocessed": debug.get("match_type_preprocessed", ""),
                "score_preprocessed": debug.get("score_preprocessed", 0.0),
            })

    df_results = pl.DataFrame(results)

    if df_results.height > 0 and verbose:
        summary = (df_results
                   .group_by("error_classification")
                   .len()
                   .sort("error_classification"))
        print("\n--- Error Summary ---")
        print(summary)
        print("---------------------\n")
    else:
        print("No errors (scores < 1) found in the dataset.")

    return df_results


if __name__ == "__main__":
    test_path = "sample_eval.ndjson"  # Change this to your file path
    if os.path.exists(test_path):
        df = run_funnel_analysis(test_path)
        print(df.head(20))
    else:
        print("Please specify a valid ndjson file path.")