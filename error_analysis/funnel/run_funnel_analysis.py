import os
import re
import ast
import json
import html
import unicodedata
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import polars as pl
from html_eval.util.eval_util import is_not_null, repair_and_parse

# ---------- Optional fuzzy deps ----------
try:
    from rapidfuzz import fuzz
    _HAS_RF = True
except ImportError:  # fall back to stdlib
    from difflib import SequenceMatcher
    _HAS_RF = False

    class _Fuzz:
        @staticmethod
        def ratio(a, b):     return SequenceMatcher(None, a, b).ratio() * 100
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
_NON_ALNUM_RE = re.compile(r'[^a-z0-9]+')


# ---------- Config ----------
@dataclass
class MatchingConfig:
    # Normalization
    strip_html: bool = True
    decode_entities: bool = True
    unicode_normalize: bool = True
    lowercase: bool = True
    collapse_whitespace: bool = True
    strip_punct: bool = False
    # Matching strategies
    use_substring: bool = True
    use_token_subset: bool = True
    use_prefix_match: bool = True
    use_fuzzy: bool = False
    fuzzy_threshold: float = 99.0            # for text search (partial_ratio)
    postprocessor_fuzzy_threshold: float = 99.0  # for value-vs-value (ratio)
    min_candidate_len: int = 1               # skip very short tokens for fuzzy


# ---------- Normalization ----------
def strip_html_tags(text: str) -> str:
    return _HTML_TAG_RE.sub(' ', text) if text else ""


def decode_html_entities(text: str) -> str:
    return html.unescape(text) if text else ""


def normalize_text(text: Any, cfg: MatchingConfig, strip_html: Optional[bool] = None) -> str:
    """Normalize a piece of text according to cfg."""
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
    """Return raw (un-normalized) string candidates from gt."""
    if not is_not_null(gt):
        return []
    cands = gt if isinstance(gt, (list, tuple)) else [gt]
    out = []
    for c in cands:
        if not is_not_null(c):
            continue
        s = str(c).strip()
        if s in ("", "<NULL>"):
            continue
        out.append(s)
    return out


# ---------- Matching ----------
def fuzzy_ratio(a: str, b: str, partial: bool = False) -> float:
    if not a or not b:
        return 0.0
    return (fuzz.partial_ratio(a, b) if partial else fuzz.ratio(a, b))


def match_in_text(
    gt: Any,
    text_content: Optional[str],
    cfg: MatchingConfig,
) -> Tuple[bool, str, float]:
    """
    Multi-strategy search of gt inside text_content.
    Returns (matched, match_type, best_score).
    """
    if not text_content:
        return False, "no_text", 0.0

    candidates = normalize_candidates(gt)
    if not candidates:
        return False, "no_candidates", 0.0

    text_raw  = str(text_content)
    text_norm = normalize_text(text_raw, cfg)
    text_tokens = set(text_norm.split()) if cfg.use_token_subset else set()

    best_score = 0.0
    best_type  = "no_match"

    for cand in candidates:
        cand_norm = normalize_text(cand, cfg, strip_html=False)
        if not cand_norm:
            continue

        # 1) Exact raw substring
        if cfg.use_substring and cand in text_raw:
            return True, "exact_raw", 100.0

        # 2) Normalized substring
        if cfg.use_substring and cand_norm in text_norm:
            return True, "normalized_substring", 100.0

        # 3) Token-subset (all gt tokens appear in text, any order)
        if cfg.use_token_subset:
            cand_tokens = cand_norm.split()
            if cand_tokens and all(t in text_tokens for t in cand_tokens):
                # compute coverage score (share of gt token chars covered)
                score = 100.0
                if score > best_score:
                    best_score, best_type = score, "token_subset"

        # 4) Prefix match: candidate is a prefix of some chunk, or vice versa
        if cfg.use_prefix_match:
            # split text_norm on natural separators
            for chunk in re.split(r'[|•·\n\r;]+', text_norm):
                chunk = chunk.strip()
                if not chunk:
                    continue
                if len(cand_norm) >= cfg.min_candidate_len and (
                    chunk.startswith(cand_norm) or cand_norm.startswith(chunk)
                ):
                    overlap = min(len(chunk), len(cand_norm)) / max(len(chunk), len(cand_norm)) * 100
                    if overlap > best_score:
                        best_score, best_type = overlap, "prefix_match"

        # 5) Fuzzy partial match
        if cfg.use_fuzzy and len(cand_norm) >= cfg.min_candidate_len and text_norm:
            score = fuzzy_ratio(cand_norm, text_norm, partial=True)
            if score > best_score:
                best_score, best_type = score, f"fuzzy_{score:.0f}"
            if score >= cfg.fuzzy_threshold:
                return True, f"fuzzy_{score:.0f}", score

    matched = best_score >= cfg.fuzzy_threshold or best_type in {
        "exact_raw", "normalized_substring", "token_subset", "prefix_match"
    }
    return matched, best_type, best_score


def values_match(
    a: Any,
    b_candidates: List[str],
    cfg: MatchingConfig,
    threshold: Optional[float] = None,
) -> bool:
    """Compare a single extracted value against gt candidates (value-vs-value)."""
    if not is_not_null(a):
        return False
    a_str = str(a).strip()
    a_norm = normalize_text(a_str, cfg, strip_html=False)
    thr = threshold if threshold is not None else cfg.postprocessor_fuzzy_threshold
    for cand in b_candidates:
        cand_norm = normalize_text(cand, cfg, strip_html=False)
        if not cand_norm:
            continue
        if a_str == cand or a_norm == cand_norm:
            return True
        # use ratio (not partial) so substring-via-partial doesn't fool us
        if len(a_norm) >= cfg.min_candidate_len:
            score = fuzzy_ratio(a_norm, cand_norm, partial=False)
            if score >= thr:
                return True
    return False


# ---------- Classification ----------
def classify_error(
    key: str,
    record: Dict[str, Any],
    cfg: MatchingConfig,
) -> Tuple[str, Dict[str, Any]]:
    """
    Hierarchy:
      Type 1            – Missing ground truth
      Type 2            – Postprocessor error (original_extracted matched gt but final didn't)
      Extraction Error  – gt present in filtered_html but original_extracted didn't match
      Type 3            – Filtering error (gt missing from filtered_html but in preprocessed)
      Type 4            – Preprocessing error (gt missing from preprocessed)
      Investigate       – gt not found anywhere
    """
    debug: Dict[str, Any] = {"key": key}

    # 1) Ground truth
    gt_raw = record.get("ground_truth", {})
    gt_dict = repair_and_parse(gt_raw) if isinstance(gt_raw, str) else gt_raw
    gt = gt_dict.get(key) if isinstance(gt_dict, dict) else gt_raw

    if not is_not_null(gt) or (isinstance(gt, list) and not any(is_not_null(x) for x in gt)):
        return "Type 1", debug

    candidates = normalize_candidates(gt)
    debug["gt"] = candidates[0] if candidates else ""

    # 2) Postprocessor log
    postprocessor = record.get("postprocessor")
    if not postprocessor and isinstance(record.get("step_logs"), dict):
        postprocessor = record["step_logs"].get("postprocessor")

    exact_match_log = {}
    if isinstance(postprocessor, dict):
        exact_match_log = postprocessor.get("exact_match_log", {}) or {}

    log_entry = exact_match_log.get(key, {})
    original_extracted = None
    postprocessed_value = None
    if isinstance(log_entry, dict):
        original_extracted = log_entry.get("original_extracted")
        postprocessed_value = log_entry.get("value")
        debug["original_extracted"] = str(original_extracted)
        debug["postprocessed_value"] = str(postprocessed_value)

    # Type 2: original_extracted matched gt but postprocessor broke it
    if values_match(original_extracted, candidates, cfg):
        if not values_match(postprocessed_value, candidates, cfg):
            return "Type 2", debug

    # 3) Check filtered_html
    filtered_html = record.get("filtered_html")
    found_filtered, mt_filtered, sc_filtered = match_in_text(gt, filtered_html, cfg)
    debug["found_in_filtered"] = found_filtered
    debug["match_type_filtered"] = mt_filtered
    debug["score_filtered"] = round(sc_filtered, 1)

    if found_filtered:
        # Disambiguate Extraction Error vs (legacy) Type 3
        if is_not_null(original_extracted) and not values_match(original_extracted, candidates, cfg):
            return "Extraction Error", debug
        return "Type 3", debug

    # 4) Check preprocessed_content
    pre_content = record.get("preprocessed_content")
    found_pre, mt_pre, sc_pre = match_in_text(gt, pre_content, cfg)
    debug["found_in_preprocessed"] = found_pre
    debug["match_type_preprocessed"] = mt_pre
    debug["score_preprocessed"] = round(sc_pre, 1)

    if found_pre:
        return "Type 4", debug

    return "Investigate", debug


# ---------- Main entry ----------
def run_funnel_analysis(
    ndjson_path: str,
    cfg: Optional[MatchingConfig] = None,
    verbose: bool = True,
) -> pl.DataFrame:
    cfg = cfg or MatchingConfig()
    print(f"Reading and analyzing errors in: {ndjson_path}...")
    print(f"Matching config: {asdict(cfg)}")

    results: List[Dict[str, Any]] = []

    with open(ndjson_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            record = json.loads(line_str)
            record_id = record.get("id", "unknown")
            evaluation = record.get("evaluation", {})

            # Pre-fetch commonly used substructures
            gt_raw = record.get("ground_truth", {})
            gt_dict = repair_and_parse(gt_raw) if isinstance(gt_raw, str) else gt_raw
            postprocessor = record.get("postprocessor") or (
                record.get("step_logs", {}).get("postprocessor", {}) if isinstance(record.get("step_logs"), dict) else {}
            )
            exact_match_log = postprocessor.get("exact_match_log", {}) if isinstance(postprocessor, dict) else {}

            for key, score in evaluation.items():
                if score is None or score >= 1:
                    continue

                error_type, debug = classify_error(key, record, cfg)

                gt = gt_dict.get(key) if isinstance(gt_dict, dict) else gt_raw
                log_entry = exact_match_log.get(key, {}) if isinstance(exact_match_log, dict) else {}
                original_extracted = log_entry.get("original_extracted") if isinstance(log_entry, dict) else None
                postprocessed_value = log_entry.get("value") if isinstance(log_entry, dict) else None

                results.append({
                    "id": record_id,
                    "key": key,
                    "score": score,
                    "ground_truth": str(gt),
                    "original_extracted": str(original_extracted) if is_not_null(original_extracted) else "",
                    "postprocessed_value": str(postprocessed_value) if is_not_null(postprocessed_value) else "",
                    "error_classification": error_type,
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
    test_path = "/home/abdo/PAPER/Eval/swde_auto/metric/page_level_f1_sample_eval.ndjson"
    if not os.path.exists(test_path):
        test_path = "websrc/metric/token_f1_sample_eval.ndjson"
    if os.path.exists(test_path):
        df = run_funnel_analysis(test_path)
        print(df.head(20))
    else:
        print("Please specify a valid ndjson file path.")