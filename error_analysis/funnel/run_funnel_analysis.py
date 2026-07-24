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
    # FIX: require substring matches to fall on word/number boundaries.
    # See `_bounded_contains` for the rationale — this closes a real
    # false-positive hole where e.g. gt "9.00" was reported as "found"
    # inside "109.00", or "red" inside "stored".
    require_word_boundary: bool = True


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
    """Return raw (un-normalized) string candidates from gt.

    FIX: also flattens one level of nested list/tuple values, so a
    multi-value field stored as e.g. [["a", "b"]] contributes "a" and "b"
    as separate candidates instead of being silently dropped or stringified.
    """
    if not is_not_null(gt):
        return []
    cands = gt if isinstance(gt, (list, tuple)) else [gt]
    out = []
    for c in cands:
        sub_items = c if isinstance(c, (list, tuple)) else [c]
        for item in sub_items:
            if not is_not_null(item):
                continue
            s = str(item).strip()
            if s in ("", "<NULL>"):
                continue
            out.append(s)
    return out


# ---------- Matching ----------
def fuzzy_ratio(a: str, b: str, partial: bool = False) -> float:
    if not a or not b:
        return 0.0
    return (fuzz.partial_ratio(a, b) if partial else fuzz.ratio(a, b))


def _bounded_contains(candidate: str, text: str) -> bool:
    """
    Substring containment that requires the match to sit on a word/number
    boundary (the characters immediately before/after the match, if any,
    must not be alphanumeric).

    FIX for a real false-positive bug: without this, matching gt "9.00" is
    reported as "found" inside "109.00"/"29.00" (plain substring), and gt
    "red" is "found" inside "stored". On a dataset full of numeric prices
    this was silently turning real Pruner Error / Investigate cases into
    false Extractor Error classifications.

    Punctuation-adjacent matches are still fine (e.g. "$109.00" matching
    right after "$"), since only alphanumeric neighbors block the match.
    """
    if not candidate or not text:
        return False
    pattern = r'(?<![0-9A-Za-z])' + re.escape(candidate) + r'(?![0-9A-Za-z])'
    return re.search(pattern, text) is not None


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

        # 1) Exact raw substring (FIX: boundary-aware when cfg.require_word_boundary)
        if cfg.use_substring:
            raw_hit = (_bounded_contains(cand, text_raw) if cfg.require_word_boundary
                       else cand in text_raw)
            if raw_hit:
                return True, "exact_raw", 100.0

        # 2) Normalized substring (FIX: boundary-aware)
        if cfg.use_substring:
            norm_hit = (_bounded_contains(cand_norm, text_norm) if cfg.require_word_boundary
                        else cand_norm in text_norm)
            if norm_hit:
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


def _as_scalar_list(a: Any) -> List[str]:
    """Coerce a scalar-or-list value into a flat list of strings.

    FIX: `values_match` used to call `str(a)` directly, so a list-valued
    extraction like ["red"] became the literal string "['red']" and would
    almost never equal or fuzzy-match a plain candidate "red". Since this
    codebase's own ground-truth format is list-based, list-valued
    `original_extracted`/`value` entries are a realistic case, not an edge
    case — this was silently breaking GXR Error detection and mislabeling
    correct list-valued extractions as extraction failures.
    """
    if not is_not_null(a):
        return []
    items = a if isinstance(a, (list, tuple)) else [a]
    return [str(x).strip() for x in items if is_not_null(x)]


def values_match(
    a: Any,
    b_candidates: List[str],
    cfg: MatchingConfig,
    threshold: Optional[float] = None,
) -> bool:
    """Compare an extracted value against gt candidates (value-vs-value).

    `a` may be a scalar or a list/tuple (FIX: see `_as_scalar_list`); this
    returns True if ANY element of `a` matches ANY candidate.
    """
    a_items = _as_scalar_list(a)
    if not a_items:
        return False
    thr = threshold if threshold is not None else cfg.postprocessor_fuzzy_threshold
    for a_str in a_items:
        a_norm = normalize_text(a_str, cfg, strip_html=False)
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
    Hierarchy (four tiers + an "Investigate" catch-all for unexplained cases):
      Hallucination   - gt is NULL but the extractor actually predicted a
                        non-null value anyway.
      GXR Error       - the raw extractor output matched gt, but the
                        postprocessed/GXR-resolved value doesn't (the
                        extractor got it right; postprocessing broke it).
      Extractor Error - the correct value was present in the filtered text
                        handed to the extractor, but the extractor either
                        failed to extract it or extracted the wrong value.
                        (debug["extractor_had_wrong_value"] tells you which.)
      Pruner Error    - the value isn't in the filtered text, but it WAS
                        present before pruning (in preprocessed_content) -
                        the pruner incorrectly removed it.
      Investigate     - gt isn't found in filtered OR preprocessed content
                        at all, OR gt is null with no corresponding
                        predicted value (nothing to actually attribute as
                        a hallucination).

    CHANGELOG (fixes applied):
      - Hallucination previously returned before the extractor log was even read,
        so it could label a case "hallucination" without checking that
        anything was actually predicted. Now it verifies that.
      - match_in_text's substring checks previously had no word/number
        boundary awareness, so e.g. gt "9.00" matched inside "109.00".
        Fixed via `_bounded_contains` (toggle: cfg.require_word_boundary).
      - values_match previously stringified list-valued extractions (e.g.
        ["red"] -> "['red']"), which almost never matched a plain gt
        candidate. Fixed via `_as_scalar_list`.
      - The old "Extraction Error" label (a case where filtered text had
        the value but the extractor picked the wrong one) has been folded
        into "Extractor Error" to keep the funnel at four tiers as intended, with the
        finer distinction preserved in debug["extractor_had_wrong_value"]
        rather than fragmenting the top-line classification.
    """
    debug: Dict[str, Any] = {"key": key}

    # 1) Ground truth
    gt_raw = record.get("ground_truth", {})
    gt_dict = repair_and_parse(gt_raw) if isinstance(gt_raw, str) else gt_raw
    gt = gt_dict.get(key) if isinstance(gt_dict, dict) else gt_raw
    gt_is_null = not is_not_null(gt) or (isinstance(gt, list) and not any(is_not_null(x) for x in gt))

    # 2) Postprocessor / extractor log — fetched up front (FIX: previously
    # fetched only after the Hallucination early-return, so Hallucination could never
    # actually verify a value had been predicted).
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
    debug["original_extracted"] = str(original_extracted) if is_not_null(original_extracted) else ""
    debug["postprocessed_value"] = str(postprocessed_value) if is_not_null(postprocessed_value) else ""

    if gt_is_null:
        # FIX: only a genuine hallucination if the extractor actually
        # predicted something non-null. Otherwise there's nothing here to
        # attribute as a hallucination.
        if is_not_null(original_extracted) or is_not_null(postprocessed_value):
            return "Hallucination", debug
        return "Investigate", debug

    candidates = normalize_candidates(gt)
    debug["gt"] = candidates[0] if candidates else ""

    # GXR Error: original_extracted matched gt but postprocessor broke it
    if values_match(original_extracted, candidates, cfg):
        if not values_match(postprocessed_value, candidates, cfg):
            return "GXR Error", debug

    # 3) Check filtered_html
    filtered_html = record.get("filtered_html")
    found_filtered, mt_filtered, sc_filtered = match_in_text(gt, filtered_html, cfg)
    debug["found_in_filtered"] = found_filtered
    debug["match_type_filtered"] = mt_filtered
    debug["score_filtered"] = round(sc_filtered, 1)

    if found_filtered:
        # Was the extractor's raw output present-but-wrong, or absent
        # entirely? Kept as a debug flag (not a separate top-level type) so
        # the funnel summary stays at four tiers.
        debug["extractor_had_wrong_value"] = bool(
            is_not_null(original_extracted) and not values_match(original_extracted, candidates, cfg)
        )
        return "Extractor Error", debug

    # 4) Check preprocessed_content
    pre_content = record.get("preprocessed_content")
    found_pre, mt_pre, sc_pre = match_in_text(gt, pre_content, cfg)
    debug["found_in_preprocessed"] = found_pre
    debug["match_type_preprocessed"] = mt_pre
    debug["score_preprocessed"] = round(sc_pre, 1)

    if found_pre:
        return "Pruner Error", debug

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
                    "extractor_had_wrong_value": debug.get("extractor_had_wrong_value", False),
                    "found_in_filtered": debug.get("found_in_filtered", False),
                    "match_type_filtered": debug.get("match_type_filtered", ""),
                    "score_filtered": debug.get("score_filtered", 0.0),
                    "found_in_preprocessed": debug.get("found_in_preprocessed", False),
                    "match_type_preprocessed": debug.get("match_type_preprocessed", ""),
                    "score_preprocessed": debug.get("score_preprocessed", 0.0),
                })

    df_results = pl.DataFrame(results, strict=False)

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