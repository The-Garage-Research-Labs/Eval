"""
web_analysis.py
Core processing engine for the WebSRC Error Analysis.
Incorporates ID parsing, DOM hierarchy parsing, layout taxonomy routing, and
heuristic-based funnel bottleneck classification.
"""

import re
import json
import html
import unicodedata
import pandas as pd
from bs4 import BeautifulSoup, Tag
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

# ---------- Fuzzy Match Fallback ----------
try:
    from rapidfuzz import fuzz
except ImportError:  # Standard library fallback
    from difflib import SequenceMatcher

    class _Fuzz:
        @staticmethod
        def ratio(a, b): 
            return SequenceMatcher(None, a, b).ratio() * 100
        @staticmethod
        def partial_ratio(a, b):
            if not a or not b: 
                return 0.0
            short, long_ = (a, b) if len(a) <= len(b) else (b, a)
            best = 0.0
            for i in range(len(long_) - len(short) + 1):
                score = SequenceMatcher(None, short, long_[i:i+len(short)]).ratio() * 100
                if score > best:
                    best = score
                    if best == 100.0:
                        break
            return best
    fuzz = _Fuzz()

# ---------- Regex Caches ----------
_HTML_TAG_RE  = re.compile(r'<[^>]+>')
_WS_RE        = re.compile(r'\s+')
_PUNCT_RE     = re.compile(r'[^\w\s]')

# ---------- WebSRC Core Maps ----------
DOMAIN_MAP = {
    "au": "Auto",
    "bo": "Book",
    "ca": "Camera",
    "ga": "Game",
    "ho": "Hotel",
    "jo": "Jobs",
    "mo": "Movie",
    "ph": "Phone",
    "re": "Restaurant",
    "sp": "Sports",
    "un": "University"
}

TAXONOMY_DOMAIN_MAP = {
    "au": "auto",
    "bo": "book",
    "ca": "camera",
    "ga": "game",
    "ho": "hotel",
    "jo": "jobs",
    "mo": "movie",
    "ph": "phone",
    "re": "restaurant",
    "sp": "sports",
    "un": "university"
}

TAXONOMY_MAP = {
    "auto": {
        "1": "KV",
        "2": "Table",
        "3": "KV",
        "8": "Compare",
        "9": "Table",
        "10": "Compare",
        "11": "KV",
        "12": "Compare",
        "14": "Compare",

    },
    "book": {
        "1": "Compare",
        "7": "Compare",
        "8": "KV",
        "9": "Compare",
        "10": "KV",
        "17": "Compare"
    },
    "camera": {
        "1": "KV",
        "2": "KV"
    },
    "game": {
        "1": "Compare",
        "6": "KV",
        "8": "KV",
        "9": "KV",
        "10": "Table",
        "12": "KV",
        "38": "KV",

    },
    "jobs": {
        "3": "Compare",
        "5": "KV",
        "10": "Compare",
        "11": "Compare",
        "12": "Table",
        "13": "Table"
    },
    "movie": {
        "2": "KV",
        "4": "KV",
        "6": "KV",
        "7": "KV",
        "8": "KV",
    },
    "phone": {
        "1": "KV",
        "2": "KV",
        "3": "KV",
        "4": "KV",
        "5": "KV"
    },
    "restaurant": {
        "2": "KV",
        "3": "KV",
        "5": "KV"
    },
    "sports": {
        "1": "KV",
        "2": "KV",
        "3": "KV",
        "4": "KV",
        "6": "KV",
        "7": "Table",
        "8": "Table",
        "9": "Table",
        "10": "Table",
        "11": "Table",
        "12": "Table",
        "13": "Table",
        "14": "Table",
        "15": "Table",
        "16": "Table",

    },
    "hotel": {
        "7": "Compare",
        "8": "Compare",
    
    },
    "university": {
        "2": "Table",
        "3": "KV",
        "4": "Compare",
        "6": "KV",
        "7": "KV",
        "8": "Compare",
        "9": "Table",
        "10": "Table",
        "11": "Table",
        "12": "Table"
    }
}

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

# ---------- Heuristic Extraction Helpers ----------
def is_not_null(val: Any) -> bool:
    if val is None: 
        return False
    if isinstance(val, str) and val.strip().lower() in ("", "none", "null", "<null>"): 
        return False
    return True

def _is_yes_no(val: Any) -> bool:
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
    if gt is None: 
        return []
    s = str(gt).strip()
    return [s] if s and s.lower() != "<null>" else []

def fuzzy_ratio(a: str, b: str, partial: bool = False) -> float:
    if not a or not b: 
        return 0.0
    return (fuzz.partial_ratio(a, b) if partial else fuzz.ratio(a, b))

def _bounded_contains(candidate: str, text: str) -> bool:
    if not candidate or not text: 
        return False
    pattern = r'(?<![0-9A-Za-z])' + re.escape(candidate) + r'(?![0-9A-Za-z])'
    return re.search(pattern, text) is not None

def match_in_text(gt: Any, text_content: Optional[str], cfg: MatchingConfig) -> Tuple[bool, str, float]:
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
                if not chunk: 
                    continue
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
    if not a: 
        return False
    a_str = str(a).strip()
    if not a_str or a_str == "<NULL>": 
        return False
    
    thr = threshold if threshold is not None else cfg.postprocessor_fuzzy_threshold
    a_norm = normalize_text(a_str, cfg, strip_html=False)
    
    for cand in b_candidates:
        cand_norm = normalize_text(cand, cfg, strip_html=False)
        if not cand_norm: 
            continue
        if a_str == cand or a_norm == cand_norm:
            return True
        if len(a_norm) >= cfg.min_candidate_len:
            score = fuzzy_ratio(a_norm, cand_norm, partial=False)
            if score >= thr:
                return True
    return False

def _parse_raw_response(raw_resp: Any) -> Optional[str]:
    if not raw_resp: 
        return None
    s = str(raw_resp).strip()
    if s.startswith("```json"): 
        s = s[7:]
    elif s.startswith("```"): 
        s = s[3:]
    if s.endswith("```"): 
        s = s[:-3]
    try:
        parsed = json.loads(s.strip())
        if isinstance(parsed, dict) and "answer" in parsed:
            return parsed["answer"]
    except Exception:
        pass
    return None

# ---------- Classifier Logic ----------
def classify_error(record: Dict[str, Any], cfg: MatchingConfig) -> Tuple[str, Dict[str, Any]]:
    debug: Dict[str, Any] = {"id": record.get("id", "unknown")}

    gt_raw = record.get("ground_truth", "")
    pred_raw = record.get("prediction", "")
    gt_is_null = not is_not_null(gt_raw) or str(gt_raw).strip() == "" or str(gt_raw).strip().lower() == "<null>"

    step_logs = record.get("step_logs", {}) if isinstance(record.get("step_logs"), dict) else {}
    extractor_log = step_logs.get("extractor", {}) if isinstance(step_logs.get("extractor"), dict) else {}
    postprocessor_log = step_logs.get("postprocessor", {}) if isinstance(step_logs.get("postprocessor"), dict) else {}

    original_extracted = _parse_raw_response(extractor_log.get("raw_response"))
    postprocessed_value = _parse_raw_response(postprocessor_log.get("raw_response"))
    
    if not postprocessed_value:
        postprocessed_value = pred_raw

    debug["original_extracted"] = str(original_extracted) if original_extracted else ""
    debug["postprocessed_value"] = str(postprocessed_value) if postprocessed_value else ""
    debug["gt"] = str(gt_raw) if gt_raw else ""

    if _is_yes_no(gt_raw) or _is_yes_no(original_extracted) or _is_yes_no(postprocessed_value):
        return "Skipped (Yes/No)", debug

    if gt_is_null:
        if original_extracted or postprocessed_value:
            return "Hallucination", debug
        return "Investigate", debug

    # Empty Prediction: GT exists but prediction is blank/null
    pred_str = str(pred_raw).strip() if pred_raw is not None else ""
    if not pred_str or pred_str.lower() in ("", "none", "null", "<null>"):
        return "Empty Prediction", debug

    candidates = normalize_candidates(gt_raw)

    # 1. GXR Error
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

def lookup_taxonomy(sample_id: str) -> str:
    if not isinstance(sample_id, str) or len(sample_id) < 4:
        return "Unknown"
    
    domain_code = sample_id[:2].lower()
    tax_domain_key = TAXONOMY_DOMAIN_MAP.get(domain_code)
    if not tax_domain_key:
        return "Unknown"
        
    website_code_raw = sample_id[2:4]
    try:
        website_idx_str = str(int(website_code_raw))
    except ValueError:
        return "Unknown"
        
    return TAXONOMY_MAP.get(tax_domain_key, {}).get(website_idx_str, "Unknown")

def parse_metadata(sample_id: str) -> dict:
    if not isinstance(sample_id, str) or len(sample_id) < 4:
        return {
            "domain": "Unknown",
            "domain_code": "unknown",
            "website": "Unknown",
            "website_code": "unknown",
            "page_id": "Unknown"
        }
    
    domain_code = sample_id[:2].lower()
    domain_name = DOMAIN_MAP.get(domain_code, f"Domain ({domain_code.upper()})")
    
    website_code = sample_id[2:4]
    website_name = f"Website {website_code}"
    
    page_id = sample_id[2:9] if len(sample_id) >= 9 else "Unknown"
    
    return {
        "domain": domain_name,
        "domain_code": domain_code,
        "website": website_name,
        "website_code": website_code,
        "page_id": page_id
    }

def classify_query_type(query_str: str, ground_truth: str = "") -> str:
    """
    Classifies WebSRC query pattern as 'Yes/No' or 'Extraction'.
    Detects yes/no indicators or closed question start patterns.
    """
    if not query_str:
        return "Unknown"
    
    gt_clean = str(ground_truth).strip().lower()
    if gt_clean in ["yes", "no", "true", "false"]:
        return "Yes/No"
    
    # Matches common English leading auxiliary or modal verbs (e.g. Is, Are, Can)
    yes_no_pattern = re.compile(
        r'^(is|was|are|were|do|does|did|can|could|should|would|will|has|have|had|if)\b', 
        re.IGNORECASE
    )
    if yes_no_pattern.match(query_str.strip()):
        return "Yes/No"
    
    return "Extraction"

def analyze_dom(html_content: str) -> dict:
    if not html_content or not isinstance(html_content, str):
        return {
            "max_depth": 0,
            "total_tags": 0,
            "tag_frequencies": {}
        }
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    tag_frequencies = {}
    for tag in soup.find_all():
        tag_frequencies[tag.name] = tag_frequencies.get(tag.name, 0) + 1
        
    total_tags = sum(tag_frequencies.values())
    
    def compute_depth(element):
        if not isinstance(element, Tag):
            return 0
        children = [child for child in element.children if isinstance(child, Tag)]
        if not children:
            return 1
        return 1 + max(compute_depth(child) for child in children)
    
    top_level_elements = [child for child in soup.children if isinstance(child, Tag)]
    max_depth = max([compute_depth(el) for el in top_level_elements]) if top_level_elements else 0
    
    return {
        "max_depth": max_depth,
        "total_tags": total_tags,
        "tag_frequencies": tag_frequencies
    }

# Column schema used to guarantee a consistent DataFrame even when no records parse.
_DF_COLUMNS = [
    "id", "domain", "domain_code", "website", "website_code", "page_id",
    "query", "ground_truth", "prediction", "f1", "precision", "recall",
    "query_type", "taxonomy", "max_dom_depth", "total_dom_tags", "is_correct",
    "error_classification", "extractor_had_wrong_value",
    "found_in_filtered", "match_type_filtered", "score_filtered",
    "found_in_preprocessed", "match_type_preprocessed", "score_preprocessed",
    "reranker_top_score",
]


def _extract_reranker_top_score(step_logs: dict) -> float:
    """Extract the highest reranker chunk score from step_logs."""
    reranker = step_logs.get("reranker", {})
    if not isinstance(reranker, dict):
        return 0.0
    chunks = reranker.get("chunks", [])
    if not chunks or not isinstance(chunks, list):
        return 0.0
    try:
        return max(float(c.get("score", 0.0)) for c in chunks if isinstance(c, dict))
    except (ValueError, TypeError):
        return 0.0


def compute_content_stats(records: list) -> pd.DataFrame:
    """
    Extract per-sample token/content statistics from raw log records.
    Returns a DataFrame with columns:
      id, raw_len, cleaned_len, filtered_len,
      raw_tokens, cleaned_tokens, filtered_tokens,
      num_chunks, reduction_ratio
    """
    rows = []
    for sample in records:
        if not isinstance(sample, dict):
            continue
        sample_id = sample.get("id", "")
        if not sample_id:
            continue

        step_logs = sample.get("step_logs", {})
        prep = step_logs.get("preprocessor", {}) if isinstance(step_logs, dict) else {}

        raw_len     = prep.get("raw_len", 0) or 0
        cleaned_len = prep.get("cleaned_len", 0) or 0
        num_chunks  = prep.get("num_chunks", 1) or 1

        filtered_html = sample.get("filtered_html", "") or ""
        filtered_len  = len(filtered_html)

        rows.append({
            "id":             sample_id,
            "raw_len":        raw_len,
            "cleaned_len":    cleaned_len,
            "filtered_len":   filtered_len,
            "raw_tokens":     raw_len // 4,
            "cleaned_tokens": cleaned_len // 4,
            "filtered_tokens":filtered_len // 4,
            "num_chunks":     num_chunks,
            "reduction_ratio":cleaned_len / max(raw_len, 1),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "id", "raw_len", "cleaned_len", "filtered_len",
            "raw_tokens", "cleaned_tokens", "filtered_tokens",
            "num_chunks", "reduction_ratio"
        ])
    return pd.DataFrame(rows)


def process_log_records(records: list) -> tuple:
    parsed_data = []
    raw_samples_dict = {}
    cfg = MatchingConfig()
    
    for sample in records:
        if not isinstance(sample, dict):
            continue
            
        sample_id = sample.get("id", "")
        if not sample_id:
            continue
            
        raw_samples_dict[sample_id] = sample
        
        meta = parse_metadata(sample_id)
        taxonomy_type = lookup_taxonomy(sample_id)
        
        query = sample.get("query", "")
        gt = sample.get("ground_truth", "")
        pred = sample.get("prediction", "")
        
        # Fallback: extract prediction from postprocessor/extractor raw_response answer key
        if not pred or (isinstance(pred, str) and pred.strip().lower() in ("", "none", "null", "<null>")):
            step_logs_raw = sample.get("step_logs", {}) if isinstance(sample.get("step_logs"), dict) else {}
            post_log = step_logs_raw.get("postprocessor", {}) if isinstance(step_logs_raw.get("postprocessor"), dict) else {}
            ext_log = step_logs_raw.get("extractor", {}) if isinstance(step_logs_raw.get("extractor"), dict) else {}
            extracted_pred = _parse_raw_response(post_log.get("raw_response"))
            if extracted_pred is None:
                extracted_pred = _parse_raw_response(ext_log.get("raw_response"))
            if extracted_pred is not None:
                pred = str(extracted_pred)
        
        evaluation = sample.get("evaluation", {})
        f1 = evaluation.get("f1", 0.0)
        precision = evaluation.get("precision", 0.0)
        recall = evaluation.get("recall", 0.0)
        
        query_type = classify_query_type(query, gt)
        
        preprocessed_html = sample.get("preprocessed_content", "")
        dom_stats = analyze_dom(preprocessed_html)

        step_logs = sample.get("step_logs", {}) if isinstance(sample.get("step_logs"), dict) else {}
        reranker_top_score = _extract_reranker_top_score(step_logs)
        
        # Determine Error Category
        if f1 >= 0.99:
            error_class = "Success"
            debug_info = {}
        elif 0.0 < f1 < 0.99:
            # Partial Match: prediction has some overlap with ground truth
            error_class, debug_info = classify_error(sample, cfg)
            # Sub-label the pipeline stage while preserving that this is partial
            error_class = f"Partial Match ({error_class})"
        else:
            error_class, debug_info = classify_error(sample, cfg)
            
        parsed_data.append({
            "id": sample_id,
            "domain": meta["domain"],
            "domain_code": meta["domain_code"],
            "website": meta["website"],
            "website_code": meta["website_code"],
            "page_id": meta["page_id"],
            "query": query,
            "ground_truth": gt,
            "prediction": pred,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "query_type": query_type,
            "taxonomy": taxonomy_type,
            "max_dom_depth": dom_stats["max_depth"],
            "total_dom_tags": dom_stats["total_tags"],
            "tag_frequencies": dom_stats["tag_frequencies"],
            "is_correct": 1.0 if f1 >= 0.99 else 0.0,
            
            # Diagnostic Fields
            "error_classification": error_class,
            "extractor_had_wrong_value": debug_info.get("extractor_had_wrong_value", False),
            "found_in_filtered": debug_info.get("found_in_filtered", False),
            "match_type_filtered": debug_info.get("match_type_filtered", ""),
            "score_filtered": debug_info.get("score_filtered", 0.0),
            "found_in_preprocessed": debug_info.get("found_in_preprocessed", False),
            "match_type_preprocessed": debug_info.get("match_type_preprocessed", ""),
            "score_preprocessed": debug_info.get("score_preprocessed", 0.0),
            "reranker_top_score": reranker_top_score,
        })
    
    # Guard: return a DataFrame with expected columns even when no records parsed
    if not parsed_data:
        df = pd.DataFrame(columns=_DF_COLUMNS)
    else:
        df = pd.DataFrame(parsed_data)
    return df, raw_samples_dict