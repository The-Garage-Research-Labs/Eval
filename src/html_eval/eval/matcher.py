# master_evaluator.py
from __future__ import annotations
import re
import ast
from dataclasses import dataclass
from typing import Any, List, Optional
import html
import textwrap
from collections import Counter
from html_eval.util.eval_util import is_not_null
from html_eval.util.html_util import normalize_text

# Try to use rapidfuzz (faster) else fallback to fuzzywuzzy
try:
    from rapidfuzz import fuzz as _rfuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False
    try:
        from fuzzywuzzy import fuzz as _fwuzz  # type: ignore
    except Exception:
        _fwuzz = None

# --------------------------
# Configs / Utilities
# --------------------------
@dataclass
class MatcherConfig:
    is_fuzzy: bool = False
    fuzzy_threshold: int = 90  # 0-100

# --------------------------
# Matcher (exact / fuzzy)
# --------------------------
class Matcher:
    def __init__(self, cfg: Optional[MatcherConfig] = None):
        self.cfg = cfg or MatcherConfig()



    def _normalize_gt(self, gt: Any) -> List[Any]:
        """
        Turn GT into list of candidates:
        - If list -> return list filtered for non-null
        - If string begins with '[' parse literal -> list
        - Else single-item list

        Normalization includes:
        - HTML unescaping (&amp; -> &, &nbsp; -> ' ')
        - Dedenting multi-line text
        - Stripping leading/trailing whitespace
        - Collapsing multiple spaces
        """
        def clean_text(x: str) -> str:
            if not isinstance(x, str):
                return x
            x = textwrap.dedent(x)              # remove indentation
            x = html.unescape(x)                # decode &amp;, &lt;, etc.
            x = x.replace('\xa0', ' ')          # convert non-breaking spaces to normal spaces
            x = re.sub(r'\s+', ' ', x).strip()  # normalize whitespace
            return x

        # early exit if not null and not string-literal
        if not is_not_null(gt) and not (isinstance(gt, str) and gt.strip().startswith("[")):
            return []

        # parse GT into list
        if isinstance(gt, list):
            items = gt
        elif isinstance(gt, str):
            try:
                evaluated = ast.literal_eval(gt)
                if isinstance(evaluated, list):
                    items = evaluated
                else:
                    items = [evaluated]
            except Exception:
                items = [gt]
        else:
            items = [gt]

        # normalize each item and filter out nulls
        return [clean_text(it) for it in items if is_not_null(it)]

    def compare(self, gt: Any, pred: Any) -> bool:
        """
        Return True if pred matches any gt candidate.
        Exact match by default, fuzzy if configured.
        """
        # if not is_not_null(pred):
        #     return False
        candidates = self._normalize_gt(gt)
        if not candidates:
            if not is_not_null(pred):
                return True
            return False
        
        pred_s = normalize_text(str(pred))
        
        if self.cfg.is_fuzzy:
            threshold = max(0, min(100, self.cfg.fuzzy_threshold))
            for c in candidates:
                try:
                    cs = normalize_text(str(c))
                    if _HAS_RAPIDFUZZ:
                        score = _rfuzz.ratio(cs.lower(), pred_s.lower())
                    else:
                        if _fwuzz is None:
                            # no fuzzy lib installed -> fallback to simple equality
                            continue
                        score = _fwuzz.ratio(cs.lower(), pred_s.lower())
                    if score >= threshold:
                        return True
                except Exception:
                    continue
            return False
        else:

            # tmp_pred = pred_s
            
            # candidates = sorted(candidates, key=lambda x: len(str(x)), reverse=True)

            # for c in candidates:
            #     if isinstance(pred, (int, float)) and isinstance(c, (int, float)):
            #         if pred == c:
            #             return True
                
            #     tmp_c = normalize_text(str(c))
            #     if tmp_c in tmp_pred:
            #         tmp_pred = tmp_pred.replace(tmp_c, '')
            #         if not tmp_pred.strip():
            #             return True
                
                
            # return False

                        # Convert the prediction string into a Counter of tokens
            pred_counts = Counter(pred_s.split())
            
            candidates = sorted(candidates, key=lambda x: len(str(x)), reverse=True)

            for c in candidates:
                # Keep original numeric check
                if isinstance(pred, (int, float)) and isinstance(c, (int, float)):
                    if pred == c:
                        return True
                
                tmp_c = normalize_text(str(c))
                c_counts = Counter(tmp_c.split())
                
                # Check if the candidate tokens are fully contained within the current prediction counts
                # (pred_counts & c_counts) returns the intersection (min counts of common elements)
                if (pred_counts & c_counts) == c_counts:
                    # Subtract candidate tokens from the prediction counter
                    pred_counts = pred_counts - c_counts
                    
                    # Check if pred_counts is empty (sum of all counts is 0)
                    if sum(pred_counts.values()) == 0:
                        return True
                
            return False

