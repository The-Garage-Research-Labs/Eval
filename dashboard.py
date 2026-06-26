import streamlit as st
import pandas as pd
import json
import ast
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional
import numpy as np
from collections import Counter, defaultdict
import difflib
import re
import string
import textwrap
import html as html_lib

# -----------------------------------------------------------------------------
# Configuration & Styles
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Eval Inspector Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def apply_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* Global */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 98%;
        }

        /* Nav buttons */
        .nav-container {
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 0.5rem 0;
        }
        .nav-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 8px 20px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.2s ease;
            letter-spacing: 0.3px;
        }
        .nav-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        .nav-btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none;
        }
        .nav-counter {
            font-weight: 700;
            font-size: 16px;
            padding: 6px 16px;
            background: var(--secondary-background-color);
            border-radius: 8px;
            border: 1px solid rgba(102, 126, 234, 0.3);
            min-width: 80px;
            text-align: center;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: var(--secondary-background-color);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            padding: 16px 20px;
        }
        div[data-testid="stMetric"] label {
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-size: 11px !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-weight: 800 !important;
            font-size: 28px !important;
        }

        /* Status pills */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 13px;
            letter-spacing: 0.3px;
        }
        .status-match {
            background: rgba(0, 200, 83, 0.12);
            color: #00c853;
            border: 1px solid rgba(0, 200, 83, 0.25);
        }
        .status-mismatch {
            background: rgba(255, 75, 75, 0.12);
            color: #ff4b4b;
            border: 1px solid rgba(255, 75, 75, 0.25);
        }
        .status-missing {
            background: rgba(255, 145, 0, 0.12);
            color: #ff9100;
            border: 1px solid rgba(255, 145, 0, 0.25);
        }
        .status-partial {
            background: rgba(0, 176, 255, 0.12);
            color: #00b0ff;
            border: 1px solid rgba(0, 176, 255, 0.25);
        }

        /* Diff blocks */
        .diff-container {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 13px;
            line-height: 1.6;
            background: var(--secondary-background-color);
            border-radius: 10px;
            padding: 16px;
            border: 1px solid rgba(255,255,255,0.06);
            overflow-x: auto;
        }
        .diff-add { color: #00c853; background: rgba(0,200,83,0.08); }
        .diff-del { color: #ff4b4b; background: rgba(255,75,75,0.08); }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 8px 20px;
            font-weight: 600;
        }

        /* Score bar */
        .score-bar-bg {
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.06);
            border-radius: 4px;
            overflow: hidden;
        }
        .score-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }

        /* Expander headers */
        .streamlit-expanderHeader {
            font-weight: 600 !important;
        }
        
        /* iframes */
        iframe {
            border: 1px solid var(--border-color) !important;
            border-radius: 10px;
        }
        
        /* Keyboard shortcut hint */
        .kbd-hint {
            display: inline-block;
            padding: 2px 8px;
            background: var(--secondary-background-color);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 5px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            font-weight: 600;
            color: rgba(255,255,255,0.6);
            margin: 0 2px;
        }
    </style>
    """, unsafe_allow_html=True)


apply_custom_css()

# -----------------------------------------------------------------------------
# Data Loading & Parsing Logic
# -----------------------------------------------------------------------------


@st.cache_data
def parse_ndjson(file) -> List[Dict[str, Any]]:
    data = []
    try:
        content = file.getvalue().decode("utf-8").strip().split('\n')
        for line in content:
            if not line.strip():
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                try:
                    data.append(ast.literal_eval(line))
                except Exception:
                    continue
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return data


@st.cache_data
def load_ndjson_from_path(path: str) -> List[Dict[str, Any]]:
    """Load NDJSON directly from a filesystem path (for auto-discovery)."""
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    try:
                        data.append(ast.literal_eval(line))
                    except Exception:
                        continue
    except Exception as e:
        st.error(f"Error reading file {path}: {e}")
    return data


def squad_normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lower, remove punctuation, articles, extra whitespace.
    Copied from html_eval/util/eval_util.py to avoid import."""
    if s is None:
        return ""
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text: str) -> str:
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def clean_gt_text(x: str) -> str:
    """Normalize a single GT/pred string the same way matcher._normalize_gt does.
    Copied from html_eval/eval/matcher.py to avoid import.
    - HTML unescaping (&amp; -> &, &nbsp; -> ' ')
    - Dedenting multi-line text
    - Converting non-breaking spaces to normal spaces
    - Collapsing multiple spaces and stripping
    """
    if not isinstance(x, str):
        return x
    x = textwrap.dedent(x)          # remove indentation
    x = html_lib.unescape(x)        # decode &amp;, &lt;, etc.
    x = x.replace('\xa0', ' ')      # non-breaking space -> normal space
    x = re.sub(r'\s+', ' ', x).strip()  # normalize whitespace
    return x


def is_null_value(v: Any) -> bool:
    """Return True if v represents a null/missing value.
    Handles: None, NaN, '<NULL>', 'MISSING', 'none', 'null', 'n/a', '', 'na', etc.
    Case-insensitive. Mirrors eval_util.is_not_null logic.
    """
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        return len(v) == 0
    s = str(v).strip()
    # Check raw form first (catches <NULL>, MISSING, etc.)
    if s.upper() in ('<NULL>', 'MISSING', 'NULL', 'NONE', 'N/A', 'NA', ''):
        return True
    # Check after squad normalization (strips punctuation like < >)
    return squad_normalize_answer(s) in ('null', 'missing', 'none', 'na', '')


def normalize_gt_candidates(gt: Any) -> List[str]:
    """Turn a GT value into a list of cleaned candidate strings.
    Mirrors matcher._normalize_gt logic.
    """
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

    return [clean_gt_text(it) for it in items if not is_null_value(it)]


def compute_fuzzy_score(gt: str, pred: str) -> float:
    """Compute normalized fuzzy similarity ratio between ground truth and prediction.
    Both sides are cleaned via clean_gt_text (HTML unescape, dedent, whitespace collapse)
    then normalized via squad_normalize_answer (lower, remove punc/articles).
    """
    if not isinstance(gt, str):
        gt = str(gt)
    if not isinstance(pred, str):
        pred = str(pred)

    gt_norm = squad_normalize_answer(clean_gt_text(gt))
    pred_norm = squad_normalize_answer(clean_gt_text(pred))

    # Post-normalize null forms: angle brackets are stripped by squad_normalize,
    # so '<NULL>' -> 'null', 'n/a' -> 'na', 'MISSING' -> 'missing'
    NULL_NORM = ('null', 'missing', 'none', 'na', '')
    if gt_norm in NULL_NORM and pred_norm in NULL_NORM:
        return 1.0 if gt_norm == pred_norm else 0.0

    return difflib.SequenceMatcher(None, gt_norm, pred_norm).ratio()


def safe_parse_gt(gt_val):
    if isinstance(gt_val, (dict, list)):
        return gt_val
    try:
        return json.loads(gt_val)
    except Exception:
        try:
            return ast.literal_eval(gt_val)
        except Exception:
            return gt_val


def detect_file_type(record: Dict) -> str:
    eval_keys = record.get("evaluation", {}).keys()
    if any(k in eval_keys for k in ["f1", "precision", "recall"]):
        return "Type 2: Text Generation (QA)"
    return "Type 1: Structured Extraction (Schema)"


def merge_predictions_with_metrics(predictions_path: str, metrics_path: str) -> List[Dict[str, Any]]:
    """Merge predictions (with step_logs) and metric files (with evaluation)."""
    preds = load_ndjson_from_path(predictions_path)
    metrics = load_ndjson_from_path(metrics_path)
    
    # Index metrics by id
    metrics_by_id = {m['id']: m for m in metrics}
    
    merged = []
    for pred in preds:
        record_id = pred.get('id')
        metric = metrics_by_id.get(record_id, {})
        # Start with prediction data (has step_logs, preprocessed_content, content)
        combined = {**pred}
        # Overlay evaluation from metrics
        if 'evaluation' in metric:
            combined['evaluation'] = metric['evaluation']
        # Overlay filtered_html from metrics if not in pred
        if 'filtered_html' not in combined and 'filtered_html' in metric:
            combined['filtered_html'] = metric['filtered_html']
        merged.append(combined)
    
    return merged


def discover_output_dirs() -> Dict[str, Dict[str, str]]:
    """Auto-discover output directories with predictions and metrics."""
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    results = {}
    
    for entry in sorted(os.listdir(base)):
        full = os.path.join(base, entry)
        if not os.path.isdir(full):
            continue
        pred_path = os.path.join(full, 'predictions.ndjson')
        if not os.path.exists(pred_path):
            continue
        
        # Find metric file
        metric_path = None
        metric_dir = os.path.join(full, 'metric')
        if os.path.isdir(metric_dir):
            for mf in os.listdir(metric_dir):
                if mf.endswith('.ndjson'):
                    metric_path = os.path.join(metric_dir, mf)
                    break
        
        results_json = os.path.join(full, 'results.json')
        results[entry] = {
            'predictions': pred_path,
            'metrics': metric_path,
            'results': results_json if os.path.exists(results_json) else None,
            'config': os.path.join(full, 'experiment_config.json') if os.path.exists(os.path.join(full, 'experiment_config.json')) else None,
        }
    
    return results

# -----------------------------------------------------------------------------
# Error Analysis Engine
# -----------------------------------------------------------------------------

class ErrorAnalyzer:
    """Deep error analysis for structured extraction results."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._compute_metrics()
    
    def _compute_metrics(self):
        """Pre-compute all error metrics."""
        if 'evaluation' not in self.df.columns:
            return
        
        eval_df = pd.json_normalize(self.df['evaluation'])
        eval_df.index = self.df.index
        self.eval_df = eval_df
        self.fields = list(eval_df.columns)
        self.n_records = len(self.df)
        
        # Per-field accuracy
        self.field_accuracy = eval_df.mean().to_dict()
        
        # Per-record error count
        self.df['error_count'] = eval_df.apply(lambda x: (x == 0).sum(), axis=1).values
        self.df['total_fields'] = len(self.fields)
        self.df['accuracy'] = eval_df.mean(axis=1).values
        
        # Extract source site from ID (e.g., "aol_0000" -> "aol")
        self.df['source_site'] = self.df['id'].apply(lambda x: '_'.join(str(x).split('_')[:-1]) if '_' in str(x) else str(x))
        
        # Compute fuzzy matching scores
        fuzzy_scores = {}
        for field in self.fields:
            field_scores = []
            for _, row in self.df.iterrows():
                gt = safe_parse_gt(row.get('ground_truth', {}))
                pred = row.get('prediction', {})
                
                gt_val = gt.get(field, 'MISSING') if isinstance(gt, dict) else 'MISSING'
                pred_val = pred.get(field, 'MISSING') if isinstance(pred, dict) else 'MISSING'
                
                if isinstance(gt_val, list):
                    display_gt = clean_gt_text(gt_val[0]) if gt_val else 'MISSING'
                else:
                    display_gt = clean_gt_text(str(gt_val))
                
                display_pred = str(pred_val) if pred_val is not None else 'null'
                
                field_scores.append(compute_fuzzy_score(display_gt, display_pred))
            fuzzy_scores[field] = field_scores
        
        self.fuzzy_scores_df = pd.DataFrame(fuzzy_scores, index=self.df.index)
        self.field_fuzzy = self.fuzzy_scores_df.mean().to_dict()
        
        # Error patterns: which field combinations fail together
        self.error_patterns = self._compute_error_patterns(eval_df)
        
        # Per-site accuracy
        self.site_accuracy = {}
        for site, group in self.df.groupby('source_site'):
            site_eval = eval_df.loc[group.index]
            site_fuzzy = self.fuzzy_scores_df.loc[group.index]
            self.site_accuracy[site] = {
                'overall': site_eval.values.mean(),
                'fields': site_eval.mean().to_dict(),
                'fuzzy_overall': site_fuzzy.values.mean(),
                'fuzzy_fields': site_fuzzy.mean().to_dict(),
                'count': len(group)
            }
    
    def _compute_error_patterns(self, eval_df: pd.DataFrame) -> List[Dict]:
        """Find co-occurring error patterns."""
        patterns = Counter()
        for _, row in eval_df.iterrows():
            failed = tuple(sorted(col for col in eval_df.columns if row[col] == 0))
            if failed:
                patterns[failed] += 1
        
        result = []
        for fields, count in patterns.most_common(15):
            result.append({
                'fields': list(fields),
                'count': count,
                'pct': count / len(eval_df) * 100
            })
        return result
    
    def get_field_confusion(self, field: str) -> Dict:
        """Analyze error characteristics for a specific field."""
        errors = self.df[self.eval_df[field] == 0]
        analysis = {
            'total_errors': len(errors),
            'error_rate': 1 - self.field_accuracy.get(field, 0),
            'error_by_site': {},
            'common_issues': []
        }
        
        for site, group in errors.groupby('source_site'):
            analysis['error_by_site'][site] = len(group)
        
        # Analyze prediction vs ground truth patterns
        for _, row in errors.head(20).iterrows():
            gt = safe_parse_gt(row.get('ground_truth', {}))
            pred = row.get('prediction', {})
            
            gt_val = gt.get(field, 'N/A') if isinstance(gt, dict) else 'N/A'
            pred_val = pred.get(field, 'N/A') if isinstance(pred, dict) else 'N/A'
            
            # Handle list ground truths 
            if isinstance(gt_val, list):
                gt_val = gt_val[0] if gt_val else 'N/A'
            
            issue = classify_error(str(gt_val), str(pred_val))
            analysis['common_issues'].append(issue)
        
        # Aggregate issues
        issue_counts = Counter(analysis['common_issues'])
        analysis['issue_breakdown'] = dict(issue_counts.most_common())
        
        return analysis


def classify_error(gt: str, pred: str) -> str:
    """Classify the type of extraction error."""
    gt_clean  = clean_gt_text(gt)  if isinstance(gt, str)  else str(gt)
    pred_clean = clean_gt_text(pred) if isinstance(pred, str) else str(pred)

    gt_null   = is_null_value(gt_clean)
    pred_null = is_null_value(pred_clean)

    if pred_null and not gt_null:
        return '🚫 Missing Extraction'
    if gt_null and not pred_null:
        return '👻 Hallucination (GT is null)'
    if gt_null and pred_null:
        return '✅ Both Null'

    # Normalize for comparison
    gt_norm   = squad_normalize_answer(gt_clean)
    pred_norm = squad_normalize_answer(pred_clean)

    if gt_norm == pred_norm:
        return '✅ Exact Match (eval bug?)'
    if gt_norm in pred_norm:
        return '📏 Over-extraction (superset)'
    if pred_norm in gt_norm:
        return '✂️ Under-extraction (subset)'

    ratio = difflib.SequenceMatcher(None, gt_norm, pred_norm).ratio()
    if ratio > 0.7:
        return '🔄 Near-miss (formatting)'
    if ratio > 0.4:
        return '🎯 Partial overlap'

    return '❌ Completely wrong'


def generate_text_diff(gt: str, pred: str) -> str:
    """Generate an HTML diff between ground truth and prediction."""
    if not gt or not pred:
        return ""
    
    d = difflib.unified_diff(
        gt.splitlines(keepends=True),
        pred.splitlines(keepends=True),
        fromfile='Ground Truth',
        tofile='Prediction',
        lineterm=''
    )
    
    diff_lines = []
    for line in d:
        if line.startswith('+') and not line.startswith('+++'):
            diff_lines.append(f'<span class="diff-add">{_escape_html(line)}</span>')
        elif line.startswith('-') and not line.startswith('---'):
            diff_lines.append(f'<span class="diff-del">{_escape_html(line)}</span>')
        elif line.startswith('@@'):
            diff_lines.append(f'<span style="color:#888">{_escape_html(line)}</span>')
        else:
            diff_lines.append(_escape_html(line))
    
    return '<br>'.join(diff_lines)

def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

# -----------------------------------------------------------------------------
# UI Components
# -----------------------------------------------------------------------------

def render_score_bar(score: float, label: str = ""):
    """Render a colored score bar."""
    pct = score * 100
    if score >= 0.8:
        color = "linear-gradient(90deg, #00c853, #69f0ae)"
    elif score >= 0.5:
        color = "linear-gradient(90deg, #ff9100, #ffab40)"
    else:
        color = "linear-gradient(90deg, #ff4b4b, #ff8a80)"
    
    st.markdown(f"""
    <div style="margin: 4px 0;">
        <div style="display:flex; justify-content:space-between; font-size:12px; font-weight:600; margin-bottom:3px;">
            <span>{label}</span>
            <span>{pct:.1f}%</span>
        </div>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_html_preview(html_content: str, key_suffix: str = ""):
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        col1.caption("📄 Document Context (HTML)")
        invert = col2.toggle("Invert (Dark Mode)", value=False, key=f"invert_{key_suffix}")
        filter_css = "filter: invert(1) hue-rotate(180deg);" if invert else ""
        
        wrapped = f"""
        <div style="background: white; {filter_css} padding: 10px; height: 100%;">
            {html_content}
        </div>
        """
        st.components.v1.html(wrapped, height=550, scrolling=True)


def render_field_comparison(record: Dict, eval_data: Dict):
    """Render a detailed field-by-field comparison with diff."""
    gt = safe_parse_gt(record.get('ground_truth', {}))
    pred = record.get('prediction', {})
    
    if not isinstance(gt, dict) or not isinstance(pred, dict):
        st.json({"ground_truth": gt, "prediction": pred, "evaluation": eval_data})
        return
    
    for field, score in eval_data.items():
        gt_val = gt.get(field, 'MISSING')
        pred_val = pred.get(field, 'MISSING')
        
        # Handle list ground truths (SWDE format)
        if isinstance(gt_val, list):
            display_gt = clean_gt_text(gt_val[0]) if gt_val else 'MISSING'
        else:
            display_gt = clean_gt_text(str(gt_val))
        
        display_pred = str(pred_val) if pred_val is not None else 'null'
        
        if score == 1:
            status_class = "status-match"
            icon = "✅"
        elif score == 0:
            error_type = classify_error(display_gt, display_pred)
            status_class = "status-mismatch"
            icon = "❌"
        else:
            status_class = "status-partial"
            icon = "⚠️"
        
        fuzzy_score = compute_fuzzy_score(display_gt, display_pred)
        
        with st.expander(f"{icon} **{field}** — Score: {score} | Fuzzy Match: {fuzzy_score:.1%}", expanded=(score != 1)):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🎯 Ground Truth**")
                st.code(display_gt, language=None)
            with c2:
                st.markdown("**🤖 Prediction**")
                st.code(display_pred, language=None)
            
            render_score_bar(fuzzy_score, "Fuzzy Match Score")
            
            if score != 1:
                error_type = classify_error(display_gt, display_pred)
                st.markdown(f"**Error Type:** {error_type}")
                
                # Show postprocessor details if available
                step_logs = record.get('step_logs', {})
                if isinstance(step_logs, str):
                    try:
                        step_logs = json.loads(step_logs)
                    except Exception:
                        step_logs = {}
                
                post = step_logs.get('postprocessor', {}) if isinstance(step_logs, dict) else {}
                exact_match = post.get('exact_match_log', {}) if isinstance(post, dict) else {}
                field_match = exact_match.get(field, {})
                
                if field_match:
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Match Score", f"{field_match.get('score', 0):.3f}")
                    mc2.metric("Status", field_match.get('status', 'N/A'))
                    mc3.metric("XPath", field_match.get('xpath', 'N/A')[:40])
                    
                    if field_match.get('original_extracted') != field_match.get('value'):
                        st.info(f"**Original LLM extraction:** `{field_match.get('original_extracted')}`  \n"
                                f"**After alignment:** `{field_match.get('value')}`")


def render_diagnostics(record: Dict[str, Any]):
    """Render step-by-step pipeline diagnostics."""
    step_logs = record.get("step_logs")
    if not step_logs:
        return
    
    if isinstance(step_logs, str):
        try:
            step_logs = json.loads(step_logs)
        except Exception:
            try:
                step_logs = ast.literal_eval(step_logs)
            except Exception:
                st.warning("Could not parse step logs.")
                return
    
    if not isinstance(step_logs, dict):
        return

    st.markdown("---")
    st.subheader("🔬 Pipeline Diagnostics")
    
    tabs = st.tabs(["🔍 Preprocessor", "⚖️ Reranker", "✂️ LLM Pruner", "🤖 Extractor", "⚙️ Postprocessor"])
    
    record_id = record.get('id', 'unknown')
    
    # 1. Preprocessing
    with tabs[0]:
        prep = step_logs.get("preprocessor")
        if prep:
            pc1, pc2, pc3 = st.columns(3)
            raw_len = prep.get('raw_len', 0)
            cleaned_len = prep.get('cleaned_len', 0)
            reduction = 1 - (cleaned_len / max(1, raw_len))
            
            pc1.metric("Raw Length", f"{raw_len:,} chars")
            pc2.metric("Cleaned Length", f"{cleaned_len:,} chars")
            pc3.metric("Size Reduction", f"{reduction:.1%}")
            
            st.write(f"**Chunks Generated:** {prep.get('num_chunks', 0)}")
            
            if prep.get("error"):
                st.error(f"Preprocessing error: {prep['error']}")
        else:
            st.info("No preprocessing logs available.")
        
        preprocessed_html = record.get("preprocessed_content")
        if preprocessed_html:
            with st.expander("👁️ View Preprocessed HTML"):
                st.components.v1.html(
                    f'<div style="background:white; color:black; padding:10px; height:100%; overflow:auto;">{preprocessed_html}</div>',
                    height=400, scrolling=True
                )
    
    # 2. Reranking
    with tabs[1]:
        rerank = step_logs.get("reranker")
        if rerank and rerank.get("chunks"):
            df_chunks = pd.DataFrame(rerank["chunks"])
            
            if 'score' in df_chunks.columns:
                fig = px.bar(
                    df_chunks, x='chunkid', y='score',
                    color='score', color_continuous_scale='Viridis',
                    title='Chunk Relevance Scores'
                )
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, width="stretch")
            
            st.dataframe(df_chunks, width="stretch", hide_index=True)
        else:
            st.info("No reranking details (reranker may have been disabled).")
    
    # 3. LLM Pruning
    with tabs[2]:
        pruner = step_logs.get("pruner")
        if pruner:
            if isinstance(pruner, list):
                for idx, p_log in enumerate(pruner):
                    if not p_log:
                        continue
                    with st.expander(f"Chunk {idx + 1} Pruning Details"):
                        st.text_area("Pruner Prompt", p_log.get("prompt", ""), height=150,
                                     key=f"prn_prompt_{record_id}_{idx}")
                        st.text_area("LLM Response", p_log.get("response", ""), height=80,
                                     key=f"prn_resp_{record_id}_{idx}")
                        selected = p_log.get('selected_indices', [])
                        st.write(f"**Selected Indices:** {selected} ({len(selected)} chunks kept)")
            elif isinstance(pruner, dict):
                st.text_area("Pruner Prompt", pruner.get("prompt", ""), height=150,
                             key=f"prn_prompt_{record_id}")
                st.text_area("LLM Response", pruner.get("response", ""), height=80,
                             key=f"prn_resp_{record_id}")
        else:
            st.info("No LLM pruner logs available.")
    
    # 4. Extraction
    with tabs[3]:
        extractor = step_logs.get("extractor")
        if extractor:
            st.text_area("Generator Prompt", extractor.get("prompt", ""), height=200,
                         key=f"ext_prompt_{record_id}")
            st.text_area("Raw Generator Response", extractor.get("raw_response", ""), height=150,
                         key=f"ext_resp_{record_id}")
            
            # Parse and show structured response
            raw = extractor.get("raw_response", "")
            if raw:
                with st.expander("📝 Parsed Response"):
                    # Try to extract JSON from markdown code blocks
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw)
                    if json_match:
                        try:
                            parsed = json.loads(json_match.group(1))
                            st.json(parsed)
                        except Exception:
                            st.code(json_match.group(1))
                    else:
                        st.code(raw)
        else:
            st.info("No extractor logs available.")
    
    # 5. Postprocessing
    with tabs[4]:
        post = step_logs.get("postprocessor")
        if post:
            if post.get("error"):
                st.error(f"Postprocessing Error: {post['error']}")
            
            exact_match = post.get("exact_match_log")
            if exact_match:
                st.write("**Exact Extraction / Fuzzy Alignment Results:**")
                match_data = []
                for field, details in exact_match.items():
                    score = details.get("score", 0)
                    match_data.append({
                        "Field": field,
                        "Status": details.get("status", ""),
                        "Score": f"{score:.3f}" if isinstance(score, float) else str(score),
                        "Original": str(details.get("original_extracted", ""))[:80],
                        "Aligned": str(details.get("value", ""))[:80],
                        "XPath": str(details.get("xpath", ""))[:50]
                    })
                df_match = pd.DataFrame(match_data)
                
                def color_match_status(val):
                    if val == "success":
                        return "color: #00c853; font-weight: bold"
                    if val == "not_found":
                        return "color: #ff9100; font-weight: bold"
                    if val == "error":
                        return "color: #ff4b4b; font-weight: bold"
                    return ""
                
                st.dataframe(
                    df_match.style.map(color_match_status, subset=['Status']),
                    width="stretch", hide_index=True
                )
            else:
                st.info("No exact matching logs (exact extraction might have been disabled).")
        else:
            st.info("No postprocessor logs available.")


# -----------------------------------------------------------------------------
# Sample Navigation
# -----------------------------------------------------------------------------

def init_nav_state():
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
    if 'sort_mode' not in st.session_state:
        st.session_state.sort_mode = 'errors_desc'
    if 'filter_field' not in st.session_state:
        st.session_state.filter_field = 'All Fields'
    if 'filter_site' not in st.session_state:
        st.session_state.filter_site = 'All Sites'


def render_sample_navigator(df: pd.DataFrame, sorted_indices: list):
    """Render prev/next navigation with jump-to-sample."""
    n = len(sorted_indices)
    if n == 0:
        st.warning("No records match current filters.")
        return None
    
    # Clamp index
    st.session_state.current_idx = max(0, min(st.session_state.current_idx, n - 1))
    
    col_prev, col_counter, col_next, col_jump, col_hint = st.columns([1, 1, 1, 2, 2])
    
    with col_prev:
        if st.button("◀ Prev", key="nav_prev", width="stretch",
                      disabled=(st.session_state.current_idx <= 0)):
            st.session_state.current_idx -= 1
            st.rerun()
    
    with col_counter:
        st.markdown(
            f'<div class="nav-counter">{st.session_state.current_idx + 1} / {n}</div>',
            unsafe_allow_html=True
        )
    
    with col_next:
        if st.button("Next ▶", key="nav_next", width="stretch",
                      disabled=(st.session_state.current_idx >= n - 1)):
            st.session_state.current_idx += 1
            st.rerun()
    
    with col_jump:
        # Build display labels for jump selector
        jump_options = []
        for i, idx in enumerate(sorted_indices):
            row = df.loc[idx]
            err = row.get('error_count', 0)
            jump_options.append(f"{i+1}. {row['id']} (❌{err})")
        
        selected_jump = st.selectbox(
            "Jump to sample",
            options=jump_options,
            index=st.session_state.current_idx,
            key="jump_select",
            label_visibility="collapsed"
        )
        
        new_idx = jump_options.index(selected_jump)
        if new_idx != st.session_state.current_idx:
            st.session_state.current_idx = new_idx
            st.rerun()
    
    with col_hint:
        record = df.loc[sorted_indices[st.session_state.current_idx]]
        acc = record.get('accuracy', 0)
        color = "#00c853" if acc >= 0.8 else "#ff9100" if acc >= 0.5 else "#ff4b4b"
        st.markdown(f'<span style="font-size:14px; font-weight:700; color:{color};">'
                    f'Accuracy: {acc:.1%}</span>', unsafe_allow_html=True)
    
    return sorted_indices[st.session_state.current_idx]


# -----------------------------------------------------------------------------
# Views
# -----------------------------------------------------------------------------

def view_overview(analyzer: ErrorAnalyzer, df: pd.DataFrame):
    """High-level metrics overview."""
    eval_df = analyzer.eval_df
    
    # Top metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    global_acc = eval_df.values.mean()
    global_fuzzy = analyzer.fuzzy_scores_df.values.mean()
    perfect = len(df[eval_df.sum(axis=1) == len(eval_df.columns)])
    zero_acc = len(df[eval_df.sum(axis=1) == 0])
    avg_errors = eval_df.shape[1] - eval_df.sum(axis=1).mean()
    
    m1.metric("Total Records", len(df))
    m2.metric("Global Accuracy", f"{global_acc:.1%}")
    m3.metric("Avg Fuzzy Match", f"{global_fuzzy:.1%}")
    m4.metric("Perfect Records", f"{perfect} ({perfect/len(df):.0%})")
    m5.metric("Zero-Accuracy", f"{zero_acc}")
    m6.metric("Avg Errors/Doc", f"{avg_errors:.2f}")
    
    st.markdown("---")
    
    # Two-column layout: field accuracy + error distribution
    col_chart, col_dist = st.columns([1.2, 1])
    
    with col_chart:
        st.subheader("📊 Field Accuracy vs. Fuzzy Match Breakdown")
        field_acc = eval_df.mean()
        field_fuzzy = analyzer.fuzzy_scores_df.mean()
        
        plot_df = pd.DataFrame({
            'Field': field_acc.index,
            'Strict Accuracy': field_acc.values,
            'Fuzzy Match Score': [field_fuzzy[f] for f in field_acc.index]
        }).sort_values('Strict Accuracy', ascending=True)
        
        melted_df = plot_df.melt(id_vars='Field', var_name='Metric', value_name='Score')
        
        fig = px.bar(
            melted_df,
            x='Score',
            y='Field',
            color='Metric',
            barmode='group',
            orientation='h',
            color_discrete_map={
                'Strict Accuracy': '#ff4b4b',
                'Fuzzy Match Score': '#667eea'
            },
            labels={'Score': 'Score', 'Field': 'Field'},
        )
        fig.update_layout(
            height=max(300, len(field_acc) * 55),
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, width="stretch")
    
    with col_dist:
        st.subheader("📈 Error Count Distribution")
        error_counts = df['error_count'].value_counts().sort_index()
        
        fig = px.bar(
            x=error_counts.index,
            y=error_counts.values,
            labels={'x': 'Number of Errors', 'y': 'Records'},
            color=error_counts.index,
            color_continuous_scale=[[0, '#00c853'], [0.5, '#ff9100'], [1, '#ff4b4b']],
        )
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=10, b=20),
            showlegend=False,
            coloraxis_showscale=False,
            xaxis=dict(dtick=1)
        )
        st.plotly_chart(fig, width="stretch")
    
    # Site-level heatmap
    if len(analyzer.site_accuracy) > 1:
        st.markdown("---")
        st.subheader("🗺️ Accuracy by Source × Field")
        
        sites = sorted(analyzer.site_accuracy.keys())
        fields = analyzer.fields
        
        z_data = []
        for site in sites:
            row = []
            for field in fields:
                row.append(analyzer.site_accuracy[site]['fields'].get(field, 0))
            z_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=fields,
            y=sites,
            colorscale=[[0, '#ff4b4b'], [0.5, '#ff9100'], [0.8, '#ffd600'], [1, '#00c853']],
            zmin=0, zmax=1,
            text=[[f"{v:.0%}" for v in row] for row in z_data],
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False,
        ))
        fig.update_layout(
            height=max(300, len(sites) * 35 + 100),
            margin=dict(l=20, r=20, t=10, b=20),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, width="stretch")
    
    # Error patterns
    if analyzer.error_patterns:
        st.markdown("---")
        st.subheader("🔗 Co-occurring Error Patterns")
        st.caption("Fields that frequently fail together — suggests systematic issues.")
        
        pattern_data = []
        for p in analyzer.error_patterns[:10]:
            pattern_data.append({
                'Failed Fields': ', '.join(p['fields']),
                'Occurrences': p['count'],
                'Frequency': f"{p['pct']:.1f}%"
            })
        st.dataframe(pd.DataFrame(pattern_data), width="stretch", hide_index=True)


def view_field_deep_dive(analyzer: ErrorAnalyzer, df: pd.DataFrame):
    """Deep dive into errors for a specific field."""
    st.subheader("🔎 Field Error Deep Dive")
    
    selected_field = st.selectbox(
        "Select field to analyze",
        options=analyzer.fields,
        format_func=lambda f: f"{f} — {analyzer.field_accuracy.get(f, 0):.1%} accuracy"
    )
    
    analysis = analyzer.get_field_confusion(selected_field)
    field_fuzzy_score = analyzer.field_fuzzy.get(selected_field, 0.0)
    
    # Metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Errors", analysis['total_errors'])
    mc2.metric("Error Rate", f"{analysis['error_rate']:.1%}")
    mc3.metric("Avg Fuzzy Match", f"{field_fuzzy_score:.1%}")
    mc4.metric("Sites Affected", len(analysis['error_by_site']))
    
    col_issues, col_sites = st.columns([1, 1])
    
    with col_issues:
        st.markdown("**Error Type Breakdown**")
        if analysis.get('issue_breakdown'):
            issue_df = pd.DataFrame([
                {'Error Type': k, 'Count': v}
                for k, v in analysis['issue_breakdown'].items()
            ]).sort_values('Count', ascending=False)
            
            fig = px.bar(
                issue_df, x='Count', y='Error Type',
                orientation='h',
                color='Count',
                color_continuous_scale='Reds'
            )
            fig.update_layout(
                height=max(200, len(issue_df) * 40),
                margin=dict(l=20, r=20, t=10, b=20),
                showlegend=False, coloraxis_showscale=False,
            )
            st.plotly_chart(fig, width="stretch")
    
    with col_sites:
        st.markdown("**Errors by Source Site**")
        if analysis['error_by_site']:
            site_df = pd.DataFrame([
                {'Site': k, 'Errors': v}
                for k, v in analysis['error_by_site'].items()
            ]).sort_values('Errors', ascending=False)
            
            fig = px.bar(
                site_df, x='Errors', y='Site',
                orientation='h',
                color='Errors',
                color_continuous_scale='OrRd'
            )
            fig.update_layout(
                height=max(200, len(site_df) * 35),
                margin=dict(l=20, r=20, t=10, b=20),
                showlegend=False, coloraxis_showscale=False,
            )
            st.plotly_chart(fig, width="stretch")
    
    # Show specific error examples
    st.markdown("---")
    st.markdown(f"**Sample Errors for `{selected_field}`**")
    
    errors = df[analyzer.eval_df[selected_field] == 0].head(10)
    for _, row in errors.iterrows():
        gt = safe_parse_gt(row.get('ground_truth', {}))
        pred = row.get('prediction', {})
        
        gt_val = gt.get(selected_field, 'N/A') if isinstance(gt, dict) else 'N/A'
        pred_val = pred.get(selected_field, 'N/A') if isinstance(pred, dict) else 'N/A'
        
        if isinstance(gt_val, list):
            gt_val = gt_val[0] if gt_val else 'N/A'
        
        display_gt = clean_gt_text(str(gt_val))
        display_pred = str(pred_val) if pred_val is not None else 'null'
        
        error_type = classify_error(display_gt, display_pred)
        fuzzy_score = compute_fuzzy_score(display_gt, display_pred)
        
        with st.expander(f"📌 {row['id']} — {error_type} (Fuzzy Match: {fuzzy_score:.1%})"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Ground Truth**")
                st.code(display_gt, language=None)
            with c2:
                st.markdown("**Prediction**")
                st.code(display_pred, language=None)
            
            render_score_bar(fuzzy_score, "Fuzzy Match Score")


def view_record_inspector(df: pd.DataFrame, file_type: str):
    """Record-by-record inspector with prev/next navigation."""
    init_nav_state()
    
    is_structured = "Type 1" in file_type
    
    # Sidebar-style filters in columns
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
    
    with filter_col1:
        sort_options = {
            'errors_desc': '❌ Most Errors First',
            'errors_asc': '✅ Fewest Errors First',
            'id': '🔤 By ID',
            'accuracy': '📊 By Accuracy (ascending)'
        }
        sort_mode = st.selectbox("Sort by", options=list(sort_options.keys()),
                                 format_func=lambda x: sort_options[x],
                                 key="sort_select")
    
    with filter_col2:
        if is_structured and 'evaluation' in df.columns:
            eval_fields = list(pd.json_normalize(df['evaluation'].iloc[:1]).columns)
            filter_field = st.selectbox("Filter by failed field", 
                                        options=['All Fields'] + eval_fields,
                                        key="field_filter")
        else:
            filter_field = 'All Fields'
    
    with filter_col3:
        if 'source_site' in df.columns:
            sites = ['All Sites'] + sorted(df['source_site'].unique().tolist())
            filter_site = st.selectbox("Filter by source", options=sites, key="site_filter")
        else:
            filter_site = 'All Sites'
    
    # Apply filters
    mask = pd.Series(True, index=df.index)
    
    if filter_field != 'All Fields' and 'evaluation' in df.columns:
        mask &= df['evaluation'].apply(lambda x: x.get(filter_field, 1) == 0)
    
    if filter_site != 'All Sites' and 'source_site' in df.columns:
        mask &= df['source_site'] == filter_site
    
    filtered_df = df[mask]
    
    if len(filtered_df) == 0:
        st.info("No records match current filters. Try relaxing your filter criteria.")
        return
    
    # Sort
    if sort_mode == 'errors_desc':
        sorted_df = filtered_df.sort_values('error_count', ascending=False)
    elif sort_mode == 'errors_asc':
        sorted_df = filtered_df.sort_values('error_count', ascending=True)
    elif sort_mode == 'accuracy':
        sorted_df = filtered_df.sort_values('accuracy', ascending=True)
    else:
        sorted_df = filtered_df.sort_values('id')
    
    sorted_indices = sorted_df.index.tolist()
    
    # Navigation
    current_df_idx = render_sample_navigator(df, sorted_indices)
    if current_df_idx is None:
        return
    
    record = df.loc[current_df_idx]
    eval_data = record.get('evaluation', {})
    
    st.markdown("---")
    
    # Record header
    error_count = record.get('error_count', 0)
    total_fields = record.get('total_fields', 0)
    accuracy = record.get('accuracy', 0)
    source = record.get('source_site', '')
    
    hdr1, hdr2, hdr3, hdr4 = st.columns([2, 1, 1, 1])
    hdr1.markdown(f"### 📋 `{record['id']}`")
    hdr2.metric("Errors", f"{error_count}/{total_fields}")
    hdr3.metric("Accuracy", f"{accuracy:.1%}")
    hdr4.metric("Source", source)
    
    # Main content area
    if is_structured:
        col_left, col_right = st.columns([1.3, 1])
        
        with col_left:
            tab_compare, tab_raw = st.tabs(["🔍 Field Comparison", "🛠️ Raw Data"])
            
            with tab_compare:
                render_field_comparison(record, eval_data)
            
            with tab_raw:
                st.json(record.to_dict())
            
            render_diagnostics(record)
        
        with col_right:
            if 'filtered_html' in record and record['filtered_html']:
                render_html_preview(record['filtered_html'], key_suffix=str(current_df_idx))
            elif 'content' in record and record['content']:
                render_html_preview(record['content'], key_suffix=str(current_df_idx))
            else:
                st.info("No HTML content available for this record.")
    else:
        # QA / Text generation view
        col_l, col_r = st.columns([1, 1])
        
        with col_l:
            st.info(f"**Query:** {record.get('query', 'N/A')}")
            
            c1, c2 = st.columns(2)
            with c1:
                st.success("**Prediction**")
                st.write(record.get('prediction', ''))
            with c2:
                st.warning("**Ground Truth**")
                st.write(record.get('ground_truth', ''))
            
            st.divider()
            
            # Metrics
            fuzzy_score = compute_fuzzy_score(record.get('ground_truth', ''), record.get('prediction', ''))
            
            st.markdown("### Similarity Metrics")
            render_score_bar(fuzzy_score, "Fuzzy Match Score")
            st.divider()
            
            if eval_data:
                ec1, ec2, ec3, ec4 = st.columns(4)
                ec1.metric("F1", f"{eval_data.get('f1', 0):.3f}")
                ec2.metric("Precision", f"{eval_data.get('precision', 0):.3f}")
                ec3.metric("Recall", f"{eval_data.get('recall', 0):.3f}")
                ec4.metric("Fuzzy Match", f"{fuzzy_score:.1%}")
            else:
                st.metric("Fuzzy Match Score", f"{fuzzy_score:.1%}")
            
            render_diagnostics(record)
        
        with col_r:
            if 'filtered_html' in record and record['filtered_html']:
                render_html_preview(record['filtered_html'], key_suffix=str(current_df_idx))


def view_type2_overview(df: pd.DataFrame):
    """Overview for QA / text generation results."""
    eval_df = pd.json_normalize(df['evaluation'])
    
    # Compute fuzzy matching scores
    fuzzy_scores = []
    for _, row in df.iterrows():
        gt_val = str(row.get('ground_truth', ''))
        pred_val = str(row.get('prediction', ''))
        fuzzy_scores.append(compute_fuzzy_score(gt_val, pred_val))
    avg_fuzzy = np.mean(fuzzy_scores) if fuzzy_scores else 0.0
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Samples", len(df))
    m2.metric("Avg F1", f"{eval_df['f1'].mean():.3f}")
    m3.metric("Avg Precision", f"{eval_df['precision'].mean():.3f}")
    m4.metric("Avg Recall", f"{eval_df['recall'].mean():.3f}")
    m5.metric("Avg Fuzzy Match", f"{avg_fuzzy:.1%}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            eval_df, x="f1", nbins=25,
            title="F1 Score Distribution",
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Precision", "Recall"))
        fig.add_trace(go.Histogram(x=eval_df['precision'], nbinsx=25,
                                    marker_color='#00c853', name='Precision'), row=1, col=1)
        fig.add_trace(go.Histogram(x=eval_df['recall'], nbinsx=25,
                                    marker_color='#ff9100', name='Recall'), row=1, col=2)
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig, width="stretch")


# -----------------------------------------------------------------------------
# Pipeline Analysis View
# -----------------------------------------------------------------------------

def _parse_step_logs(record) -> dict:
    """Safely parse step_logs from a record (handles str, dict, None)."""
    sl = record.get('step_logs')
    if sl is None:
        return {}
    if isinstance(sl, str):
        try:
            sl = json.loads(sl)
        except Exception:
            try:
                sl = ast.literal_eval(sl)
            except Exception:
                return {}
    return sl if isinstance(sl, dict) else {}


def view_pipeline_analysis(df: pd.DataFrame):
    """Aggregate pipeline-step-level error analysis driven by step_logs."""
    # Check if step_logs are available
    has_step_logs = 'step_logs' in df.columns and df['step_logs'].notna().any()
    if not has_step_logs:
        st.warning("⚠️ No `step_logs` data found in the loaded records. "
                   "Load data in **Metrics + Predictions (full)** mode, or re-run evaluation "
                   "with the latest `html_eval` package to include step_logs in metric output.")
        return

    # Parse all step_logs into a working list
    parsed_logs = df.apply(lambda row: _parse_step_logs(row), axis=1)
    logs_available = parsed_logs.apply(lambda x: len(x) > 0).sum()

    st.caption(f"📋 Step logs available for **{logs_available}** / {len(df)} records")

    # ======================== A. PIPELINE FUNNEL & FORENSICS ========================
    st.subheader("🔻 Pipeline Funnels & Error Forensics")
    st.caption("Compare execution flow completion against the retention of target ground truth data.")

    funnel_data = {
        'Total Records': len(df),
        'Preprocessed': 0,
        'Reranked': 0,
        'Extracted': 0,
        'Postprocessed': 0,
        'Fully Correct': 0,
    }

    for idx, sl in parsed_logs.items():
        if sl.get('preprocessor'):
            funnel_data['Preprocessed'] += 1
        if sl.get('reranker') or sl.get('preprocessor'):  # reranker may be skipped
            funnel_data['Reranked'] += 1
        if sl.get('extractor'):
            funnel_data['Extracted'] += 1
        post = sl.get('postprocessor')
        if post and not post.get('error'):
            funnel_data['Postprocessed'] += 1

    # Fully correct = all eval scores are 1
    if 'evaluation' in df.columns:
        for _, row in df.iterrows():
            ev = row.get('evaluation', {})
            if isinstance(ev, dict) and ev:
                if all(v == 1 for v in ev.values()):
                    funnel_data['Fully Correct'] += 1

    # Ground Truth Survival Funnel computation
    is_structured = "Type 1" in detect_file_type(df.iloc[0].to_dict())
    
    total_gt_fields = 0
    in_raw_count = 0
    in_prep_count = 0
    in_prompt_count = 0
    in_response_count = 0
    in_pred_count = 0

    for idx, row in df.iterrows():
        gt = safe_parse_gt(row.get('ground_truth', {}))
        eval_scores = row.get('evaluation', {})
        sl = parsed_logs.get(idx, {})
        
        # Get raw content - normalize HTML entities so &nbsp; etc. don't cause false misses
        raw_content = clean_gt_text(str(row.get('content', '') or row.get('filtered_html', '') or ''))
        
        # Get preprocessed content
        prep_content = clean_gt_text(str(row.get('preprocessed_content', '') or sl.get('preprocessor', {}).get('cleaned_content', '') or ''))
        if not prep_content:
            prep_content = raw_content  # Fallback
            
        # Get prompt sent to extractor
        prompt = clean_gt_text(str(sl.get('extractor', {}).get('prompt', '') or ''))
        
        # Get LLM response
        llm_response = clean_gt_text(str(sl.get('extractor', {}).get('raw_response', '') or ''))
        
        # Check structured fields
        if is_structured and isinstance(gt, dict) and gt:
            for field, gt_val in gt.items():
                # Use all candidates from list GT, or wrap scalar in list
                if isinstance(gt_val, list):
                    candidates = [clean_gt_text(str(v)) for v in gt_val if v is not None]
                else:
                    candidates = [clean_gt_text(str(gt_val))]
                
                # Filter empty / null candidates
                candidates = [
                    c for c in candidates
                    if not is_null_value(c)
                ]
                if not candidates:
                    continue
                
                total_gt_fields += 1
                # A GT value is considered present at a stage if ANY candidate matches
                in_raw  = any(c.lower() in raw_content.lower()     for c in candidates)
                in_prep = any(c.lower() in prep_content.lower()    for c in candidates)
                in_prompt   = any(c.lower() in prompt.lower()      for c in candidates)
                in_response = any(c.lower() in llm_response.lower() for c in candidates)
                
                is_correct = eval_scores.get(field, 0) == 1 if isinstance(eval_scores, dict) else False
                
                if in_raw:      in_raw_count += 1
                if in_prep:     in_prep_count += 1
                if in_prompt:   in_prompt_count += 1
                if in_response: in_response_count += 1
                if is_correct:  in_pred_count += 1
        else:
            # Type 2 / QA dataset or string ground truth
            display_gt = clean_gt_text(str(gt)).strip()
            if display_gt and not is_null_value(display_gt):
                total_gt_fields += 1
                gt_lower = display_gt.lower()
                
                in_raw = gt_lower in raw_content.lower()
                in_prep = gt_lower in prep_content.lower()
                in_prompt = gt_lower in prompt.lower()
                in_response = gt_lower in llm_response.lower()
                
                is_correct = eval_scores.get('f1', 0) >= 0.8 if isinstance(eval_scores, dict) else False
                
                if in_raw: in_raw_count += 1
                if in_prep: in_prep_count += 1
                if in_prompt: in_prompt_count += 1
                if in_response: in_response_count += 1
                if is_correct: in_pred_count += 1

    col_fun1, col_fun2 = st.columns(2)
    
    with col_fun1:
        st.subheader("🔻 Pipeline Execution Funnel")
        st.caption("Tracks how many records survive each pipeline stage code execution.")
        
        stages_exec = list(funnel_data.keys())
        counts_exec = list(funnel_data.values())
        colors_exec = ['#667eea', '#7c6dd8', '#9b59b6', '#e67e22', '#2ecc71', '#00c853']
        
        fig_exec = go.Figure(go.Funnel(
            y=stages_exec,
            x=counts_exec,
            textinfo="value+percent initial",
            marker=dict(color=colors_exec[:len(stages_exec)]),
            connector=dict(line=dict(color="rgba(255,255,255,0.1)", width=1)),
        ))
        fig_exec.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=10, b=20),
            font=dict(size=12),
        )
        st.plotly_chart(fig_exec, use_container_width=True)
        
    with col_fun2:
        st.subheader("🔑 Ground Truth Retention Funnel")
        st.caption("Tracks the exact stage where the ground truth text was dropped by the pipeline.")
        
        stages_gt = [
            'Total Gold Values',
            'Present in Raw HTML',
            'Survived Preprocess',
            'Survived Rerank (Context)',
            'Survived Extract (LLM)',
            'Survived Normalize (Correct)'
        ]
        counts_gt = [
            total_gt_fields,
            in_raw_count,
            in_prep_count,
            in_prompt_count,
            in_response_count,
            in_pred_count
        ]
        colors_gt = ['#4f46e5', '#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444']
        
        fig_gt = go.Figure(go.Funnel(
            y=stages_gt,
            x=counts_gt,
            textinfo="value+percent initial",
            marker=dict(color=colors_gt),
            connector=dict(line=dict(color="rgba(255,255,255,0.1)", width=1)),
        ))
        fig_gt.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=10, b=20),
            font=dict(size=12),
        )
        st.plotly_chart(fig_gt, use_container_width=True)

    st.markdown("### 🔍 Ground Truth Loss Analysis")
    st.caption("Breakdown of data drop-offs by stage across the pipeline.")
    
    dt1, dt2, dt3, dt4, dt5 = st.columns(5)
    
    # Not in raw
    not_in_raw = total_gt_fields - in_raw_count
    pct_not_in_raw = (not_in_raw / max(total_gt_fields, 1)) * 100
    dt1.metric("Absent in Raw HTML", f"{not_in_raw}", f"-{pct_not_in_raw:.1f}%", delta_color="off")
    
    # Lost in HTML cleaning
    lost_prep = in_raw_count - in_prep_count
    pct_prep = (lost_prep / max(in_raw_count, 1)) * 100
    dt2.metric("Lost in HTML Cleaning", f"{lost_prep}", f"-{pct_prep:.1f}%", delta_color="inverse")
    
    # Lost in context pruning
    lost_rerank = in_prep_count - in_prompt_count
    pct_rerank = (lost_rerank / max(in_prep_count, 1)) * 100
    dt3.metric("Lost in Reranking", f"{lost_rerank}", f"-{pct_rerank:.1f}%", delta_color="inverse")
    
    # Lost in LLM recall
    lost_extract = in_prompt_count - in_response_count
    pct_extract = (lost_extract / max(in_prompt_count, 1)) * 100
    dt4.metric("Lost in LLM Extraction", f"{lost_extract}", f"-{pct_extract:.1f}%", delta_color="inverse")
    
    # Lost in normalization
    lost_post = in_response_count - in_pred_count
    pct_post = (lost_post / max(in_response_count, 1)) * 100
    dt5.metric("Lost in Postprocessing", f"{lost_post}", f"-{pct_post:.1f}%", delta_color="inverse")

    st.markdown("---")

    # ======================== B. PREPROCESSING STATISTICS ========================
    col_prep, col_extract = st.columns(2)

    with col_prep:
        st.subheader("📦 Preprocessing Statistics")

        prep_data = []
        for idx, sl in parsed_logs.items():
            prep = sl.get('preprocessor')
            if prep and isinstance(prep, dict):
                raw_len = prep.get('raw_len', 0)
                cleaned_len = prep.get('cleaned_len', 0)
                compression = 1 - (cleaned_len / max(1, raw_len)) if raw_len > 0 else 0
                prep_data.append({
                    'id': df.loc[idx, 'id'] if 'id' in df.columns else str(idx),
                    'raw_len': raw_len,
                    'cleaned_len': cleaned_len,
                    'compression': compression,
                    'num_chunks': prep.get('num_chunks', 0),
                    'error': prep.get('error'),
                })

        if prep_data:
            prep_df = pd.DataFrame(prep_data)

            # Summary metrics
            pm1, pm2, pm3 = st.columns(3)
            pm1.metric("Avg Raw Size", f"{prep_df['raw_len'].mean():,.0f} chars")
            pm2.metric("Avg Cleaned Size", f"{prep_df['cleaned_len'].mean():,.0f} chars")
            pm3.metric("Avg Compression", f"{prep_df['compression'].mean():.1%}")

            # Size distributions
            fig = make_subplots(rows=1, cols=2, subplot_titles=("HTML Size Distribution", "Chunk Count Distribution"))

            fig.add_trace(
                go.Histogram(x=prep_df['raw_len'], name='Raw', marker_color='#ff6b6b', opacity=0.7),
                row=1, col=1
            )
            fig.add_trace(
                go.Histogram(x=prep_df['cleaned_len'], name='Cleaned', marker_color='#51cf66', opacity=0.7),
                row=1, col=1
            )
            fig.add_trace(
                go.Histogram(x=prep_df['num_chunks'], name='Chunks', marker_color='#667eea',
                             nbinsx=max(1, int(prep_df['num_chunks'].max()))),
                row=1, col=2
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), showlegend=True,
                              legend=dict(orientation='h', yanchor='bottom', y=1.02))
            st.plotly_chart(fig, use_container_width=True)

            # Preprocessing errors
            errors = prep_df[prep_df['error'].notna()]
            if len(errors) > 0:
                st.error(f"⚠️ {len(errors)} records had preprocessing errors")
                with st.expander("View preprocessing errors"):
                    st.dataframe(errors[['id', 'error']], use_container_width=True, hide_index=True)
        else:
            st.info("No preprocessor logs available.")

    # ======================== C. EXTRACTION ERROR CATEGORIZATION ========================
    with col_extract:
        st.subheader("🤖 Extraction Analysis")

        extract_data = []
        for idx, sl in parsed_logs.items():
            ext = sl.get('extractor')
            if ext and isinstance(ext, dict):
                raw_resp = ext.get('raw_response', '')
                # Check if extraction produced valid JSON
                has_json = bool(re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_resp)) if raw_resp else False
                if has_json:
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_resp)
                    try:
                        json.loads(json_match.group(1))
                        parse_ok = True
                    except Exception:
                        parse_ok = False
                else:
                    # Try direct JSON parse
                    try:
                        json.loads(raw_resp)
                        parse_ok = True
                        has_json = True
                    except Exception:
                        parse_ok = False

                extract_data.append({
                    'id': df.loc[idx, 'id'] if 'id' in df.columns else str(idx),
                    'has_json_block': has_json,
                    'json_parseable': parse_ok,
                    'response_len': len(raw_resp) if raw_resp else 0,
                })

        if extract_data:
            ext_df = pd.DataFrame(extract_data)

            em1, em2, em3 = st.columns(3)
            total_ext = len(ext_df)
            em1.metric("Total Extractions", total_ext)
            em2.metric("Valid JSON", f"{ext_df['json_parseable'].sum()} ({ext_df['json_parseable'].mean():.0%})")
            em3.metric("Parse Failures", f"{(~ext_df['json_parseable']).sum()}")

            # Response length distribution
            fig = px.histogram(
                ext_df, x='response_len', nbins=30,
                title='LLM Response Length Distribution',
                color_discrete_sequence=['#667eea'],
                labels={'response_len': 'Response Length (chars)'}
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            # Show parse failure examples
            parse_failures = ext_df[~ext_df['json_parseable']]
            if len(parse_failures) > 0:
                st.warning(f"⚠️ {len(parse_failures)} extractions failed JSON parsing")
                with st.expander(f"View parse failure IDs ({len(parse_failures)} total)"):
                    st.dataframe(parse_failures[['id', 'has_json_block', 'response_len']],
                                 use_container_width=True, hide_index=True)
        else:
            st.info("No extractor logs available.")

    st.markdown("---")

    # ======================== D. POSTPROCESSOR ALIGNMENT IMPACT ========================
    st.subheader("⚙️ Postprocessor Alignment Impact")
    st.caption("Analyze how the exact-match alignment step changes LLM outputs and its effect on final accuracy.")

    alignment_data = []
    field_status_counts = defaultdict(lambda: defaultdict(int))
    field_alignment_scores = defaultdict(list)
    field_drift_counts = defaultdict(lambda: {'drifted': 0, 'unchanged': 0})

    for idx, sl in parsed_logs.items():
        post = sl.get('postprocessor')
        if not post or not isinstance(post, dict):
            continue
        eml = post.get('exact_match_log')
        if not eml or not isinstance(eml, dict):
            continue

        row_eval = df.loc[idx].get('evaluation', {}) if 'evaluation' in df.columns else {}

        for field, details in eml.items():
            if not isinstance(details, dict):
                continue
            status = details.get('status', 'unknown')
            score = details.get('score', 0)
            original = str(details.get('original_extracted', ''))
            aligned = str(details.get('value', ''))
            eval_score = row_eval.get(field, None) if isinstance(row_eval, dict) else None

            field_status_counts[field][status] += 1
            field_alignment_scores[field].append(score)

            drifted = original != aligned
            if drifted:
                field_drift_counts[field]['drifted'] += 1
            else:
                field_drift_counts[field]['unchanged'] += 1

            alignment_data.append({
                'id': df.loc[idx, 'id'] if 'id' in df.columns else str(idx),
                'field': field,
                'status': status,
                'alignment_score': score,
                'drifted': drifted,
                'original': original[:80],
                'aligned': aligned[:80],
                'eval_correct': eval_score,
            })

    if alignment_data:
        align_df = pd.DataFrame(alignment_data)

        # Per-field status breakdown chart
        col_status, col_drift = st.columns(2)

        with col_status:
            st.markdown("**Alignment Status per Field**")
            status_data = []
            for field in sorted(field_status_counts.keys()):
                for status, count in field_status_counts[field].items():
                    status_data.append({'Field': field, 'Status': status, 'Count': count})

            if status_data:
                status_df = pd.DataFrame(status_data)
                color_map = {
                    'success': '#00c853',
                    'not_found': '#ff9100',
                    'error': '#ff4b4b',
                    'unknown': '#888888'
                }
                fig = px.bar(
                    status_df, x='Field', y='Count', color='Status',
                    color_discrete_map=color_map,
                    barmode='stack',
                    title='Alignment Status Distribution'
                )
                fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

        with col_drift:
            st.markdown("**Alignment Drift per Field**")
            st.caption("How often does fuzzy matching change the LLM's original extraction?")

            drift_data = []
            for field in sorted(field_drift_counts.keys()):
                total = field_drift_counts[field]['drifted'] + field_drift_counts[field]['unchanged']
                drift_pct = field_drift_counts[field]['drifted'] / max(1, total)
                drift_data.append({
                    'Field': field,
                    'Drift Rate': drift_pct,
                    'Drifted': field_drift_counts[field]['drifted'],
                    'Unchanged': field_drift_counts[field]['unchanged'],
                })

            if drift_data:
                drift_df = pd.DataFrame(drift_data)
                fig = px.bar(
                    drift_df, x='Field', y='Drift Rate',
                    color='Drift Rate',
                    color_continuous_scale=[[0, '#00c853'], [0.5, '#ff9100'], [1, '#ff4b4b']],
                    text=[f"{r['Drifted']}/{r['Drifted']+r['Unchanged']}" for _, r in drift_df.iterrows()],
                    title='Alignment Drift Rate'
                )
                fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20),
                                  coloraxis_showscale=False)
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

        # Drift impact on accuracy
        if 'eval_correct' in align_df.columns and align_df['eval_correct'].notna().any():
            st.markdown("**Alignment Drift vs. Final Accuracy**")
            st.caption("Did alignment changes help or hurt accuracy?")

            impact_data = align_df[align_df['eval_correct'].notna()].copy()
            impact_data['Category'] = impact_data.apply(
                lambda r: ('Drifted → Correct' if r['eval_correct'] == 1 else 'Drifted → Wrong')
                          if r['drifted']
                          else ('Unchanged → Correct' if r['eval_correct'] == 1 else 'Unchanged → Wrong'),
                axis=1
            )

            impact_counts = impact_data['Category'].value_counts()
            cat_colors = {
                'Unchanged → Correct': '#00c853',
                'Drifted → Correct': '#69f0ae',
                'Unchanged → Wrong': '#ff8a80',
                'Drifted → Wrong': '#ff4b4b',
            }

            fig = px.bar(
                x=impact_counts.index,
                y=impact_counts.values,
                color=impact_counts.index,
                color_discrete_map=cat_colors,
                labels={'x': 'Category', 'y': 'Count'},
                title='Alignment Drift Impact on Final Accuracy'
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Detailed alignment table
        with st.expander("📋 Detailed Alignment Data"):
            display_cols = ['id', 'field', 'status', 'alignment_score', 'drifted', 'original', 'aligned']
            if 'eval_correct' in align_df.columns:
                display_cols.append('eval_correct')

            def color_status(val):
                if val == 'success':
                    return 'color: #00c853; font-weight: bold'
                if val == 'not_found':
                    return 'color: #ff9100; font-weight: bold'
                if val == 'error':
                    return 'color: #ff4b4b; font-weight: bold'
                return ''

            def color_drift(val):
                return 'color: #ff9100; font-weight: bold' if val else ''

            styled = align_df[display_cols].style.map(color_status, subset=['status'])
            styled = styled.map(color_drift, subset=['drifted'])
            st.dataframe(styled, use_container_width=True, hide_index=True, height=400)
    else:
        st.info("No postprocessor alignment logs available.")


# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------

def main():
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
        <span style="font-size:36px;">🔬</span>
        <div>
            <h1 style="margin:0; font-size:32px; font-weight:800; 
                        background: linear-gradient(135deg, #667eea, #764ba2);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Eval Inspector Pro
            </h1>
            <p style="margin:0; font-size:14px; color:rgba(255,255,255,0.5); font-weight:500;">
                Production-grade evaluation analysis & error forensics
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Data Source")
        
        # Auto-discover output directories
        discovered = discover_output_dirs()
        
        source_mode = st.radio(
            "Load data from",
            options=["📁 Auto-discover results", "📤 Upload file"],
            index=0 if discovered else 1,
            key="source_mode"
        )
        
        raw_data = None
        
        if "Auto-discover" in source_mode and discovered:
            selected_dir = st.selectbox(
                "Select experiment",
                options=list(discovered.keys()),
                key="experiment_select"
            )
            
            info = discovered[selected_dir]
            
            # Load mode: merged or metric-only
            if info['metrics']:
                load_mode = st.radio(
                    "Data to load",
                    ["📊 Metrics + Predictions (full)", "📊 Metrics only (faster)"],
                    key="load_mode"
                )
                
                if "full" in load_mode:
                    raw_data = merge_predictions_with_metrics(info['predictions'], info['metrics'])
                else:
                    raw_data = load_ndjson_from_path(info['metrics'])
            else:
                raw_data = load_ndjson_from_path(info['predictions'])
                st.warning("No metric file found — evaluation scores may be missing.")
            
            # Show experiment config
            if info.get('config'):
                with st.expander("📋 Experiment Config"):
                    try:
                        with open(info['config'], 'r') as f:
                            config = json.load(f)
                        st.json(config)
                    except Exception:
                        st.info("Could not load config.")
            
            # Show aggregate results
            if info.get('results'):
                with st.expander("📈 Aggregate Results"):
                    try:
                        with open(info['results'], 'r') as f:
                            results = json.load(f)
                        # Show top-level metrics
                        if 'page_level_f1' in results.get('results', {}):
                            plf1 = results['results']['page_level_f1']
                            rc1, rc2, rc3 = st.columns(3)
                            rc1.metric("Page F1", f"{plf1.get('f1', 0):.3f}")
                            rc2.metric("Page Precision", f"{plf1.get('precision', 0):.3f}")
                            rc3.metric("Page Recall", f"{plf1.get('recall', 0):.3f}")
                        st.json(results)
                    except Exception:
                        st.info("Could not load results.")
        
        else:
            uploaded = st.file_uploader("Upload NDJSON / JSONL", type=["ndjson", "jsonl", "txt"])
            if uploaded:
                raw_data = parse_ndjson(uploaded)
        
        st.divider()
        
        if raw_data:
            st.success(f"✅ Loaded **{len(raw_data)}** records")
    
    if not raw_data:
        # Landing page
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.container(border=True).markdown("""
            ### Welcome to Eval Inspector Pro 🔬
            
            **Auto-discovery:** Place experiment output directories (containing `predictions.ndjson`) 
            alongside this dashboard to auto-discover them.
            
            **Upload:** Or upload an `.ndjson` / `.jsonl` file manually.
            
            **Expected record format:**
            - `id`: Unique identifier
            - `prediction`: Model output (dict or string)  
            - `ground_truth`: Gold standard
            - `evaluation`: Dict of scores (0/1 or F1/Precision/Recall)
            - `filtered_html`: Source context HTML
            - `step_logs`: *(optional)* Pipeline diagnostic logs
            """)
        return
    
    df = pd.DataFrame(raw_data)
    file_type = detect_file_type(raw_data[0])
    is_structured = "Type 1" in file_type
    
    # Compute derived columns
    if 'evaluation' in df.columns:
        eval_df = pd.json_normalize(df['evaluation'])
        if is_structured:
            df['error_count'] = eval_df.apply(lambda x: (x == 0).sum(), axis=1).values
            df['total_fields'] = len(eval_df.columns)
            df['accuracy'] = eval_df.mean(axis=1).values
        else:
            df['error_count'] = 0
            df['accuracy'] = eval_df.get('f1', pd.Series([0]*len(df))).values
            df['total_fields'] = 3  # f1, precision, recall
    else:
        df['error_count'] = 0
        df['accuracy'] = 0
        df['total_fields'] = 0
    
    # Extract source site
    df['source_site'] = df['id'].apply(
        lambda x: '_'.join(str(x).split('_')[:-1]) if '_' in str(x) else str(x)
    )
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("### 🎛️ Filters")
        
        if is_structured and 'evaluation' in df.columns:
            show_errors_only = st.checkbox("Show Errors Only", value=False)
            if show_errors_only:
                df = df[df['error_count'] > 0]
            
            min_errors = st.slider("Min errors to show", 0, int(df['error_count'].max()) if len(df) > 0 else 0, 0)
            if min_errors > 0:
                df = df[df['error_count'] >= min_errors]
        
        elif not is_structured and 'evaluation' in df.columns:
            f1_threshold = st.slider("Max F1 score", 0.0, 1.0, 1.0, 0.05)
            df = df[df['evaluation'].apply(lambda x: x.get('f1', 0) <= f1_threshold)]
        
        if 'source_site' in df.columns and df['source_site'].nunique() > 1:
            sites = ['All'] + sorted(df['source_site'].unique().tolist())
            site_filter = st.selectbox("Source site", sites, key="sidebar_site")
            if site_filter != 'All':
                df = df[df['source_site'] == site_filter]
        
        st.caption(f"Showing **{len(df)}** records")
    
    if len(df) == 0:
        st.warning("No records match current filters.")
        return
    
    # Main tabs
    if is_structured and 'evaluation' in df.columns:
        analyzer = ErrorAnalyzer(df)
        
        tab_overview, tab_fields, tab_pipeline, tab_inspect, tab_data = st.tabs([
            "📊 Overview",
            "🔎 Field Analysis",
            "🔧 Pipeline Analysis",
            "🔬 Record Inspector",
            "🗃️ Raw Data"
        ])
        
        with tab_overview:
            view_overview(analyzer, df)
        
        with tab_fields:
            view_field_deep_dive(analyzer, df)
        
        with tab_pipeline:
            view_pipeline_analysis(df)
        
        with tab_inspect:
            view_record_inspector(df, file_type)
        
        with tab_data:
            st.dataframe(df.drop(columns=['content', 'preprocessed_content', 'filtered_html'], errors='ignore'),
                         width="stretch")
    
    elif not is_structured and 'evaluation' in df.columns:
        tab_overview, tab_pipeline, tab_inspect, tab_data = st.tabs([
            "📊 Overview",
            "🔧 Pipeline Analysis",
            "🔬 Record Inspector",
            "🗃️ Raw Data"
        ])
        
        with tab_overview:
            view_type2_overview(df)
        
        with tab_pipeline:
            view_pipeline_analysis(df)
        
        with tab_inspect:
            view_record_inspector(df, file_type)
        
        with tab_data:
            st.dataframe(df.drop(columns=['content', 'preprocessed_content', 'filtered_html'], errors='ignore'),
                         width="stretch")
    
    else:
        st.warning("No `evaluation` key found in records. Showing raw data.")
        st.dataframe(df, width="stretch")


if __name__ == "__main__":
    main()