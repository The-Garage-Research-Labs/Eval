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
        
        # Error patterns: which field combinations fail together
        self.error_patterns = self._compute_error_patterns(eval_df)
        
        # Per-site accuracy
        self.site_accuracy = {}
        for site, group in self.df.groupby('source_site'):
            site_eval = eval_df.loc[group.index]
            self.site_accuracy[site] = {
                'overall': site_eval.values.mean(),
                'fields': site_eval.mean().to_dict(),
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
    if pred in ('N/A', 'None', 'null', '', 'MISSING'):
        return '🚫 Missing Extraction'
    if gt in ('<NULL>', 'N/A', 'None', 'null', ''):
        return '👻 Hallucination (GT is null)'
    
    # Normalize for comparison
    gt_norm = gt.lower().strip()
    pred_norm = pred.lower().strip()
    
    if gt_norm == pred_norm:
        return '✅ Exact Match (eval bug?)'
    if gt_norm in pred_norm:
        return '📏 Over-extraction (superset)'
    if pred_norm in gt_norm:
        return '✂️ Under-extraction (subset)'
    
    # Check similarity
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
            display_gt = gt_val[0] if gt_val else 'MISSING'
        else:
            display_gt = str(gt_val)
        
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
        
        with st.expander(f"{icon} **{field}** — Score: {score}", expanded=(score != 1)):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🎯 Ground Truth**")
                st.code(display_gt, language=None)
            with c2:
                st.markdown("**🤖 Prediction**")
                st.code(display_pred, language=None)
            
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
    m1, m2, m3, m4, m5 = st.columns(5)
    global_acc = eval_df.values.mean()
    perfect = len(df[eval_df.sum(axis=1) == len(eval_df.columns)])
    zero_acc = len(df[eval_df.sum(axis=1) == 0])
    avg_errors = eval_df.shape[1] - eval_df.sum(axis=1).mean()
    
    m1.metric("Total Records", len(df))
    m2.metric("Global Accuracy", f"{global_acc:.1%}")
    m3.metric("Perfect Records", f"{perfect} ({perfect/len(df):.0%})")
    m4.metric("Zero-Accuracy", f"{zero_acc}")
    m5.metric("Avg Errors/Doc", f"{avg_errors:.2f}")
    
    st.markdown("---")
    
    # Two-column layout: field accuracy + error distribution
    col_chart, col_dist = st.columns([1.2, 1])
    
    with col_chart:
        st.subheader("📊 Field Accuracy Breakdown")
        field_acc = eval_df.mean().sort_values(ascending=True)
        
        fig = px.bar(
            x=field_acc.values,
            y=field_acc.index,
            orientation='h',
            labels={'x': 'Accuracy', 'y': 'Field'},
            color=field_acc.values,
            color_continuous_scale=[[0, '#ff4b4b'], [0.5, '#ff9100'], [0.8, '#ffd600'], [1, '#00c853']],
            range_color=[0, 1],
        )
        fig.update_layout(
            height=max(300, len(field_acc) * 45),
            margin=dict(l=20, r=20, t=10, b=20),
            showlegend=False,
            coloraxis_showscale=False,
        )
        fig.update_traces(
            text=[f"{v:.1%}" for v in field_acc.values],
            textposition='outside',
            textfont=dict(size=12, color='white')
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
    
    # Metrics
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Total Errors", analysis['total_errors'])
    mc2.metric("Error Rate", f"{analysis['error_rate']:.1%}")
    mc3.metric("Sites Affected", len(analysis['error_by_site']))
    
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
        
        error_type = classify_error(str(gt_val), str(pred_val))
        
        with st.expander(f"📌 {row['id']} — {error_type}"):
            c1, c2 = st.columns(2)
            c1.markdown("**Ground Truth**")
            c1.code(str(gt_val), language=None)
            c2.markdown("**Prediction**")
            c2.code(str(pred_val), language=None)


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
            if eval_data:
                ec1, ec2, ec3 = st.columns(3)
                ec1.metric("F1", f"{eval_data.get('f1', 0):.3f}")
                ec2.metric("Precision", f"{eval_data.get('precision', 0):.3f}")
                ec3.metric("Recall", f"{eval_data.get('recall', 0):.3f}")
            
            render_diagnostics(record)
        
        with col_r:
            if 'filtered_html' in record and record['filtered_html']:
                render_html_preview(record['filtered_html'], key_suffix=str(current_df_idx))


def view_type2_overview(df: pd.DataFrame):
    """Overview for QA / text generation results."""
    eval_df = pd.json_normalize(df['evaluation'])
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Samples", len(df))
    m2.metric("Avg F1", f"{eval_df['f1'].mean():.3f}")
    m3.metric("Avg Precision", f"{eval_df['precision'].mean():.3f}")
    m4.metric("Avg Recall", f"{eval_df['recall'].mean():.3f}")
    
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
        
        tab_overview, tab_fields, tab_inspect, tab_data = st.tabs([
            "📊 Overview",
            "🔎 Field Analysis",
            "🔬 Record Inspector",
            "🗃️ Raw Data"
        ])
        
        with tab_overview:
            view_overview(analyzer, df)
        
        with tab_fields:
            view_field_deep_dive(analyzer, df)
        
        with tab_inspect:
            view_record_inspector(df, file_type)
        
        with tab_data:
            st.dataframe(df.drop(columns=['content', 'preprocessed_content', 'filtered_html'], errors='ignore'),
                         width="stretch")
    
    elif not is_structured and 'evaluation' in df.columns:
        tab_overview, tab_inspect, tab_data = st.tabs([
            "📊 Overview",
            "🔬 Record Inspector",
            "🗃️ Raw Data"
        ])
        
        with tab_overview:
            view_type2_overview(df)
        
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