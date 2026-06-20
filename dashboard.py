import streamlit as st
import pandas as pd
import json
import ast
import plotly.express as px
from typing import Dict, Any, List

# -----------------------------------------------------------------------------
# Configuration & Styles
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Metric Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    st.markdown("""
    <style>
        /* Main container adjustments */
        .main .block-container {
            padding-top: 2rem;
            max-width: 95%;
        }
        
        /* Custom card style that adapts to theme */
        .status-card {
            background-color: var(--secondary-background-color);
            border-radius: 10px;
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            margin-bottom: 1rem;
        }

        /* Metrics labels */
        .stMetric label {
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Iframe border for HTML preview */
        iframe {
            border: 1px solid var(--border-color) !important;
            border-radius: 8px;
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
            if not line.strip(): continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                try:
                    data.append(ast.literal_eval(line))
                except:
                    continue
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return data

def safe_parse_gt(gt_val):
    if isinstance(gt_val, (dict, list)): return gt_val
    try: return json.loads(gt_val)
    except:
        try: return ast.literal_eval(gt_val)
        except: return gt_val

def detect_file_type(record: Dict) -> str:
    eval_keys = record.get("evaluation", {}).keys()
    if any(k in eval_keys for k in ["f1", "precision", "recall"]):
        return "Type 2: Text Generation (QA)"
    return "Type 1: Structured Extraction (Schema)"

# -----------------------------------------------------------------------------
# UI Components
# -----------------------------------------------------------------------------

def render_html_preview(html_content):
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        col1.caption("Document Context (HTML Snippet)")
        
        # HTML Preview Dark/Light Toggle
        invert = col2.toggle("Invert Preview (Dark Mode)", value=False)
        filter_css = "filter: invert(1) hue-rotate(180deg);" if invert else ""
        
        wrapped_html = f"""
        <div style="background: white; {filter_css} padding: 10px; height: 100%;">
            {html_content}
        </div>
        """
        st.components.v1.html(wrapped_html, height=600, scrolling=True)

def render_diagnostics(record: Dict[str, Any]):
    if "step_logs" not in record or not record["step_logs"]:
        return

    st.markdown("---")
    st.subheader("📝 Step-by-Step Diagnostics")
    
    logs = record["step_logs"]
    if isinstance(logs, str):
        try:
            logs = json.loads(logs)
        except Exception:
            try:
                logs = ast.literal_eval(logs)
            except Exception:
                st.warning("Could not parse step logs.")
                return
    
    # We can create tabs:
    tabs = st.tabs(["🔍 Preprocessing", "⚖️ Reranking", "✂️ LLM Pruning", "🤖 Extraction", "⚙️ Postprocessing"])
    
    # 1. Preprocessing Tab
    with tabs[0]:
        prep = logs.get("preprocessor")
        if prep:
            st.write(f"**Raw Content Length:** {prep.get('raw_len', 0)} characters")
            st.write(f"**Cleaned Content Length:** {prep.get('cleaned_len', 0)} characters")
            reduction = 1 - (prep.get('cleaned_len', 0) / max(1, prep.get('raw_len', 0)))
            st.write(f"**Size Reduction:** {reduction:.1%}")
            st.write(f"**Chunks Generated:** {prep.get('num_chunks', 0)}")
            if prep.get("error"):
                st.error(f"Error during preprocessing: {prep['error']}")
        else:
            st.info("No preprocessing logs available.")

        # Also let's show the Preprocessed HTML under a togglable expander or iframe!
        preprocessed_html = record.get("preprocessed_content")
        if preprocessed_html:
            with st.expander("👁️ View Preprocessed HTML"):
                st.components.v1.html(
                    f'<div style="background: white; color: black; padding: 10px; height: 100%; overflow: auto;">{preprocessed_html}</div>',
                    height=400,
                    scrolling=True
                )
                
    # 2. Reranking Tab
    with tabs[1]:
        rerank = logs.get("reranker")
        if rerank and rerank.get("chunks"):
            chunks_data = rerank["chunks"]
            df_chunks = pd.DataFrame(chunks_data)
            st.dataframe(df_chunks, use_container_width=True, hide_index=True)
        else:
            st.info("No reranking details available (reranker might have been disabled).")
            
    # 3. LLM Pruning Tab
    with tabs[2]:
        pruner = logs.get("pruner")
        if pruner:
            # pruner is a list of logs per chunk
            for idx, p_log in enumerate(pruner):
                if not p_log: continue
                with st.expander(f"Chunk {idx + 1} Pruning Details"):
                    st.text_area("Pruner Prompt", p_log.get("prompt", ""), height=150, key=f"prn_prompt_{record['id']}_{idx}")
                    st.text_area("LLM Response", p_log.get("response", ""), height=80, key=f"prn_resp_{record['id']}_{idx}")
                    st.write(f"**Selected Indices:** {p_log.get('selected_indices', [])}")
        else:
            st.info("No LLM pruner logs available.")
            
    # 4. Extraction Tab
    with tabs[3]:
        extractor = logs.get("extractor")
        if extractor:
            st.text_area("Generator Prompt", extractor.get("prompt", ""), height=200, key=f"ext_prompt_{record['id']}")
            st.text_area("Raw Generator Response", extractor.get("raw_response", ""), height=150, key=f"ext_resp_{record['id']}")
        else:
            st.info("No extractor logs available.")
            
    # 5. Postprocessing Tab
    with tabs[4]:
        post = logs.get("postprocessor")
        if post:
            if post.get("error"):
                st.error(f"Postprocessing Error: {post['error']}")
            
            exact_match = post.get("exact_match_log")
            if exact_match:
                st.write("**Exact Extraction / Fuzzy Alignment Log:**")
                match_data = []
                for field, details in exact_match.items():
                    match_data.append({
                        "Attribute/Field": field,
                        "Status": details.get("status"),
                        "Original Extracted": str(details.get("original_extracted")),
                        "Aligned Text": str(details.get("value")),
                        "Score": details.get("score"),
                        "XPath": details.get("xpath")
                    })
                df_match = pd.DataFrame(match_data)
                
                def color_match_status(val):
                    if val == "success": return "color: #00c853; font-weight: bold"
                    if val == "not_found": return "color: #ff9100; font-weight: bold"
                    if val == "error": return "color: #ff4b4b; font-weight: bold"
                    return ""
                
                st.dataframe(
                    df_match.style.map(color_match_status, subset=['Status']),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No exact matching logs (exact extraction might have been disabled).")
        else:
            st.info("No postprocessor logs available.")

# -----------------------------------------------------------------------------
# Views
# -----------------------------------------------------------------------------

def view_type_1_structured(df):
    st.header("🎯 Structured Extraction Analysis")
    
    # Calculate Metrics
    eval_df = pd.json_normalize(df['evaluation'])
    field_acc = eval_df.mean().sort_values(ascending=True)
    global_acc = eval_df.values.mean()

    # Top Row Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", len(df))
    m2.metric("Global Accuracy", f"{global_acc:.1%}")
    m3.metric("Perfect Records", len(df[eval_df.sum(axis=1) == len(eval_df.columns)]))
    m4.metric("Avg Errors/Doc", f"{eval_df.shape[1] - eval_df.sum(axis=1).mean():.1f}")

    # Plotly Chart (Streamlit Theme)
    fig = px.bar(
        field_acc, 
        orientation='h', 
        title="Field Accuracy (Lowest to Highest)",
        labels={'value': 'Accuracy Score', 'index': 'Field'},
        color=field_acc.values,
        color_continuous_scale="RdYlGn",
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"
    )
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # Record Inspector
    st.markdown("---")
    st.subheader("🔍 Record Inspector")
    
    # Prepare sorting
    df['error_count'] = df['evaluation'].apply(lambda x: sum(1 for v in x.values() if v == 0))
    df['display_label'] = df.apply(lambda x: f"ID: {x['id']} | ❌ {x['error_count']} Errors", axis=1)
    
    selected_label = st.selectbox("Select Record (Sorted by Error Count)", 
                                  df.sort_values('error_count', ascending=False)['display_label'])
    
    record = df[df['display_label'] == selected_label].iloc[0]
    
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        gt = safe_parse_gt(record['ground_truth'])
        pred = record['prediction']
        evals = record['evaluation']
        
        comp_data = []
        for key, score in evals.items():
            comp_data.append({
                "Field": key,
                "Status": "✅ Match" if score == 1 else "❌ Mismatch",
                "Prediction": str(pred.get(key, "MISSING")),
                "Ground Truth": str(gt.get(key, "MISSING"))
            })
        
        # Styled Table
        res_df = pd.DataFrame(comp_data)
        def color_status(val):
            if "❌" in val: return "color: #ff4b4b; font-weight: bold"
            if "✅" in val: return "color: #00c853; font-weight: bold"
            return ""

        st.dataframe(
            res_df.style.map(color_status, subset=['Status']),
            use_container_width=True,
            hide_index=True
        )
        
        with st.expander("🛠️ Raw Schema/JSON Data"):
            st.json(record.to_dict())
            
        render_diagnostics(record)

    with col_right:
        render_html_preview(record['filtered_html'])

def view_type_2_unstructured(df):
    st.header("📝 Text Generation Analysis")
    
    eval_df = pd.json_normalize(df['evaluation'])
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Samples", len(df))
    m2.metric("Avg F1", f"{eval_df['f1'].mean():.3f}")
    m3.metric("Avg Precision", f"{eval_df['precision'].mean():.3f}")
    m4.metric("Avg Recall", f"{eval_df['recall'].mean():.3f}")

    fig = px.histogram(
        eval_df, x="f1", nbins=20, 
        title="F1 Score Distribution",
        color_discrete_sequence=['#00d4ff']
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    
    st.markdown("---")
    st.subheader("🔍 Record Inspector")
    
    df['display_label'] = df.apply(lambda x: f"ID: {x['id']} (F1: {x['evaluation'].get('f1', 0):.2f})", axis=1)
    selected_label = st.selectbox("Select Record (Sorted by F1)", df.sort_values("display_label")['display_label'])
    record = df[df['display_label'] == selected_label].iloc[0]
    
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.info(f"**Query:** {record['query']}")
        
        c1, c2 = st.columns(2)
        with c1:
            st.success("**Prediction**")
            st.write(record['prediction'])
        with c2:
            st.warning("**Ground Truth**")
            st.write(record['ground_truth'])
            
        st.divider()
        st.json(record['evaluation'])
        
        render_diagnostics(record)

    with col_r:
        render_html_preview(record['filtered_html'])

# -----------------------------------------------------------------------------
# Main Execution Flow
# -----------------------------------------------------------------------------

def main():
    st.title("📂 Model Evaluation Inspector")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        uploaded_file = st.file_uploader("Upload NDJSON Results", type=["ndjson", "jsonl", "txt"])
        st.divider()
        
    if uploaded_file:
        raw_data = parse_ndjson(uploaded_file)
        if not raw_data:
            st.error("Could not parse file.")
            return

        df = pd.DataFrame(raw_data)
        file_type = detect_file_type(raw_data[0])
        st.sidebar.info(f"Detected: {file_type}")
        
        # Global Filters in Sidebar
        if "Type 2" in file_type:
            f1_threshold = st.sidebar.slider("Show F1 below", 0.0, 1.0, 1.0)
            df = df[df['evaluation'].apply(lambda x: x.get('f1', 0) <= f1_threshold)]
        else:
            if st.sidebar.checkbox("Show Errors Only", value=False):
                df = df[df['evaluation'].apply(lambda x: any(v == 0 for v in x.values()))]

        tab_dash, tab_data = st.tabs(["📊 Dashboard", "🔠 Raw Data Explorer"])
        
        with tab_dash:
            if "Type 1" in file_type:
                view_type_1_structured(df)
            else:
                view_type_2_unstructured(df)
                
        with tab_data:
            st.dataframe(df, use_container_width=True)
            
    else:
        # Landing Page
        st.empty()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.container(border=True).markdown("""
            ### Welcome! 
            Please upload an `.ndjson` file to begin the evaluation review. 
            
            **Expected format:**
            - `id`: Unique identifier
            - `query`: The input prompt/schema
            - `prediction`: The model output
            - `ground_truth`: The gold standard
            - `evaluation`: Dictionary of scores (0/1 or F1/Precision)
            - `filtered_html`: Source context
            """)

if __name__ == "__main__":
    main()