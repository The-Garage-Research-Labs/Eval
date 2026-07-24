"""
app.py
Streamlit User Interface for WebSRC Error Analysis.
Incorporates visual rendering, customizable scatter plots, domain/website heatmaps,
sorting filters, website filters, prev/next buttons, and error classifications.
"""

import json
import html as html_mod
import difflib
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px

# Imports directly from your backend helper file
import web_analysis as analysis

st.set_page_config(
    page_title="WebSRC Error Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Reference sample matches WebSRC structural specs (Taxonomy: KV)
SAMPLE_LOG = {
    "id": "au010007101567",
    "query": "How many seats are there in it.",
    "ground_truth": "2",
    "prediction": "2",
    "filtered_html": "<ul> <li> <strong> Drive Train: </strong> Rear Wheel Drive </li> <li> <strong> Passengers: </strong> 2 </li> <li> <strong> Doors: </strong> 2 </li> </ul>",
    "preprocessed_content": "<div> <h3> Vehicle Highlights </h3> <ul> <li> <strong> Fuel Economy: </strong> 25 mpg City, 32 mpg Hwy </li> <li> <strong> Engine: </strong> 2.0 L Premium Unleaded I-4 </li> </ul> <ul> <li> <strong> Drive Train: </strong> Rear Wheel Drive </li> <li> <strong> Passengers: </strong> 2 </li> <li> <strong> Doors: </strong> 2 </li> </ul>   <a> View More Features </a>  <h3> Warranty </h3> <ul> <li> <strong> Basic Warranty: </strong> 4 Years </li> </ul> </div>",
    "step_logs": {
        "preprocessor": {
            "raw_len": 1692,
            "cleaned_len": 1157,
            "num_chunks": 1
        },
        "reranker": {
            "chunks": [{"chunkid": "3-1", "score": 1.0, "score_norm": 1.0}]
        },
        "pruner": [
            {
                "prompt": "\nYou are a Smart Context Selector.\n\nQuery:\nHow many seats are there in it.\n\nContent:\n0 ('/h3', '<h3> Vehicle Highlights </h3>')\n1 ('/ul', '<ul> <li> Passengers: 2 </li> </ul>')\n\nResponse Format:\n[indices]",
                "response": "[1]",
                "selected_indices": [1]
            }
        ],
        "extractor": {
            "prompt": "Extract: How many seats are there in it based on HTML details.",
            "raw_response": "```json\n{\"answer\": \"2\"}\n```"
        },
        "postprocessor": {
            "raw_response": "```json\n{\"answer\": \"2\"}\n```",
            "error": None,
            "exact_match_log": {}
        }
    },
    "evaluation": {
        "f1": 1.0,
        "precision": 1.0,
        "recall": 1.0
    }
}

# WebSRC synthetic variations spanning different directories, taxonomy classes, and diagnostics
SYNTHETIC_MOCKS = [
    SAMPLE_LOG,
    {
        "id": "sp020008502311", # Domain: Sports, Website: 02 (Taxonomy: KV) -> Extractor Error
        "query": "Who is the head coach?",
        "ground_truth": "John Doe",
        "prediction": "Basketball",
        "filtered_html": "<p>Coach: John Doe, Sport: Basketball</p>",
        "preprocessed_content": "<div class='sports-card'><h2>New York Knicks</h2><p>Coach: John Doe</p><p>Sport: Basketball</p></div>",
        "step_logs": {
            "preprocessor": {"raw_len": 400, "cleaned_len": 280, "num_chunks": 1},
            "reranker": {"chunks": [{"chunkid": "1", "score": 0.81, "score_norm": 0.81}]},
            "pruner": [{"prompt": "Filter coach info", "response": "[1]", "selected_indices": [1]}],
            "extractor": {"prompt": "Identify coach.", "raw_response": "```json\n{\"answer\": \"Basketball\"}\n```"},
            "postprocessor": {"raw_response": "{\"answer\": \"Basketball\"}", "error": None, "exact_match_log": {}}
        },
        "evaluation": {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    },
    {
        "id": "ga010012204561", # Domain: Game, Website: 01 (Taxonomy: Compare) -> Pruner Error
        "query": "What is the game release date?",
        "ground_truth": "November 2021",
        "prediction": "Unknown",
        "filtered_html": "<span>Release Date: Unknown</span>",
        "preprocessed_content": "<div><h3>Game Info</h3><p>Release Date: November 2021</p></div>",
        "step_logs": {
            "preprocessor": {"raw_len": 350, "cleaned_len": 250, "num_chunks": 1},
            "reranker": {"chunks": [{"chunkid": "1", "score": 0.95, "score_norm": 0.95}]},
            "pruner": [{"prompt": "Find release details", "response": "[1]", "selected_indices": [1]}],
            "extractor": {"prompt": "Extract date details", "raw_response": "```json\n{\"answer\": \"Unknown\"}\n```"},
            "postprocessor": {"raw_response": "{\"answer\": \"Unknown\"}", "error": None, "exact_match_log": {}}
        },
        "evaluation": {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    },
    {
        "id": "un020001201599", # Domain: University, Website: 02 (Taxonomy: Table) -> GXR Error
        "query": "What is the tuition fee?",
        "ground_truth": "$15,000",
        "prediction": "$12,000",
        "filtered_html": "<span>Tuition: $15,000</span>",
        "preprocessed_content": "<div><table><tr><td>Tuition</td><td>$15,000</td></tr></table></div>",
        "step_logs": {
            "preprocessor": {"raw_len": 350, "cleaned_len": 250, "num_chunks": 1},
            "reranker": {"chunks": [{"chunkid": "1", "score": 0.95, "score_norm": 0.95}]},
            "pruner": [{"prompt": "Tuition details", "response": "[1]", "selected_indices": [1]}],
            "extractor": {"prompt": "Extract tuition details", "raw_response": "```json\n{\"answer\": \"$15,000\"}\n```"},
            "postprocessor": {"raw_response": "{\"answer\": \"$12,000\"}", "error": "Forced mapping failure", "exact_match_log": {}}
        },
        "evaluation": {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    }
]

st.title("📊 WebSRC Pipeline Error Analysis")
st.markdown("Analyze extraction precision, DOM structures, page layouts, and pipeline bottleneck errors across WebSRC.")

# Sidebar Configuration
st.sidebar.header("📁 WebSRC Data Loader")
uploaded_file = st.sidebar.file_uploader("Upload WebSRC evaluation logs (.ndjson)", type=["ndjson"])

# 🎨 Visualization Settings
st.sidebar.markdown("---")
st.sidebar.header("🎨 Visualization Settings")
dot_size = st.sidebar.slider("Scatter Plot Dot Size", min_value=4, max_value=30, value=12, step=1)

if uploaded_file is not None:
    try:
        raw_lines = [json.loads(line) for line in uploaded_file if line.strip()]
        df, lookup_dict = analysis.process_log_records(raw_lines)
        content_stats_df = analysis.compute_content_stats(raw_lines)
        st.sidebar.success(f"Parsed {len(df)} records.")
    except Exception as e:
        st.sidebar.error(f"Error parsing file: {e}")
        df, lookup_dict = analysis.process_log_records(SYNTHETIC_MOCKS)
        content_stats_df = analysis.compute_content_stats(SYNTHETIC_MOCKS)
else:
    st.sidebar.info("Displaying default WebSRC log samples. Upload an NDJSON file to load custom results.")
    df, lookup_dict = analysis.process_log_records(SYNTHETIC_MOCKS)
    content_stats_df = analysis.compute_content_stats(SYNTHETIC_MOCKS)

# 🔍 Interactive Filtering & Sorting Panel in Sidebar
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filter & Sort Samples")

all_error_types = ["All"] + sorted(list(df["error_classification"].unique()))
f1_filter = st.sidebar.selectbox("Error Funnel Category Filter", options=all_error_types)

domain_list = ["All"] + sorted(list(df["domain"].unique()))
domain_filter = st.sidebar.selectbox("WebSRC Domain Filter", options=domain_list)

# Dynamically filters website options based on the chosen WebSRC domain
if domain_filter != "All":
    available_websites = sorted(list(df[df["domain"] == domain_filter]["website"].unique()))
else:
    available_websites = sorted(list(df["website"].unique()))
website_list = ["All"] + available_websites
website_filter = st.sidebar.selectbox("WebSRC Website Filter", options=website_list)

qtype_filter = st.sidebar.selectbox("Structural Layout Type", options=["All", "KV", "Compare", "Table"])

sort_by = st.sidebar.selectbox(
    "Sort Order",
    options=[
        "F1 Score (Ascending)", 
        "F1 Score (Descending)", 
        "Max DOM Depth (Descending)", 
        "Total Tags (Descending)",
        "Sample ID (Ascending)"
    ]
)

search_query = st.sidebar.text_input("📝 Search (Query, Ground Truth, or Prediction Text)", "").strip().lower()

# Apply Filters to local DataFrame copy
filtered_df = df.copy()

# 1. Error Bottleneck Filters
if f1_filter != "All":
    filtered_df = filtered_df[filtered_df["error_classification"] == f1_filter]
    
# 2. Domain Filter
if domain_filter != "All":
    filtered_df = filtered_df[filtered_df["domain"] == domain_filter]
    
# 3. Website Filter
if website_filter != "All":
    filtered_df = filtered_df[filtered_df["website"] == website_filter]
    
# 4. Layout Type Filter
if qtype_filter != "All":
    filtered_df = filtered_df[filtered_df["taxonomy"] == qtype_filter]
    
# 5. Text Search
if search_query:
    filtered_df = filtered_df[
        filtered_df["query"].str.lower().str.contains(search_query, regex=False) |
        filtered_df["ground_truth"].str.lower().str.contains(search_query, regex=False) |
        filtered_df["prediction"].str.lower().str.contains(search_query, regex=False)
    ]
    
# 6. Apply Sorting
if sort_by == "F1 Score (Ascending)":
    filtered_df = filtered_df.sort_values(by="f1", ascending=True)
elif sort_by == "F1 Score (Descending)":
    filtered_df = filtered_df.sort_values(by="f1", ascending=False)
elif sort_by == "Max DOM Depth (Descending)":
    filtered_df = filtered_df.sort_values(by="max_dom_depth", ascending=False)
elif sort_by == "Total Tags (Descending)":
    filtered_df = filtered_df.sort_values(by="total_dom_tags", ascending=False)
elif sort_by == "Sample ID (Ascending)":
    filtered_df = filtered_df.sort_values(by="id", ascending=True)

# Tab views for clean visual navigation
tabs = st.tabs([
    "WebSRC Overview & Metrics", 
    "DOM Complexity & Layouts", 
    "RAG Diagnostic Bottlenecks", 
    "Granular Case Explorer",
    "DOM Element Frequency Analysis"
])

# ----------------------------------------------------
# TAB 1: OVERVIEW & PERFORMANCE ANALYSIS
# ----------------------------------------------------
with tabs[0]:
    st.header("Overall WebSRC Insights")
    
    if filtered_df.empty:
        st.warning("No samples match the selected filtering choices. Please adjust the sidebar filters.")
    else:
        kpi_cols = st.columns(4)
        avg_f1 = filtered_df["f1"].mean()
        avg_precision = filtered_df["precision"].mean()
        avg_recall = filtered_df["recall"].mean()
        accuracy = filtered_df["is_correct"].mean()

        kpi_cols[0].metric("Average F1 Score", f"{avg_f1:.2%}")
        kpi_cols[1].metric("Average Precision", f"{avg_precision:.2%}")
        kpi_cols[2].metric("Average Recall", f"{avg_recall:.2%}")
        kpi_cols[3].metric("Accuracy (F1 = 1.0)", f"{accuracy:.2%}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Performance by WebSRC Domain")
            domain_df = filtered_df.groupby("domain")[["f1", "precision", "recall"]].mean().reset_index()
            fig_domain = px.bar(
                domain_df, 
                x="domain", 
                y=["f1", "precision", "recall"], 
                barmode="group",
                labels={"value": "Score", "variable": "Metric", "domain": "WebSRC Domain"},
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_domain.update_layout(yaxis_range=[0, 1.05])
            st.plotly_chart(fig_domain, use_container_width=True)

        with col2:
            st.subheader("Performance by Question Style")
            query_type_df = filtered_df.groupby("query_type")[["f1", "precision", "recall"]].mean().reset_index()
            fig_query = px.bar(
                query_type_df,
                x="query_type",
                y=["f1", "precision", "recall"],
                barmode="group",
                labels={"value": "Score", "variable": "Metric", "query_type": "Query Category"},
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_query.update_layout(yaxis_range=[0, 1.05])
            st.plotly_chart(fig_query, use_container_width=True)

        st.markdown("---")
        # ── Token & Content Length Statistics ──────────────────────────────────
        st.subheader("🔢 Token & Content Length Statistics")
        st.markdown(
            "Character lengths and approximate token counts (chars ÷ 4) for each "
            "preprocessing stage across the **filtered** sample set."
        )

        # Join content_stats_df with filtered_df on id so stats respect sidebar filters
        filtered_ids = set(filtered_df["id"].tolist())
        cs = content_stats_df[content_stats_df["id"].isin(filtered_ids)]

        if cs.empty:
            st.info("No content statistics available for the current filter.")
        else:
            # ── Row 1: KPI tiles ──
            stat_cols = st.columns(6)
            stat_cols[0].metric(
                "Avg Raw Tokens",
                f"{cs['raw_tokens'].mean():,.0f}",
                help="Mean approx. token count of the raw HTML before preprocessing (chars ÷ 4)"
            )
            stat_cols[1].metric(
                "Avg Preprocessed Tokens",
                f"{cs['cleaned_tokens'].mean():,.0f}",
                help="Mean approx. token count after preprocessing / cleaning"
            )
            stat_cols[2].metric(
                "Avg Filtered Tokens",
                f"{cs['filtered_tokens'].mean():,.0f}",
                help="Mean approx. token count of the final filtered HTML segment sent to the extractor"
            )
            stat_cols[3].metric(
                "Avg Reduction Ratio",
                f"{cs['reduction_ratio'].mean():.1%}",
                help="Average ratio of cleaned_len / raw_len – how much the preprocessor compresses the HTML"
            )
            stat_cols[4].metric(
                "Avg Chunks",
                f"{cs['num_chunks'].mean():.1f}",
                help="Average number of reranker chunks produced by the chunker"
            )
            stat_cols[5].metric(
                "Median Filtered Tokens",
                f"{cs['filtered_tokens'].median():,.0f}",
                help="Median token count of the filtered HTML (robust to outliers)"
            )

            st.markdown("")
            # ── Row 2: Histograms ──
            hist_col1, hist_col2, hist_col3 = st.columns(3)

            with hist_col1:
                fig_hist_raw = px.histogram(
                    cs, x="raw_tokens",
                    nbins=30,
                    labels={"raw_tokens": "Raw Approx. Tokens"},
                    title="Raw HTML Token Distribution",
                    color_discrete_sequence=["#6366f1"]
                )
                fig_hist_raw.update_layout(showlegend=False, margin=dict(t=40, b=20))
                st.plotly_chart(fig_hist_raw, use_container_width=True)

            with hist_col2:
                fig_hist_clean = px.histogram(
                    cs, x="cleaned_tokens",
                    nbins=30,
                    labels={"cleaned_tokens": "Preprocessed Approx. Tokens"},
                    title="Preprocessed Token Distribution",
                    color_discrete_sequence=["#22c55e"]
                )
                fig_hist_clean.update_layout(showlegend=False, margin=dict(t=40, b=20))
                st.plotly_chart(fig_hist_clean, use_container_width=True)

            with hist_col3:
                fig_hist_filt = px.histogram(
                    cs, x="filtered_tokens",
                    nbins=30,
                    labels={"filtered_tokens": "Filtered Approx. Tokens"},
                    title="Filtered HTML Token Distribution",
                    color_discrete_sequence=["#f59e0b"]
                )
                fig_hist_filt.update_layout(showlegend=False, margin=dict(t=40, b=20))
                st.plotly_chart(fig_hist_filt, use_container_width=True)

            # ── Row 3: Per-stage comparison bar ──
            stage_summary = pd.DataFrame({
                "Stage": ["Raw HTML", "Preprocessed", "Filtered HTML"],
                "Avg Tokens":  [
                    cs["raw_tokens"].mean(),
                    cs["cleaned_tokens"].mean(),
                    cs["filtered_tokens"].mean(),
                ],
                "Median Tokens": [
                    cs["raw_tokens"].median(),
                    cs["cleaned_tokens"].median(),
                    cs["filtered_tokens"].median(),
                ],
                "Max Tokens": [
                    cs["raw_tokens"].max(),
                    cs["cleaned_tokens"].max(),
                    cs["filtered_tokens"].max(),
                ],
            })

            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                fig_stage = px.bar(
                    stage_summary.melt(id_vars="Stage", var_name="Statistic", value_name="Tokens"),
                    x="Stage", y="Tokens", color="Statistic",
                    barmode="group",
                    title="Token Count by Pipeline Stage",
                    labels={"Tokens": "Approx. Token Count"},
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                st.plotly_chart(fig_stage, use_container_width=True)

            with comp_col2:
                fig_red = px.histogram(
                    cs, x="reduction_ratio",
                    nbins=20,
                    labels={"reduction_ratio": "Reduction Ratio (cleaned / raw)"},
                    title="HTML Reduction Ratio Distribution",
                    color_discrete_sequence=["#ec4899"]
                )
                fig_red.update_layout(showlegend=False, margin=dict(t=40, b=20))
                st.plotly_chart(fig_red, use_container_width=True)

            # ── Summary table ──
            with st.expander("📋 Full Content Statistics Summary Table"):
                display_cs = cs[[
                    "id", "raw_len", "cleaned_len", "filtered_len",
                    "raw_tokens", "cleaned_tokens", "filtered_tokens",
                    "num_chunks", "reduction_ratio"
                ]].copy()
                display_cs["reduction_ratio"] = display_cs["reduction_ratio"].map("{:.1%}".format)
                st.dataframe(display_cs, use_container_width=True, hide_index=True)


        
        heatmap_data = filtered_df.groupby(["domain", "website"])["f1"].mean().reset_index()
        if not heatmap_data.empty:
            pivot_matrix = heatmap_data.pivot(index="domain", columns="website", values="f1")
            
            fig_heatmap = px.imshow(
                pivot_matrix,
                labels=dict(x="Website ID", y="Domain Name", color="Avg F1 Score"),
                x=pivot_matrix.columns,
                y=pivot_matrix.index,
                color_continuous_scale="RdYlGn",
                zmin=0,
                zmax=1,
                text_auto=".2f"
            )
            fig_heatmap.update_layout(
                xaxis_title="WebSRC Website Code",
                yaxis_title="WebSRC Domain",
                coloraxis_colorbar=dict(title="Average F1")
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("No data available for the heatmap.")

# ----------------------------------------------------
# TAB 2: DOM COMPLEXITY & TAXONOMY INSIGHTS
# ----------------------------------------------------
with tabs[1]:
    st.header("DOM Complexity & Page Layout Taxonomy Analysis")
    st.markdown("Assess how structural layout types (KV, Compare, Table) and DOM depth hierarchies correlate with model predictions.")
    
    if filtered_df.empty:
        st.warning("No samples match the selected filtering choices. Please adjust the sidebar filters.")
    else:
        col_tax1, col_tax2 = st.columns(2)
        with col_tax1:
            st.subheader("Metrics by Page Structural Layout (Taxonomy)")
            tax_perf = filtered_df.groupby("taxonomy")[["f1", "precision", "recall"]].mean().reset_index()
            fig_tax_perf = px.bar(
                tax_perf,
                x="taxonomy",
                y=["f1", "precision", "recall"],
                barmode="group",
                labels={"value": "Average Score", "variable": "Metric", "taxonomy": "Structural Layout"},
                color_discrete_sequence=px.colors.qualitative.T10
            )
            fig_tax_perf.update_layout(yaxis_range=[0, 1.05])
            st.plotly_chart(fig_tax_perf, use_container_width=True)
            
        with col_tax2:
            st.subheader("Structural Layout Composition")
            fig_pie = px.pie(
                filtered_df,
                names="taxonomy",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")
        col_dom1, col_dom2 = st.columns(2)
        with col_dom1:
            st.subheader("Nesting Depth vs. F1 Score")
            fig_depth = px.scatter(
                filtered_df,
                x="max_dom_depth",
                y="f1",
                color="taxonomy",
                hover_data=["id", "query"],
                labels={"max_dom_depth": "Max Tree Depth", "f1": "F1 Score", "taxonomy": "Layout Type"}
            )
            fig_depth.update_traces(marker=dict(size=dot_size))
            fig_depth.update_layout(yaxis_range=[-0.05, 1.05])
            st.plotly_chart(fig_depth, use_container_width=True)
            
        with col_dom2:
            st.subheader("DOM Nesting Distribution across Structural Layouts")
            fig_box = px.box(
                filtered_df,
                x="taxonomy",
                y="max_dom_depth",
                color="taxonomy",
                points="all",
                labels={"taxonomy": "Layout Type", "max_dom_depth": "Max DOM Depth"},
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # Reranker Confidence Correlation
        if "reranker_top_score" in filtered_df.columns and filtered_df["reranker_top_score"].notna().any():
            st.markdown("---")
            st.subheader("Reranker Confidence vs. F1 Score")
            st.markdown("Does low reranker confidence predict downstream extraction errors?")
            fig_reranker = px.scatter(
                filtered_df,
                x="reranker_top_score",
                y="f1",
                color="error_classification",
                hover_data=["id", "query"],
                labels={"reranker_top_score": "Top Reranker Chunk Score", "f1": "F1 Score", "error_classification": "Error Type"},
                opacity=0.7
            )
            fig_reranker.update_traces(marker=dict(size=dot_size))
            fig_reranker.update_layout(yaxis_range=[-0.05, 1.05], xaxis_range=[-0.05, 1.05])
            st.plotly_chart(fig_reranker, use_container_width=True)

# ----------------------------------------------------
# TAB 3: RAG DIAGNOSTIC BOTTLENECK ANALYSIS
# ----------------------------------------------------
with tabs[2]:
    st.header("Pipeline Error Funnel & Failure Bottlenecks")
    st.markdown("Diagnostics highlighting exactly where errors occur in your RAG execution chain (Pruner vs. Extractor vs. Postprocessor).")
    
    if filtered_df.empty:
        st.warning("No samples match the selected filtering choices. Please adjust the sidebar filters.")
    else:
        # 1. Broad Error Categories Breakdown
        col_err1, col_err2 = st.columns(2)
        with col_err1:
            st.subheader("RAG Funnel Distribution")
            err_counts = filtered_df["error_classification"].value_counts().reset_index()
            err_counts.columns = ["Error Classification", "Count"]
            
            fig_err_pie = px.pie(
                err_counts,
                values="Count",
                names="Error Classification",
                color="Error Classification",
                color_discrete_map={
                    "Success": "green",
                    "Extractor Error": "red",
                    "Pruner Error": "orange",
                    "GXR Error": "purple",
                    "Hallucination": "brown",
                    "Skipped (Yes/No)": "grey",
                    "Investigate": "pink",
                    "Empty Prediction": "#4a4a4a",
                },
                hole=0.4
            )
            st.plotly_chart(fig_err_pie, use_container_width=True)
            
        with col_err2:
            st.subheader("Error Classifications Summary")
            st.dataframe(err_counts, use_container_width=True, hide_index=True)

        # 2. Sectioned Error breakdowns across Domains and Structural taxonomies
        error_subset = filtered_df[filtered_df["error_classification"] != "Success"]
        if not error_subset.empty:
            st.markdown("---")
            col_bar1, col_bar2 = st.columns(2)
            
            with col_bar1:
                st.subheader("Pipeline Failures by WebSRC Domain")
                grouped_err_dom = error_subset.groupby(["domain", "error_classification"]).size().reset_index(name="Count")
                fig_err_dom = px.bar(
                    grouped_err_dom,
                    x="domain",
                    y="Count",
                    color="error_classification",
                    barmode="stack",
                    labels={"domain": "Domain Name", "Count": "Error Count", "error_classification": "Error Category"}
                )
                st.plotly_chart(fig_err_dom, use_container_width=True)
                
            with col_bar2:
                st.subheader("Pipeline Failures by Layout Style")
                grouped_err_tax = error_subset.groupby(["taxonomy", "error_classification"]).size().reset_index(name="Count")
                fig_err_tax = px.bar(
                    grouped_err_tax,
                    x="taxonomy",
                    y="Count",
                    color="error_classification",
                    barmode="stack",
                    labels={"taxonomy": "Layout Taxonomy", "Count": "Error Count", "error_classification": "Error Category"}
                )
                st.plotly_chart(fig_err_tax, use_container_width=True)
                
            st.markdown("---")
            st.subheader("DOM Nesting Depth distribution by Pipeline Failure Classification")
            fig_err_box = px.box(
                error_subset,
                x="error_classification",
                y="max_dom_depth",
                color="error_classification",
                points="all",
                labels={"error_classification": "Error Category", "max_dom_depth": "Max DOM Depth"}
            )
            st.plotly_chart(fig_err_box, use_container_width=True)
        else:
            st.success("No system failures or extraction errors detected in the loaded dataset.")

# ----------------------------------------------------
# TAB 4: CASE EXPLORER
# ----------------------------------------------------
with tabs[3]:
    st.header("Granular Sample Analysis")
    st.markdown("Search, filter, and sort across log executions to examine RAG pipeline behaviors.")

    # UI Flow for matching datasets
    id_list = filtered_df["id"].tolist()
    
    # Aggregate statistics for the current filter
    _n_filtered = len(filtered_df)
    _n_errors = int((filtered_df["is_correct"] == 0.0).sum()) if _n_filtered else 0
    _avg_f1 = filtered_df["f1"].mean() if _n_filtered else 0.0
    st.markdown(
        f"<div style='background: linear-gradient(90deg, #1e293b 0%, #334155 100%); "
        f"border-radius: 8px; padding: 10px 20px; margin-bottom: 12px; "
        f"display: flex; gap: 30px; color: #e2e8f0; font-size: 0.95em;'>"
        f"<span>📋 <b>{_n_filtered}</b> samples</span>"
        f"<span>❌ <b>{_n_errors}</b> errors</span>"
        f"<span>📊 Avg F1: <b>{_avg_f1:.2%}</b></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not id_list:
        st.warning("No samples match the selected filtering or sorting choices.")
    else:
        # Generate options matching the filtered subset
        sample_options = filtered_df.apply(
            lambda row: f"{row['id']} | F1: {row['f1']:.2f} | Error: {row['error_classification']} | Msg: {row['query'][:40]}...", axis=1
        ).tolist()

        # Stateful Sample Initialization
        if "selected_sample_id" not in st.session_state or st.session_state.selected_sample_id not in id_list:
            st.session_state.selected_sample_id = id_list[0]
            
        current_id = st.session_state.selected_sample_id
        current_idx = id_list.index(current_id)

        # 🔄 Previous and Next Sample Navigation Buttons
        st.markdown("### Navigation Controls")
        col_prev, col_status, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("⬅️ Previous Sample", use_container_width=True):
                current_idx = (current_idx - 1) % len(id_list)
                new_id = id_list[current_idx]
                st.session_state.selected_sample_id = new_id
                st.session_state.sample_selectbox = sample_options[current_idx]
                st.rerun()
        with col_status:
            st.markdown(
                f"<p style='text-align: center; font-size: 1.1em; font-weight: bold; margin-top: 5px;'>"
                f"Sample {current_idx + 1} of {len(id_list)}"
                f"</p>", 
                unsafe_allow_html=True
            )
        with col_next:
            if st.button("Next Sample ➡️", use_container_width=True):
                current_idx = (current_idx + 1) % len(id_list)
                new_id = id_list[current_idx]
                st.session_state.selected_sample_id = new_id
                st.session_state.sample_selectbox = sample_options[current_idx]
                st.rerun()

        # Keep dropdown selected index synced with button actions
        matching_options = [opt for opt in sample_options if opt.startswith(current_id)]
        if matching_options:
            matching_option = matching_options[0]
            matching_index = sample_options.index(matching_option)
        else:
            matching_index = 0

        # Safety check: if sample_selectbox is not in sample_options (e.g. after filter change), reset it
        if "sample_selectbox" in st.session_state and st.session_state.sample_selectbox not in sample_options:
            st.session_state.sample_selectbox = sample_options[matching_index]

        # Callback helper to change state when selecting from selectbox manually
        def handle_selectbox_change():
            if "sample_selectbox" in st.session_state:
                selected_option = st.session_state.sample_selectbox
                if selected_option:
                    st.session_state.selected_sample_id = selected_option.split(" | ")[0]

        selected_option = st.selectbox(
            "Jump to Sample ID", 
            options=sample_options,
            index=matching_index,
            key="sample_selectbox",
            on_change=handle_selectbox_change,
            help="Select a sample to inspect its execution logs, HTML renders, and metric details."
        )
        
        # Load active row details
        selected_id = selected_option.split(" | ")[0]
        sample_data = lookup_dict[selected_id]
        sample_row = df[df["id"] == selected_id].iloc[0]

        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.markdown(f"**WebSRC Directory:** `{sample_row['domain'].lower()}`")
            st.markdown(f"**Sub-website Name:** `{sample_row['website_code']}`")
            st.markdown(f"**Page Target ID:** `{sample_row['page_id']}.html`")
        with col_info2:
            st.markdown(f"**Structural Layout:** `{sample_row['taxonomy']}`")
            st.markdown(f"**DOM Max Nesting Depth:** {sample_row['max_dom_depth']}")
            st.markdown(f"**Tag Count:** {sample_row['total_dom_tags']}")
        with col_info3:
            st.markdown(f"**RAG Bottleneck Type:** `{sample_row['error_classification']}`")
            st.markdown(f"**F1 Accuracy:** {sample_row['f1']:.2f}")
            st.markdown(f"**Recall Accuracy:** {sample_row['recall']:.2f}")

        # Bottleneck matching details inspector card
        st.markdown("---")
        if sample_row["error_classification"] != "Success":
            st.subheader("🔍 Bottleneck Diagnostic Information")
            col_diag1, col_diag2, col_diag3 = st.columns(3)
            with col_diag1:
                st.markdown(f"**Found In filtered_html:** `{sample_row['found_in_filtered']}`")
                st.markdown(f"**Match Type (Filtered):** `{sample_row['match_type_filtered']}`")
                st.markdown(f"**Fuzzy Match Score (Filtered):** `{sample_row['score_filtered']}`")
            with col_diag2:
                st.markdown(f"**Found In preprocessed_content:** `{sample_row['found_in_preprocessed']}`")
                st.markdown(f"**Match Type (Preprocessed):** `{sample_row['match_type_preprocessed']}`")
                st.markdown(f"**Fuzzy Match Score (Preprocessed):** `{sample_row['score_preprocessed']}`")
            with col_diag3:
                st.markdown(f"**Extractor Output Error Checked:** `{sample_row['extractor_had_wrong_value']}`")
            st.markdown("---")

        col_qa1, col_qa2 = st.columns(2)
        with col_qa1:
            st.info(f"**Query:**\n\n{sample_row['query']}")
        with col_qa2:
            is_match = sample_row["is_correct"] == 1.0
            color_block = "green" if is_match else "red"
            match_label = "MATCH" if is_match else "MISMATCH"
            gt_safe = html_mod.escape(str(sample_row['ground_truth']))
            pred_safe = html_mod.escape(str(sample_row['prediction']))
            
            st.markdown(
                f"""
                <div style="border-left: 5px solid {color_block}; padding-left: 15px; margin-top: 10px;">
                    <strong>Ground Truth:</strong> <code style="font-size: 1.1em;">{gt_safe}</code><br>
                    <strong>Prediction:</strong> <code style="font-size: 1.1em;">{pred_safe}</code><br>
                    <span style="color: {color_block}; font-weight: bold;">Result: {match_label}</span>
                </div>
                """, 
                unsafe_allow_html=True
            )

        st.markdown("### Execution Pipeline Steps")
        step_logs = sample_data.get("step_logs", {})
        
        step_tabs = st.tabs([
            "1. Preprocessor", 
            "2. Reranker", 
            "3. Pruner", 
            "4. Extractor", 
            "5. Postprocessor"
        ])
        
        # 1. Preprocessor tab
        with step_tabs[0]:
            prep_logs = step_logs.get("preprocessor", {})
            st.markdown("**Parser Tag Distributions**")
            
            html_analysis = analysis.analyze_dom(sample_data.get("preprocessed_content", ""))
            col_prep1, col_prep2 = st.columns([1, 2])
            
            with col_prep1:
                st.write(prep_logs)
                ratio = prep_logs.get('cleaned_len', 0) / max(prep_logs.get('raw_len', 1), 1)
                st.metric("HTML Reduction Ratio", f"{ratio:.2%}")
            with col_prep2:
                freq_df = pd.DataFrame(
                    list(html_analysis["tag_frequencies"].items()), 
                    columns=["HTML Tag", "Count"]
                ).sort_values(by="Count", ascending=False)
                st.dataframe(freq_df, height=180, hide_index=True)
                
        # 2. Reranker tab
        with step_tabs[1]:
            st.markdown("**Scored Rerank Iterations**")
            st.json(step_logs.get("reranker", {}))
            
        # 3. Pruner tab
        with step_tabs[2]:
            prune_logs = step_logs.get("pruner", [])
            if prune_logs and isinstance(prune_logs, list):
                for idx, prune_run in enumerate(prune_logs):
                    st.markdown(f"**Pruning Run Instance #{idx + 1}**")
                    col_prune1, col_prune2 = st.columns([2, 1])
                    with col_prune1:
                        st.text_area("System Selector Prompt", value=prune_run.get("prompt", ""), height=250, disabled=True, key=f"prune_prompt_{idx}")
                    with col_prune2:
                        st.json({
                            "Selected indices": prune_run.get("selected_indices", []),
                            "Raw selector response": prune_run.get("response", "")
                        })
            else:
                st.warning("No pruner stage execution log present.")

        # 4. Extractor tab
        with step_tabs[3]:
            ext_logs = step_logs.get("extractor", {})
            col_ext1, col_ext2 = st.columns([2, 1])
            with col_ext1:
                st.text_area("Instruction Prompt", value=ext_logs.get("prompt", ""), height=250, disabled=True)
            with col_ext2:
                st.markdown("**LLM Output**")
                st.code(ext_logs.get("raw_response", ""), language="json")

        # 5. Postprocessor tab
        with step_tabs[4]:
            post_logs = step_logs.get("postprocessor", {})
            col_post1, col_post2 = st.columns(2)
            with col_post1:
                st.markdown("**Input Extraction Response**")
                st.code(post_logs.get("raw_response", ""), language="json")
            with col_post2:
                st.markdown("**Postprocessor Errors & Flags**")
                st.json({
                    "error": post_logs.get("error"),
                    "exact_match_log": post_logs.get("exact_match_log", {})
                })

        # Token-level Diff Viewer for non-exact matches
        if sample_row["is_correct"] != 1.0 and str(sample_row['ground_truth']).strip() and str(sample_row['prediction']).strip():
            st.markdown("### 🔬 Token-Level Diff (Ground Truth vs. Prediction)")
            gt_tokens = str(sample_row['ground_truth']).split()
            pred_tokens = str(sample_row['prediction']).split()
            diff = list(difflib.ndiff(gt_tokens, pred_tokens))
            
            diff_html_parts = []
            for token in diff:
                tag = token[:2]
                word = html_mod.escape(token[2:])
                if tag == '- ':
                    diff_html_parts.append(f'<span style="background:#fca5a5;color:#7f1d1d;padding:2px 4px;border-radius:3px;margin:1px;text-decoration:line-through;" title="In GT, missing from Prediction">{word}</span>')
                elif tag == '+ ':
                    diff_html_parts.append(f'<span style="background:#86efac;color:#14532d;padding:2px 4px;border-radius:3px;margin:1px;" title="In Prediction, not in GT">{word}</span>')
                elif tag == '  ':
                    diff_html_parts.append(f'<span style="padding:2px 4px;margin:1px;">{word}</span>')
            
            st.markdown(
                f'<div style="font-family: monospace; font-size: 1.05em; line-height: 2; padding: 12px; '
                f'border: 1px solid #444; border-radius: 8px; background: #1e1e2e;">'
                f'{" ".join(diff_html_parts)}</div>'
                f'<div style="margin-top:6px;font-size:0.85em;color:#888;">'
                f'<span style="background:#fca5a5;color:#7f1d1d;padding:1px 4px;border-radius:3px;">strikethrough</span> = in GT only &nbsp; '
                f'<span style="background:#86efac;color:#14532d;padding:1px 4px;border-radius:3px;">green</span> = in prediction only &nbsp; '
                f'plain = shared</div>',
                unsafe_allow_html=True
            )

        # Content source inspector with Rendered Preview & Raw Code toggles
        st.markdown("### HTML Source Code & Rendered Preview")
        source_tabs = st.tabs(["Preprocessed Content", "Filtered HTML Segment"])
        
        with source_tabs[0]:
            prep_sub_tabs = st.tabs(["Rendered Preview", "Raw Source Code"])
            with prep_sub_tabs[0]:
                html_code = sample_data.get("preprocessed_content", "")
                styled_html = f"""
                <div style="font-family: system-ui, -apple-system, sans-serif; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #fcfcfc; color: #1a1a1a;">
                    {html_code}
                </div>
                """
                components.html(styled_html, height=300, scrolling=True)
            with prep_sub_tabs[1]:
                st.code(html_code, language="html")
                
        with source_tabs[1]:
            filt_sub_tabs = st.tabs(["Rendered Preview", "Raw Source Code"])
            with filt_sub_tabs[0]:
                html_code_filt = sample_data.get("filtered_html", "")
                styled_html_filt = f"""
                <div style="font-family: system-ui, -apple-system, sans-serif; padding: 15px; border: 1px solid #ddd; border-radius: 8px; background-color: #fcfcfc; color: #1a1a1a;">
                    {html_code_filt}
                </div>
                """
                components.html(styled_html_filt, height=200, scrolling=True)
            with filt_sub_tabs[1]:
                st.code(html_code_filt, language="html")

        # CSV Export
        st.markdown("---")
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Error Report (CSV)",
            data=csv_data,
            file_name="websrc_error_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ----------------------------------------------------
# TAB 5: DOM ELEMENT FREQUENCY ANALYSIS
# ----------------------------------------------------
with tabs[4]:
    st.header("DOM Element Frequency vs. Downstream F1 Score")
    st.markdown("Analyze how the presence and density of specific HTML tags correlate with question answering performance.")

    if filtered_df.empty:
        st.warning("No samples match the selected filtering choices. Please adjust the sidebar filters.")
    else:
        # Get all unique HTML tags in the dataset
        all_tags = set()
        for tag_freq in df["tag_frequencies"]:
            if isinstance(tag_freq, dict):
                all_tags.update(tag_freq.keys())
        all_tags = sorted(list(all_tags))

        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            selected_tags = st.multiselect(
                "Select HTML Tags to Analyze", 
                options=all_tags, 
                default=[t for t in ["table", "div", "span", "a", "button"] if t in all_tags]
            )
        with col_sel2:
            freq_metric = st.radio(
                "Frequency Metric",
                options=["Raw Count", "Relative Density (% of total tags)"],
                horizontal=True
            )

        if not selected_tags:
            st.info("Select one or more HTML tags from the dropdown above to generate the analysis.")
        else:
            # Create a plotting dataframe
            plot_records = []
            for _, row in filtered_df.iterrows():
                tag_freq = row.get("tag_frequencies") or {}
                total_tags = row.get("total_dom_tags") or 1
                if total_tags == 0:
                    total_tags = 1
                
                record = {
                    "id": row["id"],
                    "f1": row["f1"],
                    "domain": row["domain"],
                    "taxonomy": row["taxonomy"],
                    "error_classification": row["error_classification"]
                }
                
                # Compute frequency for each selected tag
                for tag in selected_tags:
                    count = tag_freq.get(tag, 0)
                    if freq_metric == "Raw Count":
                        record[f"{tag}_freq"] = count
                    else:
                        record[f"{tag}_freq"] = (count / total_tags) * 100
                
                plot_records.append(record)
                
            plot_df = pd.DataFrame(plot_records)

            # Let the user choose which tag to plot on X-axis (if multiple are selected)
            if len(selected_tags) > 1:
                x_tag = st.selectbox("Select Tag for X-axis", options=selected_tags)
            else:
                x_tag = selected_tags[0]

            x_col = f"{x_tag}_freq"
            x_label = f"Raw Count of <{x_tag}>" if freq_metric == "Raw Count" else f"Density of <{x_tag}> (% of total tags)"

            st.markdown("---")
            st.subheader(f"Scatter Plot: {x_label} vs. F1 Score")
            
            # Color by options
            color_by = st.selectbox(
                "Color Points By",
                options=["taxonomy", "domain", "error_classification"]
            )

            fig_freq_scatter = px.scatter(
                plot_df,
                x=x_col,
                y="f1",
                color=color_by,
                hover_data=["id", "domain", "taxonomy"],
                labels={x_col: x_label, "f1": "F1 Score", color_by: color_by.capitalize()},
                opacity=0.7
            )
            fig_freq_scatter.update_traces(marker=dict(size=dot_size))
            fig_freq_scatter.update_layout(yaxis_range=[-0.05, 1.05])
            st.plotly_chart(fig_freq_scatter, use_container_width=True)

            # Show correlation
            correlation_value = plot_df[x_col].corr(plot_df["f1"])
            if pd.isna(correlation_value):
                corr_text = "No variation in values to compute correlation."
            else:
                corr_text = f"Pearson correlation coefficient: **{correlation_value:.3f}**"
                if correlation_value < -0.2:
                    corr_text += " (Significant negative correlation: more of this element tends to decrease F1 score)"
                elif correlation_value > 0.2:
                    corr_text += " (Significant positive correlation: more of this element tends to increase F1 score)"
                else:
                    corr_text += " (Weak or no linear correlation)"
            
            st.markdown(f"📊 **Correlation Analysis:** {corr_text}")

            st.markdown("---")
            st.subheader(f"F1 Score Binned by {x_tag} Frequency")
            st.markdown("Average F1 score grouped by ranges of tag frequency.")
            
            # Create bins dynamically
            if plot_df[x_col].max() == 0:
                st.info("All samples have a frequency of 0 for this tag.")
            else:
                if freq_metric == "Raw Count":
                    max_val = int(plot_df[x_col].max())
                    if max_val <= 5:
                        bins = [-1, 0, 1, 2, 3, 4, max_val + 1]
                        labels = ["0", "1", "2", "3", "4", f"5+"]
                    else:
                        bins = [-1, 0, 2, 5, 10, 20, max_val + 1]
                        labels = ["0", "1-2", "3-5", "6-10", "11-20", f"21+"]
                else:
                    max_val = plot_df[x_col].max()
                    bins = [-0.1, 0.0, 5.0, 10.0, 20.0, 50.0, 100.1]
                    labels = ["0%", "0.1% - 5%", "5% - 10%", "10% - 20%", "20% - 50%", "50% - 100%"]

                # Ensure bins are unique and sorted
                unique_bins = sorted(list(set(bins)))
                actual_labels = labels[:len(unique_bins)-1]
                
                if len(unique_bins) > 1:
                    plot_df["bin"] = pd.cut(plot_df[x_col], bins=unique_bins, labels=actual_labels, include_lowest=True)
                    binned_df = plot_df.groupby("bin")["f1"].agg(["mean", "count"]).reset_index()
                    
                    fig_freq_bar = px.bar(
                        binned_df,
                        x="bin",
                        y="mean",
                        hover_data=["count"],
                        labels={"bin": f"{x_tag} Frequency Range", "mean": "Average F1 Score"},
                        color="mean",
                        color_continuous_scale="RdYlGn",
                        range_color=[0, 1],
                        text_auto=".2f"
                    )
                    fig_freq_bar.update_layout(yaxis_range=[0, 1.05])
                    st.plotly_chart(fig_freq_bar, use_container_width=True)
                else:
                    st.info("Not enough data variation to bin tag frequency.")