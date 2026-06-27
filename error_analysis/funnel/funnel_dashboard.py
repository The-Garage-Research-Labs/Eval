# dashboard_controls.py
import streamlit as st
import polars as pl
# pyrefly: ignore [missing-import]
from run_funnel_analysis import MatchingConfig, run_funnel_analysis

st.sidebar.title("Matching Settings")

with st.sidebar.expander("Normalization", expanded=True):
    strip_html          = st.checkbox("Strip HTML tags",            value=True)
    decode_entities     = st.checkbox("Decode HTML entities",       value=True)
    unicode_normalize   = st.checkbox("Unicode NFKC normalize",     value=True)
    lowercase           = st.checkbox("Lowercase",                  value=True)
    collapse_whitespace = st.checkbox("Collapse whitespace",        value=True)
    strip_punct         = st.checkbox("Strip punctuation",          value=False)

with st.sidebar.expander("Matching Strategies", expanded=True):
    use_substring      = st.checkbox("Substring match",   value=True)
    use_token_subset   = st.checkbox("Token-subset match",value=True)
    use_prefix_match   = st.checkbox("Prefix match",      value=True)
    use_fuzzy          = st.checkbox("Fuzzy match (rapidfuzz / difflib)", value=True)
    fuzzy_threshold    = st.slider("Fuzzy threshold (text search, partial_ratio)",
                                   0, 100, 90)
    pp_fuzzy_threshold = st.slider("Postprocessor compare threshold (ratio)",
                                   0, 100, 95)
    min_candidate_len  = st.slider("Min candidate length for fuzzy", 1, 10, 3)

cfg = MatchingConfig(
    strip_html=strip_html,
    decode_entities=decode_entities,
    unicode_normalize=unicode_normalize,
    lowercase=lowercase,
    collapse_whitespace=collapse_whitespace,
    strip_punct=strip_punct,
    use_substring=use_substring,
    use_token_subset=use_token_subset,
    use_prefix_match=use_prefix_match,
    use_fuzzy=use_fuzzy,
    fuzzy_threshold=float(fuzzy_threshold),
    postprocessor_fuzzy_threshold=float(pp_fuzzy_threshold),
    min_candidate_len=min_candidate_len,
)

ndjson_path = st.text_input("NDJSON path",
                            "/home/abdo/PAPER/Eval/swde_camera/metric/page_level_f1_sample_eval.ndjson")

if st.button("Run funnel analysis") and ndjson_path:
    df = run_funnel_analysis(ndjson_path, cfg=cfg, verbose=True)

    # ---- Summary ----
    st.subheader("Error Summary")
    st.dataframe(df.group_by("error_classification").len().sort("error_classification"))

    # ---- Filters ----
    st.subheader("Errors")
    types = st.multiselect("Filter by classification",
                           options=df["error_classification"].unique().to_list(),
                           default=df["error_classification"].unique().to_list())
    keys  = st.multiselect("Filter by key",
                           options=df["key"].unique().to_list(),
                           default=df["key"].unique().to_list())
    view = df.filter(
        pl.col("error_classification").is_in(types)
        & pl.col("key").is_in(keys)
    )
    st.dataframe(view.to_pandas(), use_container_width=True)

    # ---- Drill-down ----
    st.subheader("Drill-down")
    if view.height > 0:
        rid = st.selectbox("Record id", view["id"].unique().to_list())
        k   = st.selectbox("Key", view.filter(pl.col("id")==rid)["key"].to_list())
        row = view.filter((pl.col("id")==rid) & (pl.col("key")==k)).row(0, named=True)
        st.json({k: v for k, v in row.items()})