"""
generate_websrc_latex_table.py
==============================
Generate professional ACL-style LaTeX tables for **WebSRC Taxonomy-Driven Results**.

Two tables are produced:
  1. **Taxonomy Results Table** -- rows = taxonomy types (KV, Table, Compare),
     columns = Test F1 / Dev F1 (and optionally precision/recall), with a final
     aggregate row.

  2. **Domain × Taxonomy Table** (optional, --full) -- rows = domain, columns =
     taxonomy type, cells = F1 score for that split.

The script uses the ``process_log_records`` function from the sibling
``web_analysis.py`` module, so it must be run from this directory (or with
PYTHONPATH set appropriately).

Usage
-----
    # Default: uses hard-coded paths defined in DEFAULT_* below
    python generate_websrc_latex_table.py

    # Custom paths:
    python generate_websrc_latex_table.py \\
        --test /path/to/websrc_test/metric/token_f1_sample_eval.ndjson \\
        --dev  /path/to/websrc_dev/metric/token_f1_sample_eval.ndjson \\
        --out  websrc_taxonomy_table.tex

    # Also produce the full domain x taxonomy breakdown:
    python generate_websrc_latex_table.py --full

Requires: pandas, beautifulsoup4, (optionally) rapidfuzz
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# ---------------------------------------------------------------------------
# Make sibling web_analysis importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))  # project root

from web_analysis import process_log_records, DOMAIN_MAP  # noqa: E402

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
BASE = Path("/home/abdo/PAPER/Eval/axe_final_output")
DEFAULT_TEST_PATH = str(
    BASE / "websrc_test/metric/token_f1_sample_eval.ndjson"
)
DEFAULT_DEV_PATH = str(
    BASE / "websrc_dev/metric/token_f1_sample_eval.ndjson"
)

# Canonical taxonomy order
TAXONOMY_ORDER = ["KV", "Table", "Compare"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("_", r"\_")
             .replace("#", r"\#")
             .replace("$", r"\$")
             .replace("{", r"\{")
             .replace("}", r"\}")
             .replace("~", r"\textasciitilde{}")
             .replace("^", r"\textasciicircum{}"))


def load_ndjson(path: str) -> list:
    """Load all records from an ndjson file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_taxonomy_stats(df) -> Dict[str, Dict[str, float]]:
    """
    Given a DataFrame from process_log_records, compute per-taxonomy-type:
      { taxonomy_type: {f1, precision, recall, n, n_correct} }
    All samples are included (no Yes/No filtering) so that aggregate F1
    matches the official token_f1 from results.json.
    """
    stats: Dict[str, Dict[str, float]] = {}
    for tax in TAXONOMY_ORDER:
        sub = df[df["taxonomy"] == tax]
        if len(sub) == 0:
            stats[tax] = {"f1": float("nan"), "precision": float("nan"),
                          "recall": float("nan"), "n": 0, "n_correct": 0}
            continue
        stats[tax] = {
            "f1":        sub["f1"].mean(),
            "precision": sub["precision"].mean() if "precision" in sub.columns else float("nan"),
            "recall":    sub["recall"].mean()    if "recall" in sub.columns else float("nan"),
            "n":         len(sub),
            "n_correct": int((sub["f1"] >= 0.99).sum()),
        }

    # Aggregate (all samples)
    stats["All"] = {
        "f1":        df["f1"].mean(),
        "precision": df["precision"].mean() if "precision" in df.columns else float("nan"),
        "recall":    df["recall"].mean()    if "recall" in df.columns else float("nan"),
        "n":         len(df),
        "n_correct": int((df["f1"] >= 0.99).sum()),
    }
    return stats


def compute_domain_taxonomy_stats(df) -> Dict[str, Dict[str, float]]:
    """
    Returns { domain_name: { taxonomy: avg_f1 } } over all taxonomy types.
    All samples included (no Yes/No filtering).
    """
    result: Dict[str, Dict[str, float]] = defaultdict(dict)

    domains = sorted(df["domain"].unique())
    for domain in domains:
        d_df = df[df["domain"] == domain]
        for tax in TAXONOMY_ORDER:
            sub = d_df[d_df["taxonomy"] == tax]
            result[domain][tax] = sub["f1"].mean() if len(sub) > 0 else float("nan")
        result[domain]["All"] = d_df["f1"].mean() if len(d_df) > 0 else float("nan")

    return dict(result)


# ---------------------------------------------------------------------------
# LaTeX: Taxonomy Results Table (main table)
# ---------------------------------------------------------------------------

def generate_taxonomy_latex(
    test_stats: Dict[str, Dict[str, float]],
    dev_stats:  Optional[Dict[str, Dict[str, float]]],
    caption: str,
    label: str,
    show_pr: bool = False,
) -> str:
    """
    Produces the main taxonomy-driven results table.
    Rows: KV, Table, Compare, (midrule), All
    Columns: Test F1 [P R] | Dev F1 [P R]
    """
    has_dev = dev_stats is not None

    def _f(v: float) -> str:
        if v != v:  # nan
            return "--"
        return f"{v * 100:.1f}"

    # Build column spec and header
    if show_pr:
        if has_dev:
            col_spec = r"lrrrrrr"
            header = (r"\textbf{Layout} & "
                      r"\multicolumn{3}{c}{\textbf{Test}} & "
                      r"\multicolumn{3}{c}{\textbf{Dev}} \\")
            sub_header = (r" & \textbf{F1} & \textbf{P} & \textbf{R} & "
                          r"\textbf{F1} & \textbf{P} & \textbf{R} \\")
            cmidrule = r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}"
        else:
            col_spec = r"lrrr"
            header = r"\textbf{Layout} & \textbf{F1} & \textbf{P} & \textbf{R} \\"
            sub_header = None
            cmidrule = None
    else:
        if has_dev:
            col_spec = r"lrr"
            header = (r"\textbf{Layout} & "
                      r"\textbf{Test F1} & \textbf{Dev F1} \\")
            sub_header = None
            cmidrule = None
        else:
            col_spec = r"lr"
            header = r"\textbf{Layout} & \textbf{F1} \\"
            sub_header = None
            cmidrule = None

    def make_row(tax_label: str, tstat: dict, dstat: Optional[dict], bold: bool) -> str:
        label_cell = (r"\textbf{" + _tex_escape(tax_label) + r"}"
                      if bold else _tex_escape(tax_label))

        if show_pr:
            cells = [label_cell,
                     _f(tstat["f1"]), _f(tstat["precision"]), _f(tstat["recall"])]
            if has_dev and dstat:
                cells += [_f(dstat["f1"]), _f(dstat["precision"]), _f(dstat["recall"])]
        else:
            cells = [label_cell, _f(tstat["f1"])]
            if has_dev and dstat:
                cells.append(_f(dstat["f1"]))

        return " & ".join(cells) + r" \\"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        header,
    ]
    if cmidrule:
        lines.append(cmidrule)
    if sub_header:
        lines.append(sub_header)
    lines.append(r"\midrule")

    # Data rows
    for tax in TAXONOMY_ORDER:
        tstat = test_stats.get(tax, {"f1": float("nan"), "precision": float("nan"),
                                      "recall": float("nan"), "n": 0})
        dstat = dev_stats.get(tax) if dev_stats else None
        lines.append(make_row(tax, tstat, dstat, bold=False))

    # Aggregate row
    lines.append(r"\midrule")
    tstat_all = test_stats["All"]
    dstat_all = dev_stats["All"] if dev_stats else None
    lines.append(make_row("All", tstat_all, dstat_all, bold=True))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX: Domain x Taxonomy breakdown table
# ---------------------------------------------------------------------------

def generate_domain_taxonomy_latex(
    test_domain_tax: Dict[str, Dict[str, float]],
    dev_domain_tax:  Optional[Dict[str, Dict[str, float]]],
    caption: str,
    label: str,
) -> str:
    """
    Produces a domain × taxonomy breakdown table.
    Rows: domains
    Columns: taxonomy types (KV, Table, Compare) + All
    With Test / Dev sub-columns if dev data available.
    """
    has_dev = dev_domain_tax is not None
    tax_cols = [t for t in TAXONOMY_ORDER if t != "Unknown"] + ["All"]

    def _f(v: float) -> str:
        if v != v:  # nan
            return "--"
        return f"{v * 100:.1f}"

    if has_dev:
        n_cols = len(tax_cols) * 2
        col_spec = "l" + "rr" * len(tax_cols)
        # Build multi-column header
        mc_parts = [r"\textbf{Domain}"] + [
            r"\multicolumn{2}{c}{\textbf{" + _tex_escape(t) + r"}}" for t in tax_cols
        ]
        header1 = " & ".join(mc_parts) + r" \\"
        cmidrules = " ".join(
            r"\cmidrule(lr){" + str(2 + i * 2) + r"-" + str(3 + i * 2) + r"}"
            for i in range(len(tax_cols))
        )
        sub_cols = [""] + [r"\textbf{Te} & \textbf{De}"] * len(tax_cols)
        header2 = " & ".join(sub_cols) + r" \\"
    else:
        col_spec = "l" + "r" * len(tax_cols)
        header1 = (r"\textbf{Domain} & "
                   + " & ".join(r"\textbf{" + _tex_escape(t) + r"}" for t in tax_cols)
                   + r" \\")
        cmidrules = None
        header2 = None

    all_domains = sorted(test_domain_tax.keys())

    def make_row(domain: str) -> str:
        tdata = test_domain_tax.get(domain, {})
        ddata = (dev_domain_tax.get(domain, {}) if dev_domain_tax else {})
        cells = [_tex_escape(domain)]
        for t in tax_cols:
            tf = _f(tdata.get(t, float("nan")))
            if has_dev:
                df = _f(ddata.get(t, float("nan")))
                cells += [tf, df]
            else:
                cells.append(tf)
        return " & ".join(cells) + r" \\"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        header1,
    ]
    if cmidrules:
        lines.append(cmidrules)
    if header2:
        lines.append(header2)
    lines.append(r"\midrule")

    for domain in all_domains:
        lines.append(make_row(domain))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_taxonomy_summary(
    test_stats: Dict[str, Dict[str, float]],
    dev_stats:  Optional[Dict[str, Dict[str, float]]],
    split_name: str = "Test",
) -> None:
    def _f(v: float) -> str:
        return "--" if v != v else f"{v * 100:.2f}"

    header = f"{'Layout':<12}{'Test F1':>10}"
    if dev_stats:
        header += f"{'Dev F1':>10}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print("WebSRC Taxonomy-Driven Results")
    print(sep)
    print(header)
    print(sep)
    for tax in TAXONOMY_ORDER + ["All"]:
        t = test_stats.get(tax, {})
        line = f"{tax:<12}{_f(t.get('f1', float('nan'))):>10}"
        if dev_stats:
            d = dev_stats.get(tax, {})
            line += f"{_f(d.get('f1', float('nan'))):>10}"
        print(line)
    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate ACL LaTeX taxonomy-driven results tables for WebSRC.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--test", default=DEFAULT_TEST_PATH, metavar="PATH",
        help="Path to WebSRC Test metric ndjson.",
    )
    p.add_argument(
        "--dev", default=DEFAULT_DEV_PATH, metavar="PATH",
        help="Path to WebSRC Dev metric ndjson (omit to produce test-only table).",
    )
    p.add_argument(
        "--no-dev", action="store_true",
        help="Skip dev data even if --dev path exists.",
    )
    p.add_argument(
        "--out", default="websrc_taxonomy_table.tex",
        help="Output .tex file for the main taxonomy table (default: websrc_taxonomy_table.tex).",
    )
    p.add_argument(
        "--caption",
        default=(
            r"Taxonomy-driven results on WebSRC Test and Dev splits. "
            r"F1 scores are macro-averaged over all questions in each layout category. "
            r"\textbf{KV} = key-value lists, \textbf{Table} = tabular data, "
            r"\textbf{Compare} = comparative/multi-attribute layouts."
        ),
        help="LaTeX caption for the taxonomy table.",
    )
    p.add_argument(
        "--label", default="tab:websrc_taxonomy",
        help="LaTeX label for the taxonomy table (default: tab:websrc_taxonomy).",
    )
    p.add_argument(
        "--show-pr", action="store_true",
        help="Include Precision and Recall columns alongside F1.",
    )
    p.add_argument(
        "--full", action="store_true",
        help="Also produce a domain x taxonomy breakdown table.",
    )
    p.add_argument(
        "--full-out", default="websrc_domain_taxonomy_table.tex",
        help="Output path for the full domain x taxonomy table (default: websrc_domain_taxonomy_table.tex).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load Test split ---
    if not os.path.exists(args.test):
        print(f"[ERROR] Test file not found: {args.test}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Test split: {args.test} ...", file=sys.stderr)
    test_records = load_ndjson(args.test)
    test_df, _ = process_log_records(test_records)
    test_stats = compute_taxonomy_stats(test_df)
    test_domain_tax = compute_domain_taxonomy_stats(test_df)

    # --- Load Dev split (optional) ---
    dev_stats = None
    dev_domain_tax = None
    if not args.no_dev and os.path.exists(args.dev):
        print(f"Loading Dev split:  {args.dev} ...", file=sys.stderr)
        dev_records = load_ndjson(args.dev)
        dev_df, _ = process_log_records(dev_records)
        dev_stats = compute_taxonomy_stats(dev_df)
        dev_domain_tax = compute_domain_taxonomy_stats(dev_df)
    elif not args.no_dev:
        print(f"[WARNING] Dev file not found, skipping: {args.dev}", file=sys.stderr)

    # --- Print console summary ---
    print_taxonomy_summary(test_stats, dev_stats)

    # --- Generate main taxonomy table ---
    latex_main = generate_taxonomy_latex(
        test_stats,
        dev_stats,
        caption=args.caption,
        label=args.label,
        show_pr=args.show_pr,
    )
    out_main = Path(args.out)
    out_main.write_text(latex_main, encoding="utf-8")
    print(f"[OK] Taxonomy table written to: {out_main.resolve()}")
    print("\n--- Taxonomy Table LaTeX ---")
    print(latex_main)

    # --- Generate domain x taxonomy table (optional) ---
    if args.full:
        latex_full = generate_domain_taxonomy_latex(
            test_domain_tax,
            dev_domain_tax,
            caption=(
                r"Domain-level taxonomy breakdown on WebSRC. "
                r"Values are average F1 (\%) per layout type. "
                r"Te = Test, De = Dev. ``--'' indicates no samples of that type."
            ),
            label="tab:websrc_domain_taxonomy",
        )
        out_full = Path(args.full_out)
        out_full.write_text(latex_full, encoding="utf-8")
        print(f"\n[OK] Domain x Taxonomy table written to: {out_full.resolve()}")
        print("\n--- Domain x Taxonomy Table LaTeX ---")
        print(latex_full)


if __name__ == "__main__":
    main()
