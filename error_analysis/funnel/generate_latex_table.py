"""
generate_latex_table.py
=======================
Generate a professional ACL-style LaTeX table for the **SWDE Funnel Error Analysis**.

For each specified domain ndjson file this script:
  1. Runs the funnel classifier on all failing samples.
  2. Counts errors by type (Hallucination, GXR Error, Extractor Error, Pruner Error).
  3. Reports raw error counts per pipeline stage.
  4. Produces a final aggregate row across all domains.
  5. Outputs a self-contained .tex snippet ready to paste into an ACL paper.

Usage
-----
    python generate_latex_table.py                  # uses DEFAULT_FILES below
    python generate_latex_table.py --help

    # Override with explicit paths:
    python generate_latex_table.py \\
        --files /path/to/auto/metric/page_level_f1_sample_eval.ndjson \\
                /path/to/book/metric/page_level_f1_sample_eval.ndjson \\
        --names Auto Book \\
        --out   swde_funnel_table.tex

The script resolves the ``run_funnel_analysis`` function from the sibling
``run_funnel_analysis.py`` module, so it must be run from this directory
(or with PYTHONPATH set appropriately).

Requires: polars, (optionally) rapidfuzz
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# ---------------------------------------------------------------------------
# Make sibling modules importable regardless of CWD
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

# Also add project src so html_eval can be found
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
for candidate in [_PROJECT_ROOT / "src", _PROJECT_ROOT]:
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from run_funnel_analysis import run_funnel_analysis, MatchingConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Default file configuration
# Re-edit this dict to point at your ndjson files.
# Keys = display name for the row; values = path to the metric ndjson.
# ---------------------------------------------------------------------------
BASE = Path("/home/abdo/PAPER/Eval/axe_final_output")

DEFAULT_FILES: Dict[str, str] = {
    "Auto":         str(BASE / "swde_auto/metric/page_level_f1_sample_eval.ndjson"),
    "Book":         str(BASE / "swde_book/metric/page_level_f1_sample_eval.ndjson"),
    "Camera":       str(BASE / "swde_camera/metric/page_level_f1_sample_eval.ndjson"),
    "Job":          str(BASE / "swde_job/metric/page_level_f1_sample_eval.ndjson"),
    "Movie":        str(BASE / "swde_movie/metric/page_level_f1_sample_eval.ndjson"),
    "NBA Player":   str(BASE / "swde_nbaplayer/metric/page_level_f1_sample_eval.ndjson"),
    "Restaurant":   str(BASE / "swde_restaurant/metric/page_level_f1_sample_eval.ndjson"),
    "University":   str(BASE / "swde_university/metric/page_level_f1_sample_eval.ndjson"),
}

# Canonical column order and display names
ERROR_TYPES: List[Tuple[str, str]] = [
    ("Hallucination",  "Halluc."),
    ("GXR Error",      "GXR"),
    ("Extractor Error","Extract."),
    ("Pruner Error",   "Pruner"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tex_escape(s: str) -> str:
    """Minimal LaTeX special-character escaping for table cell text."""
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


def analyze_domain(
    ndjson_path: str,
    cfg: Optional[MatchingConfig] = None,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Run funnel analysis on *ndjson_path* and return a dict mapping
    error_classification -> count (only for samples with score < 1).
    """
    df = run_funnel_analysis(ndjson_path, cfg=cfg, verbose=verbose)

    counts: Dict[str, int] = defaultdict(int)
    if df.height == 0:
        return counts

    for row in df["error_classification"].to_list():
        counts[row] += 1
    return dict(counts)


def build_table_data(
    domain_files: Dict[str, str],
    cfg: Optional[MatchingConfig] = None,
    verbose: bool = False,
) -> Tuple[List[dict], dict]:
    rows = []
    agg_counts: Dict[str, int] = defaultdict(int)
    agg_total = 0

    for domain_name, path in domain_files.items():
        if not os.path.exists(path):
            print(f"[WARNING] File not found, skipping: {path}", file=sys.stderr)
            continue

        print(f"Processing {domain_name} ...", file=sys.stderr)
        counts = analyze_domain(path, cfg=cfg, verbose=verbose)

        # ✅ FIX: Only count the 4 displayed error types toward the total
        #          (ignores any "Match"/"Correct"/"None" classifications)
        total_errors = sum(counts.get(etype, 0) for etype, _ in ERROR_TYPES)

        row = {"domain": domain_name, "total_errors": total_errors}
        for etype, _ in ERROR_TYPES:
            n = counts.get(etype, 0)
            row[f"{etype}_count"] = n
            row[f"{etype}_pct"]   = (n / total_errors * 100) if total_errors > 0 else 0.0
            agg_counts[etype]    += n
        agg_total += total_errors
        rows.append(row)

    for row in rows:
        row["domain_share_pct"] = (row["total_errors"] / agg_total * 100) if agg_total > 0 else 0.0

    agg: dict = {"domain": r"\textbf{All}", "total_errors": agg_total, "domain_share_pct": 100.0}
    for etype, _ in ERROR_TYPES:
        n = agg_counts[etype]
        agg[f"{etype}_count"] = n
        agg[f"{etype}_pct"]   = (n / agg_total * 100) if agg_total > 0 else 0.0

    return rows, agg


# ---------------------------------------------------------------------------
# LaTeX generation
# ---------------------------------------------------------------------------

def generate_latex(
    rows: List[dict],
    agg: dict,
    caption: str = (
        "Funnel error analysis on SWDE across domains. "
        "Values for error pipeline stages are reported as percentages of each domain's errors. "
        "The Total column reports each domain's percentage share of all dataset errors. "
        r"\textbf{Bold} marks the dominant error type per domain."
    ),
    label: str = "tab:swde_funnel",
) -> str:
    """
    Produce a complete, compilable LaTeX table environment.

    Parameters
    ----------
    rows        : per-domain data rows from build_table_data()
    agg         : aggregate row from build_table_data()
    caption     : LaTeX caption text
    label       : LaTeX \\label{} value
    """
    n_ecols = len(ERROR_TYPES)
    # Column spec: Domain | N error cols | Total errors %
    col_spec = "l" + "r" * n_ecols + "r"

    # Column headers for error types (percentages mentioned in caption)
    header_parts = [
        r"\textbf{Domain}",
    ] + [
        r"\textbf{" + _tex_escape(label_short) + r"}"
        for _, label_short in ERROR_TYPES
    ] + [r"\textbf{Total}"]
    header_line = " & ".join(header_parts) + r" \\"

    def fmt_cell(row: dict, etype: str, is_max: bool) -> str:
        pct = row[f"{etype}_pct"]
        cell = f"{pct:.1f}"
        if is_max:
            cell = r"\textbf{" + cell + r"}"
        return cell

    def make_data_row(row: dict, bold_domain: bool = False) -> str:
        domain_cell = row["domain"] if bold_domain else _tex_escape(row["domain"])
        # Find the maximum pct to bold it (skip bolding if it's the aggregate row)
        if not bold_domain:
            max_pct = max(row[f"{etype}_pct"] for etype, _ in ERROR_TYPES)
        else:
            max_pct = -1  # never bold individual cells in the aggregate row

        cells = [domain_cell]
        for etype, _ in ERROR_TYPES:
            is_max = (not bold_domain) and (row[f"{etype}_pct"] == max_pct)
            cells.append(fmt_cell(row, etype, is_max))
        cells.append(f"{row['domain_share_pct']:.1f}")
        return " & ".join(cells) + r" \\"

    # Build the table body
    body_lines = [make_data_row(r) for r in rows]

    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        header_line,
        r"\midrule",
    ] + body_lines + [
        r"\midrule",
        make_data_row(agg, bold_domain=True),
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\end{table}",
    ]

    return "\n".join(latex_lines)


# ---------------------------------------------------------------------------
# Summary console output
# ---------------------------------------------------------------------------

def print_summary(rows: List[dict], agg: dict) -> None:
    col_w = 14
    sep = "-" * (20 + col_w * (len(ERROR_TYPES) + 1))
    print("\n" + "=" * len(sep))
    print("SWDE Funnel Error Analysis Summary (%)")
    print("=" * len(sep))
    header = f"{'Domain':<20}" + "".join(f"{s:>{col_w}}" for _, s in ERROR_TYPES) + f"{'Total':>{col_w}}"
    print(header)
    print(sep)
    for row in rows:
        cols = [f"{row[f'{e}_pct']:.1f}" for e, _ in ERROR_TYPES]
        val_str = f"{row['domain_share_pct']:.1f}"
        line = (f"{row['domain']:<20}"
                + "".join(f"{c:>{col_w}}" for c in cols)
                + f"{val_str:>{col_w}}")
        print(line)
    print(sep)
    agg_cols = [f"{agg[f'{e}_pct']:.1f}" for e, _ in ERROR_TYPES]
    agg_val_str = f"{agg['domain_share_pct']:.1f}"
    agg_line = (f"{'ALL':<20}"
                + "".join(f"{c:>{col_w}}" for c in agg_cols)
                + f"{agg_val_str:>{col_w}}")
    print(agg_line)
    print("=" * len(sep) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate an ACL LaTeX funnel error table for SWDE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--files", nargs="+", metavar="PATH",
        help="Paths to metric ndjson files. Must match --names in order.",
    )
    p.add_argument(
        "--names", nargs="+", metavar="NAME",
        help="Display names for each domain (same order as --files).",
    )
    p.add_argument(
        "--out", default="swde_funnel_table.tex",
        help="Output .tex file path (default: swde_funnel_table.tex).",
    )
    p.add_argument(
        "--caption",
        default=(
            r"Funnel error analysis on SWDE across domains. "
            r"Values for error pipeline stages are reported as percentages of each domain's errors. "
            r"The Total column reports each domain's percentage share of all dataset errors. "
            r"\textbf{Bold} marks the dominant error type per domain."
        ),
        help="LaTeX caption for the table.",
    )
    p.add_argument(
        "--label", default="tab:swde_funnel",
        help="LaTeX label value (default: tab:swde_funnel).",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-file classifier summary.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build domain -> path mapping
    if args.files:
        if not args.names or len(args.names) != len(args.files):
            print(
                "[ERROR] --names must be provided with the same length as --files.",
                file=sys.stderr,
            )
            sys.exit(1)
        domain_files = dict(zip(args.names, args.files))
    else:
        domain_files = DEFAULT_FILES

    cfg = MatchingConfig()
    rows, agg = build_table_data(domain_files, cfg=cfg, verbose=args.verbose)

    if not rows:
        print("[ERROR] No data could be loaded. Check your file paths.", file=sys.stderr)
        sys.exit(1)

    print_summary(rows, agg)

    latex = generate_latex(
        rows,
        agg,
        caption=args.caption,
        label=args.label,
    )

    out_path = Path(args.out)
    out_path.write_text(latex, encoding="utf-8")
    print(f"[OK] LaTeX table written to: {out_path.resolve()}")
    print("\n--- LaTeX Preview ---")
    print(latex)


if __name__ == "__main__":
    main()
