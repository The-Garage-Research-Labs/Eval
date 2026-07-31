#!/usr/bin/env python3
"""
Generate a LaTeX table from SWDE results.
Each row = one SWDE domain (swde_auto, swde_book, …).
Final row = macro-average aggregate across all domains.
"""

import json
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/home/abdo/PAPER/Eval/axe_final_output")
OUTPUT_TEX = Path("/home/abdo/PAPER/Eval/latex_data_generation/swde_results_table.tex")

DOMAINS = [
    "swde_auto",
    "swde_book",
    "swde_camera",
    "swde_job",
    "swde_movie",
    "swde_nbaplayer",
    "swde_restaurant",
    "swde_university",
]

DOMAIN_LABELS = {
    "swde_auto":       "Auto",
    "swde_book":       "Book",
    "swde_camera":     "Camera",
    "swde_job":        "Job",
    "swde_movie":      "Movie",
    "swde_nbaplayer":  "NBA Player",
    "swde_restaurant": "Restaurant",
    "swde_university": "University",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def fmt(value: float) -> str:
    """Format a 0–1 float as a percentage string with 1 decimal place."""
    return f"{value * 100:.2f}"


def load_metrics(domain: str) -> dict:
    """Return the top-level page_level_f1 metrics for a domain."""
    path = BASE_DIR / domain / "results.json"
    with path.open() as f:
        data = json.load(f)
    pl = data["page_level_f1"]
    return {
        "precision": pl["precision"],
        "recall":    pl["recall"],
        "f1":        pl["f1"],
    }


def make_table(rows: list[dict]) -> str:
    """Build the LaTeX table string.

    Aggregation strategy:
      - Raw (unrounded) precision / recall / f1 floats are collected from JSON.
      - The average is computed entirely on raw floats.
      - fmt() is only ever called at the final string-formatting step,
        so rounding never influences the aggregate.
    """
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \caption{SWDE Automatic Evaluation Results per Domain}",
        r"  \label{tab:swde_results}",
        r"  \renewcommand{\arraystretch}{1.2}",
        r"  \begin{tabular}{lrrr}",
        r"    \toprule",
        r"    \textbf{Domain} & \textbf{Precision (\%)} & \textbf{Recall (\%)} & \textbf{F1 (\%)} \\",
        r"    \midrule",
    ]

    # ── Step 1: collect all raw floats ─────────────────────────────────────────
    raw_p  = [row["precision"] for row in rows]
    raw_r  = [row["recall"]    for row in rows]
    raw_f1 = [row["f1"]        for row in rows]

    # ── Step 2: per-domain rows — format only here ──────────────────────────────
    for row, p, r, f1 in zip(rows, raw_p, raw_r, raw_f1):
        label = DOMAIN_LABELS.get(row["domain"], row["domain"])
        lines.append(f"    {label} & {fmt(p)} & {fmt(r)} & {fmt(f1)} \\\\")

    # ── Step 3: aggregate on raw values, format once ────────────────────────────
    n = len(rows)
    avg_p  = sum(raw_p)  / n
    avg_r  = sum(raw_r)  / n
    avg_f1 = sum(raw_f1) / n

    lines += [
        r"    \midrule",
        f"    \\textbf{{Average}} & \\textbf{{{fmt(avg_p)}}} & \\textbf{{{fmt(avg_r)}}} & \\textbf{{{fmt(avg_f1)}}} \\\\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines) + "\n"


def main():
    rows = []
    for domain in DOMAINS:
        metrics = load_metrics(domain)
        metrics["domain"] = domain
        rows.append(metrics)

    table = make_table(rows)

    OUTPUT_TEX.write_text(table)
    print(f"LaTeX table written to: {OUTPUT_TEX}")
    print()
    print("=" * 60)
    print(table)
    print("=" * 60)


if __name__ == "__main__":
    main()
