"""
generate_charts.py
==================
Generate publication-quality charts for the AXE paper (ACL submission).

Design Philosophy:
  - Pure white background to match ACL/LaTeX templates.
  - Refined AXE brand palette (warm creams, AXE orange, deep burnt sienna).
  - Elegant serif typography with strong visual hierarchy.
  - High-contrast annotations and subtle point edges for perfect readability.

Outputs (saved to ./charts/):
  1. SWDE Heatmaps       — fig_heatmap_swde_<domain>.pdf
  2. DOM Token Scatter   — fig_scatter_dom_tokens.pdf
  3. Retention Scatter   — fig_scatter_retention_f1.pdf

Usage:
    python generate_charts.py
"""

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = Path("/home/abdo/PAPER/Eval/axe_final_output")
OUTPUT_DIR = Path(__file__).resolve().parent / "charts"
OUTPUT_DIR.mkdir(exist_ok=True)

SWDE_DOMAINS = {
    "auto":       "Auto",
    "book":       "Book",
    "camera":     "Camera",
    "job":        "Job",
    "movie":      "Movie",
    "nbaplayer":  "NBA Player",
    "restaurant": "Restaurant",
    "university": "University",
}

WEBSRC_SUBSETS = {
    "websrc_dev":  "WebSRC Dev",
    "websrc_test": "WebSRC Test",
}

SCATTER_SAMPLE_SIZE = 2000
CHARS_PER_TOKEN = 4

# ---------------------------------------------------------------------------
# AXE Brand Palette (Adapted for White Background)
# ---------------------------------------------------------------------------
AXE_ORANGE     = "#E8531E"   # Primary brand orange
AXE_ORANGE_DK  = "#7A2A0E"   # Deep burnt sienna for high values
AXE_BLACK      = "#1A1A1A"   # Near-black for text
AXE_GRAY       = "#666666"   # Medium gray for secondary elements
AXE_LIGHT_GRAY = "#EAEAEA"   # Light gray for subtle gridlines
AXE_POINT_EDGE = "#2C2C2C"   # Subtle dark edge for scatter points

# ---------------------------------------------------------------------------
# Global matplotlib styling — Professional ACL Theme
# ---------------------------------------------------------------------------
plt.rcParams.update({
    # Use serif fonts to match LaTeX ACL templates
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    # White backgrounds
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    # Text and spines
    "text.color": AXE_BLACK,
    "axes.labelcolor": AXE_BLACK,
    "axes.titlecolor": AXE_BLACK,
    "xtick.color": AXE_BLACK,
    "ytick.color": AXE_BLACK,
    "axes.edgecolor": AXE_GRAY,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

ATTR_LABELS = {
    "fuel_economy": "Fuel Economy", "engine": "Engine", "price": "Price",
    "model": "Model", "title": "Title", "author": "Author",
    "publisher": "Publisher", "isbn_13": "ISBN-13",
    "publication_date": "Pub. Date", "manufacturer": "Manufacturer",
    "product_title": "Product Title", "company": "Company",
    "location": "Location", "date_posted": "Date Posted",
    "director": "Director", "genre": "Genre", "mpaa_rating": "MPAA Rating",
    "name": "Name", "height": "Height", "weight": "Weight",
    "team": "Team", "phone": "Phone", "address": "Address",
    "cuisine": "Cuisine", "website": "Website", "type": "Type",
}

# ---------------------------------------------------------------------------
# AXE-themed heatmap colormap: cream (0) -> peach -> AXE orange -> sienna (1)
# No pure black, which creates a smooth, elegant sequential ramp.
# ---------------------------------------------------------------------------
AXE_HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "axe_heatmap_designer",
    [
        (0.00, "#FCF2EB"),   # Pale cream
        (0.25, "#FAD3C0"),   # Soft peach
        (0.50, "#F58A3A"),   # Bright orange
        (0.75, "#E8531E"),   # AXE orange
        (1.00, "#7A2A0E"),   # Deep burnt sienna
    ],
    N=256,
)


# ---------------------------------------------------------------------------
# 1. SWDE Heatmaps
# ---------------------------------------------------------------------------

def compute_per_website_per_field_f1(results_json: dict) -> dict:
    """Compute F1 per (website, field) from results.json."""
    results = results_json["page_level_f1"]["results"]
    out = {}
    for website, fields in results.items():
        out[website] = {}
        for field, stats in fields.items():
            hits = stats["page_hits"]
            ext = stats["extracted_pages"]
            gt = stats["ground_truth_pages"]
            prec = hits / ext if ext > 0 else 0
            rec = hits / gt if gt > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            out[website][field] = f1
    return out


def generate_heatmap(domain_key: str, domain_name: str):
    """Generate an elegant heatmap for a single SWDE domain."""
    results_path = BASE / f"swde_{domain_key}" / "results.json"
    if not results_path.exists():
        print(f"  [SKIP] {results_path} not found")
        return

    with open(results_path) as f:
        data = json.load(f)

    f1_data = compute_per_website_per_field_f1(data)
    websites = sorted(f1_data.keys())
    if not websites:
        return
    fields = sorted(next(iter(f1_data.values())).keys())

    matrix = np.zeros((len(fields), len(websites)))
    for j, website in enumerate(websites):
        for i, field in enumerate(fields):
            matrix[i, j] = f1_data[website].get(field, 0)

    # Expanded dimensions for better whitespace
    fig_width = max(9, len(websites) * 1.1 + 2.5)
    fig_height = max(4, len(fields) * 0.8 + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(matrix, cmap=AXE_HEATMAP_CMAP, vmin=0, vmax=1, aspect="auto")

    # Annotate cells
    for i in range(len(fields)):
        for j in range(len(websites)):
            val = matrix[i, j]
            # Use white text on dark cells, black text on light cells
            text_color = "white" if val > 0.55 else AXE_BLACK
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

    # Labels
    field_labels = [ATTR_LABELS.get(f, f.replace("_", " ").title()) for f in fields]
    ax.set_xticks(range(len(websites)))
    ax.set_xticklabels([w.title() for w in websites], rotation=35, ha="right")
    ax.set_yticks(range(len(fields)))
    ax.set_yticklabels(field_labels)

    # Title
    ax.set_title(f"SWDE {domain_name}: Page-Level F1 by Website and Attribute",
                 fontweight="bold", pad=14, loc="left")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Page-Level F1", fontsize=11, color=AXE_BLACK, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color=AXE_BLACK)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=AXE_BLACK)
    cbar.outline.set_edgecolor(AXE_GRAY)
    cbar.outline.set_linewidth(0.8)

    # Subtle white separators (tile effect)
    ax.set_xticks(np.arange(-0.5, len(websites), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(fields), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", size=0)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", length=0, pad=5)

    # Remove spines for a clean look
    for spine in ax.spines.values():
        spine.set_visible(False)

    out_path = OUTPUT_DIR / f"fig_heatmap_swde_{domain_key}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ Saved {out_path.name}")


# ---------------------------------------------------------------------------
# 2. Scatter Plots: DOM Tokens Analysis
# ---------------------------------------------------------------------------

def sample_ndjson_dom_stats(ndjson_path: str, n: int = SCATTER_SAMPLE_SIZE) -> list:
    """Reservoir-sample n rows from an ndjson; return dicts with token counts + F1."""
    samples = []
    with open(ndjson_path) as f:
        for idx, line in enumerate(f):
            d = json.loads(line)
            sl = d["step_logs"]["preprocessor"]
            ev = d["evaluation"]
            if "f1" in ev:
                f1 = ev["f1"]
            else:
                vals = list(ev.values())
                f1 = sum(vals) / len(vals) if vals else 0

            entry = {
                "raw_tokens":     sl["raw_len"] / CHARS_PER_TOKEN,
                "filtered_tokens": len(d["filtered_html"]) / CHARS_PER_TOKEN,
                "f1": f1,
            }
            if idx < n:
                samples.append(entry)
            else:
                j = random.randint(0, idx)
                if j < n:
                    samples[j] = entry
    return samples


def find_ndjson_metric(domain_dir: str) -> str:
    metric_dir = Path(domain_dir) / "metric"
    if not metric_dir.exists():
        return ""
    for f in metric_dir.iterdir():
        if f.suffix == ".ndjson":
            return str(f)
    return ""


def collect_all_scatter_data() -> list:
    all_samples = []
    for key, name in {**SWDE_DOMAINS, **WEBSRC_SUBSETS}.items():
        prefix = "swde_" if key in SWDE_DOMAINS else ""
        domain_dir = BASE / f"{prefix}{key}"
        ndjson_path = find_ndjson_metric(str(domain_dir))
        if not ndjson_path:
            print(f"  [SKIP] No ndjson found for {name}")
            continue
        print(f"  Sampling {name} ...")
        samples = sample_ndjson_dom_stats(ndjson_path)
        for s in samples:
            s["domain"] = name
        all_samples.extend(samples)
    return all_samples


def _style_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontweight="bold", labelpad=10)
    ax.set_ylabel(ylabel, fontweight="bold", labelpad=10)
    ax.set_title(title, fontweight="bold", pad=14, loc="left")
    # Extremely subtle gridlines for elegance
    ax.grid(True, color=AXE_LIGHT_GRAY, alpha=1.0, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.spines['bottom'].set_color(AXE_GRAY)
    ax.spines['left'].set_color(AXE_GRAY)


# Updated F1 colormap: starts at light gray so 0 values are visible on white
AXE_SCATTER_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "axe_scatter_f1",
    [
        (0.00, "#D3D3D3"),   # Light gray (F1 = 0)
        (0.25, "#FAD3C0"),   # Soft peach
        (0.50, "#F58A3A"),   # Bright orange
        (0.75, "#E8531E"),   # AXE orange
        (1.00, "#7A2A0E"),   # Deep burnt sienna (F1 = 1)
    ],
    N=256,
)

def generate_scatter_original_vs_pruned(data: list):
    """x = Original DOM tokens, y = Pruned DOM tokens, color = F1."""
    raw  = np.array([d["raw_tokens"]        for d in data])
    filt = np.array([d["filtered_tokens"]   for d in data])
    f1s  = np.array([d["f1"]                for d in data])

    # Wide, cinematic aspect ratio
    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Sort by F1 so high-F1 points render on top
    order = np.argsort(f1s)
    
    scatter = ax.scatter(
        raw[order], filt[order], c=f1s[order],
        cmap=AXE_SCATTER_CMAP, vmin=0, vmax=1,
        s=12, alpha=0.4, 
        edgecolors="none", 
        rasterized=True,
    )

    # Decoupled zooming: Cut extreme outliers independently for X and Y
    x_max = np.percentile(raw, 98) * 1.02
    y_max = np.percentile(filt, 98) * 1.02  # Y is strictly limited to its own percentile
    
    _style_axes(
        ax,
        xlabel="Original DOM (tokens)",
        ylabel="Pruned DOM (tokens)",
        title="DOM Reduction: Original vs. Pruned Tokens",
    )

    # Apply tight limits to zoom in on the dense cluster
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)  # Y-axis is now properly bounded
    
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("F1 Score", fontsize=11, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color=AXE_BLACK)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=AXE_BLACK)
    cbar.outline.set_edgecolor(AXE_GRAY)
    cbar.outline.set_linewidth(0.8)

    out_path = OUTPUT_DIR / "fig_scatter_dom_tokens.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ Saved {out_path.name}")


def generate_scatter_retention_vs_f1(data: list):
    """x = Retention Ratio (Pruned / Original), y = F1."""
    retention, f1s = [], []
    for d in data:
        if d["raw_tokens"] > 0:
            retention.append(d["filtered_tokens"] / d["raw_tokens"])
            f1s.append(d["f1"])

    retention = np.array(retention)
    f1s = np.array(f1s)

    # Wide, cinematic aspect ratio
    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Single AXE Orange color with high transparency to create a clean density cloud
    # This entirely eliminates the "rainbow vomit" clutter of multi-domain scatter plots
    ax.scatter(
        retention, f1s,
        c=AXE_ORANGE, s=12, alpha=0.25, 
        edgecolors="none", 
        rasterized=True,
    )

    _style_axes(
        ax,
        xlabel="Retention Ratio (Pruned / Original Tokens)",
        ylabel="F1 Score",
        title="Retention Ratio vs. F1 Score",
    )
    
    # Zoom X to 98th percentile to ignore extreme outlier ratios and focus on the bulk
    x_max = min(1.0, np.percentile(retention, 98) * 1.02)
    ax.set_xlim(-0.01, x_max)
    ax.set_ylim(-0.02, 1.02)

    out_path = OUTPUT_DIR / "fig_scatter_retention_f1.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  ✓ Saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(42)

    print("=" * 64)
    print("AXE — Generating SWDE Heatmaps")
    print("=" * 64)
    for key, name in SWDE_DOMAINS.items():
        print(f"\n[{name}]")
        generate_heatmap(key, name)

    print("\n" + "=" * 64)
    print("AXE — Collecting sample data for scatter plots")
    print("=" * 64)
    scatter_data = collect_all_scatter_data()
    print(f"\n  Total samples collected: {len(scatter_data)}")

    print("\n" + "=" * 64)
    print("AXE — Generating Scatter: Original vs. Pruned DOM Tokens")
    print("=" * 64)
    generate_scatter_original_vs_pruned(scatter_data)

    print("\n" + "=" * 64)
    print("AXE — Generating Scatter: Retention Ratio vs. F1")
    print("=" * 64)
    generate_scatter_retention_vs_f1(scatter_data)

    print("\n" + "=" * 64)
    print(f"All charts saved to: {OUTPUT_DIR}")
    print("=" * 64)


if __name__ == "__main__":
    main()