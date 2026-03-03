"""
QC Visualization for rish-harmonize

Generates publication-ready figures from pipeline output JSON files.
Requires matplotlib (install with: pip install rish-harmonize[viz])
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _check_matplotlib():
    """Check matplotlib availability and return it."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for QC visualization. "
            "Install it with: pip install rish-harmonize[viz]"
        )


def plot_site_effect_comparison(
    comparison_dir: str,
    output_path: str,
    dpi: int = 150,
) -> str:
    """Plot pre- vs post-harmonization site effect comparison.

    Reads summary.json files from b*/pre/ and b*/post/ subdirectories
    and creates a grouped bar chart showing mean eta-squared and
    percent significant voxels.

    Parameters
    ----------
    comparison_dir : str
        Directory containing b*/pre/summary.json and b*/post/summary.json
    output_path : str
        Output PNG file path.
    dpi : int
        Figure resolution.

    Returns
    -------
    str
        Path to saved figure.
    """
    plt = _check_matplotlib()
    import numpy as np

    comparison_dir = Path(comparison_dir)

    # Discover b-shells
    shells = {}
    for bdir in sorted(comparison_dir.iterdir()):
        if not bdir.is_dir() or not bdir.name.startswith("b"):
            continue
        pre_json = bdir / "pre" / "summary.json"
        post_json = bdir / "post" / "summary.json"
        if pre_json.exists() and post_json.exists():
            with open(pre_json) as f:
                pre = json.load(f)
            with open(post_json) as f:
                post = json.load(f)
            shells[bdir.name] = {"pre": pre, "post": post}

    if not shells:
        raise FileNotFoundError(
            f"No b*/pre/summary.json + b*/post/summary.json found in {comparison_dir}"
        )

    shell_names = sorted(shells.keys())
    n_shells = len(shell_names)
    x = np.arange(n_shells)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3 + 2 * n_shells, 8), sharex=True)

    # Top panel: Mean eta-squared
    pre_eta = [shells[s]["pre"]["mean_effect_size"] for s in shell_names]
    post_eta = [shells[s]["post"]["mean_effect_size"] for s in shell_names]

    bars_pre = ax1.bar(x - width / 2, pre_eta, width, label="Pre-harmonization",
                       color="#2c7bb6", edgecolor="black", linewidth=0.5)
    bars_post = ax1.bar(x + width / 2, post_eta, width, label="Post-harmonization",
                        color="#abd9e9", edgecolor="black", linewidth=0.5)

    # Annotate reduction percentage
    for i in range(n_shells):
        if pre_eta[i] > 0:
            reduction = (pre_eta[i] - post_eta[i]) / pre_eta[i] * 100
            y_max = max(pre_eta[i], post_eta[i])
            ax1.annotate(
                f"{reduction:+.0f}%",
                xy=(x[i], y_max),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center", fontsize=10, fontweight="bold",
                color="#d7191c" if reduction > 0 else "#1a9641",
            )

    ax1.set_ylabel("Mean Effect Size (η²)", fontsize=12)
    ax1.set_title("Site Effect: Pre vs Post Harmonization", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_ylim(bottom=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Bottom panel: % significant voxels (permutation-based)
    pre_pct = [shells[s]["pre"]["percent_significant_permutation"] for s in shell_names]
    post_pct = [shells[s]["post"]["percent_significant_permutation"] for s in shell_names]

    ax2.bar(x - width / 2, pre_pct, width, label="Pre-harmonization",
            color="#2c7bb6", edgecolor="black", linewidth=0.5)
    ax2.bar(x + width / 2, post_pct, width, label="Post-harmonization",
            color="#abd9e9", edgecolor="black", linewidth=0.5)

    # Annotate values
    for i in range(n_shells):
        ax2.annotate(f"{pre_pct[i]:.1f}%", xy=(x[i] - width / 2, pre_pct[i]),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
        ax2.annotate(f"{post_pct[i]:.1f}%", xy=(x[i] + width / 2, post_pct[i]),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)

    ax2.set_ylabel("Significant Voxels (%)\n(permutation p<0.05)", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(shell_names, fontsize=12)
    ax2.set_xlabel("Shell", fontsize=12)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Add sample info from first shell
    first = shells[shell_names[0]]["pre"]
    fig.text(
        0.5, 0.01,
        f"n={first['n_subjects']} subjects, {first['n_sites']} sites, "
        f"{first['n_voxels']:,} voxels, {first['n_permutations']} permutations",
        ha="center", fontsize=9, style="italic", color="gray",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return str(output_path)


def plot_scale_map_heatmap(
    glm_output_dir: str,
    output_dir: str,
    dpi: int = 150,
) -> List[str]:
    """Plot scale map diagnostics as annotated heatmaps.

    Creates one heatmap per b-shell showing mean scale factors
    across target sites and SH orders, with clipping percentages.

    Parameters
    ----------
    glm_output_dir : str
        GLM output directory containing scale_maps/*/b*/scale_map_diagnostics.json
    output_dir : str
        Output directory for PNG files.
    dpi : int
        Figure resolution.

    Returns
    -------
    list of str
        Paths to saved figures.
    """
    plt = _check_matplotlib()
    from matplotlib.colors import TwoSlopeNorm
    import numpy as np

    glm_output_dir = Path(glm_output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_maps_dir = glm_output_dir / "scale_maps"
    if not scale_maps_dir.exists():
        raise FileNotFoundError(f"scale_maps directory not found in {glm_output_dir}")

    # Collect all diagnostics: {b_shell: {site: {order: stats}}}
    all_data: Dict[str, Dict[str, Dict[str, Dict]]] = {}
    for site_dir in sorted(scale_maps_dir.iterdir()):
        if not site_dir.is_dir():
            continue
        site = site_dir.name
        for bdir in sorted(site_dir.iterdir()):
            if not bdir.is_dir() or not bdir.name.startswith("b"):
                continue
            diag_file = bdir / "scale_map_diagnostics.json"
            if not diag_file.exists():
                continue
            with open(diag_file) as f:
                diag = json.load(f)
            b_shell = bdir.name
            if b_shell not in all_data:
                all_data[b_shell] = {}
            all_data[b_shell][site] = diag

    if not all_data:
        raise FileNotFoundError(
            f"No scale_map_diagnostics.json found under {scale_maps_dir}"
        )

    saved = []
    for b_shell in sorted(all_data.keys()):
        shell_data = all_data[b_shell]
        sites = sorted(shell_data.keys())
        # Gather all SH orders across sites
        all_orders = set()
        for site_diag in shell_data.values():
            all_orders.update(site_diag.keys())
        orders = sorted(all_orders, key=lambda o: int(o.lstrip("l")))

        n_sites = len(sites)
        n_orders = len(orders)

        # Build matrices
        mean_matrix = np.full((n_sites, n_orders), np.nan)
        clip_matrix = np.full((n_sites, n_orders), np.nan)

        for i, site in enumerate(sites):
            for j, order in enumerate(orders):
                if order in shell_data[site]:
                    stats = shell_data[site][order]
                    mean_matrix[i, j] = stats["mean"]
                    clip_matrix[i, j] = stats["pct_clipped_total"]

        # Plot
        fig_width = max(6, 1.5 * n_orders + 2)
        fig_height = max(3, 0.8 * n_sites + 2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Diverging colormap centered at 1.0
        vmin = max(0.5, np.nanmin(mean_matrix) - 0.05)
        vmax = min(2.0, np.nanmax(mean_matrix) + 0.05)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

        im = ax.imshow(mean_matrix, cmap="RdBu_r", norm=norm, aspect="auto")

        # Annotate cells
        for i in range(n_sites):
            for j in range(n_orders):
                if not np.isnan(mean_matrix[i, j]):
                    mean_val = mean_matrix[i, j]
                    clip_val = clip_matrix[i, j]

                    # Choose text color based on background
                    text_color = "white" if abs(mean_val - 1.0) > 0.3 else "black"

                    ax.text(j, i, f"{mean_val:.2f}\n({clip_val:.0f}% clip)",
                            ha="center", va="center", fontsize=9,
                            color=text_color, fontweight="bold")

        # Labels
        order_labels = [f"θ{o.lstrip('l')}" for o in orders]
        ax.set_xticks(range(n_orders))
        ax.set_xticklabels(order_labels, fontsize=11)
        ax.set_yticks(range(n_sites))
        ax.set_yticklabels([f"Site {s}" for s in sites], fontsize=11)
        ax.set_xlabel("SH Order", fontsize=12)
        ax.set_ylabel("Target Site", fontsize=12)
        ax.set_title(f"Scale Map Diagnostics — {b_shell}", fontsize=14, fontweight="bold")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Mean Scale Factor", fontsize=10)

        plt.tight_layout()

        out_path = output_dir / f"scale_map_heatmap_{b_shell}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(out_path))

    return saved
