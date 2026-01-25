"""
Visualization module for Phase 2 experiment results.

Generates slide-quality plots for comparing voting method performance
across different experimental conditions.

Plots generated:
1. Bar chart: Epsilon by voting method
2. Comparison: Alternative distributions
3. Comparison: Voter distributions (uniform vs ideology clusters)
4. Heatmap: Method × Alternative distribution
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from .config import PHASE2_FIGURES_DIR, ALT_DISTRIBUTIONS
from .results_aggregator import (
    collect_all_results,
    compute_summary_stats,
    compare_voter_distributions,
    compare_alt_distributions,
)

logger = logging.getLogger(__name__)

# Set style for slide-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Slide-quality figure sizes
FIGURE_SIZE_WIDE = (14, 6)
FIGURE_SIZE_SQUARE = (10, 8)
FIGURE_SIZE_TALL = (10, 10)

# Method categories for coloring
TRADITIONAL_METHODS = ["schulze", "borda", "irv", "plurality", "veto_by_consumption"]
CHATGPT_METHODS = ["chatgpt", "chatgpt_rankings", "chatgpt_personas"]
CHATGPT_STAR_METHODS = ["chatgpt_star", "chatgpt_star_rankings", "chatgpt_star_personas"]
CHATGPT_DOUBLE_STAR_METHODS = ["chatgpt_double_star", "chatgpt_double_star_rankings", "chatgpt_double_star_personas"]

# Colors for method categories
METHOD_COLORS = {
    "traditional": "#2ecc71",  # Green
    "chatgpt": "#3498db",      # Blue
    "chatgpt_star": "#9b59b6", # Purple
    "chatgpt_double_star": "#e74c3c",  # Red
}

# Alternative distribution labels
ALT_DIST_LABELS = {
    "persona_no_context": "Alt1: Persona Only",
    "persona_context": "Alt2: Persona + Context",
    "no_persona_context": "Alt3: Context Only",
    "no_persona_no_context": "Alt4: Blind",
}

# Voter distribution labels
VOTER_DIST_LABELS = {
    "uniform": "Uniform",
    "progressive_liberal": "Progressive/Liberal",
    "conservative_traditional": "Conservative/Traditional",
}


def get_method_color(method: str) -> str:
    """Get color for a voting method based on its category."""
    if method in TRADITIONAL_METHODS:
        return METHOD_COLORS["traditional"]
    elif method in CHATGPT_METHODS:
        return METHOD_COLORS["chatgpt"]
    elif method in CHATGPT_STAR_METHODS:
        return METHOD_COLORS["chatgpt_star"]
    elif method in CHATGPT_DOUBLE_STAR_METHODS:
        return METHOD_COLORS["chatgpt_double_star"]
    else:
        return "#95a5a6"  # Gray for unknown


def plot_epsilon_by_method(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Epsilon by Voting Method"
) -> None:
    """
    Create bar chart of mean epsilon by voting method.
    
    Slide-quality plot with method categories color-coded.
    """
    if df.empty:
        logger.warning("No data to plot")
        return
    
    # Compute means and stds
    valid_df = df[df["epsilon"].notna()]
    if valid_df.empty:
        logger.warning("No valid epsilon values")
        return
    
    summary = valid_df.groupby("method")["epsilon"].agg(["mean", "std", "count"])
    summary = summary.sort_values("mean")
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    
    methods = summary.index.tolist()
    means = summary["mean"].values
    stds = summary["std"].values
    colors = [get_method_color(m) for m in methods]
    
    # Plot bars with error bars
    x = np.arange(len(methods))
    yerr_lower = np.minimum(stds, means)  # Clip to not go below 0
    yerr_upper = stds
    
    bars = ax.bar(x, means, yerr=[yerr_lower, yerr_upper], capsize=3,
                  color=colors, alpha=0.8, edgecolor='black')
    
    # Add mean line
    overall_mean = valid_df["epsilon"].mean()
    ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2,
               label=f'Overall mean ε = {overall_mean:.3f}')
    
    # Customize
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Epsilon (ε)', fontsize=12)
    ax.set_xlabel('Voting Method', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Legend for categories
    legend_patches = [
        mpatches.Patch(color=METHOD_COLORS["traditional"], label='Traditional'),
        mpatches.Patch(color=METHOD_COLORS["chatgpt"], label='ChatGPT'),
        mpatches.Patch(color=METHOD_COLORS["chatgpt_star"], label='ChatGPT*'),
        mpatches.Patch(color=METHOD_COLORS["chatgpt_double_star"], label='ChatGPT**'),
        plt.Line2D([0], [0], color='red', linestyle='--', label=f'Mean ε = {overall_mean:.3f}'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved bar chart to {output_path}")


def plot_comparison_by_alt_dist(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Epsilon by Alternative Distribution"
) -> None:
    """
    Create grouped bar chart comparing alternative distributions.
    """
    if df.empty:
        logger.warning("No data to plot")
        return
    
    valid_df = df[df["epsilon"].notna()]
    if valid_df.empty:
        return
    
    # Pivot to get method × alt_dist
    pivot = valid_df.pivot_table(
        values="epsilon",
        index="method",
        columns="alt_dist",
        aggfunc="mean"
    )
    
    if pivot.empty:
        return
    
    # Sort by overall mean
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean")
    pivot = pivot.drop("_mean", axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    
    # Plot grouped bars
    methods = pivot.index.tolist()
    x = np.arange(len(methods))
    width = 0.2
    
    colors = sns.color_palette("husl", len(ALT_DISTRIBUTIONS))
    
    for i, alt_dist in enumerate(ALT_DISTRIBUTIONS):
        if alt_dist in pivot.columns:
            values = pivot[alt_dist].values
            offset = (i - len(ALT_DISTRIBUTIONS)/2 + 0.5) * width
            label = ALT_DIST_LABELS.get(alt_dist, alt_dist)
            ax.bar(x + offset, values, width, label=label, color=colors[i], alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Mean Epsilon (ε)', fontsize=12)
    ax.set_xlabel('Voting Method', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved alt distribution comparison to {output_path}")


def plot_comparison_by_voter_dist(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Epsilon by Voter Distribution"
) -> None:
    """
    Create grouped bar chart comparing voter distributions.
    """
    if df.empty:
        logger.warning("No data to plot")
        return
    
    valid_df = df[df["epsilon"].notna()]
    if valid_df.empty:
        return
    
    # Pivot to get method × voter_dist
    pivot = valid_df.pivot_table(
        values="epsilon",
        index="method",
        columns="voter_dist",
        aggfunc="mean"
    )
    
    if pivot.empty:
        return
    
    # Sort by overall mean
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean")
    pivot = pivot.drop("_mean", axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    
    # Plot grouped bars
    methods = pivot.index.tolist()
    voter_dists = pivot.columns.tolist()
    x = np.arange(len(methods))
    width = 0.25
    
    colors = {"uniform": "#3498db", "progressive_liberal": "#2ecc71", "conservative_traditional": "#e74c3c"}
    
    for i, voter_dist in enumerate(voter_dists):
        values = pivot[voter_dist].values
        offset = (i - len(voter_dists)/2 + 0.5) * width
        label = VOTER_DIST_LABELS.get(voter_dist, voter_dist)
        color = colors.get(voter_dist, sns.color_palette("husl")[i])
        ax.bar(x + offset, values, width, label=label, color=color, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Mean Epsilon (ε)', fontsize=12)
    ax.set_xlabel('Voting Method', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved voter distribution comparison to {output_path}")


def plot_heatmap_method_alt(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Mean Epsilon: Method × Alternative Distribution"
) -> None:
    """
    Create heatmap of mean epsilon: voting method × alternative distribution.
    """
    if df.empty:
        logger.warning("No data to plot")
        return
    
    valid_df = df[df["epsilon"].notna()]
    if valid_df.empty:
        return
    
    # Pivot to get method × alt_dist
    pivot = valid_df.pivot_table(
        values="epsilon",
        index="method",
        columns="alt_dist",
        aggfunc="mean"
    )
    
    if pivot.empty:
        return
    
    # Sort by overall mean
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean")
    pivot = pivot.drop("_mean", axis=1)
    
    # Rename columns for display
    pivot = pivot.rename(columns=ALT_DIST_LABELS)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
                ax=ax, cbar_kws={'label': 'Mean Epsilon'})
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Alternative Distribution', fontsize=12)
    ax.set_ylabel('Voting Method', fontsize=12)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved heatmap to {output_path}")


def plot_heatmap_method_voter(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Mean Epsilon: Method × Voter Distribution"
) -> None:
    """
    Create heatmap of mean epsilon: voting method × voter distribution.
    """
    if df.empty:
        logger.warning("No data to plot")
        return
    
    valid_df = df[df["epsilon"].notna()]
    if valid_df.empty:
        return
    
    # Pivot to get method × voter_dist
    pivot = valid_df.pivot_table(
        values="epsilon",
        index="method",
        columns="voter_dist",
        aggfunc="mean"
    )
    
    if pivot.empty:
        return
    
    # Sort by overall mean
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean")
    pivot = pivot.drop("_mean", axis=1)
    
    # Rename columns for display
    pivot = pivot.rename(columns=VOTER_DIST_LABELS)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r',
                ax=ax, cbar_kws={'label': 'Mean Epsilon'})
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Voter Distribution', fontsize=12)
    ax.set_ylabel('Voting Method', fontsize=12)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved heatmap to {output_path}")


def plot_cdf_epsilon(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "CDF of Critical Epsilon"
) -> None:
    """
    Create CDF plot of epsilon for each voting method.
    
    Includes a "Random" baseline which represents mean epsilon across all alternatives.
    """
    if df.empty:
        logger.warning("No data to plot")
        return
    
    valid_df = df[df["epsilon"].notna()]
    if valid_df.empty:
        logger.warning("No valid epsilon values")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    
    # Method colors for CDF
    method_colors = {
        'schulze': '#3498db',      # Blue
        'borda': '#e67e22',        # Orange
        'irv': '#2ecc71',          # Green
        'plurality': '#e74c3c',    # Red
        'veto_by_consumption': '#9b59b6',  # Purple
    }
    
    method_labels = {
        'schulze': 'Schulze',
        'borda': 'Borda',
        'irv': 'IRV',
        'plurality': 'Plurality',
        'veto_by_consumption': 'VBC',
    }
    
    # Plot CDF for each method
    methods = [m for m in TRADITIONAL_METHODS if m in valid_df["method"].unique()]
    
    for method in methods:
        method_data = valid_df[valid_df["method"] == method]["epsilon"].values
        if len(method_data) == 0:
            continue
        
        sorted_data = np.sort(method_data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        color = method_colors.get(method, '#95a5a6')
        label = method_labels.get(method, method)
        ax.plot(sorted_data, cdf, label=label, color=color, linewidth=2)
    
    # Add "Random" baseline - mean epsilon of all alternatives
    # This simulates randomly picking an alternative
    # We approximate this by taking all epsilon values and computing CDF
    all_epsilons = valid_df["epsilon"].values
    if len(all_epsilons) > 0:
        # For random, we use the distribution of all epsilons
        sorted_random = np.sort(all_epsilons)
        cdf_random = np.arange(1, len(sorted_random) + 1) / len(sorted_random)
        ax.plot(sorted_random, cdf_random, label='Random', color='black', 
                linewidth=2, linestyle='--')
    
    # Customize
    ax.set_xlabel('Critical Epsilon (ε*)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(0.1, valid_df["epsilon"].max() * 1.1))
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved CDF plot to {output_path}")


def plot_heatmap_method_topic(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Mean Epsilon: Method × Topic"
) -> None:
    """
    Create heatmap of mean epsilon: voting method × topic.
    """
    if df.empty:
        logger.warning("No data to plot")
        return
    
    valid_df = df[df["epsilon"].notna()]
    if valid_df.empty:
        return
    
    # Pivot to get method × topic
    pivot = valid_df.pivot_table(
        values="epsilon",
        index="method",
        columns="topic",
        aggfunc="mean"
    )
    
    if pivot.empty:
        return
    
    # Sort by overall mean
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean")
    pivot = pivot.drop("_mean", axis=1)
    
    # Capitalize topic names
    pivot = pivot.rename(columns=lambda x: x.title())
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SQUARE)
    
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r',
                ax=ax, cbar_kws={'label': 'Mean Epsilon'})
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Topic', fontsize=12)
    ax.set_ylabel('Voting Method', fontsize=12)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved heatmap to {output_path}")


def plot_heatmap_method_rep(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Mean Epsilon: Method × Rep"
) -> None:
    """
    Create heatmap of mean epsilon: voting method × rep_id.
    """
    if df.empty:
        logger.warning("No data to plot")
        return
    
    valid_df = df[df["epsilon"].notna()]
    if valid_df.empty:
        return
    
    # Pivot to get method × rep_id
    pivot = valid_df.pivot_table(
        values="epsilon",
        index="method",
        columns="rep_id",
        aggfunc="mean"
    )
    
    if pivot.empty:
        return
    
    # Sort by overall mean
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean")
    pivot = pivot.drop("_mean", axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r',
                ax=ax, cbar_kws={'label': 'Mean Epsilon'})
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Rep ID', fontsize=12)
    ax.set_ylabel('Voting Method', fontsize=12)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved heatmap to {output_path}")


def generate_all_plots(
    df: pd.DataFrame = None,
    output_dir: Path = None
) -> None:
    """
    Generate all visualization plots.
    
    Args:
        df: Results DataFrame (loads from disk if not provided)
        output_dir: Output directory (defaults to PHASE2_FIGURES_DIR)
    """
    if df is None:
        logger.info("Loading results...")
        df = collect_all_results()
    
    if output_dir is None:
        output_dir = PHASE2_FIGURES_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating plots from {len(df)} results...")
    
    # 1. Overall epsilon by method
    plot_epsilon_by_method(
        df,
        output_dir / "bar_by_method.png",
        title="Voting Method Performance (Mean Epsilon)"
    )
    
    # 2. Comparison by alternative distribution
    plot_comparison_by_alt_dist(
        df,
        output_dir / "comparison_alt_dists.png",
        title="Performance by Alternative Distribution"
    )
    
    # 3. Comparison by voter distribution
    plot_comparison_by_voter_dist(
        df,
        output_dir / "comparison_voter_dists.png",
        title="Performance by Voter Distribution"
    )
    
    # 4. Heatmap: method × alt distribution
    plot_heatmap_method_alt(
        df,
        output_dir / "heatmap_method_alt.png",
        title="Mean Epsilon: Method × Alternative Distribution"
    )
    
    # 5. Heatmap: method × voter distribution
    plot_heatmap_method_voter(
        df,
        output_dir / "heatmap_method_voter.png",
        title="Mean Epsilon: Method × Voter Distribution"
    )
    
    # 6. CDF plot overall
    plot_cdf_epsilon(
        df,
        output_dir / "cdf_epsilon.png",
        title="CDF of Critical Epsilon - All Conditions"
    )
    
    # Per-topic plots
    for topic in df["topic"].unique():
        topic_df = df[df["topic"] == topic]
        topic_dir = output_dir / topic
        topic_dir.mkdir(exist_ok=True)
        
        plot_epsilon_by_method(
            topic_df,
            topic_dir / "bar_by_method.png",
            title=f"{topic.title()}: Voting Method Performance"
        )
        
        plot_heatmap_method_alt(
            topic_df,
            topic_dir / "heatmap_method_alt.png",
            title=f"{topic.title()}: Method × Alternative Distribution"
        )
        
        plot_cdf_epsilon(
            topic_df,
            topic_dir / "cdf_epsilon.png",
            title=f"{topic.title()}: CDF of Critical Epsilon"
        )
    
    # Per topic × (alt_dist, voter_dist) combination plots
    # Get unique combinations
    alt_dists = df["alt_dist"].unique()
    voter_dists = df["voter_dist"].unique()
    topics = df["topic"].unique()
    
    for topic in topics:
        topic_dir = output_dir / topic
        
        for alt_dist in alt_dists:
            for voter_dist in voter_dists:
                # Filter data for this combination
                combo_df = df[
                    (df["topic"] == topic) & 
                    (df["alt_dist"] == alt_dist) & 
                    (df["voter_dist"] == voter_dist)
                ]
                
                if len(combo_df) == 0:
                    continue
                
                # Create subfolder within topic
                combo_dir = topic_dir / f"{alt_dist}_{voter_dist}"
                combo_dir.mkdir(parents=True, exist_ok=True)
                
                alt_label = ALT_DIST_LABELS.get(alt_dist, alt_dist)
                voter_label = VOTER_DIST_LABELS.get(voter_dist, voter_dist)
                
                plot_epsilon_by_method(
                    combo_df,
                    combo_dir / "bar_by_method.png",
                    title=f"{topic.title()}: {alt_label} × {voter_label}"
                )
                
                # Heatmap by rep for this combo (since we're already filtered by topic)
                plot_heatmap_method_rep(
                    combo_df,
                    combo_dir / "heatmap_method_rep.png",
                    title=f"{topic.title()}: {alt_label} × {voter_label}"
                )
                
                # CDF plot for this combo
                plot_cdf_epsilon(
                    combo_df,
                    combo_dir / "cdf_epsilon.png",
                    title=f"{topic.title()}: {alt_label} × {voter_label}"
                )
    
    logger.info(f"All plots saved to {output_dir}")


def main():
    """CLI entry point for visualization."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description="Generate Phase 2 experiment visualizations"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all plots"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for plots"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    if args.all:
        generate_all_plots(output_dir=output_dir)
    else:
        # Default: generate all
        generate_all_plots(output_dir=output_dir)


if __name__ == "__main__":
    main()
