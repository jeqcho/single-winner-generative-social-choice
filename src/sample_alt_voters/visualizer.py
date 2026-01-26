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

# Fixed method order for bar charts
METHOD_ORDER = [
    "veto_by_consumption",
    "borda",
    "schulze",
    "irv",
    "plurality",
    "chatgpt",
    "chatgpt_rankings",
    "chatgpt_personas",
    "chatgpt_star",
    "chatgpt_star_rankings",
    "chatgpt_star_personas",
    "chatgpt_double_star",
    "chatgpt_double_star_rankings",
    "chatgpt_double_star_personas",
]

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


def reorder_methods(methods: List[str]) -> List[str]:
    """Reorder methods according to METHOD_ORDER."""
    ordered = [m for m in METHOD_ORDER if m in methods]
    # Add any unknown methods at the end
    unknown = [m for m in methods if m not in METHOD_ORDER]
    return ordered + unknown


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
    # Use fixed method order instead of sorting by mean
    ordered_methods = reorder_methods(summary.index.tolist())
    summary = summary.reindex(ordered_methods)
    
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
    title: str = "CDF of Critical Epsilon",
    x_max: float = 1.0,
    y_min: float = 0.0
) -> None:
    """
    Create 2x2 grouped CDF plot of epsilon by method category.
    
    Subplots: Traditional, ChatGPT, ChatGPT*, ChatGPT**
    Includes a "Random" baseline in each subplot.
    
    Args:
        df: DataFrame with epsilon values
        output_path: Path to save the plot
        title: Plot title
        x_max: Maximum value for x-axis (default 1.0)
        y_min: Minimum value for y-axis (default 0.0)
    """
    if df.empty:
        logger.warning("No data to plot")
        return
    
    valid_df = df[df["epsilon"].notna()]
    if valid_df.empty:
        logger.warning("No valid epsilon values")
        return
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Standard contrasting colors for within-subplot differentiation
    CONTRAST_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Method display names
    method_labels = {
        'schulze': 'Schulze',
        'borda': 'Borda',
        'irv': 'IRV',
        'plurality': 'Plurality',
        'veto_by_consumption': 'VBC',
        'chatgpt': 'GPT',
        'chatgpt_rankings': 'GPT+Rank',
        'chatgpt_personas': 'GPT+Pers',
        'chatgpt_star': 'GPT*',
        'chatgpt_star_rankings': 'GPT*+Rank',
        'chatgpt_star_personas': 'GPT*+Pers',
        'chatgpt_double_star': 'GPT**',
        'chatgpt_double_star_rankings': 'GPT**+Rank',
        'chatgpt_double_star_personas': 'GPT**+Pers',
    }
    
    # Method groups for each subplot
    groups = [
        ('Traditional Methods', TRADITIONAL_METHODS, axes[0, 0]),
        ('ChatGPT Methods', CHATGPT_METHODS, axes[0, 1]),
        ('ChatGPT* Methods', CHATGPT_STAR_METHODS, axes[1, 0]),
        ('ChatGPT** Methods', CHATGPT_DOUBLE_STAR_METHODS, axes[1, 1]),
    ]
    
    # Get random baseline (all epsilons combined)
    all_epsilons = valid_df["epsilon"].values
    
    for group_name, methods, ax in groups:
        color_idx = 0
        has_data = False
        
        for method in methods:
            method_data = valid_df[valid_df["method"] == method]["epsilon"].values
            if len(method_data) == 0:
                continue
            
            has_data = True
            sorted_data = np.sort(method_data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            color = CONTRAST_COLORS[color_idx % len(CONTRAST_COLORS)]
            label = method_labels.get(method, method)
            ax.step(sorted_data, cdf, where='post', label=label, color=color, linewidth=2.5)
            color_idx += 1
        
        # Add Random baseline (black line)
        if len(all_epsilons) > 0:
            sorted_random = np.sort(all_epsilons)
            cdf_random = np.arange(1, len(sorted_random) + 1) / len(sorted_random)
            ax.step(sorted_random, cdf_random, where='post', 
                    label='Random', color='black', linewidth=2.5)
        
        ax.set_xlabel('Epsilon', fontsize=10)
        ax.set_ylabel('Cumulative Probability', fontsize=10)
        ax.set_title(group_name, fontsize=12)
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min, 1.05)
        ax.grid(True, alpha=0.3)
        if has_data or len(all_epsilons) > 0:
            ax.legend(loc='lower right', fontsize=9)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
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
    
    # 6. CDF plots overall (standard and zoomed)
    plot_cdf_epsilon(
        df,
        output_dir / "cdf_epsilon.png",
        title="CDF of Critical Epsilon - All Conditions",
        x_max=1.0,
        y_min=0.0
    )
    plot_cdf_epsilon(
        df,
        output_dir / "cdf_epsilon_zoomed.png",
        title="CDF of Critical Epsilon - All Conditions (Zoomed)",
        x_max=0.5,
        y_min=0.5
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
        
        # CDF plots (standard and zoomed)
        plot_cdf_epsilon(
            topic_df,
            topic_dir / "cdf_epsilon.png",
            title=f"{topic.title()}: CDF of Critical Epsilon",
            x_max=1.0,
            y_min=0.0
        )
        plot_cdf_epsilon(
            topic_df,
            topic_dir / "cdf_epsilon_zoomed.png",
            title=f"{topic.title()}: CDF of Critical Epsilon (Zoomed)",
            x_max=0.5,
            y_min=0.5
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
                
                # CDF plots for this combo (standard and zoomed)
                plot_cdf_epsilon(
                    combo_df,
                    combo_dir / "cdf_epsilon.png",
                    title=f"{topic.title()}: {alt_label} × {voter_label}",
                    x_max=1.0,
                    y_min=0.0
                )
                plot_cdf_epsilon(
                    combo_df,
                    combo_dir / "cdf_epsilon_zoomed.png",
                    title=f"{topic.title()}: {alt_label} × {voter_label} (Zoomed)",
                    x_max=0.5,
                    y_min=0.5
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
