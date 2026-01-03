"""
Plotting functions for epsilon-100 values.

Generates bar plots and strip plots showing epsilon computed with respect to
100 personas using winners from 20 personas.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import VOTING_METHODS, OUTPUT_DIR, TOPIC_SHORT_NAMES, TOPIC_DISPLAY_NAMES
from .visualizer import (
    METHOD_COLORS,
    METHOD_NAMES,
    BARPLOT_METHOD_ORDER,
    compute_cluster_ci,
)
from .epsilon_100 import (
    collect_all_epsilon_100,
    collect_all_epsilon_100_clustered,
    collect_epsilon_100_for_topic,
    collect_all_epsilon_100_for_n_personas,
    collect_all_epsilon_100_for_n_personas_clustered,
    PERSONA_COUNTS,
)

logger = logging.getLogger(__name__)


def plot_epsilon_100_barplot(
    clustered_results: Dict[str, List[List[float]]],
    title: str = "Average Epsilon (100 Personas) by Voting Method",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot bar chart of average epsilon-100 with 95% CI error bars (cluster-aware).
    
    Args:
        clustered_results: Dict mapping method to list of lists (outer reps → inner samples)
        title: Plot title
        output_path: Path to save figure (None = show)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    means = []
    cis = []
    colors = []
    n_clusters_list = []
    
    # Use custom order for bar plots
    for method in BARPLOT_METHOD_ORDER:
        clusters = clustered_results.get(method, [])
        if clusters:
            mean, ci, n_clusters = compute_cluster_ci(clusters)
            if mean is not None:
                methods.append(METHOD_NAMES.get(method, method))
                means.append(mean)
                cis.append(ci if ci is not None else 0)
                colors.append(METHOD_COLORS.get(method, "#333333"))
                n_clusters_list.append(n_clusters)
    
    if not methods:
        logger.warning("No data to plot")
        return
    
    x = np.arange(len(methods))
    
    # Asymmetric error bars: clip to [0, 1] (epsilon ∈ [0, 1])
    lower_errors = [min(ci, mean) for mean, ci in zip(means, cis)]  # Can't go below 0
    upper_errors = [min(ci, 1 - mean) for mean, ci in zip(means, cis)]  # Can't go above 1
    yerr = [lower_errors, upper_errors]
    
    bars = ax.bar(x, means, yerr=yerr, capsize=5, color=colors, alpha=0.8)
    
    ax.set_xlabel("Voting Method", fontsize=12)
    ax.set_ylabel("Average Epsilon (ε)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, ci in zip(bars, means, cis):
        height = bar.get_height()
        ax.annotate(
            f'{mean:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=9
        )
    
    # Add note about error bars
    n_clusters = n_clusters_list[0] if n_clusters_list else 0
    ax.text(0.02, 0.98, f"Error bars: 95% CI (n={n_clusters} outer reps)",
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            color='gray')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved epsilon-100 barplot to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_epsilon_100_stripplot(
    results: Dict[str, List[float]],
    title: str = "Epsilon (100 Personas) Distribution by Voting Method",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot horizontal strip plot of epsilon-100 values with methods as rows.
    
    Args:
        results: Dict mapping method to list of epsilon-100 values
        title: Plot title
        output_path: Path to save figure (None = show)
    """
    # Prepare data for seaborn
    data = []
    for method in BARPLOT_METHOD_ORDER:
        values = results.get(method, [])
        display_name = METHOD_NAMES.get(method, method)
        for v in values:
            if v is not None and v >= 0:  # Filter out sentinel values
                data.append({"Method": display_name, "Epsilon": v, "method_key": method})
    
    if not data:
        logger.warning("No epsilon-100 values to plot")
        return
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create color palette based on method order
    method_order = [METHOD_NAMES.get(m, m) for m in BARPLOT_METHOD_ORDER if METHOD_NAMES.get(m, m) in df["Method"].values]
    palette = {METHOD_NAMES.get(m, m): METHOD_COLORS.get(m, "#333333") for m in BARPLOT_METHOD_ORDER}
    
    # Plot strip plot with jitter
    sns.stripplot(
        data=df,
        x="Epsilon",
        y="Method",
        hue="Method",
        order=method_order,
        hue_order=method_order,
        palette=palette,
        jitter=0.3,
        alpha=0.6,
        size=4,
        ax=ax,
        legend=False
    )
    
    # Add mean markers
    for i, method in enumerate(method_order):
        method_data = df[df["Method"] == method]["Epsilon"]
        if len(method_data) > 0:
            mean_val = method_data.mean()
            ax.scatter([mean_val], [i], color='black', s=100, marker='|', linewidths=2, zorder=10)
    
    ax.set_xlabel("Epsilon (ε)", fontsize=12)
    ax.set_ylabel("Voting Method", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add note about sample size
    n_samples = len(df)
    ax.text(0.98, 0.02, f"n={n_samples} samples | bars = means",
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            horizontalalignment='right', color='gray')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved epsilon-100 stripplot to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_epsilon_100_multi_persona_barplot(
    results_by_n_personas: Dict[int, Dict[str, List[List[float]]]],
    title: str = "Average Epsilon (100 Personas) by Persona Count",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot bar chart comparing average epsilon-100 across different persona counts.
    
    Args:
        results_by_n_personas: Dict mapping n_personas to clustered results
        title: Plot title
        output_path: Path to save figure (None = show)
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Get persona counts and methods
    persona_counts = sorted(results_by_n_personas.keys())
    n_groups = len(BARPLOT_METHOD_ORDER)
    n_bars = len(persona_counts)
    
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)
    
    # Colors for different persona counts
    persona_colors = {
        5: '#e74c3c',   # red
        10: '#f39c12',  # orange
        20: '#27ae60',  # green
    }
    
    for i, n_personas in enumerate(persona_counts):
        clustered_results = results_by_n_personas[n_personas]
        
        means = []
        cis = []
        
        for method in BARPLOT_METHOD_ORDER:
            clusters = clustered_results.get(method, [])
            if clusters:
                mean, ci, _ = compute_cluster_ci(clusters)
                means.append(mean if mean is not None else 0)
                cis.append(ci if ci is not None else 0)
            else:
                means.append(0)
                cis.append(0)
        
        # Asymmetric error bars
        lower_errors = [min(ci, mean) for mean, ci in zip(means, cis)]
        upper_errors = [min(ci, 1 - mean) for mean, ci in zip(means, cis)]
        yerr = [lower_errors, upper_errors]
        
        offset = (i - (n_bars - 1) / 2) * bar_width
        color = persona_colors.get(n_personas, '#333333')
        
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=yerr,
            capsize=3,
            color=color,
            alpha=0.8,
            label=f'{n_personas} personas'
        )
    
    ax.set_xlabel("Voting Method", fontsize=12)
    ax.set_ylabel("Average Epsilon (ε)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_NAMES.get(m, m) for m in BARPLOT_METHOD_ORDER], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved epsilon-100 multi-persona barplot to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_epsilon_100_multi_persona_stripplot(
    results_by_n_personas: Dict[int, Dict[str, List[float]]],
    title: str = "Epsilon (100 Personas) Distribution by Persona Count",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot strip plot comparing epsilon-100 distributions across different persona counts.
    
    Args:
        results_by_n_personas: Dict mapping n_personas to flat results
        title: Plot title
        output_path: Path to save figure (None = show)
    """
    # Prepare data for seaborn
    data = []
    for n_personas, results in results_by_n_personas.items():
        for method in BARPLOT_METHOD_ORDER:
            values = results.get(method, [])
            display_name = METHOD_NAMES.get(method, method)
            for v in values:
                if v is not None and v >= 0:
                    data.append({
                        "Method": display_name,
                        "Epsilon": v,
                        "Personas": f"{n_personas}",
                        "method_key": method
                    })
    
    if not data:
        logger.warning("No epsilon-100 values to plot")
        return
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get method order
    method_order = [METHOD_NAMES.get(m, m) for m in BARPLOT_METHOD_ORDER if METHOD_NAMES.get(m, m) in df["Method"].values]
    
    # Colors for different persona counts
    persona_palette = {
        '5': '#e74c3c',   # red
        '10': '#f39c12',  # orange
        '20': '#27ae60',  # green
    }
    
    # Plot strip plot with hue for persona counts
    sns.stripplot(
        data=df,
        x="Epsilon",
        y="Method",
        hue="Personas",
        order=method_order,
        hue_order=['5', '10', '20'],
        palette=persona_palette,
        jitter=0.3,
        alpha=0.6,
        size=4,
        ax=ax,
        dodge=True
    )
    
    ax.set_xlabel("Epsilon (ε)", fontsize=12)
    ax.set_ylabel("Voting Method", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(title="Personas", loc='lower right')
    
    # Add note about sample size
    n_samples = len(df)
    ax.text(0.98, 0.02, f"n={n_samples} total samples",
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            horizontalalignment='right', color='gray')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved epsilon-100 multi-persona stripplot to {output_path}")
        plt.close()
    else:
        plt.show()


def generate_epsilon_100_plots(
    output_dir: Path = OUTPUT_DIR,
    topics: Optional[List[str]] = None,
    ablations: Optional[List[str]] = None
) -> None:
    """
    Generate epsilon-100 plots for all ablations.
    
    Args:
        output_dir: Output directory
        topics: List of topics (None = auto-detect)
        ablations: List of ablations to plot
    """
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    if ablations is None:
        ablations = ["full", "no_bridging", "no_filtering"]
    
    for ablation in ablations:
        logger.info(f"Generating epsilon-100 plots for ablation: {ablation}")
        
        # Create subfolder for this ablation
        ablation_dir = figures_dir / ablation
        ablation_dir.mkdir(parents=True, exist_ok=True)
        
        ablation_label = f" ({ablation.replace('_', ' ')})" if ablation != "full" else ""
        
        # Aggregate plots in ablation/aggregate/
        aggregate_dir = ablation_dir / "aggregate"
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect epsilon-100 results
        flat_results = collect_all_epsilon_100(output_dir, ablation, topics)
        clustered_results = collect_all_epsilon_100_clustered(output_dir, ablation, topics)
        
        # Log some stats
        total_samples = sum(len(v) for v in flat_results.values())
        logger.info(f"  Collected {total_samples} epsilon-100 values")
        
        # Generate barplot
        plot_epsilon_100_barplot(
            clustered_results,
            title=f"Average Epsilon (100 Personas) by Voting Method{ablation_label}",
            output_path=aggregate_dir / "epsilon_100_barplot.png"
        )
        
        # Generate stripplot
        plot_epsilon_100_stripplot(
            flat_results,
            title=f"Epsilon (100 Personas) Distribution by Voting Method{ablation_label}",
            output_path=aggregate_dir / "epsilon_100_stripplot.png"
        )
        
        # Per-topic plots in ablation/topic_short_name/
        data_dir = output_dir / "data"
        if topics is None and data_dir.exists():
            topics_to_plot = [d.name for d in data_dir.iterdir() if d.is_dir()]
        else:
            topics_to_plot = topics or []
        
        for topic in topics_to_plot:
            # Use short name for folder, display name for title
            short_name = TOPIC_SHORT_NAMES.get(topic, topic[:20])
            display_name = TOPIC_DISPLAY_NAMES.get(topic, topic[:50])
            
            # Create subfolder for this topic
            topic_dir = ablation_dir / short_name
            topic_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect epsilon-100 results for this topic
            topic_results = collect_epsilon_100_for_topic(topic, output_dir, ablation)
            
            # Per-topic epsilon-100 strip plot
            plot_epsilon_100_stripplot(
                topic_results,
                title=f"Epsilon (100 Personas): {display_name}{ablation_label}",
                output_path=topic_dir / "epsilon_100_stripplot.png"
            )
    
    logger.info("Epsilon-100 plot generation complete!")


def generate_multi_persona_epsilon_100_plots(
    output_dir: Path = OUTPUT_DIR,
    topics: Optional[List[str]] = None,
    ablations: Optional[List[str]] = None,
    persona_counts: Optional[List[int]] = None
) -> None:
    """
    Generate multi-persona comparison plots for epsilon-100.
    
    Args:
        output_dir: Output directory
        topics: List of topics (None = auto-detect)
        ablations: List of ablations to plot
        persona_counts: List of persona counts to compare (default: [5, 10, 20])
    """
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    if ablations is None:
        ablations = ["full"]
    
    if persona_counts is None:
        persona_counts = PERSONA_COUNTS
    
    for ablation in ablations:
        logger.info(f"Generating multi-persona epsilon-100 plots for ablation: {ablation}")
        
        # Create subfolder for this ablation
        ablation_dir = figures_dir / ablation
        ablation_dir.mkdir(parents=True, exist_ok=True)
        
        ablation_label = f" ({ablation.replace('_', ' ')})" if ablation != "full" else ""
        
        # Aggregate plots in ablation/aggregate/
        aggregate_dir = ablation_dir / "aggregate"
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect epsilon-100 results for each persona count
        flat_results_by_n = {}
        clustered_results_by_n = {}
        
        for n_personas in persona_counts:
            flat_results = collect_all_epsilon_100_for_n_personas(
                n_personas, output_dir, ablation, topics
            )
            clustered_results = collect_all_epsilon_100_for_n_personas_clustered(
                n_personas, output_dir, ablation, topics
            )
            
            # Only include if we have data
            total_samples = sum(len(v) for v in flat_results.values())
            if total_samples > 0:
                flat_results_by_n[n_personas] = flat_results
                clustered_results_by_n[n_personas] = clustered_results
                logger.info(f"  {n_personas} personas: {total_samples} epsilon-100 values")
        
        if not flat_results_by_n:
            logger.warning(f"No multi-persona data found for ablation {ablation}")
            continue
        
        # Generate multi-persona barplot
        plot_epsilon_100_multi_persona_barplot(
            clustered_results_by_n,
            title=f"Average Epsilon (100 Personas) by Persona Count{ablation_label}",
            output_path=aggregate_dir / "epsilon_100_multi_persona_barplot.png"
        )
        
        # Generate multi-persona stripplot
        plot_epsilon_100_multi_persona_stripplot(
            flat_results_by_n,
            title=f"Epsilon (100 Personas) Distribution by Persona Count{ablation_label}",
            output_path=aggregate_dir / "epsilon_100_multi_persona_stripplot.png"
        )
    
    logger.info("Multi-persona epsilon-100 plot generation complete!")


if __name__ == "__main__":
    # Allow running as a script
    import sys
    logging.basicConfig(level=logging.INFO)
    
    output_dir = Path("outputs/full_experiment")
    data_dir = output_dir / "data"
    
    # Auto-discover topics
    if data_dir.exists():
        topics = [d.name for d in sorted(data_dir.iterdir()) if d.is_dir()]
        print(f"Found {len(topics)} topics: {topics}")
    else:
        topics = None
        print("Data directory not found, will auto-detect topics")
    
    generate_epsilon_100_plots(
        output_dir=output_dir,
        topics=topics,
        ablations=["full", "no_bridging", "no_filtering"]
    )
    
    # Also generate multi-persona comparison plots
    generate_multi_persona_epsilon_100_plots(
        output_dir=output_dir,
        topics=topics,
        ablations=["full"]
    )

