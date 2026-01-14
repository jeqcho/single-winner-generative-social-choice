"""
Visualization for the sampling experiment.

Generates plots with:
- Bar charts with horizontal red line showing mean epsilon
- Heatmaps of epsilon across (K, P) grid
- Comparison plots for ChatGPT variants
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from .config import (
    K_VALUES,
    P_VALUES,
    TRADITIONAL_METHODS,
    CHATGPT_METHODS,
    CHATGPT_STAR_METHODS,
    CHATGPT_DOUBLE_STAR_METHODS,
    ALL_METHODS,
    TOPIC_SHORT_NAMES,
)

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_all_results(output_dir: Path, topic_slug: str, n_reps: int) -> Dict:
    """
    Load all results for a topic across all reps, (K, P) combinations, and samples.
    
    Supports both old format (results.json directly in k{k}_p{p}/) and 
    new format (results.json in k{k}_p{p}/sample{idx}/).
    
    Returns:
        Dict with structure: {(rep_idx, sample_idx): {(k, p): {method: {winner, epsilon, ...}}}}
        For backward compatibility with old format, sample_idx will be 0.
    """
    all_results = {}
    
    for rep_idx in range(n_reps):
        rep_dir = output_dir / "data" / topic_slug / f"rep{rep_idx}"
        if not rep_dir.exists():
            continue
        
        for k in K_VALUES:
            for p in P_VALUES:
                kp_dir = rep_dir / f"k{k}_p{p}"
                if not kp_dir.exists():
                    continue
                
                # Check for new format (sample subdirectories)
                sample_dirs = sorted(kp_dir.glob("sample*"))
                
                if sample_dirs:
                    # New format: load from each sample subdirectory
                    for sample_dir in sample_dirs:
                        results_file = sample_dir / "results.json"
                        if results_file.exists():
                            # Extract sample index from directory name
                            sample_idx = int(sample_dir.name.replace("sample", ""))
                            key = (rep_idx, sample_idx)
                            
                            if key not in all_results:
                                all_results[key] = {}
                            
                            with open(results_file) as f:
                                all_results[key][(k, p)] = json.load(f)
                else:
                    # Old format: results.json directly in k{k}_p{p}/
                    results_file = kp_dir / "results.json"
                    if results_file.exists():
                        key = (rep_idx, 0)  # Treat as sample 0
                        
                        if key not in all_results:
                            all_results[key] = {}
                        
                        with open(results_file) as f:
                            all_results[key][(k, p)] = json.load(f)
    
    return all_results


def collect_epsilons_by_method(all_results: Dict) -> Dict[str, List[float]]:
    """
    Collect all epsilon values grouped by method.
    
    Args:
        all_results: Dict with keys (rep_idx, sample_idx) or rep_idx
    
    Returns:
        Dict mapping method name to list of epsilon values
    """
    epsilons = {method: [] for method in ALL_METHODS}
    
    for key, rep_results in all_results.items():
        for (k, p), kp_results in rep_results.items():
            for method, result in kp_results.items():
                if method in epsilons and result.get("epsilon") is not None:
                    epsilons[method].append(result["epsilon"])
    
    return epsilons


def collect_epsilons_by_kp(all_results: Dict) -> Dict[Tuple[int, int], Dict[str, List[float]]]:
    """
    Collect epsilon values grouped by (K, P) combination, then by method.
    
    Args:
        all_results: Dict with keys (rep_idx, sample_idx) or rep_idx
    
    Returns:
        Dict mapping (k, p) to {method: [epsilons]}
    """
    kp_epsilons = {}
    
    for k in K_VALUES:
        for p in P_VALUES:
            kp_epsilons[(k, p)] = {method: [] for method in ALL_METHODS}
    
    for key, rep_results in all_results.items():
        for (k, p), kp_results in rep_results.items():
            for method, result in kp_results.items():
                if method in kp_epsilons[(k, p)] and result.get("epsilon") is not None:
                    kp_epsilons[(k, p)][method].append(result["epsilon"])
    
    return kp_epsilons


def plot_epsilon_bar_chart(
    epsilons: Dict[str, List[float]],
    title: str,
    output_path: Path,
    show_mean_line: bool = True
) -> None:
    """
    Create bar chart of mean epsilon by voting method.
    
    Includes a horizontal red line showing the overall mean epsilon.
    """
    # Calculate means and stds
    methods = []
    means = []
    stds = []
    
    for method in ALL_METHODS:
        if epsilons.get(method) and len(epsilons[method]) > 0:
            methods.append(method)
            means.append(np.mean(epsilons[method]))
            stds.append(np.std(epsilons[method]))
    
    if not methods:
        logger.warning("No data to plot")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Color by method category
    colors = []
    for method in methods:
        if method in TRADITIONAL_METHODS:
            colors.append('#2ecc71')  # Green for traditional
        elif method in CHATGPT_METHODS:
            colors.append('#3498db')  # Blue for ChatGPT
        elif method in CHATGPT_STAR_METHODS:
            colors.append('#9b59b6')  # Purple for ChatGPT*
        elif method in CHATGPT_DOUBLE_STAR_METHODS:
            colors.append('#e74c3c')  # Red for ChatGPT**
        else:
            colors.append('#95a5a6')  # Gray for unknown
    
    # Plot bars with asymmetric error bars (clipped to not go below 0)
    x = np.arange(len(methods))
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    # Clip lower error bar so it doesn't go below 0
    yerr_lower = np.minimum(stds_arr, means_arr)
    yerr_upper = stds_arr
    bars = ax.bar(x, means, yerr=[yerr_lower, yerr_upper], capsize=3, color=colors, alpha=0.8, edgecolor='black')
    
    # Add mean line
    if show_mean_line:
        all_eps = [e for eps_list in epsilons.values() for e in eps_list if e is not None]
        if all_eps:
            overall_mean = np.mean(all_eps)
            ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean ε = {overall_mean:.3f}')
    
    # Customize
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Epsilon (ε)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    
    # Add legend for categories
    legend_patches = [
        mpatches.Patch(color='#2ecc71', label='Traditional'),
        mpatches.Patch(color='#3498db', label='ChatGPT'),
        mpatches.Patch(color='#9b59b6', label='ChatGPT*'),
        mpatches.Patch(color='#e74c3c', label='ChatGPT**'),
    ]
    ax.legend(handles=legend_patches + [plt.Line2D([0], [0], color='red', linestyle='--', 
              label=f'Mean ε')], loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved bar chart to {output_path}")


def plot_epsilon_heatmap(
    kp_epsilons: Dict[Tuple[int, int], Dict[str, List[float]]],
    method: str,
    title: str,
    output_path: Path
) -> None:
    """
    Create heatmap of mean epsilon across (K, P) grid for a single method.
    """
    # Build matrix
    data = np.zeros((len(K_VALUES), len(P_VALUES)))
    
    for i, k in enumerate(K_VALUES):
        for j, p in enumerate(P_VALUES):
            eps_list = kp_epsilons.get((k, p), {}).get(method, [])
            if eps_list:
                data[i, j] = np.mean(eps_list)
            else:
                data[i, j] = np.nan
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn_r',
                xticklabels=[f'P={p}' for p in P_VALUES],
                yticklabels=[f'K={k}' for k in K_VALUES],
                ax=ax, cbar_kws={'label': 'Mean Epsilon'})
    
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Number of Alternatives (P)', fontsize=10)
    ax.set_ylabel('Number of Voters (K)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved heatmap to {output_path}")


def plot_chatgpt_comparison(
    epsilons: Dict[str, List[float]],
    title: str,
    output_path: Path
) -> None:
    """
    Create comparison plot for ChatGPT vs ChatGPT* vs ChatGPT** variants.
    """
    # Group variants
    groups = {
        'Base': ['chatgpt', 'chatgpt_star', 'chatgpt_double_star'],
        '+Rankings': ['chatgpt_rankings', 'chatgpt_star_rankings', 'chatgpt_double_star_rankings'],
        '+Personas': ['chatgpt_personas', 'chatgpt_star_personas', 'chatgpt_double_star_personas'],
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    colors = {'chatgpt': '#3498db', 'chatgpt_star': '#9b59b6', 'chatgpt_double_star': '#e74c3c'}
    labels = {'chatgpt': 'ChatGPT', 'chatgpt_star': 'ChatGPT*', 'chatgpt_double_star': 'ChatGPT**'}
    
    for idx, (group_name, methods) in enumerate(groups.items()):
        ax = axes[idx]
        
        x = np.arange(3)
        means = []
        stds = []
        bar_colors = []
        bar_labels = []
        
        for method in methods:
            eps_list = epsilons.get(method, [])
            if eps_list:
                means.append(np.mean(eps_list))
                stds.append(np.std(eps_list))
            else:
                means.append(0)
                stds.append(0)
            
            # Determine base type
            if 'double_star' in method:
                bar_colors.append(colors['chatgpt_double_star'])
                bar_labels.append(labels['chatgpt_double_star'])
            elif 'star' in method:
                bar_colors.append(colors['chatgpt_star'])
                bar_labels.append(labels['chatgpt_star'])
            else:
                bar_colors.append(colors['chatgpt'])
                bar_labels.append(labels['chatgpt'])
        
        # Clip lower error bar so it doesn't go below 0
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        yerr_lower = np.minimum(stds_arr, means_arr)
        yerr_upper = stds_arr
        bars = ax.bar(x, means, yerr=[yerr_lower, yerr_upper], capsize=3, color=bar_colors, alpha=0.8, edgecolor='black')
        
        # Mean line
        all_means = [m for m in means if m > 0]
        if all_means:
            ax.axhline(y=np.mean(all_means), color='red', linestyle='--', linewidth=1.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=10)
        ax.set_title(group_name, fontsize=12)
        ax.set_ylabel('Epsilon (ε)' if idx == 0 else '', fontsize=10)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved ChatGPT comparison to {output_path}")


def plot_kp_bar_charts(
    kp_epsilons: Dict[Tuple[int, int], Dict[str, List[float]]],
    topic_short: str,
    output_dir: Path
) -> None:
    """
    Create separate bar charts for each (K, P) combination.
    """
    for (k, p), method_epsilons in kp_epsilons.items():
        title = f"{topic_short}: K={k}, P={p}"
        output_path = output_dir / f"bar_k{k}_p{p}.png"
        plot_epsilon_bar_chart(method_epsilons, title, output_path)


def generate_all_visualizations(
    output_dir: Path,
    topic_slug: str,
    n_reps: int
) -> None:
    """
    Generate all visualizations for a topic.
    """
    logger.info(f"Generating visualizations for {topic_slug}...")
    
    # Create figures directory
    figures_dir = output_dir / "figures" / topic_slug
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    all_results = load_all_results(output_dir, topic_slug, n_reps)
    
    if not all_results:
        logger.warning(f"No results found for {topic_slug}")
        return
    
    # Collect epsilons
    epsilons = collect_epsilons_by_method(all_results)
    kp_epsilons = collect_epsilons_by_kp(all_results)
    
    topic_short = TOPIC_SHORT_NAMES.get(topic_slug, topic_slug[:10])
    
    # 1. Overall bar chart
    plot_epsilon_bar_chart(
        epsilons,
        f"{topic_short}: Epsilon by Voting Method",
        figures_dir / "bar_overall.png"
    )
    
    # 2. Bar charts per (K, P)
    plot_kp_bar_charts(kp_epsilons, topic_short, figures_dir)
    
    # 3. Heatmaps for all methods
    for method in ALL_METHODS:
        if any(epsilons.get(method, [])):
            plot_epsilon_heatmap(
                kp_epsilons,
                method,
                f"{topic_short}: {method} Epsilon by (K, P)",
                figures_dir / f"heatmap_{method}.png"
            )
    
    # 4. ChatGPT comparison
    plot_chatgpt_comparison(
        epsilons,
        f"{topic_short}: ChatGPT Variants Comparison",
        figures_dir / "chatgpt_comparison.png"
    )
    
    logger.info(f"Completed visualizations for {topic_slug}")


def plot_summary_across_topics(
    output_dir: Path,
    topics: List[str],
    n_reps: int
) -> None:
    """
    Generate summary visualizations across all topics.
    """
    logger.info("Generating summary visualizations across topics...")
    
    figures_dir = output_dir / "figures" / "summary"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all epsilons across topics
    all_epsilons = {method: [] for method in ALL_METHODS}
    
    for topic_slug in topics:
        all_results = load_all_results(output_dir, topic_slug, n_reps)
        topic_epsilons = collect_epsilons_by_method(all_results)
        
        for method, eps_list in topic_epsilons.items():
            all_epsilons[method].extend(eps_list)
    
    # Summary bar chart
    plot_epsilon_bar_chart(
        all_epsilons,
        "All Topics: Epsilon by Voting Method",
        figures_dir / "bar_all_topics.png"
    )
    
    # Summary ChatGPT comparison
    plot_chatgpt_comparison(
        all_epsilons,
        "All Topics: ChatGPT Variants Comparison",
        figures_dir / "chatgpt_comparison_all.png"
    )
    
    logger.info("Completed summary visualizations")
