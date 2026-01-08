"""
Generate multi-persona barplots with different error bar types:
- Standard Error (SE)
- Interquartile Range (25th-75th percentile)

Usage:
    uv run python -m src.full_experiment.generate_errorbar_variants
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .config import OUTPUT_DIR, ABLATIONS
from .visualizer import (
    BARPLOT_METHOD_ORDER,
    METHOD_NAMES,
    collect_all_results_for_n_personas_clustered,
)
from .epsilon_100_plotter import (
    collect_all_epsilon_100_for_n_personas_clustered,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PERSONA_COUNTS = [5, 10, 20]


def compute_cluster_se(clustered_values: List[List[float]]) -> Tuple[Optional[float], Optional[float], int]:
    """
    Compute cluster-aware standard error.
    
    Returns:
        Tuple of (grand_mean, standard_error, n_clusters)
    """
    if not clustered_values:
        return None, None, 0
    
    # Compute mean within each cluster
    cluster_means = []
    for cluster in clustered_values:
        valid = [v for v in cluster if v is not None]
        if valid:
            cluster_means.append(np.mean(valid))
    
    if not cluster_means:
        return None, None, 0
    
    n = len(cluster_means)
    
    if n < 2:
        return np.mean(cluster_means), None, n
    
    grand_mean = np.mean(cluster_means)
    se = np.std(cluster_means, ddof=1) / np.sqrt(n)
    
    return grand_mean, se, n


def compute_cluster_iqr(clustered_values: List[List[float]]) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    """
    Compute cluster-aware IQR (25th and 75th percentiles).
    
    Returns:
        Tuple of (median, p25, p75, n_clusters)
    """
    if not clustered_values:
        return None, None, None, 0
    
    # Compute mean within each cluster
    cluster_means = []
    for cluster in clustered_values:
        valid = [v for v in cluster if v is not None]
        if valid:
            cluster_means.append(np.mean(valid))
    
    if not cluster_means:
        return None, None, None, 0
    
    n = len(cluster_means)
    
    if n < 2:
        return np.mean(cluster_means), None, None, n
    
    # Use mean as center (not median) for consistency with other plots
    grand_mean = np.mean(cluster_means)
    p25 = np.percentile(cluster_means, 25)
    p75 = np.percentile(cluster_means, 75)
    
    return grand_mean, p25, p75, n


def plot_multi_persona_barplot_se(
    results_by_n_personas: Dict[int, Dict[str, List[List[float]]]],
    title: str,
    output_path: Path
) -> None:
    """Plot multi-persona barplot with Standard Error bars."""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    persona_counts = sorted(results_by_n_personas.keys())
    n_bars = len(persona_counts)
    n_methods = len(BARPLOT_METHOD_ORDER)
    
    x = np.arange(n_methods)
    bar_width = 0.8 / n_bars
    
    persona_colors = {
        5: '#3498db',   # blue
        10: '#e67e22',  # orange
        20: '#27ae60',  # green
    }
    
    for i, n_personas in enumerate(persona_counts):
        clustered_results = results_by_n_personas[n_personas]
        
        means = []
        ses = []
        
        for method in BARPLOT_METHOD_ORDER:
            clusters = clustered_results.get(method, [])
            if clusters:
                mean, se, _ = compute_cluster_se(clusters)
                means.append(mean if mean is not None else 0)
                ses.append(se if se is not None else 0)
            else:
                means.append(0)
                ses.append(0)
        
        # Asymmetric error bars (bounded by 0 and 1)
        lower_errors = [min(se, mean) for mean, se in zip(means, ses)]
        upper_errors = [min(se, 1 - mean) for mean, se in zip(means, ses)]
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
    ax.set_ylabel("Average Epsilon (ε) ± SE", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_NAMES.get(m, m) for m in BARPLOT_METHOD_ORDER], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved SE barplot to {output_path}")
    plt.close()


def plot_multi_persona_barplot_iqr(
    results_by_n_personas: Dict[int, Dict[str, List[List[float]]]],
    title: str,
    output_path: Path
) -> None:
    """Plot multi-persona barplot with IQR (25th-75th percentile) bars."""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    persona_counts = sorted(results_by_n_personas.keys())
    n_bars = len(persona_counts)
    n_methods = len(BARPLOT_METHOD_ORDER)
    
    x = np.arange(n_methods)
    bar_width = 0.8 / n_bars
    
    persona_colors = {
        5: '#3498db',   # blue
        10: '#e67e22',  # orange
        20: '#27ae60',  # green
    }
    
    for i, n_personas in enumerate(persona_counts):
        clustered_results = results_by_n_personas[n_personas]
        
        means = []
        lower_errs = []
        upper_errs = []
        
        for method in BARPLOT_METHOD_ORDER:
            clusters = clustered_results.get(method, [])
            if clusters:
                mean, p25, p75, _ = compute_cluster_iqr(clusters)
                if mean is not None:
                    means.append(mean)
                    # Error bars: distance from mean to percentiles (ensure non-negative)
                    lower_errs.append(max(0, mean - p25) if p25 is not None else 0)
                    upper_errs.append(max(0, p75 - mean) if p75 is not None else 0)
                else:
                    means.append(0)
                    lower_errs.append(0)
                    upper_errs.append(0)
            else:
                means.append(0)
                lower_errs.append(0)
                upper_errs.append(0)
        
        # Bound error bars by 0 and 1 (ensure non-negative)
        lower_errs = [max(0, min(le, m)) for le, m in zip(lower_errs, means)]
        upper_errs = [max(0, min(ue, 1 - m)) for ue, m in zip(upper_errs, means)]
        yerr = [lower_errs, upper_errs]
        
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
    ax.set_ylabel("Average Epsilon (ε) [25th-75th %ile]", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_NAMES.get(m, m) for m in BARPLOT_METHOD_ORDER], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved IQR barplot to {output_path}")
    plt.close()


def generate_all_errorbar_variants(
    output_dir: Path = OUTPUT_DIR,
    ablations: Optional[List[str]] = None,
    persona_counts: Optional[List[int]] = None
) -> None:
    """Generate SE and IQR variants for all ablations."""
    
    if ablations is None:
        ablations = ABLATIONS
    if persona_counts is None:
        persona_counts = PERSONA_COUNTS
    
    for ablation in ablations:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing ablation: {ablation}")
        logger.info(f"{'='*60}")
        
        figures_dir = output_dir / "figures" / ablation / "aggregate"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        ablation_label = f" ({ablation.replace('_', ' ')})" if ablation != "full" else ""
        
        # Collect epsilon data
        logger.info("Collecting epsilon data...")
        clustered_results_by_n = {}
        for n_personas in persona_counts:
            clustered_results = collect_all_results_for_n_personas_clustered(
                n_personas, output_dir, ablation, None
            )
            total_samples = sum(
                sum(len(c) for c in clusters) 
                for clusters in clustered_results.values()
            )
            if total_samples > 0:
                clustered_results_by_n[n_personas] = clustered_results
                logger.info(f"  {n_personas} personas: {total_samples} samples")
        
        if clustered_results_by_n:
            # Generate SE variant
            plot_multi_persona_barplot_se(
                clustered_results_by_n,
                title=f"Average Epsilon by Persona Count{ablation_label} (±SE)",
                output_path=figures_dir / "epsilon_multi_persona_barplot_se.png"
            )
            
            # Generate IQR variant
            plot_multi_persona_barplot_iqr(
                clustered_results_by_n,
                title=f"Average Epsilon by Persona Count{ablation_label} (IQR)",
                output_path=figures_dir / "epsilon_multi_persona_barplot_iqr.png"
            )
        
        # Collect epsilon-100 data
        logger.info("Collecting epsilon-100 data...")
        clustered_100_by_n = {}
        for n_personas in persona_counts:
            clustered_results = collect_all_epsilon_100_for_n_personas_clustered(
                n_personas, output_dir, ablation, None
            )
            total_samples = sum(
                sum(len(c) for c in clusters) 
                for clusters in clustered_results.values()
            )
            if total_samples > 0:
                clustered_100_by_n[n_personas] = clustered_results
                logger.info(f"  {n_personas} personas: {total_samples} samples")
        
        if clustered_100_by_n:
            # Generate SE variant
            plot_multi_persona_barplot_se(
                clustered_100_by_n,
                title=f"Average Epsilon (100 Personas) by Persona Count{ablation_label} (±SE)",
                output_path=figures_dir / "epsilon_100_multi_persona_barplot_se.png"
            )
            
            # Generate IQR variant
            plot_multi_persona_barplot_iqr(
                clustered_100_by_n,
                title=f"Average Epsilon (100 Personas) by Persona Count{ablation_label} (IQR)",
                output_path=figures_dir / "epsilon_100_multi_persona_barplot_iqr.png"
            )
    
    logger.info("\n" + "="*60)
    logger.info("All error bar variant plots generated!")
    logger.info("="*60)


if __name__ == "__main__":
    generate_all_errorbar_variants()

