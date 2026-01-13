"""
CDF (Cumulative Distribution Function) plots for the sampling experiment.

Creates CDF versions of all bar plots showing the distribution of epsilon values.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

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

# Method colors matching the bar charts
METHOD_COLORS = {
    # Traditional (green shades)
    'schulze': '#27ae60',
    'borda': '#2ecc71',
    'irv': '#1abc9c',
    'plurality': '#16a085',
    'veto_by_consumption': '#138d75',
    # ChatGPT (blue shades)
    'chatgpt': '#2980b9',
    'chatgpt_rankings': '#3498db',
    'chatgpt_personas': '#5dade2',
    # ChatGPT* (purple shades)
    'chatgpt_star': '#8e44ad',
    'chatgpt_star_rankings': '#9b59b6',
    'chatgpt_star_personas': '#a569bd',
    # ChatGPT** (red shades)
    'chatgpt_double_star': '#c0392b',
    'chatgpt_double_star_rankings': '#e74c3c',
    'chatgpt_double_star_personas': '#ec7063',
}

METHOD_DISPLAY = {
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


def load_all_results(output_dir: Path, topic_slug: str, n_reps: int) -> Dict:
    """Load all results for a topic."""
    all_results = {}
    
    for rep_idx in range(n_reps):
        rep_dir = output_dir / "data" / topic_slug / f"rep{rep_idx}"
        if not rep_dir.exists():
            continue
        
        all_results[rep_idx] = {}
        
        for k in K_VALUES:
            for p in P_VALUES:
                sample_dir = rep_dir / f"k{k}_p{p}"
                results_file = sample_dir / "results.json"
                
                if results_file.exists():
                    with open(results_file) as f:
                        all_results[rep_idx][(k, p)] = json.load(f)
    
    return all_results


def collect_epsilons_by_method(all_results: Dict) -> Dict[str, List[float]]:
    """Collect all epsilon values grouped by method."""
    epsilons = {method: [] for method in ALL_METHODS}
    
    for rep_idx, rep_results in all_results.items():
        for (k, p), kp_results in rep_results.items():
            for method, result in kp_results.items():
                if method in epsilons and result.get("epsilon") is not None:
                    epsilons[method].append(result["epsilon"])
    
    return epsilons


def collect_epsilons_by_kp(all_results: Dict) -> Dict[Tuple[int, int], Dict[str, List[float]]]:
    """Collect epsilon values grouped by (K, P) combination."""
    kp_epsilons = {}
    
    for k in K_VALUES:
        for p in P_VALUES:
            kp_epsilons[(k, p)] = {method: [] for method in ALL_METHODS}
    
    for rep_idx, rep_results in all_results.items():
        for (k, p), kp_results in rep_results.items():
            for method, result in kp_results.items():
                if method in kp_epsilons[(k, p)] and result.get("epsilon") is not None:
                    kp_epsilons[(k, p)][method].append(result["epsilon"])
    
    return kp_epsilons


def plot_epsilon_cdf(
    epsilons: Dict[str, List[float]],
    title: str,
    output_path: Path
) -> None:
    """
    Create CDF plot of epsilon values by voting method.
    
    Each method gets its own line showing the cumulative distribution.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot CDF for each method
    for method in ALL_METHODS:
        if epsilons.get(method) and len(epsilons[method]) > 0:
            values = sorted(epsilons[method])
            n = len(values)
            cdf = np.arange(1, n + 1) / n
            
            color = METHOD_COLORS.get(method, '#333333')
            label = METHOD_DISPLAY.get(method, method)
            
            ax.step(values, cdf, where='post', 
                   label=label, color=color, linewidth=2)
    
    ax.set_xlabel('Epsilon', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved CDF to {output_path}")


def plot_epsilon_cdf_grouped(
    epsilons: Dict[str, List[float]],
    title: str,
    output_path: Path
) -> None:
    """
    Create grouped CDF plot showing method categories.
    
    4 subplots: Traditional, ChatGPT, ChatGPT*, ChatGPT**
    Uses standard contrasting colors within each subplot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Standard contrasting colors for within-subplot differentiation
    CONTRAST_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    groups = [
        ('Traditional Methods', TRADITIONAL_METHODS, axes[0, 0]),
        ('ChatGPT Methods', CHATGPT_METHODS, axes[0, 1]),
        ('ChatGPT* Methods', CHATGPT_STAR_METHODS, axes[1, 0]),
        ('ChatGPT** Methods', CHATGPT_DOUBLE_STAR_METHODS, axes[1, 1]),
    ]
    
    for group_name, methods, ax in groups:
        color_idx = 0
        for method in methods:
            if epsilons.get(method) and len(epsilons[method]) > 0:
                values = sorted(epsilons[method])
                n = len(values)
                cdf = np.arange(1, n + 1) / n
                
                # Use contrasting colors within each subplot
                color = CONTRAST_COLORS[color_idx % len(CONTRAST_COLORS)]
                label = METHOD_DISPLAY.get(method, method)
                
                ax.step(values, cdf, where='post', 
                       label=label, color=color, linewidth=2.5)
                color_idx += 1
        
        ax.set_xlabel('Epsilon', fontsize=10)
        ax.set_ylabel('Cumulative Probability', fontsize=10)
        ax.set_title(group_name, fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved grouped CDF to {output_path}")


def plot_epsilon_cdf_grouped_log(
    epsilons: Dict[str, List[float]],
    title: str,
    output_path: Path
) -> None:
    """
    Create grouped CDF plot with log-scale epsilon axis.
    
    4 subplots: Traditional, ChatGPT, ChatGPT*, ChatGPT**
    X-axis is log-scaled from global minimum epsilon to 1.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Standard contrasting colors for within-subplot differentiation
    CONTRAST_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Small epsilon floor for log scale (values below this are treated as zero)
    EPS_FLOOR = 1e-3
    
    # Find global min epsilon across all methods
    all_eps = []
    for method_eps in epsilons.values():
        if method_eps:
            all_eps.extend(method_eps)
    
    if not all_eps:
        logger.warning("No epsilon values for log plot")
        return
    
    # Use floor for negative/zero values
    global_min = min(max(e, EPS_FLOOR) for e in all_eps)
    # Use a small buffer below the minimum and above 1.0
    x_min = global_min * 0.5
    x_max = 1.5  # Buffer so lines at 1.0 aren't cut off
    
    groups = [
        ('Traditional Methods', TRADITIONAL_METHODS, axes[0, 0]),
        ('ChatGPT Methods', CHATGPT_METHODS, axes[0, 1]),
        ('ChatGPT* Methods', CHATGPT_STAR_METHODS, axes[1, 0]),
        ('ChatGPT** Methods', CHATGPT_DOUBLE_STAR_METHODS, axes[1, 1]),
    ]
    
    for group_name, methods, ax in groups:
        color_idx = 0
        for method in methods:
            if epsilons.get(method) and len(epsilons[method]) > 0:
                # For log scale: replace zero/negative values with floor value
                values = sorted([max(e, EPS_FLOOR) for e in epsilons[method]])
                    
                n = len(values)
                cdf = np.arange(1, n + 1) / n
                
                # Use contrasting colors within each subplot
                color = CONTRAST_COLORS[color_idx % len(CONTRAST_COLORS)]
                label = METHOD_DISPLAY.get(method, method)
                
                ax.step(values, cdf, where='post', 
                       label=label, color=color, linewidth=2.5)
                color_idx += 1
        
        ax.set_xscale('log')
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel('Epsilon (log scale)', fontsize=10)
        ax.set_ylabel('Cumulative Probability', fontsize=10)
        ax.set_title(group_name, fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='lower right', fontsize=9)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved log-scale grouped CDF to {output_path}")


def plot_epsilon_cdf_grouped_log_single(
    epsilons: Dict[str, List[float]],
    title: str,
    output_path: Path
) -> None:
    """
    Create grouped CDF plot with log-scale epsilon axis for a single (K, P).
    
    4 subplots: Traditional, ChatGPT, ChatGPT*, ChatGPT**
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Standard contrasting colors for within-subplot differentiation
    CONTRAST_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Small epsilon floor for log scale (values below this are treated as zero)
    EPS_FLOOR = 1e-3
    
    # Find global min epsilon across all methods
    all_eps = []
    for method_eps in epsilons.values():
        if method_eps:
            all_eps.extend(method_eps)
    
    if not all_eps:
        plt.close()
        return
    
    # Use floor for negative/zero values
    global_min = min(max(e, EPS_FLOOR) for e in all_eps)
    x_min = global_min * 0.5
    x_max = 1.5
    
    groups = [
        ('Traditional Methods', TRADITIONAL_METHODS, axes[0, 0]),
        ('ChatGPT Methods', CHATGPT_METHODS, axes[0, 1]),
        ('ChatGPT* Methods', CHATGPT_STAR_METHODS, axes[1, 0]),
        ('ChatGPT** Methods', CHATGPT_DOUBLE_STAR_METHODS, axes[1, 1]),
    ]
    
    for group_name, methods, ax in groups:
        color_idx = 0
        for method in methods:
            if epsilons.get(method) and len(epsilons[method]) > 0:
                # For log scale: replace zero/negative values with floor value
                values = sorted([max(e, EPS_FLOOR) for e in epsilons[method]])
                    
                n = len(values)
                cdf = np.arange(1, n + 1) / n
                
                color = CONTRAST_COLORS[color_idx % len(CONTRAST_COLORS)]
                label = METHOD_DISPLAY.get(method, method)
                
                ax.step(values, cdf, where='post', 
                       label=label, color=color, linewidth=2.5)
                color_idx += 1
        
        ax.set_xscale('log')
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel('Epsilon (log scale)', fontsize=10)
        ax.set_ylabel('Cumulative Probability', fontsize=10)
        ax.set_title(group_name, fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='lower right', fontsize=9)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved log-scale CDF to {output_path}")


def generate_cdf_plots(output_dir: Path, topic_slug: str, n_reps: int) -> None:
    """Generate all CDF visualizations for a topic."""
    logger.info(f"Generating CDF plots for {topic_slug}...")
    
    # Create cdf subfolder
    figures_dir = output_dir / "figures" / topic_slug
    cdf_dir = figures_dir / "cdf"
    cdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    all_results = load_all_results(output_dir, topic_slug, n_reps)
    
    if not all_results:
        logger.warning(f"No results found for {topic_slug}")
        return
    
    # Collect data
    epsilons = collect_epsilons_by_method(all_results)
    kp_epsilons = collect_epsilons_by_kp(all_results)
    
    topic_short = TOPIC_SHORT_NAMES.get(topic_slug, topic_slug[:15])
    
    # 1. Overall CDF (corresponding to bar_overall.png)
    plot_epsilon_cdf(
        epsilons,
        f"{topic_short}: CDF of Epsilon by Method (Overall)",
        cdf_dir / "cdf_overall.png"
    )
    
    # 2. Grouped CDF
    plot_epsilon_cdf_grouped(
        epsilons,
        f"{topic_short}: CDF by Method Category",
        cdf_dir / "cdf_grouped.png"
    )
    
    # 3. Grouped CDF with log scale
    plot_epsilon_cdf_grouped_log(
        epsilons,
        f"{topic_short}: CDF by Method Category (Log Scale)",
        cdf_dir / "cdf_grouped_log.png"
    )
    
    # 4. CDF per (K, P) (corresponding to bar_k*_p*.png)
    for k in K_VALUES:
        for p in P_VALUES:
            if (k, p) in kp_epsilons:
                plot_epsilon_cdf(
                    kp_epsilons[(k, p)],
                    f"{topic_short}: CDF of Epsilon (K={k}, P={p})",
                    cdf_dir / f"cdf_k{k}_p{p}.png"
                )
    
    # 5. Log-scale grouped CDF per (K, P)
    for k in K_VALUES:
        for p in P_VALUES:
            if (k, p) in kp_epsilons:
                plot_epsilon_cdf_grouped_log_single(
                    kp_epsilons[(k, p)],
                    f"{topic_short}: CDF (Log Scale) K={k}, P={p}",
                    cdf_dir / f"cdf_k{k}_p{p}_log.png"
                )
    
    logger.info(f"Completed CDF plots for {topic_slug}")


def main():
    """Main entry point."""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    output_dir = Path(__file__).parent.parent.parent / 'outputs' / 'sampling_experiment'
    topic_slug = 'how-should-we-increase-the-general-publics-trust-i'
    n_reps = 5
    
    generate_cdf_plots(output_dir, topic_slug, n_reps)


if __name__ == '__main__':
    main()
