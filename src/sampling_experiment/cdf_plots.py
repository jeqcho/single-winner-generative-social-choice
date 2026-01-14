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
    """Load all results for a topic (using sample0 only)."""
    all_results = {}
    
    for rep_idx in range(n_reps):
        rep_dir = output_dir / "data" / topic_slug / f"rep{rep_idx}"
        if not rep_dir.exists():
            continue
        
        all_results[rep_idx] = {}
        
        for k in K_VALUES:
            for p in P_VALUES:
                # Use sample0 subfolder
                sample_dir = rep_dir / f"k{k}_p{p}" / "sample0"
                results_file = sample_dir / "results.json"
                
                if results_file.exists():
                    with open(results_file) as f:
                        all_results[rep_idx][(k, p)] = json.load(f)
    
    return all_results


def load_sample_info_and_epsilons(output_dir: Path, topic_slug: str, n_reps: int) -> Dict:
    """
    Load sample_info.json and precomputed_epsilons.json for each rep/(k,p) combination.
    
    Returns:
        Dict[rep_idx] -> Dict[(k, p)] -> {'alt_sample': [...], 'precomputed_epsilons': {...}}
    """
    all_sample_data = {}
    
    for rep_idx in range(n_reps):
        rep_dir = output_dir / "data" / topic_slug / f"rep{rep_idx}"
        if not rep_dir.exists():
            continue
        
        # Load precomputed epsilons for this rep
        precomputed_file = rep_dir / "precomputed_epsilons.json"
        if not precomputed_file.exists():
            continue
        
        with open(precomputed_file) as f:
            precomputed_epsilons = json.load(f)
        
        all_sample_data[rep_idx] = {}
        
        for k in K_VALUES:
            for p in P_VALUES:
                sample_dir = rep_dir / f"k{k}_p{p}" / "sample0"
                sample_info_file = sample_dir / "sample_info.json"
                
                if sample_info_file.exists():
                    with open(sample_info_file) as f:
                        sample_info = json.load(f)
                    
                    all_sample_data[rep_idx][(k, p)] = {
                        'alt_sample': sample_info.get('alt_sample', []),
                        'precomputed_epsilons': precomputed_epsilons
                    }
    
    return all_sample_data


def compute_random_epsilons(all_sample_data: Dict) -> Dict[str, List[float]]:
    """
    Compute random baseline epsilon for each sample.
    
    Random epsilon = mean of epsilons for all alternatives in the sample.
    
    Returns:
        Dict with 'random' key containing list of mean epsilons across all samples.
    """
    random_epsilons = []
    
    for rep_idx, rep_data in all_sample_data.items():
        for (k, p), sample_data in rep_data.items():
            alt_sample = sample_data['alt_sample']
            precomputed = sample_data['precomputed_epsilons']
            
            if alt_sample and precomputed:
                # Compute mean epsilon of alternatives in the sample
                sample_epsilons = []
                for alt_id in alt_sample:
                    eps = precomputed.get(str(alt_id))
                    if eps is not None:
                        sample_epsilons.append(eps)
                
                if sample_epsilons:
                    random_epsilons.append(np.mean(sample_epsilons))
    
    return {'random': random_epsilons}


def compute_random_epsilons_by_kp(all_sample_data: Dict) -> Dict[Tuple[int, int], List[float]]:
    """
    Compute random baseline epsilon for each (K, P) combination.
    
    Returns:
        Dict[(k, p)] -> list of random epsilons for that (k, p) across all reps.
    """
    kp_random = {(k, p): [] for k in K_VALUES for p in P_VALUES}
    
    for rep_idx, rep_data in all_sample_data.items():
        for (k, p), sample_data in rep_data.items():
            alt_sample = sample_data['alt_sample']
            precomputed = sample_data['precomputed_epsilons']
            
            if alt_sample and precomputed:
                sample_epsilons = []
                for alt_id in alt_sample:
                    eps = precomputed.get(str(alt_id))
                    if eps is not None:
                        sample_epsilons.append(eps)
                
                if sample_epsilons:
                    kp_random[(k, p)].append(np.mean(sample_epsilons))
    
    return kp_random


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
    random_epsilons: List[float],
    title: str,
    output_path: Path,
    y_min: float = 0,
    x_max: float = 1.0
) -> None:
    """
    Create CDF plot of epsilon values by voting method.
    
    Uses 4 subplots (Traditional, ChatGPT, ChatGPT*, ChatGPT**) like grouped plot.
    Each method gets its own line showing the cumulative distribution.
    Includes black 'Random' baseline line in each subplot.
    
    Args:
        y_min: Minimum value for y-axis (0 or 0.5)
        x_max: Maximum value for x-axis (1.0 or 0.2)
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
        
        # Add Random baseline (black line)
        if random_epsilons and len(random_epsilons) > 0:
            values = sorted(random_epsilons)
            n = len(values)
            cdf = np.arange(1, n + 1) / n
            ax.step(values, cdf, where='post', 
                   label='Random', color='black', linewidth=2.5)
        
        ax.set_xlabel('Epsilon', fontsize=10)
        ax.set_ylabel('Cumulative Probability', fontsize=10)
        ax.set_title(group_name, fontsize=12)
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min, 1.05)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved CDF to {output_path}")


def plot_epsilon_cdf_grouped(
    epsilons: Dict[str, List[float]],
    random_epsilons: List[float],
    title: str,
    output_path: Path,
    y_min: float = 0,
    x_max: float = 1.0
) -> None:
    """
    Create grouped CDF plot showing method categories.
    
    4 subplots: Traditional, ChatGPT, ChatGPT*, ChatGPT**
    Uses standard contrasting colors within each subplot.
    Includes black 'Random' baseline line in each subplot.
    
    Args:
        y_min: Minimum value for y-axis (0 or 0.5)
        x_max: Maximum value for x-axis (1.0 or 0.2)
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
        
        # Add Random baseline (black line)
        if random_epsilons and len(random_epsilons) > 0:
            values = sorted(random_epsilons)
            n = len(values)
            cdf = np.arange(1, n + 1) / n
            ax.step(values, cdf, where='post', 
                   label='Random', color='black', linewidth=2.5)
        
        ax.set_xlabel('Epsilon', fontsize=10)
        ax.set_ylabel('Cumulative Probability', fontsize=10)
        ax.set_title(group_name, fontsize=12)
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min, 1.05)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved grouped CDF to {output_path}")


def generate_cdf_plots(output_dir: Path, topic_slug: str, n_reps: int) -> None:
    """Generate all CDF visualizations for a topic.
    
    Generates four versions of each plot:
    - Standard: x 0-1, y 0-1
    - Y-zoomed: x 0-1, y 0.5-1 (suffix _y05)
    - X-zoomed: x 0-0.2, y 0-1 (suffix _x02)
    - Both zoomed: x 0-0.2, y 0.5-1 (suffix _x02_y05)
    """
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
    
    # Load sample info and precomputed epsilons for random baseline
    all_sample_data = load_sample_info_and_epsilons(output_dir, topic_slug, n_reps)
    
    # Collect data
    epsilons = collect_epsilons_by_method(all_results)
    kp_epsilons = collect_epsilons_by_kp(all_results)
    
    # Compute random epsilons (overall and per-kp)
    random_eps_data = compute_random_epsilons(all_sample_data)
    random_epsilons = random_eps_data.get('random', [])
    kp_random_epsilons = compute_random_epsilons_by_kp(all_sample_data)
    
    topic_short = TOPIC_SHORT_NAMES.get(topic_slug, topic_slug[:15])
    
    # Generate plots with all axis range combinations
    # (x_max, x_suffix), (y_min, y_suffix)
    axis_configs = [
        (1.0, '', 0, ''),           # Standard: x 0-1, y 0-1
        (1.0, '', 0.5, '_y05'),     # Y-zoomed: x 0-1, y 0.5-1
        (0.2, '_x02', 0, ''),       # X-zoomed: x 0-0.2, y 0-1
        (0.2, '_x02', 0.5, '_y05'), # Both zoomed: x 0-0.2, y 0.5-1
    ]
    
    for x_max, x_suffix, y_min, y_suffix in axis_configs:
        suffix = f"{x_suffix}{y_suffix}"
        
        # 1. Overall CDF (corresponding to bar_overall.png) - now with 4 subplots
        plot_epsilon_cdf(
            epsilons,
            random_epsilons,
            f"{topic_short}: CDF of Epsilon by Method (Overall)",
            cdf_dir / f"cdf_overall{suffix}.png",
            y_min=y_min,
            x_max=x_max
        )
        
        # 2. Grouped CDF
        plot_epsilon_cdf_grouped(
            epsilons,
            random_epsilons,
            f"{topic_short}: CDF by Method Category",
            cdf_dir / f"cdf_grouped{suffix}.png",
            y_min=y_min,
            x_max=x_max
        )
        
        # 3. CDF per (K, P) (corresponding to bar_k*_p*.png) - now with 4 subplots
        for k in K_VALUES:
            for p in P_VALUES:
                if (k, p) in kp_epsilons:
                    kp_random = kp_random_epsilons.get((k, p), [])
                    plot_epsilon_cdf(
                        kp_epsilons[(k, p)],
                        kp_random,
                        f"{topic_short}: CDF of Epsilon (K={k}, P={p})",
                        cdf_dir / f"cdf_k{k}_p{p}{suffix}.png",
                        y_min=y_min,
                        x_max=x_max
                    )
    
    logger.info(f"Completed CDF plots for {topic_slug}")


def main():
    """Main entry point."""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    output_dir = Path(__file__).parent.parent.parent / 'outputs' / 'sampling_experiment'
    topic_slug = 'how-should-we-increase-the-general-publics-trust-i'
    n_reps = 10
    
    generate_cdf_plots(output_dir, topic_slug, n_reps)


if __name__ == '__main__':
    main()
