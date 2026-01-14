"""
Line plot visualizations for the sampling experiment.

Creates three types of line plots:
1. By P: x=K, lines=methods, one file per P
2. By K: x=P, lines=methods, one file per K  
3. By Method: x=K, lines=P values, one file per method
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from .config import K_VALUES, P_VALUES, ALL_METHODS

logger = logging.getLogger(__name__)

# Color palette for methods
METHOD_COLORS = {
    'schulze': '#1f77b4',
    'borda': '#ff7f0e',
    'irv': '#2ca02c',
    'plurality': '#d62728',
    'veto_by_consumption': '#9467bd',
    'chatgpt': '#8c564b',
    'chatgpt_rankings': '#e377c2',
    'chatgpt_personas': '#7f7f7f',
    'chatgpt_star': '#bcbd22',
    'chatgpt_star_rankings': '#17becf',
    'chatgpt_star_personas': '#aec7e8',
    'chatgpt_double_star': '#ffbb78',
    'chatgpt_double_star_rankings': '#98df8a',
    'chatgpt_double_star_personas': '#ff9896',
}

# Shorter display names
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

P_COLORS = {10: '#1f77b4', 20: '#ff7f0e', 50: '#2ca02c', 100: '#d62728'}
P_MARKERS = {10: 'o', 20: 's', 50: '^', 100: 'D'}


def load_all_results(output_dir: Path, topic_slug: str, n_reps: int) -> Dict:
    """Load all results for a topic.
    
    Handles both old structure (results.json directly in kX_pY/) and 
    new structure (results.json in kX_pY/sample0/).
    """
    all_results = {}
    
    for rep_idx in range(n_reps):
        rep_dir = output_dir / "data" / topic_slug / f"rep{rep_idx}"
        if not rep_dir.exists():
            continue
        
        all_results[rep_idx] = {}
        
        for k in K_VALUES:
            for p in P_VALUES:
                # Try new structure first (sample0 subfolder)
                results_file = rep_dir / f"k{k}_p{p}" / "sample0" / "results.json"
                if not results_file.exists():
                    # Fall back to old structure
                    results_file = rep_dir / f"k{k}_p{p}" / "results.json"
                
                if results_file.exists():
                    with open(results_file) as f:
                        all_results[rep_idx][(k, p)] = json.load(f)
    
    return all_results


def load_sample_info_and_epsilons(output_dir: Path, topic_slug: str, n_reps: int) -> Dict:
    """
    Load sample_info.json and precomputed_epsilons.json for computing random baseline.
    
    Handles both old structure (results.json directly in kX_pY/) and 
    new structure (results.json in kX_pY/sample0/).
    
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
                # Try new structure first (sample0 subfolder)
                sample_info_file = rep_dir / f"k{k}_p{p}" / "sample0" / "sample_info.json"
                if not sample_info_file.exists():
                    # Fall back to old structure
                    sample_info_file = rep_dir / f"k{k}_p{p}" / "sample_info.json"
                
                if sample_info_file.exists():
                    with open(sample_info_file) as f:
                        sample_info = json.load(f)
                    
                    all_sample_data[rep_idx][(k, p)] = {
                        'alt_sample': sample_info.get('alt_sample', []),
                        'precomputed_epsilons': precomputed_epsilons
                    }
    
    return all_sample_data


def compute_mean_random_epsilon_grid(all_sample_data: Dict) -> Dict[Tuple[int, int], float]:
    """
    Compute mean random epsilon for each (K, P) combination.
    
    Random epsilon = mean of epsilons for all alternatives in the sample.
    
    Returns:
        Dict[(k, p)] -> mean_random_epsilon
    """
    random_by_kp = defaultdict(list)
    
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
                    random_by_kp[(k, p)].append(np.mean(sample_epsilons))
    
    # Compute mean across reps
    mean_random_grid = {}
    for (k, p), random_list in random_by_kp.items():
        if random_list:
            mean_random_grid[(k, p)] = np.mean(random_list)
    
    return mean_random_grid


def compute_mean_epsilon_grid(all_results: Dict) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Compute mean epsilon for each (K, P, method) combination across reps.
    
    Returns:
        Dict[(k, p)] -> Dict[method] -> mean_epsilon
    """
    # Collect epsilons
    eps_by_kp_method = defaultdict(lambda: defaultdict(list))
    
    for rep_idx, rep_data in all_results.items():
        for (k, p), results in rep_data.items():
            for method in ALL_METHODS:
                if method in results and results[method].get('epsilon') is not None:
                    eps_by_kp_method[(k, p)][method].append(results[method]['epsilon'])
    
    # Compute means
    mean_grid = {}
    for (k, p), method_eps in eps_by_kp_method.items():
        mean_grid[(k, p)] = {}
        for method, eps_list in method_eps.items():
            if eps_list:
                mean_grid[(k, p)][method] = np.mean(eps_list)
    
    return mean_grid


def plot_by_p(mean_grid: Dict, random_grid: Dict, topic_short: str, output_dir: Path) -> None:
    """
    Create line plots: x=K, lines=methods, one file per P.
    Includes black 'Random' baseline line.
    """
    folder = output_dir / "lines_by_p"
    folder.mkdir(parents=True, exist_ok=True)
    
    for p in P_VALUES:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method in ALL_METHODS:
            k_vals = []
            eps_vals = []
            
            for k in K_VALUES:
                if (k, p) in mean_grid and method in mean_grid[(k, p)]:
                    k_vals.append(k)
                    eps_vals.append(mean_grid[(k, p)][method])
            
            if k_vals:
                ax.plot(k_vals, eps_vals, 
                       marker='o', 
                       label=METHOD_DISPLAY.get(method, method),
                       color=METHOD_COLORS.get(method, None),
                       linewidth=2,
                       markersize=6)
        
        # Add Random baseline (black line)
        random_k_vals = []
        random_eps_vals = []
        for k in K_VALUES:
            if (k, p) in random_grid:
                random_k_vals.append(k)
                random_eps_vals.append(random_grid[(k, p)])
        
        if random_k_vals:
            ax.plot(random_k_vals, random_eps_vals,
                   marker='o',
                   label='Random',
                   color='black',
                   linewidth=2,
                   markersize=6)
        
        ax.set_xlabel('K (Number of Voters)', fontsize=12)
        ax.set_ylabel('Mean Epsilon', fontsize=12)
        ax.set_title(f'{topic_short}: Epsilon vs K (P={p})', fontsize=14)
        ax.set_xticks(K_VALUES)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(folder / f'p{p}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {folder / f'p{p}.png'}")


def plot_by_k(mean_grid: Dict, random_grid: Dict, topic_short: str, output_dir: Path) -> None:
    """
    Create line plots: x=P, lines=methods, one file per K.
    Includes black 'Random' baseline line.
    """
    folder = output_dir / "lines_by_k"
    folder.mkdir(parents=True, exist_ok=True)
    
    for k in K_VALUES:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method in ALL_METHODS:
            p_vals = []
            eps_vals = []
            
            for p in P_VALUES:
                if (k, p) in mean_grid and method in mean_grid[(k, p)]:
                    p_vals.append(p)
                    eps_vals.append(mean_grid[(k, p)][method])
            
            if p_vals:
                ax.plot(p_vals, eps_vals, 
                       marker='o', 
                       label=METHOD_DISPLAY.get(method, method),
                       color=METHOD_COLORS.get(method, None),
                       linewidth=2,
                       markersize=6)
        
        # Add Random baseline (black line)
        random_p_vals = []
        random_eps_vals = []
        for p in P_VALUES:
            if (k, p) in random_grid:
                random_p_vals.append(p)
                random_eps_vals.append(random_grid[(k, p)])
        
        if random_p_vals:
            ax.plot(random_p_vals, random_eps_vals,
                   marker='o',
                   label='Random',
                   color='black',
                   linewidth=2,
                   markersize=6)
        
        ax.set_xlabel('P (Number of Alternatives)', fontsize=12)
        ax.set_ylabel('Mean Epsilon', fontsize=12)
        ax.set_title(f'{topic_short}: Epsilon vs P (K={k})', fontsize=14)
        ax.set_xticks(P_VALUES)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(folder / f'k{k}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {folder / f'k{k}.png'}")


def plot_by_method(mean_grid: Dict, topic_short: str, output_dir: Path) -> None:
    """
    Create line plots: x=K, lines=P values, one file per method.
    """
    folder = output_dir / "lines_by_method"
    folder.mkdir(parents=True, exist_ok=True)
    
    for method in ALL_METHODS:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        has_data = False
        for p in P_VALUES:
            k_vals = []
            eps_vals = []
            
            for k in K_VALUES:
                if (k, p) in mean_grid and method in mean_grid[(k, p)]:
                    k_vals.append(k)
                    eps_vals.append(mean_grid[(k, p)][method])
            
            if k_vals:
                has_data = True
                ax.plot(k_vals, eps_vals, 
                       marker=P_MARKERS[p], 
                       label=f'P={p}',
                       color=P_COLORS[p],
                       linewidth=2,
                       markersize=8)
        
        if not has_data:
            plt.close()
            continue
        
        ax.set_xlabel('K (Number of Voters)', fontsize=12)
        ax.set_ylabel('Mean Epsilon', fontsize=12)
        ax.set_title(f'{topic_short}: {METHOD_DISPLAY.get(method, method)}', fontsize=14)
        ax.set_xticks(K_VALUES)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(folder / f'{method}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {folder / f'{method}.png'}")


def generate_line_plots(output_dir: Path, topic_slug: str, n_reps: int) -> None:
    """Generate all line plot visualizations for a topic."""
    from .config import TOPIC_SHORT_NAMES
    
    logger.info(f"Generating line plots for {topic_slug}...")
    
    figures_dir = output_dir / "figures" / topic_slug
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    all_results = load_all_results(output_dir, topic_slug, n_reps)
    
    if not all_results:
        logger.warning(f"No results found for {topic_slug}")
        return
    
    # Load sample info and precomputed epsilons for random baseline
    all_sample_data = load_sample_info_and_epsilons(output_dir, topic_slug, n_reps)
    
    # Compute mean epsilon grid
    mean_grid = compute_mean_epsilon_grid(all_results)
    
    # Compute mean random epsilon grid
    random_grid = compute_mean_random_epsilon_grid(all_sample_data)
    
    topic_short = TOPIC_SHORT_NAMES.get(topic_slug, topic_slug[:15])
    
    # Generate all three types of plots
    plot_by_p(mean_grid, random_grid, topic_short, figures_dir)
    plot_by_k(mean_grid, random_grid, topic_short, figures_dir)
    plot_by_method(mean_grid, topic_short, figures_dir)
    
    logger.info(f"Completed line plots for {topic_slug}")


def main():
    """Main entry point."""
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    output_dir = Path(__file__).parent.parent.parent / 'outputs' / 'sampling_experiment'
    topic_slug = 'how-should-we-increase-the-general-publics-trust-i'
    n_reps = 10
    
    generate_line_plots(output_dir, topic_slug, n_reps)


if __name__ == '__main__':
    main()
