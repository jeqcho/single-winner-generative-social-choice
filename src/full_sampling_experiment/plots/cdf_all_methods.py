"""
CDF plot with all methods on one figure.
"""
import matplotlib.pyplot as plt
import numpy as np

from .config import (
    FIGURES_DIR, METHODS_ORDER, METHOD_DISPLAY_LONG,
    COLORS_GROUPED, LINESTYLES
)
from .data_loader import load_all_results, load_random_epsilons


def plot_cdf_all_methods():
    """Generate CDF plot with all methods and random baseline."""
    print("Generating CDF all methods plot...")
    
    all_results = load_all_results()
    random_epsilons = load_random_epsilons()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for method in METHODS_ORDER:
        if method not in all_results or len(all_results[method]) == 0:
            continue
        epsilons = sorted(all_results[method])
        n = len(epsilons)
        cdf = np.arange(1, n + 1) / n
        ax.step(epsilons, cdf, where='post',
                label=METHOD_DISPLAY_LONG.get(method, method),
                color=COLORS_GROUPED.get(method, '#333333'),
                linestyle=LINESTYLES.get(method, '-'),
                linewidth=2)
    
    # Add Random baseline
    if random_epsilons:
        values = sorted(random_epsilons)
        n = len(values)
        cdf = np.arange(1, n + 1) / n
        ax.step(values, cdf, where='post', label='Random',
                color='black', linewidth=2.5, linestyle='-')
    
    ax.set_xlabel('Critical Epsilon (ε*)', fontsize=14)
    ax.set_ylabel('Cumulative Probability', fontsize=14)
    ax.set_title('CDF of Critical Epsilon by Voting Method (All Topics)', fontsize=16)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    output_path = FIGURES_DIR / 'cdf_all_methods_full.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_cdf_all_methods()
