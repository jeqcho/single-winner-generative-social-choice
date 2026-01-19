"""
Individual CDF plots for each topic (all methods on one plot).
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List

from .config import (
    FIGURES_DIR, METHODS_ORDER, METHOD_DISPLAY_LONG,
    COLORS_GROUPED, LINESTYLES, TOPIC_SHORT_NAMES
)
from .data_loader import get_topic_dirs, load_results_for_topics


def plot_cdf_single_topic(epsilons: Dict[str, List[float]], title: str, output_path: Path):
    """Create CDF plot for a single topic."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for method in METHODS_ORDER:
        if method not in epsilons or len(epsilons[method]) == 0:
            continue
        values = sorted(epsilons[method])
        n = len(values)
        cdf = np.arange(1, n + 1) / n
        ax.step(values, cdf, where='post',
                label=METHOD_DISPLAY_LONG.get(method, method),
                color=COLORS_GROUPED.get(method, '#333333'),
                linestyle=LINESTYLES.get(method, '-'),
                linewidth=2)
    
    ax.set_xlabel('Critical Epsilon (ε*)', fontsize=14)
    ax.set_ylabel('Cumulative Probability', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_cdf_by_topic_plots():
    """Generate individual CDF plots for each topic."""
    print("Generating per-topic CDF plots...")
    
    output_dir = FIGURES_DIR / 'by_topic_cdf'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for topic_dir in get_topic_dirs():
        topic_name = topic_dir.name
        topic_results = load_results_for_topics([topic_dir])
        short_name = TOPIC_SHORT_NAMES.get(topic_name, topic_name[:25])
        
        plot_cdf_single_topic(
            topic_results,
            f'CDF of Critical Epsilon - {short_name}',
            output_dir / f'{topic_name}.png'
        )
        print(f"  Saved: {topic_name}.png")


if __name__ == "__main__":
    generate_cdf_by_topic_plots()
