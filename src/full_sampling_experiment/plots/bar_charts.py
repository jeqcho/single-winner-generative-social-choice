"""
Bar charts of mean epsilon.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List

from .config import (
    FIGURES_DIR, METHODS_ORDER_SHORT, METHOD_DISPLAY,
    COLORS_GROUPED, TOPIC_SHORT_NAMES
)
from .data_loader import load_all_results, get_topic_dirs, load_results_for_topics


def plot_bar_mean_epsilon(results: Dict[str, List[float]], title: str, output_path: Path):
    """Create bar chart of mean epsilon by method."""
    means = []
    stds = []
    labels = []
    colors = []
    
    for method in METHODS_ORDER_SHORT:
        if method not in results or len(results[method]) == 0:
            continue
        epsilons = results[method]
        means.append(np.mean(epsilons))
        stds.append(np.std(epsilons))
        labels.append(METHOD_DISPLAY.get(method, method))
        colors.append(COLORS_GROUPED.get(method, '#333333'))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
    ax.set_ylabel('Mean Critical Epsilon (ε*)', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_bar_charts():
    """Generate bar charts for all topics combined and each topic."""
    print("Generating bar charts...")
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # All topics combined
    all_results = load_all_results()
    plot_bar_mean_epsilon(
        all_results,
        'Mean Critical Epsilon by Voting Method',
        FIGURES_DIR / 'bar_mean_epsilon.png'
    )
    print(f"  Saved: bar_mean_epsilon.png")
    
    # Per topic
    output_dir = FIGURES_DIR / 'bar_by_topic'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for topic_dir in get_topic_dirs():
        topic_name = topic_dir.name
        topic_results = load_results_for_topics([topic_dir])
        short_name = TOPIC_SHORT_NAMES.get(topic_name, topic_name[:25])
        
        plot_bar_mean_epsilon(
            topic_results,
            f'Mean Critical Epsilon - {short_name}',
            output_dir / f'{topic_name}.png'
        )
        print(f"  Saved: {topic_name}.png")


if __name__ == "__main__":
    generate_bar_charts()
