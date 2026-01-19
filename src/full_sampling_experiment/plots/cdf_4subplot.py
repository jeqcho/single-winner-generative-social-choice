"""
4-subplot CDF plots (Traditional, ChatGPT, ChatGPT*, ChatGPT**).
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List

from .config import (
    FIGURES_DIR, TRADITIONAL_METHODS, CHATGPT_METHODS,
    CHATGPT_STAR_METHODS, CHATGPT_DOUBLE_STAR_METHODS,
    METHOD_DISPLAY, TOPIC_SHORT_NAMES, CONTRAST_COLORS
)
from .data_loader import load_all_results, load_results_by_topic, get_topic_dirs, load_results_for_topics, load_random_epsilons


def plot_cdf_4subplot(epsilons: Dict[str, List[float]], title: str, output_path: Path, 
                      random_epsilons: List[float] = None):
    """Create 4-subplot CDF with Random in all subplots and GPT*** only in GPT** subplot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    groups = [
        ('Traditional Methods', TRADITIONAL_METHODS, axes[0, 0], False),
        ('ChatGPT Methods', CHATGPT_METHODS, axes[0, 1], False),
        ('ChatGPT* Methods', CHATGPT_STAR_METHODS, axes[1, 0], False),
        ('ChatGPT** Methods', CHATGPT_DOUBLE_STAR_METHODS, axes[1, 1], True),  # GPT*** only here
    ]
    
    gpt_triple_star_eps = epsilons.get('chatgpt_triple_star', [])
    
    for group_name, methods, ax, include_gpt_triple_star in groups:
        color_idx = 0
        for method in methods:
            if epsilons.get(method) and len(epsilons[method]) > 0:
                values = sorted(epsilons[method])
                n = len(values)
                cdf = np.arange(1, n + 1) / n
                color = CONTRAST_COLORS[color_idx % len(CONTRAST_COLORS)]
                label = METHOD_DISPLAY.get(method, method)
                ax.step(values, cdf, where='post', label=label, color=color, linewidth=2.5)
                color_idx += 1
        
        # Add GPT*** line (green, dashed) - only in GPT** subplot
        if include_gpt_triple_star and gpt_triple_star_eps and len(gpt_triple_star_eps) > 0:
            values = sorted(gpt_triple_star_eps)
            n = len(values)
            cdf = np.arange(1, n + 1) / n
            ax.step(values, cdf, where='post', label='GPT***',
                    color='#2ca02c', linewidth=2.5, linestyle='--')
        
        # Add Random baseline (solid black) - in all subplots
        if random_epsilons and len(random_epsilons) > 0:
            values = sorted(random_epsilons)
            n = len(values)
            cdf = np.arange(1, n + 1) / n
            ax.step(values, cdf, where='post', label='Random',
                    color='black', linewidth=2, linestyle='-')
        
        ax.set_xlabel('Critical Epsilon (ε*)', fontsize=10)
        ax.set_ylabel('Cumulative Probability', fontsize=10)
        ax.set_title(group_name, fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_cdf_4subplot_plots():
    """Generate 4-subplot CDF for all topics combined and each topic."""
    print("Generating 4-subplot CDF plots...")
    
    output_dir = FIGURES_DIR / 'cdf_4subplot'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    topic_dirs = get_topic_dirs()
    
    # Load random baseline
    random_epsilons = load_random_epsilons()
    
    # All topics combined
    all_results = load_all_results()
    plot_cdf_4subplot(
        all_results,
        'CDF of Critical Epsilon - All Topics Combined',
        output_dir / 'all_topics.png',
        random_epsilons=random_epsilons
    )
    print(f"  Saved: all_topics.png")
    
    # Per topic
    for topic_dir in topic_dirs:
        topic_name = topic_dir.name
        topic_results = load_results_for_topics([topic_dir])
        short_name = TOPIC_SHORT_NAMES.get(topic_name, topic_name[:25])
        plot_cdf_4subplot(
            topic_results,
            f'CDF of Critical Epsilon - {short_name}',
            output_dir / f'{topic_name}.png',
            random_epsilons=random_epsilons
        )
        print(f"  Saved: {topic_name}.png")


if __name__ == "__main__":
    generate_cdf_4subplot_plots()
