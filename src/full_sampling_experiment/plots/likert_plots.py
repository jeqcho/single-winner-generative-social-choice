"""
Likert score histogram plots.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List

from .config import FIGURES_DIR, TOPIC_SHORT_NAMES
from .data_loader import load_likert_scores


def plot_likert_histograms(topic_scores: Dict[str, List[int]], output_path: Path):
    """Plot histogram for each topic in a single figure."""
    n_topics = len(topic_scores)
    
    fig, axes = plt.subplots(n_topics, 1, figsize=(12, 2.5 * n_topics))
    
    if n_topics == 1:
        axes = [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_topics))
    
    for ax, (topic_name, scores), color in zip(axes, topic_scores.items(), colors):
        short_name = TOPIC_SHORT_NAMES.get(topic_name, topic_name[:25])
        scores = np.array(scores)
        
        bins = np.arange(0.5, 11.5, 1)
        counts, _ = np.histogram(scores, bins=bins)
        normalized = counts / len(scores)
        
        ax.bar(range(1, 11), normalized, color=color, edgecolor='black', alpha=0.8)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(0, max(normalized) * 1.15)
        ax.set_xticks(range(1, 11))
        ax.set_title(short_name, fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, axis='y')
        
        mean_score = np.mean(scores)
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_score:.2f}')
        ax.legend(loc='upper right', fontsize=9)
    
    axes[-1].set_xlabel("Likert Score (1=Strongly Disagree, 10=Strongly Agree)", fontsize=12)
    
    plt.suptitle("Distribution of Likert Agreement Scores by Topic\n"
                 "(100 voters × 100 statements per topic)",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_likert_combined(topic_scores: Dict[str, List[int]], output_path: Path):
    """Plot all topics overlaid on one histogram."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(topic_scores)))
    bins = np.arange(0.5, 11.5, 1)
    
    for (topic_name, scores), color in zip(topic_scores.items(), colors):
        short_name = TOPIC_SHORT_NAMES.get(topic_name, topic_name[:20])
        scores = np.array(scores)
        counts, _ = np.histogram(scores, bins=bins)
        normalized = counts / len(scores)
        ax.plot(range(1, 11), normalized, marker='o', linewidth=2, markersize=6,
                color=color, label=short_name, alpha=0.8)
    
    ax.set_xlabel("Likert Score (1=Strongly Disagree, 10=Strongly Agree)", fontsize=12)
    ax.set_ylabel("Normalized Frequency", fontsize=12)
    ax.set_title("Likert Agreement Score Distribution by Topic", fontsize=14, fontweight='bold')
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_likert_summary(topic_scores: Dict[str, List[int]], output_path: Path):
    """Create summary statistics table for Likert scores."""
    data = []
    for topic_name, scores in topic_scores.items():
        short_name = TOPIC_SHORT_NAMES.get(topic_name, topic_name[:25])
        scores = np.array(scores)
        data.append([
            short_name,
            len(scores),
            f"{np.mean(scores):.2f}",
            f"{np.std(scores):.2f}",
            f"{np.median(scores):.1f}",
            f"{100 * np.mean(scores >= 7):.1f}%",
            f"{100 * np.mean(scores <= 4):.1f}%",
        ])
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    
    columns = ["Topic", "N", "Mean", "Std", "Median", "% Agree (≥7)", "% Disagree (≤4)"]
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    plt.title("Likert Score Summary Statistics by Topic", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_likert_plots():
    """Generate all Likert visualizations."""
    print("Generating Likert plots...")
    
    topic_scores = load_likert_scores()
    
    if not topic_scores:
        print("  No Likert data found!")
        return
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    plot_likert_histograms(topic_scores, FIGURES_DIR / 'likert_histograms.png')
    print(f"  Saved: likert_histograms.png")
    
    plot_likert_combined(topic_scores, FIGURES_DIR / 'likert_combined.png')
    print(f"  Saved: likert_combined.png")
    
    plot_likert_summary(topic_scores, FIGURES_DIR / 'likert_summary.png')
    print(f"  Saved: likert_summary.png")


if __name__ == "__main__":
    generate_likert_plots()
