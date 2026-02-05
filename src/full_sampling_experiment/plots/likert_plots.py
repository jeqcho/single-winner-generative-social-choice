"""
Likert score histogram plots.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path
from typing import Dict, List

from .config import FIGURES_DIR, TOPIC_SHORT_NAMES
from .data_loader import load_likert_scores


def plot_likert_histograms(topic_scores: Dict[str, List[int]], output_path: Path):
    """Plot histogram for each topic in a 2-column paper-quality figure."""
    n_topics = len(topic_scores)
    ncols = 2
    nrows = (n_topics + ncols - 1) // ncols  # ceiling division

    # Paper-quality matplotlib settings
    with mpl.rc_context({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.family': 'serif',
    }):
        fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.6 * nrows))
        axes = axes.flatten()

        # Use a qualitative colormap with enough distinct colors
        cmap = mpl.colormaps.get_cmap('tab20').resampled(n_topics)

        items = list(topic_scores.items())
        for idx, (topic_name, scores) in enumerate(items):
            ax = axes[idx]
            short_name = TOPIC_SHORT_NAMES.get(topic_name, topic_name[:25])
            scores = np.array(scores)

            bins = np.arange(0.5, 11.5, 1)
            counts, _ = np.histogram(scores, bins=bins)
            normalized = counts / len(scores)

            color = cmap(idx)
            ax.bar(range(1, 11), normalized, color=color, edgecolor='black',
                   linewidth=0.5, alpha=0.85)
            ax.set_xlim(0.5, 10.5)
            ax.set_ylim(0, max(normalized) * 1.2)
            ax.set_xticks(range(1, 11))
            ax.set_title(short_name, fontweight='bold', loc='left')
            ax.grid(True, alpha=0.25, axis='y', linewidth=0.5)

            mean_score = np.mean(scores)
            std_score = np.std(scores)
            ax.axvline(mean_score, color='red', linestyle='--', linewidth=1.5)

            # Annotate with mean ± std
            ax.text(
                0.97, 0.92,
                f'$\\mu={mean_score:.1f},\\ \\sigma={std_score:.1f}$',
                transform=ax.transAxes, fontsize=10,
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.8),
            )

            # Only label outer axes
            if idx >= (nrows - 1) * ncols:
                ax.set_xlabel("Likert Score")
            if idx % ncols == 0:
                ax.set_ylabel("Frequency")

        # Hide unused subplot cells
        for idx in range(n_topics, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(
            "Distribution of Likert Agreement Scores by Topic",
            fontsize=15, fontweight='bold', y=1.005,
        )
        plt.tight_layout()

        # Save both PNG and PDF
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path.name} and {pdf_path.name}")


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
