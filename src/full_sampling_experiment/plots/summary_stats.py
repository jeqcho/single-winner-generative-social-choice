"""
Summary statistics table plot.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List

from .config import FIGURES_DIR, METHODS_ORDER_SHORT, METHOD_DISPLAY
from .data_loader import load_all_results


def plot_summary_stats(all_results: Dict[str, List[float]], output_path: Path):
    """Create a summary statistics table as a figure."""
    data = []
    for method in METHODS_ORDER_SHORT:
        if method not in all_results or len(all_results[method]) == 0:
            continue
        epsilons = np.array(all_results[method])
        data.append([
            METHOD_DISPLAY.get(method, method),
            len(epsilons),
            f"{np.mean(epsilons):.4f}",
            f"{np.std(epsilons):.4f}",
            f"{np.median(epsilons):.4f}",
            f"{np.min(epsilons):.4f}",
            f"{np.max(epsilons):.4f}",
            f"{100 * np.mean(epsilons == 0):.1f}%"
        ])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    columns = ["Method", "N", "Mean", "Std", "Median", "Min", "Max", "% ε=0"]
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    plt.title("Summary Statistics: Critical Epsilon by Method", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_summary_stats():
    """Generate summary statistics table."""
    print("Generating summary stats...")
    
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = load_all_results()
    plot_summary_stats(all_results, FIGURES_DIR / 'summary_stats.png')
    print(f"  Saved: summary_stats.png")


if __name__ == "__main__":
    generate_summary_stats()
