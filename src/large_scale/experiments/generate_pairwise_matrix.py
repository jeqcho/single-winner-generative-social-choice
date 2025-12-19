"""
Generate pairwise comparison matrix for all models.

Creates a matrix where rows and columns are model names,
and cell values are the average Kendall-tau-b (or tau distance) between them.

Outputs:
- Colored heatmaps (.png)
- LaTeX tables (.tex)
- Markdown tables (.md)
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# Constants
PROJECT_ROOT = Path("/home/ec2-user/single-winner-generative-social-choice")
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "check-models"
RANKINGS_FILE = OUTPUT_BASE / "rankings.json"
OUTPUT_DIR = OUTPUT_BASE / "all"


def load_rankings() -> Dict[str, List[List[int]]]:
    """Load rankings from JSON file."""
    with open(RANKINGS_FILE, 'r') as f:
        return json.load(f)


def compute_pairwise_metrics(all_rankings: Dict[str, List[List[int]]]) -> tuple:
    """
    Compute pairwise Kendall tau-b and tau distance for all model pairs.
    
    Returns:
        Tuple of (tau_b_matrix, tau_distance_matrix, model_names)
    """
    model_names = list(all_rankings.keys())
    n_models = len(model_names)
    
    tau_b_matrix = np.zeros((n_models, n_models))
    tau_distance_matrix = np.zeros((n_models, n_models))
    
    for i, model_i in enumerate(model_names):
        for j, model_j in enumerate(model_names):
            rankings_i = all_rankings[model_i]
            rankings_j = all_rankings[model_j]
            
            # Compute average tau-b across all personas
            tau_b_values = []
            for rank_i, rank_j in zip(rankings_i, rankings_j):
                tau_b, _ = kendalltau(rank_i, rank_j)
                tau_b_values.append(tau_b)
            
            avg_tau_b = np.mean(tau_b_values)
            avg_tau_distance = (1 - avg_tau_b) / 2
            
            tau_b_matrix[i, j] = avg_tau_b
            tau_distance_matrix[i, j] = avg_tau_distance
    
    return tau_b_matrix, tau_distance_matrix, model_names


def create_heatmap(matrix: np.ndarray, model_names: List[str], output_path: Path, 
                   metric_name: str, cmap: str = 'RdYlGn'):
    """Create a colored heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # For tau-b, higher is better (more similar), so use RdYlGn
    # For tau distance, lower is better, so reverse the colormap
    if 'Distance' in metric_name:
        cmap = 'RdYlGn_r'  # Reversed - green for low values
    
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric_name, rotation=-90, va="bottom", fontsize=12)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(model_names)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([m.replace('_', '\n') for m in model_names], fontsize=10)
    ax.set_yticklabels(model_names, fontsize=10)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add cell values
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            text_color = 'white' if abs(matrix[i, j] - 0.5) > 0.3 else 'black'
            ax.text(j, i, f'{matrix[i, j]:.3f}',
                   ha="center", va="center", color=text_color, fontsize=11, fontweight='bold')
    
    ax.set_title(f'Pairwise {metric_name} Between Models', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model (Column)', fontsize=12)
    ax.set_ylabel('Model (Row)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap to {output_path}")


def create_latex_table(matrix: np.ndarray, model_names: List[str], output_path: Path, metric_name: str):
    """Create a LaTeX table."""
    n = len(model_names)
    
    # Escape underscores for LaTeX
    escaped_names = [m.replace('_', '-') for m in model_names]
    
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{Pairwise {metric_name} Between Models}}",
        r"\small",
        r"\begin{tabular}{l" + "c" * n + "}",
        r"\toprule",
        " & " + " & ".join(escaped_names) + r" \\",
        r"\midrule",
    ]
    
    for i, name in enumerate(escaped_names):
        row_values = [f"{matrix[i, j]:.4f}" for j in range(n)]
        lines.append(f"{name} & " + " & ".join(row_values) + r" \\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved LaTeX table to {output_path}")


def create_markdown_table(matrix: np.ndarray, model_names: List[str], output_path: Path, metric_name: str):
    """Create a Markdown table."""
    n = len(model_names)
    
    lines = [
        f"# Pairwise {metric_name} Between Models",
        "",
        "| Model | " + " | ".join(model_names) + " |",
        "|" + "---|" * (n + 1),
    ]
    
    for i, name in enumerate(model_names):
        row_values = [f"{matrix[i, j]:.4f}" for j in range(n)]
        lines.append(f"| {name} | " + " | ".join(row_values) + " |")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved Markdown table to {output_path}")


def main():
    print("Loading rankings...")
    all_rankings = load_rankings()
    
    print("Computing pairwise metrics...")
    tau_b_matrix, tau_distance_matrix, model_names = compute_pairwise_metrics(all_rankings)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating outputs in {OUTPUT_DIR}...")
    
    # Kendall Tau-b outputs
    print("\n--- Kendall Tau-b ---")
    create_heatmap(tau_b_matrix, model_names, OUTPUT_DIR / "kendall_tau_b_heatmap.png", "Kendall Tau-b")
    create_latex_table(tau_b_matrix, model_names, OUTPUT_DIR / "kendall_tau_b_table.tex", "Kendall Tau-b")
    create_markdown_table(tau_b_matrix, model_names, OUTPUT_DIR / "kendall_tau_b_table.md", "Kendall Tau-b")
    
    # Kendall Tau Distance outputs
    print("\n--- Kendall-Tau Distance ---")
    create_heatmap(tau_distance_matrix, model_names, OUTPUT_DIR / "kendall_tau_distance_heatmap.png", "Kendall-Tau Distance")
    create_latex_table(tau_distance_matrix, model_names, OUTPUT_DIR / "kendall_tau_distance_table.tex", "Kendall-Tau Distance")
    create_markdown_table(tau_distance_matrix, model_names, OUTPUT_DIR / "kendall_tau_distance_table.md", "Kendall-Tau Distance")
    
    print(f"\n{'='*60}")
    print("All pairwise matrix outputs generated!")
    print(f"{'='*60}")
    
    # Print the matrices for quick reference
    print("\n--- Kendall Tau-b Matrix ---")
    print(f"Models: {model_names}")
    print(tau_b_matrix.round(4))
    
    print("\n--- Kendall-Tau Distance Matrix ---")
    print(tau_distance_matrix.round(4))


if __name__ == "__main__":
    main()



