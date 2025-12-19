"""
Generate model comparison outputs for each model as reference.

Uses existing rankings data from outputs/check-models/rankings.json
and generates charts, CSV, and LaTeX files for each model as reference.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

# Constants
PROJECT_ROOT = Path("/home/ec2-user/single-winner-generative-social-choice")
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "check-models"
RANKINGS_FILE = OUTPUT_BASE / "rankings.json"


def load_rankings() -> Dict[str, List[List[int]]]:
    """Load rankings from JSON file."""
    with open(RANKINGS_FILE, 'r') as f:
        return json.load(f)


def compute_metrics(
    model_rankings: List[List[int]],
    reference_rankings: List[List[int]]
) -> Tuple[List[float], List[float]]:
    """
    Compute Kendall tau-b and tau distance for each persona.
    
    Returns:
        Tuple of (tau_b_values, tau_distance_values) for each persona
    """
    tau_b_values = []
    tau_distance_values = []
    
    for model_ranking, ref_ranking in zip(model_rankings, reference_rankings):
        tau_b, _ = kendalltau(model_ranking, ref_ranking)
        tau_distance = (1 - tau_b) / 2  # Normalized to [0, 1]
        
        tau_b_values.append(tau_b)
        tau_distance_values.append(tau_distance)
    
    return tau_b_values, tau_distance_values


def write_results_csv(results: Dict[str, Dict], output_path: Path, metric_name: str):
    """Write main results CSV with mean and std."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", f"Mean {metric_name}", f"Std {metric_name}"])
        
        for label, data in results.items():
            writer.writerow([label, f"{data['mean']:.6f}", f"{data['std']:.6f}"])
    
    print(f"  Saved {metric_name} results to {output_path}")


def write_variance_csv(results: Dict[str, Dict], output_path: Path, metric_name: str):
    """Write per-persona values CSV."""
    first_key = list(results.keys())[0]
    num_personas = len(results[first_key]['values'])
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["Model"] + [f"Persona_{i}" for i in range(num_personas)]
        writer.writerow(header)
        
        for label, data in results.items():
            row = [label] + [f"{v:.6f}" for v in data['values']]
            writer.writerow(row)
    
    print(f"  Saved {metric_name} per-persona values to {output_path}")


def write_latex_table(results: Dict[str, Dict], output_path: Path, metric_name: str, reference_model: str):
    """Write LaTeX table with mean ± std format."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        f"\\caption{{{metric_name} by Model (relative to {reference_model})}}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        f"Model & {metric_name} \\\\",
        r"\midrule",
    ]
    
    for label, data in results.items():
        mean = data['mean']
        std = data['std']
        lines.append(f"{label.replace('_', '-')} & ${mean:.4f} \\pm {std:.4f}$ \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  Saved {metric_name} LaTeX table to {output_path}")


def create_bar_chart(results: Dict[str, Dict], output_path: Path, metric_name: str, reference_model: str):
    """Create bar chart with error bars."""
    labels = list(results.keys())
    means = [results[label]['mean'] for label in labels]
    stds = [results[label]['std'] for label in labels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color='steelblue', edgecolor='black')
    
    # Highlight reference model (should be 0)
    ref_idx = labels.index(reference_model) if reference_model in labels else -1
    if ref_idx >= 0:
        bars[ref_idx].set_color('forestgreen')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name} by Model (relative to {reference_model})', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([label.replace('_', '\n') for label in labels], fontsize=10)
    
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(f'{mean:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.01),
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  Saved {metric_name} bar chart to {output_path}")


def generate_outputs_for_reference(
    all_rankings: Dict[str, List[List[int]]],
    reference_model: str,
    output_dir: Path
):
    """Generate all outputs for a specific reference model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reference_rankings = all_rankings[reference_model]
    
    tau_b_results: Dict[str, Dict] = {}
    tau_distance_results: Dict[str, Dict] = {}
    
    for model_label, model_rankings in all_rankings.items():
        tau_b_values, tau_distance_values = compute_metrics(model_rankings, reference_rankings)
        
        tau_b_results[model_label] = {
            'values': tau_b_values,
            'mean': np.mean(tau_b_values),
            'std': np.std(tau_b_values)
        }
        
        tau_distance_results[model_label] = {
            'values': tau_distance_values,
            'mean': np.mean(tau_distance_values),
            'std': np.std(tau_distance_values)
        }
    
    # Generate Kendall-tau distance outputs
    write_results_csv(tau_distance_results, output_dir / "kendall_tau_results.csv", "Kendall-Tau Distance")
    write_variance_csv(tau_distance_results, output_dir / "kendall_tau_variance.csv", "Kendall-Tau Distance")
    write_latex_table(tau_distance_results, output_dir / "kendall_tau_table.tex", "Kendall-Tau Distance", reference_model)
    create_bar_chart(tau_distance_results, output_dir / "kendall_tau_chart.png", "Kendall-Tau Distance", reference_model)
    
    # Generate Kendall tau-b outputs
    write_results_csv(tau_b_results, output_dir / "kendall_tau_b_results.csv", "Kendall Tau-b")
    write_variance_csv(tau_b_results, output_dir / "kendall_tau_b_variance.csv", "Kendall Tau-b")
    write_latex_table(tau_b_results, output_dir / "kendall_tau_b_table.tex", "Kendall Tau-b", reference_model)
    create_bar_chart(tau_b_results, output_dir / "kendall_tau_b_chart.png", "Kendall Tau-b", reference_model)


def main():
    print("Loading rankings...")
    all_rankings = load_rankings()
    
    models = list(all_rankings.keys())
    print(f"Found {len(models)} models: {models}")
    
    for reference_model in models:
        # Create directory name from model label
        dir_name = reference_model.replace(".", "-")  # gpt-5.2-none -> gpt-5-2-none
        output_dir = OUTPUT_BASE / dir_name
        
        print(f"\n{'='*60}")
        print(f"Generating outputs with reference: {reference_model}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")
        
        generate_outputs_for_reference(all_rankings, reference_model, output_dir)
    
    print(f"\n{'='*60}")
    print("All outputs generated successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()



