#!/usr/bin/env python3
"""
Plot scatter plots comparing original epsilon (m=100) vs new epsilon (m=101).

Creates scatter plots for uniform and conservative voters across all 6 topics.

Usage:
    uv run python -m src.sample_alt_voters.plot_epsilon_comparison
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "outputs" / "sample_alt_voters" / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "slides" / "epsilon_comparison"

TOPICS = ["abortion", "electoral", "environment", "healthcare", "policing", "trust"]

VOTER_CONFIGS = {
    "uniform": {
        "path_template": "{topic}/uniform/persona_no_context",
        "label": "Uniform Voters",
    },
    "conservative": {
        "path_template": "{topic}/clustered/conservative_traditional/persona_no_context",
        "label": "Conservative Voters",
    },
}

METHODS = {
    "chatgpt_double_star": {"label": "GPT**", "marker": "o", "color": "#1f77b4"},
    "chatgpt_double_star_rankings": {"label": "GPT** Rankings", "marker": "s", "color": "#ff7f0e"},
    "chatgpt_double_star_personas": {"label": "GPT** Personas", "marker": "^", "color": "#2ca02c"},
    "chatgpt_triple_star": {"label": "GPT***", "marker": "D", "color": "#d62728"},
    "random_insertion": {"label": "Random Insertion", "marker": "x", "color": "#9467bd"},
}

N_REPS = 10
N_MINI_REPS = 4


def save_figure(output_path: Path, dpi: int = 150) -> None:
    """Save figure in PNG, SVG, and PDF formats.
    
    Args:
        output_path: Path to save the PNG file (SVG/PDF will use same name with different extensions)
        dpi: Resolution for PNG output
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    svg_path = output_path.with_suffix('.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"  Saved: {output_path}, {svg_path}, {pdf_path}")


def load_epsilon_pairs(
    topic: str, voter_type: str
) -> Dict[str, List[Tuple[float, float]]]:
    """Load epsilon_original and epsilon pairs for each method."""
    config = VOTER_CONFIGS[voter_type]
    base_path = DATA_DIR / config["path_template"].format(topic=topic)
    
    method_pairs = {method: [] for method in METHODS}
    
    for rep_idx in range(N_REPS):
        rep_dir = base_path / f"rep{rep_idx}"
        if not rep_dir.exists():
            continue
        
        for mini_rep_idx in range(N_MINI_REPS):
            results_path = rep_dir / f"mini_rep{mini_rep_idx}" / "results.json"
            if not results_path.exists():
                continue
            
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            for method in METHODS:
                if method not in results.get("results", {}):
                    continue
                
                method_data = results["results"][method]
                epsilon_new = method_data.get("epsilon")
                epsilon_original = method_data.get("epsilon_original")
                
                if epsilon_new is not None and epsilon_original is not None:
                    method_pairs[method].append((epsilon_original, epsilon_new))
    
    return method_pairs


def plot_epsilon_comparison(
    topic: str,
    voter_type: str,
    method_pairs: Dict[str, List[Tuple[float, float]]],
    output_path: Path,
):
    """Create scatter plot comparing original vs new epsilon."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Find data range for reference line
    all_values = []
    for pairs in method_pairs.values():
        for orig, new in pairs:
            all_values.extend([orig, new])
    
    if not all_values:
        print(f"  No data for {topic}/{voter_type}")
        plt.close()
        return
    
    min_val = min(all_values)
    max_val = max(all_values)
    padding = (max_val - min_val) * 0.05
    
    # Plot y=x reference line
    line_range = [min_val - padding, max_val + padding]
    ax.plot(line_range, line_range, 'k--', alpha=0.5, linewidth=1, label='y = x')
    
    # Plot each method
    for method, config in METHODS.items():
        pairs = method_pairs[method]
        if not pairs:
            continue
        
        x_vals = [p[0] for p in pairs]
        y_vals = [p[1] for p in pairs]
        
        ax.scatter(
            x_vals, y_vals,
            marker=config["marker"],
            color=config["color"],
            label=config["label"],
            alpha=0.7,
            s=50,
        )
    
    # Labels and formatting
    voter_label = VOTER_CONFIGS[voter_type]["label"]
    ax.set_xlabel("Corrected Epsilon (m=100)", fontsize=12)
    ax.set_ylabel("Naive Epsilon (m=101)", fontsize=12)
    ax.set_title(f"{topic.capitalize()} - {voter_label}", fontsize=14)
    
    ax.set_xlim(line_range)
    ax.set_ylim(line_range)
    ax.set_aspect('equal')
    
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(output_path)
    plt.close()


def main():
    """Generate all epsilon comparison scatter plots."""
    print("Generating epsilon comparison scatter plots...")
    
    for voter_type in VOTER_CONFIGS:
        print(f"\n{VOTER_CONFIGS[voter_type]['label']}:")
        output_subdir = OUTPUT_DIR / voter_type
        
        for topic in TOPICS:
            method_pairs = load_epsilon_pairs(topic, voter_type)
            total_pairs = sum(len(pairs) for pairs in method_pairs.values())
            
            if total_pairs == 0:
                print(f"  {topic}: No data found")
                continue
            
            output_path = output_subdir / f"{topic}.png"
            plot_epsilon_comparison(topic, voter_type, method_pairs, output_path)
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
