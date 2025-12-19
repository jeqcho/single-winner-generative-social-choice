"""
Analyze percentile mapping across models for insertion ranking experiment.

For each persona, select statements at specific percentiles (100%, 75%, 50%, 25%, 0%)
from a reference model's rankings, then find what percentile those 
statements appear at in other models' rankings.

Generates outputs for each model as reference in separate subfolders.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "check-models-insertion"
RANKINGS_FILE = OUTPUT_BASE / "rankings.json"
OUTPUT_DIR = OUTPUT_BASE / "percentile_mapping"

PERCENTILES = [100, 75, 50, 25, 0]
NUM_PERSONAS = 10  # Analyze all 10 personas


def load_rankings() -> Dict[str, List[List[int]]]:
    """Load rankings from JSON file."""
    with open(RANKINGS_FILE, 'r') as f:
        return json.load(f)


def rank_to_percentile(rank: int, total: int) -> float:
    """Convert rank (0-indexed position) to percentile (100=top, 0=bottom)."""
    # rank 0 = 100%, rank (total-1) = 0%
    return 100 * (1 - rank / (total - 1))


def percentile_to_rank(percentile: float, total: int) -> int:
    """Convert percentile to rank (0-indexed position)."""
    # 100% = rank 0, 0% = rank (total-1)
    return int((1 - percentile / 100) * (total - 1))


def find_statement_rank(ranking: List[int], statement_idx: int) -> int:
    """Find the rank (0-indexed position) of a statement in a ranking."""
    return ranking.index(statement_idx)


def analyze_percentiles(all_rankings: Dict[str, List[List[int]]], reference_model: str):
    """Analyze percentile mappings for all personas with a specific reference model."""
    
    model_names = list(all_rankings.keys())
    other_models = [m for m in model_names if m != reference_model]
    
    ref_rankings = all_rankings[reference_model]
    num_statements = len(ref_rankings[0])
    num_personas = len(ref_rankings)
    
    print(f"Reference model: {reference_model}")
    print(f"Number of statements: {num_statements}")
    print(f"Analyzing {min(NUM_PERSONAS, num_personas)} personas")
    print(f"Percentiles: {PERCENTILES}")
    print(f"Other models: {other_models}")
    print()
    
    results = []
    
    for persona_idx in range(min(NUM_PERSONAS, num_personas)):
        ref_ranking = ref_rankings[persona_idx]
        
        persona_result = {"persona": persona_idx, "rows": []}
        
        for percentile in PERCENTILES:
            # Get statement at this percentile in reference model
            ref_rank = percentile_to_rank(percentile, num_statements)
            statement_idx = ref_ranking[ref_rank]
            
            row_data = {
                "ref_percentile": percentile,
                "statement_idx": statement_idx,
                "other_percentiles": {}
            }
            
            # Find what percentile this statement is at in other models
            for model_name in other_models:
                model_ranking = all_rankings[model_name][persona_idx]
                model_rank = find_statement_rank(model_ranking, statement_idx)
                model_percentile = rank_to_percentile(model_rank, num_statements)
                row_data["other_percentiles"][model_name] = model_percentile
            
            persona_result["rows"].append(row_data)
        
        results.append(persona_result)
    
    return results, other_models


def create_markdown_tables(results: List[Dict], other_models: List[str], output_path: Path, reference_model: str):
    """Create markdown file with tables for each persona."""
    
    lines = [
        "# Percentile Mapping Analysis (Insertion Ranking)",
        "",
        f"Reference model: **{reference_model}**",
        "",
        "For each percentile in the reference model, we show what percentile that same statement",
        "appears at in other models' rankings.",
        "",
    ]
    
    for persona_result in results:
        persona_idx = persona_result["persona"]
        lines.append(f"## Persona {persona_idx}")
        lines.append("")
        
        # Table header
        header = "| Ref Percentile | Statement | " + " | ".join(other_models) + " |"
        separator = "|" + "---|" * (2 + len(other_models))
        lines.append(header)
        lines.append(separator)
        
        for row in persona_result["rows"]:
            ref_pct = row["ref_percentile"]
            stmt = row["statement_idx"]
            other_pcts = [f"{row['other_percentiles'][m]:.1f}%" for m in other_models]
            line = f"| {ref_pct}% | {stmt} | " + " | ".join(other_pcts) + " |"
            lines.append(line)
        
        lines.append("")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nSaved markdown to {output_path}")


def create_heatmap(results: Dict, other_models: List[str], output_path: Path, reference_model: str):
    """
    Create a heatmap for a single persona's percentile mapping.
    
    Color scheme based on absolute deviation from reference:
    - <10% deviation = green
    - >20% deviation = red
    """
    persona_idx = results["persona"]
    rows = results["rows"]
    
    # Build matrix of deviations
    n_percentiles = len(rows)
    n_models = len(other_models)
    
    deviation_matrix = np.zeros((n_percentiles, n_models))
    percentile_matrix = np.zeros((n_percentiles, n_models))
    ref_percentiles = []
    statements = []
    
    for i, row in enumerate(rows):
        ref_pct = row["ref_percentile"]
        ref_percentiles.append(f"{ref_pct}%")
        statements.append(f"Stmt {row['statement_idx']}")
        
        for j, model in enumerate(other_models):
            other_pct = row["other_percentiles"][model]
            deviation = abs(other_pct - ref_pct)
            deviation_matrix[i, j] = deviation
            percentile_matrix[i, j] = other_pct
    
    # Create custom colormap: green (<10) -> yellow (10-20) -> red (>20)
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # green, yellow, red
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('deviation', colors, N=n_bins)
    
    # Normalize: 0-10 = green, 10-20 = yellow, 20+ = red
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=15, vmax=50)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(deviation_matrix, cmap=cmap, norm=norm, aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, extend='max')
    cbar.ax.set_ylabel('Absolute Deviation from Reference (%)', rotation=-90, va="bottom", fontsize=11)
    
    # Add reference lines on colorbar
    cbar.ax.axhline(y=10, color='black', linewidth=1, linestyle='--')
    cbar.ax.axhline(y=20, color='black', linewidth=1, linestyle='--')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_percentiles))
    
    # Format model names for display - shorter labels
    model_labels = []
    for m in other_models:
        label = m.replace('gpt-5.2-', '5.2-').replace('gpt-5-', '')
        label = label.replace('-t1-', '\nt1-').replace('-t0', '\nt0')
        model_labels.append(label)
    ax.set_xticklabels(model_labels, fontsize=9)
    
    # Y-axis: show reference percentile only
    ax.set_yticklabels(ref_percentiles, fontsize=10)
    
    # Add cell values (show actual percentile, not deviation)
    for i in range(n_percentiles):
        for j in range(n_models):
            deviation = deviation_matrix[i, j]
            actual_pct = percentile_matrix[i, j]
            
            # Choose text color based on deviation
            if deviation < 10:
                text_color = 'black'
            elif deviation < 20:
                text_color = 'black'
            else:
                text_color = 'white'
            
            ax.text(j, i, f'{actual_pct:.0f}%',
                   ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')
    
    ax.set_title(f'Persona {persona_idx}: Percentile Mapping\n(Reference: {reference_model})', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Reference Percentile', fontsize=12)
    
    # Add legend annotation
    ax.annotate('Green: <10% deviation | Yellow: 10-20% | Red: >20%', 
                xy=(0.5, -0.18), xycoords='axes fraction',
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap to {output_path}")


def create_averaged_heatmap(results: List[Dict], other_models: List[str], output_path: Path, reference_model: str):
    """
    Create a heatmap averaged across all personas.
    
    Shows mean deviation and mean percentile for each (reference percentile, model) cell.
    """
    n_personas = len(results)
    n_percentiles = len(PERCENTILES)
    n_models = len(other_models)
    
    # Accumulate deviations and percentiles across all personas
    deviation_sum = np.zeros((n_percentiles, n_models))
    percentile_sum = np.zeros((n_percentiles, n_models))
    
    for persona_result in results:
        rows = persona_result["rows"]
        for i, row in enumerate(rows):
            ref_pct = row["ref_percentile"]
            for j, model in enumerate(other_models):
                other_pct = row["other_percentiles"][model]
                deviation = abs(other_pct - ref_pct)
                deviation_sum[i, j] += deviation
                percentile_sum[i, j] += other_pct
    
    # Compute means
    deviation_matrix = deviation_sum / n_personas
    percentile_matrix = percentile_sum / n_personas
    
    ref_percentiles = [f"{p}%" for p in PERCENTILES]
    
    # Create custom colormap: green (<10) -> yellow (10-20) -> red (>20)
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # green, yellow, red
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('deviation', colors, N=n_bins)
    
    # Normalize: 0-10 = green, 10-20 = yellow, 20+ = red
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=15, vmax=50)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(deviation_matrix, cmap=cmap, norm=norm, aspect='auto')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, extend='max')
    cbar.ax.set_ylabel('Mean Absolute Deviation from Reference (%)', rotation=-90, va="bottom", fontsize=11)
    
    # Add reference lines on colorbar
    cbar.ax.axhline(y=10, color='black', linewidth=1, linestyle='--')
    cbar.ax.axhline(y=20, color='black', linewidth=1, linestyle='--')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_models))
    ax.set_yticks(np.arange(n_percentiles))
    
    # Format model names for display - shorter labels
    model_labels = []
    for m in other_models:
        label = m.replace('gpt-5.2-', '5.2-').replace('gpt-5-', '')
        label = label.replace('-t1-', '\nt1-').replace('-t0', '\nt0')
        model_labels.append(label)
    ax.set_xticklabels(model_labels, fontsize=9)
    
    # Y-axis: show reference percentile only
    ax.set_yticklabels(ref_percentiles, fontsize=10)
    
    # Add cell values (show mean percentile, not deviation)
    for i in range(n_percentiles):
        for j in range(n_models):
            deviation = deviation_matrix[i, j]
            mean_pct = percentile_matrix[i, j]
            
            # Choose text color based on deviation
            if deviation < 10:
                text_color = 'black'
            elif deviation < 20:
                text_color = 'black'
            else:
                text_color = 'white'
            
            ax.text(j, i, f'{mean_pct:.0f}%',
                   ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')
    
    ax.set_title(f'Average Across {n_personas} Personas: Percentile Mapping\n(Reference: {reference_model})', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Reference Percentile', fontsize=12)
    
    # Add legend annotation
    ax.annotate('Green: <10% deviation | Yellow: 10-20% | Red: >20%', 
                xy=(0.5, -0.18), xycoords='axes fraction',
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved averaged heatmap to {output_path}")


def generate_for_reference(all_rankings: Dict[str, List[List[int]]], reference_model: str):
    """Generate all percentile mapping outputs for a specific reference model."""
    # Create subfolder for this reference model
    dir_name = reference_model.replace(".", "-")
    model_output_dir = OUTPUT_DIR / dir_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Generating outputs with reference: {reference_model}")
    print(f"Output directory: {model_output_dir}")
    print(f"{'='*80}")
    
    results, other_models = analyze_percentiles(all_rankings, reference_model)
    
    # Save markdown
    create_markdown_tables(results, other_models, model_output_dir / "percentile_mapping.md", reference_model)
    
    # Create heatmaps for each persona
    for persona_result in results:
        persona_idx = persona_result["persona"]
        output_path = model_output_dir / f"percentile_mapping_persona_{persona_idx}.png"
        create_heatmap(persona_result, other_models, output_path, reference_model)
    
    # Create averaged heatmap across all personas
    create_averaged_heatmap(results, other_models, model_output_dir / "percentile_mapping_averaged.png", reference_model)
    
    print(f"  Saved {len(results) + 1} heatmaps to {model_output_dir}")


def main():
    print("Loading rankings...")
    all_rankings = load_rankings()
    
    model_names = list(all_rankings.keys())
    print(f"Found {len(model_names)} models: {model_names}")
    
    # Generate outputs for each model as reference
    for reference_model in model_names:
        generate_for_reference(all_rankings, reference_model)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

