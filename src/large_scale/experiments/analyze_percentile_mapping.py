"""
Analyze percentile mapping across models.

For each persona, select statements at specific percentiles (100%, 75%, 50%, 25%, 0%)
from gpt-5.2-none rankings, then find what percentile those statements appear at
in other models' rankings.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Constants
PROJECT_ROOT = Path("/home/ec2-user/single-winner-generative-social-choice")
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "check-models"
RANKINGS_FILE = OUTPUT_BASE / "rankings.json"
OUTPUT_DIR = OUTPUT_BASE / "all"

REFERENCE_MODEL = "gpt-5.2-none"
PERCENTILES = [100, 75, 50, 25, 0]
NUM_PERSONAS = 3


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


def analyze_percentiles(all_rankings: Dict[str, List[List[int]]]):
    """Analyze percentile mappings for first N personas."""
    
    model_names = list(all_rankings.keys())
    other_models = [m for m in model_names if m != REFERENCE_MODEL]
    
    ref_rankings = all_rankings[REFERENCE_MODEL]
    num_statements = len(ref_rankings[0])
    
    print(f"Reference model: {REFERENCE_MODEL}")
    print(f"Number of statements: {num_statements}")
    print(f"Analyzing {NUM_PERSONAS} personas")
    print(f"Percentiles: {PERCENTILES}")
    print()
    
    results = []
    
    for persona_idx in range(NUM_PERSONAS):
        print(f"\n{'='*80}")
        print(f"PERSONA {persona_idx}")
        print(f"{'='*80}")
        
        ref_ranking = ref_rankings[persona_idx]
        
        # Header
        header = f"{'Percentile':<12} | {'Statement':<10} | " + " | ".join([f"{m:<18}" for m in other_models])
        print(header)
        print("-" * len(header))
        
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
            other_percentiles = []
            for model_name in other_models:
                model_ranking = all_rankings[model_name][persona_idx]
                model_rank = find_statement_rank(model_ranking, statement_idx)
                model_percentile = rank_to_percentile(model_rank, num_statements)
                other_percentiles.append(model_percentile)
                row_data["other_percentiles"][model_name] = model_percentile
            
            # Print row
            row = f"{percentile:>10}% | {statement_idx:<10} | " + " | ".join([f"{p:>16.1f}%" for p in other_percentiles])
            print(row)
            
            persona_result["rows"].append(row_data)
        
        results.append(persona_result)
    
    return results, other_models


def create_markdown_tables(results: List[Dict], other_models: List[str], output_path: Path):
    """Create markdown file with tables for each persona."""
    
    lines = [
        "# Percentile Mapping Analysis",
        "",
        f"Reference model: **{REFERENCE_MODEL}**",
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


def create_latex_tables(results: List[Dict], other_models: List[str], output_path: Path):
    """Create LaTeX file with tables for each persona."""
    
    escaped_models = [m.replace('_', '-').replace('.', '-') for m in other_models]
    
    lines = []
    
    for persona_result in results:
        persona_idx = persona_result["persona"]
        
        lines.extend([
            r"\begin{table}[h]",
            r"\centering",
            f"\\caption{{Percentile Mapping for Persona {persona_idx} (Reference: {REFERENCE_MODEL.replace('_', '-').replace('.', '-')})}}",
            r"\small",
            r"\begin{tabular}{cc" + "c" * len(other_models) + "}",
            r"\toprule",
            "Ref \\% & Stmt & " + " & ".join(escaped_models) + r" \\",
            r"\midrule",
        ])
        
        for row in persona_result["rows"]:
            ref_pct = row["ref_percentile"]
            stmt = row["statement_idx"]
            other_pcts = [f"{row['other_percentiles'][m]:.1f}\\%" for m in other_models]
            lines.append(f"{ref_pct}\\% & {stmt} & " + " & ".join(other_pcts) + r" \\")
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved LaTeX to {output_path}")


def create_heatmap(results: Dict, other_models: List[str], output_path: Path):
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
    # Using a diverging colormap centered at 15% with bounds at 0 and 30+
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # green, yellow, red
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('deviation', colors, N=n_bins)
    
    # Normalize: 0-10 = green, 10-20 = yellow, 20+ = red
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=15, vmax=50)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
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
    
    # Format model names for display
    model_labels = [m.replace('gpt-5-', '').replace('gpt-5.', '5.').replace('-', '\n') for m in other_models]
    ax.set_xticklabels(model_labels, fontsize=10)
    
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
                   ha="center", va="center", color=text_color, fontsize=11, fontweight='bold')
    
    ax.set_title(f'Persona {persona_idx}: Percentile Mapping\n(Reference: {REFERENCE_MODEL})', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Reference Percentile', fontsize=12)
    
    # Add legend annotation
    ax.annotate('Green: <10% deviation | Yellow: 10-20% | Red: >20%', 
                xy=(0.5, -0.15), xycoords='axes fraction',
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved heatmap to {output_path}")


def main():
    print("Loading rankings...")
    all_rankings = load_rankings()
    
    results, other_models = analyze_percentiles(all_rankings)
    
    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    create_markdown_tables(results, other_models, OUTPUT_DIR / "percentile_mapping.md")
    create_latex_tables(results, other_models, OUTPUT_DIR / "percentile_mapping.tex")
    
    # Create heatmaps for each persona
    print("\nGenerating heatmaps...")
    for persona_result in results:
        persona_idx = persona_result["persona"]
        output_path = OUTPUT_DIR / f"percentile_mapping_persona_{persona_idx}.png"
        create_heatmap(persona_result, other_models, output_path)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
