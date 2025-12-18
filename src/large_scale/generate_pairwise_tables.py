"""
Generate pairwise comparison tables.

Creates:
1. Per-topic tables: 6x6 matrix showing % wins for row vs column method
2. Aggregate table: 6x6 matrix showing X/13 wins across all topics
"""

import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List


OUTPUT_BASE = Path("data/large_scale/gen-200-disc-50-eval-50-nano-low")
RESULTS_DIR = OUTPUT_BASE / "results"
TABLES_DIR = OUTPUT_BASE / "tables"

VOTING_METHODS = ["plurality", "borda", "irv", "chatgpt", "schulze", "veto_by_consumption"]
METHOD_LABELS = {
    "plurality": "Plurality",
    "borda": "Borda",
    "irv": "IRV",
    "chatgpt": "ChatGPT",
    "schulze": "Schulze",
    "veto_by_consumption": "Veto"
}


def load_all_results() -> List[Dict]:
    """Load all topic result files."""
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.stem == "all_topics_summary":
            continue
        with open(f, 'r') as fp:
            results.append(json.load(fp))
    return results


def build_pairwise_matrix(pairwise_results: Dict, num_personas: int) -> np.ndarray:
    """Build 6x6 pairwise win percentage matrix from results."""
    n = len(VOTING_METHODS)
    matrix = np.zeros((n, n))
    
    for i, m1 in enumerate(VOTING_METHODS):
        for j, m2 in enumerate(VOTING_METHODS):
            if i == j:
                matrix[i, j] = 0.5  # 50% against self
                continue
            
            # Find the pair result
            key1 = f"{m1}_vs_{m2}"
            key2 = f"{m2}_vs_{m1}"
            
            if key1 in pairwise_results:
                data = pairwise_results[key1]
                m1_wins = data.get("m1_wins", 0)
                m2_wins = data.get("m2_wins", 0)
                total = m1_wins + m2_wins
                if total > 0:
                    matrix[i, j] = m1_wins / total
                else:
                    matrix[i, j] = 0.5
            elif key2 in pairwise_results:
                data = pairwise_results[key2]
                m1_wins = data.get("m2_wins", 0)  # Reversed
                m2_wins = data.get("m1_wins", 0)
                total = m1_wins + m2_wins
                if total > 0:
                    matrix[i, j] = m1_wins / total
                else:
                    matrix[i, j] = 0.5
            else:
                matrix[i, j] = 0.5
    
    return matrix


def generate_topic_table(result: Dict, topic_idx: int):
    """Generate pairwise comparison table for a single topic."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    topic_slug = result.get("topic_slug", f"topic_{topic_idx}")
    pairwise = result.get("pairwise_results", {})
    num_personas = result.get("num_eval_personas", 100)
    
    matrix = build_pairwise_matrix(pairwise, num_personas)
    
    # Write CSV
    csv_file = TABLES_DIR / f"pairwise_{topic_slug}.csv"
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = [""] + [METHOD_LABELS.get(m, m) for m in VOTING_METHODS]
        writer.writerow(header)
        
        # Data rows
        for i, m in enumerate(VOTING_METHODS):
            row = [METHOD_LABELS.get(m, m)]
            for j in range(len(VOTING_METHODS)):
                if i == j:
                    row.append("-")
                else:
                    row.append(f"{matrix[i, j]:.1%}")
            writer.writerow(row)
    
    print(f"Saved: pairwise_{topic_slug}.csv")
    return matrix


def generate_aggregate_table(all_results: List[Dict]):
    """Generate aggregate pairwise comparison table (X/13 wins)."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    n = len(VOTING_METHODS)
    win_counts = np.zeros((n, n))  # Count of topics where row beats column
    
    for result in all_results:
        pairwise = result.get("pairwise_results", {})
        num_personas = result.get("num_eval_personas", 100)
        matrix = build_pairwise_matrix(pairwise, num_personas)
        
        # Count wins (win rate > 0.5)
        for i in range(n):
            for j in range(n):
                if i != j and matrix[i, j] > 0.5:
                    win_counts[i, j] += 1
    
    num_topics = len(all_results)
    
    # Write CSV
    csv_file = TABLES_DIR / "pairwise_aggregate.csv"
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = [""] + [METHOD_LABELS.get(m, m) for m in VOTING_METHODS]
        writer.writerow(header)
        
        # Data rows
        for i, m in enumerate(VOTING_METHODS):
            row = [METHOD_LABELS.get(m, m)]
            for j in range(len(VOTING_METHODS)):
                if i == j:
                    row.append("-")
                else:
                    wins = int(win_counts[i, j])
                    row.append(f"{wins}/{num_topics}")
            writer.writerow(row)
        
        # Add summary row: total wins per method
        writer.writerow([])
        total_wins = []
        for i in range(n):
            wins = sum(1 for j in range(n) if i != j and win_counts[i, j] > win_counts[j, i])
            total_wins.append(wins)
        
        writer.writerow(["Total head-to-head wins"] + [str(w) for w in total_wins])
    
    print(f"Saved: pairwise_aggregate.csv")
    
    # Also write LaTeX version
    latex_file = TABLES_DIR / "pairwise_aggregate.tex"
    
    with open(latex_file, 'w') as f:
        f.write("% Pairwise comparison table: X/Y means row method won X out of Y topics against column method\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l" + "c" * n + "}\n")
        f.write("\\toprule\n")
        
        # Header
        header = " & ".join([""] + [METHOD_LABELS.get(m, m) for m in VOTING_METHODS])
        f.write(header + " \\\\\n")
        f.write("\\midrule\n")
        
        # Data rows
        for i, m in enumerate(VOTING_METHODS):
            row_data = [METHOD_LABELS.get(m, m)]
            for j in range(n):
                if i == j:
                    row_data.append("-")
                else:
                    wins = int(win_counts[i, j])
                    row_data.append(f"{wins}/{num_topics}")
            f.write(" & ".join(row_data) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Pairwise comparison of voting methods across " + str(num_topics) + " topics. ")
        f.write("Each cell shows how many topics the row method outperformed the column method.}\n")
        f.write("\\label{tab:pairwise}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved: pairwise_aggregate.tex")
    
    return win_counts


def generate_all_tables():
    """Generate all pairwise comparison tables."""
    print("Loading results...")
    all_results = load_all_results()
    print(f"Found {len(all_results)} topic results")
    
    if not all_results:
        print("No results found!")
        return
    
    print("\nGenerating individual topic tables...")
    for i, result in enumerate(all_results):
        generate_topic_table(result, i)
    
    print("\nGenerating aggregate table...")
    generate_aggregate_table(all_results)
    
    print(f"\nAll tables saved to {TABLES_DIR}")


if __name__ == "__main__":
    generate_all_tables()

