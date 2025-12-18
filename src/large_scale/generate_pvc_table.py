"""
Generate LaTeX and CSV tables showing which voting methods selected PVC elements.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

# Global cache for topic mappings
_TOPIC_MAPPINGS: Optional[Dict[str, str]] = None


def load_topic_mappings() -> Dict[str, str]:
    """Load topic mappings from topic_mappings.json."""
    global _TOPIC_MAPPINGS
    if _TOPIC_MAPPINGS is not None:
        return _TOPIC_MAPPINGS
    
    mapping_file = Path("data/topic_mappings.json")
    if not mapping_file.exists():
        print("Warning: data/topic_mappings.json not found, using fallback shortening")
        _TOPIC_MAPPINGS = {}
        return _TOPIC_MAPPINGS
    
    with open(mapping_file, 'r') as f:
        _TOPIC_MAPPINGS = json.load(f)
    
    return _TOPIC_MAPPINGS


def load_results(results_dir: str = "data/large_scale/results") -> List[Dict]:
    """Load all result JSON files."""
    results: List[Dict] = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory {results_dir} does not exist")
        return results
    
    for json_file in sorted(results_path.glob("*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    return results


def shorten_topic(topic: str, max_length: int = 50) -> str:
    """Shorten topic using topic_mappings.json."""
    mappings = load_topic_mappings()
    
    # Try to find exact match in mappings
    if topic in mappings:
        return mappings[topic]
    
    # Fallback to old behavior if not found
    if len(topic) <= max_length:
        return topic
    
    # Remove question mark and common prefixes for cleaner display
    topic_clean = topic.replace("How should we ", "").replace("What are ", "").replace("What should ", "").replace("What ", "")
    topic_clean = topic_clean.replace("?", "").strip()
    
    if len(topic_clean) <= max_length:
        return topic_clean
    
    # Try to cut at a word boundary
    shortened = topic_clean[:max_length].rsplit(' ', 1)[0]
    if len(shortened) < max_length * 0.7:  # If too short, just truncate
        return topic_clean[:max_length-3] + "..."
    return shortened + "..."


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '^': r'\textasciicircum{}',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '\\': r'\textbackslash{}',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def generate_pvc_table(
    results: List[Dict],
    latex_output: str = "pvc_winner_table.tex",
    csv_output: str = "pvc_winner_table.csv"
) -> None:
    """Generate LaTeX and CSV tables from results."""
    
    methods = [
        "plurality",
        "borda",
        "irv",
        "rankedpairs",
        "chatgpt",
        "chatgpt_rankings",
        "chatgpt_profiles",
        "chatgpt_rankings_profiles"
    ]
    
    method_labels = {
        "plurality": "Plurality",
        "borda": "Borda",
        "irv": "IRV",
        "rankedpairs": "RankedPairs",
        "chatgpt": "ChatGPT",
        "chatgpt_rankings": "ChatGPT+R",
        "chatgpt_profiles": "ChatGPT+P",
        "chatgpt_rankings_profiles": "ChatGPT+R+P"
    }
    
    # Collect data
    table_data = []
    method_counts = {method: 0 for method in methods}
    total_topics = len(results)
    
    for result in results:
        topic = result.get("topic", "Unknown")
        method_results = result.get("method_results", {})
        
        row = {
            "topic": topic,
            "methods": {}
        }
        
        for method in methods:
            method_result = method_results.get(method, {})
            in_pvc = method_result.get("in_pvc", False)
            row["methods"][method] = in_pvc
            if in_pvc:
                method_counts[method] += 1
        
        table_data.append(row)
    
    # Generate LaTeX
    latex_lines = [
        "% This table requires the booktabs package: \\usepackage{booktabs}",
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{l" + "c" * len(methods) + "}",
        "\\toprule",
        "Topic & " + " & ".join([method_labels[m] for m in methods]) + " \\\\",
        "\\midrule"
    ]
    
    # Add data rows
    for row in table_data:
        topic_short = shorten_topic(row["topic"])
        topic_escaped = escape_latex(topic_short)
        
        method_cells = []
        for method in methods:
            if row["methods"][method]:
                method_cells.append("$\\checkmark$")
            else:
                method_cells.append("")
        
        latex_lines.append(f"{topic_escaped} & " + " & ".join(method_cells) + " \\\\")
    
    # Add bottom rule and summary row
    latex_lines.append("\\midrule")
    
    # Calculate proportions
    summary_cells = []
    for method in methods:
        proportion = method_counts[method] / total_topics if total_topics > 0 else 0
        summary_cells.append(f"{proportion:.2f}")
    
    latex_lines.append("Proportion & " + " & ".join(summary_cells) + " \\\\")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{Comparison of voting methods: whether each method selected an element in the Proportional Veto Core (PVC).}")
    latex_lines.append("\\label{tab:pvc_winners}")
    latex_lines.append("\\end{table}")
    
    # Write LaTeX to file
    with open(latex_output, 'w') as f:
        f.write("\n".join(latex_lines))
    
    print(f"LaTeX table written to {latex_output}")
    
    # Generate CSV
    csv_rows = []
    
    # Header row
    csv_rows.append(["Topic"] + [method_labels[m] for m in methods])
    
    # Data rows
    for row in table_data:
        topic_short = shorten_topic(row["topic"])
        csv_row = [topic_short]
        for method in methods:
            csv_row.append("1" if row["methods"][method] else "0")
        csv_rows.append(csv_row)
    
    # Summary row
    summary_row = ["Proportion"]
    for method in methods:
        proportion = method_counts[method] / total_topics if total_topics > 0 else 0
        summary_row.append(f"{proportion:.2f}")
    csv_rows.append(summary_row)
    
    # Write CSV to file
    with open(csv_output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    
    print(f"CSV table written to {csv_output}")
    
    # Print summary
    print("\nSummary:")
    print(f"Total topics: {total_topics}")
    for method in methods:
        count = method_counts[method]
        prop = count / total_topics if total_topics > 0 else 0
        print(f"  {method_labels[method]}: {count}/{total_topics} ({prop:.2%})")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate PVC winner table from experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/large_scale/results",
        help="Directory containing result JSON files (default: data/large_scale/results)"
    )
    parser.add_argument(
        "--latex-output",
        type=str,
        default="pvc_winner_table.tex",
        help="Output LaTeX file (default: pvc_winner_table.tex)"
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="pvc_winner_table.csv",
        help="Output CSV file (default: pvc_winner_table.csv)"
    )
    
    args = parser.parse_args()
    
    results = load_results(args.results_dir)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        return
    
    generate_pvc_table(results, args.latex_output, args.csv_output)


if __name__ == "__main__":
    main()

