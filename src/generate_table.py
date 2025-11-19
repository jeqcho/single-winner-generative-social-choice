"""
Generate LaTeX table showing which voting methods selected PVC elements.
"""

import json
import os
from pathlib import Path
from typing import Dict, List


def load_results(results_dir: str = "data/results") -> List[Dict]:
    """Load all result JSON files."""
    results = []
    results_path = Path(results_dir)
    
    for json_file in sorted(results_path.glob("*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    return results


def shorten_topic(topic: str, max_length: int = 50) -> str:
    """Shorten topic for table display, preserving key words."""
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


def generate_latex_table(results: List[Dict], output_file: str = "results_table.tex") -> None:
    """Generate LaTeX table from results."""
    
    methods = ["plurality", "borda", "irv", "chatgpt"]
    method_labels = {
        "plurality": "Plurality",
        "borda": "Borda",
        "irv": "IRV",
        "chatgpt": "ChatGPT"
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
    latex_lines.append("\\label{tab:voting_methods_pvc}")
    latex_lines.append("\\end{table}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(latex_lines))
    
    print(f"LaTeX table written to {output_file}")
    print(f"\nSummary:")
    print(f"Total topics: {total_topics}")
    for method in methods:
        count = method_counts[method]
        prop = count / total_topics if total_topics > 0 else 0
        print(f"  {method_labels[method]}: {count}/{total_topics} ({prop:.2%})")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LaTeX table from experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/results",
        help="Directory containing result JSON files (default: data/results)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_table.tex",
        help="Output LaTeX file (default: results_table.tex)"
    )
    
    args = parser.parse_args()
    
    results = load_results(args.results_dir)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        return
    
    generate_latex_table(results, args.output)


if __name__ == "__main__":
    main()

