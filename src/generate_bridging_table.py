"""
Generate LaTeX table showing mean is_good_bridging flag for each topic.
"""

import json
import os
from pathlib import Path
from typing import List, Dict


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
    """Shorten topic for display."""
    if len(topic) <= max_length:
        return topic
    
    # Remove question mark and common prefixes
    topic_clean = topic.replace("How should we ", "").replace("What are ", "").replace("What should ", "").replace("What ", "")
    topic_clean = topic_clean.replace("?", "").strip()
    
    if len(topic_clean) <= max_length:
        return topic_clean
    
    # Try to cut at a word boundary
    shortened = topic_clean[:max_length].rsplit(' ', 1)[0]
    if len(shortened) < max_length * 0.7:
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


def extract_bridging_flags(result: Dict) -> List[bool]:
    """Extract is_good_bridging flags from bridging evaluation."""
    bridging_eval = result.get("bridging_evaluation", [])
    flags = []
    
    for eval_item in bridging_eval:
        evaluation = eval_item.get("evaluation", {})
        is_good = evaluation.get("is_good_bridging")
        if is_good is not None:
            flags.append(bool(is_good))
    
    return flags


def generate_bridging_table(results: List[Dict], output_file: str = "bridging_table.tex") -> None:
    """Generate LaTeX table with topics and mean is_good_bridging."""
    
    # Collect data
    table_data = []
    all_means = []
    
    for result in results:
        topic = result.get("topic", "Unknown")
        flags = extract_bridging_flags(result)
        
        if flags:
            mean_bridging = sum(flags) / len(flags)
            all_means.append(mean_bridging)
        else:
            mean_bridging = None
        
        table_data.append({
            "topic": topic,
            "mean_bridging": mean_bridging,
            "n": len(flags)
        })
    
    # Generate LaTeX
    latex_lines = [
        "% This table requires the booktabs package: \\usepackage{booktabs}",
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{lc}",
        "\\toprule",
        "Topic & Mean is\\_good\\_bridging \\\\",
        "\\midrule"
    ]
    
    # Add data rows
    for row in table_data:
        topic_short = shorten_topic(row["topic"])
        topic_escaped = escape_latex(topic_short)
        
        if row["mean_bridging"] is not None:
            mean_str = f"{row['mean_bridging']:.3f}"
        else:
            mean_str = "N/A"
        
        latex_lines.append(f"{topic_escaped} & {mean_str} \\\\")
    
    # Add bottom rule and summary row
    latex_lines.append("\\midrule")
    
    # Calculate overall mean
    if all_means:
        overall_mean = sum(all_means) / len(all_means)
        latex_lines.append(f"Overall Mean & {overall_mean:.3f} \\\\")
    else:
        latex_lines.append("Overall Mean & N/A \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{Mean of is\\_good\\_bridging flag across evaluation personas for each topic.}")
    latex_lines.append("\\label{tab:bridging_flags}")
    latex_lines.append("\\end{table}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write("\n".join(latex_lines))
    
    print(f"LaTeX table written to {output_file}")
    print(f"\nSummary:")
    print(f"Total topics: {len(table_data)}")
    if all_means:
        print(f"Overall mean: {sum(all_means) / len(all_means):.3f}")
        print(f"Range: [{min(all_means):.3f}, {max(all_means):.3f}]")
    print(f"\nPer-topic means:")
    for row in table_data:
        topic_short = shorten_topic(row["topic"], 40)
        if row["mean_bridging"] is not None:
            print(f"  {topic_short}: {row['mean_bridging']:.3f} (n={row['n']})")
        else:
            print(f"  {topic_short}: N/A")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate LaTeX table of is_good_bridging means")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/results",
        help="Directory containing result JSON files (default: data/results)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bridging_table.tex",
        help="Output LaTeX file (default: bridging_table.tex)"
    )
    
    args = parser.parse_args()
    
    results = load_results(args.results_dir)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        return
    
    generate_bridging_table(results, args.output)


if __name__ == "__main__":
    main()



