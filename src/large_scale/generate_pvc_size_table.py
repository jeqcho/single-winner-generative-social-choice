"""
Generate LaTeX and CSV tables showing PVC size for each topic.
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


def generate_pvc_size_table(
    results: List[Dict],
    latex_output: str = "pvc_size_table.tex",
    csv_output: str = "pvc_size_table.csv"
) -> None:
    """Generate LaTeX and CSV tables showing PVC size per topic."""
    
    # Collect data
    table_data = []
    
    for result in results:
        topic = result.get("topic", "Unknown")
        pvc_size = result.get("pvc_size", 0)
        n_statements = result.get("n_statements", 0)
        pvc_percentage = result.get("pvc_percentage", 0)
        
        table_data.append({
            "topic": topic,
            "pvc_size": pvc_size,
            "n_statements": n_statements,
            "pvc_percentage": pvc_percentage
        })
    
    # Generate LaTeX
    latex_lines = [
        "% This table requires the booktabs package: \\usepackage{booktabs}",
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Topic & PVC Size & Total Statements & PVC \\% \\\\",
        "\\midrule"
    ]
    
    # Add data rows
    for row in table_data:
        topic_short = shorten_topic(row["topic"])
        topic_escaped = escape_latex(topic_short)
        
        latex_lines.append(
            f"{topic_escaped} & {row['pvc_size']} & {row['n_statements']} & {row['pvc_percentage']:.1f}\\% \\\\"
        )
    
    # Add bottom rule and summary row
    latex_lines.append("\\midrule")
    
    # Calculate averages
    if table_data:
        avg_pvc_size = sum(row["pvc_size"] for row in table_data) / len(table_data)
        avg_percentage = sum(row["pvc_percentage"] for row in table_data) / len(table_data)
        avg_statements = sum(row["n_statements"] for row in table_data) / len(table_data)
        
        latex_lines.append(
            f"Average & {avg_pvc_size:.1f} & {avg_statements:.1f} & {avg_percentage:.1f}\\% \\\\"
        )
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{Proportional Veto Core (PVC) size for each topic.}")
    latex_lines.append("\\label{tab:pvc_size}")
    latex_lines.append("\\end{table}")
    
    # Write LaTeX to file
    with open(latex_output, 'w') as f:
        f.write("\n".join(latex_lines))
    
    print(f"LaTeX table written to {latex_output}")
    
    # Generate CSV
    csv_rows = []
    
    # Header row
    csv_rows.append(["Topic", "PVC Size", "Total Statements", "PVC %"])
    
    # Data rows
    for row in table_data:
        topic_short = shorten_topic(row["topic"])
        csv_rows.append([
            topic_short,
            row["pvc_size"],
            row["n_statements"],
            f"{row['pvc_percentage']:.1f}"
        ])
    
    # Summary row
    if table_data:
        avg_pvc_size = sum(row["pvc_size"] for row in table_data) / len(table_data)
        avg_percentage = sum(row["pvc_percentage"] for row in table_data) / len(table_data)
        avg_statements = sum(row["n_statements"] for row in table_data) / len(table_data)
        
        csv_rows.append([
            "Average",
            f"{avg_pvc_size:.1f}",
            f"{avg_statements:.1f}",
            f"{avg_percentage:.1f}"
        ])
    
    # Write CSV to file
    with open(csv_output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    
    print(f"CSV table written to {csv_output}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total topics: {len(table_data)}")
    
    if table_data:
        print(f"Average PVC size: {avg_pvc_size:.1f}")
        print(f"Average PVC percentage: {avg_percentage:.1f}%")
        print(f"\nPer-topic PVC sizes:")
        for row in table_data:
            topic_short = shorten_topic(row["topic"], 40)
            print(f"  {topic_short}: {row['pvc_size']}/{row['n_statements']} ({row['pvc_percentage']:.1f}%)")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate PVC size table from experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/large_scale/results",
        help="Directory containing result JSON files (default: data/large_scale/results)"
    )
    parser.add_argument(
        "--latex-output",
        type=str,
        default="pvc_size_table.tex",
        help="Output LaTeX file (default: pvc_size_table.tex)"
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="pvc_size_table.csv",
        help="Output CSV file (default: pvc_size_table.csv)"
    )
    
    args = parser.parse_args()
    
    results = load_results(args.results_dir)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        return
    
    generate_pvc_size_table(results, args.latex_output, args.csv_output)


if __name__ == "__main__":
    main()


