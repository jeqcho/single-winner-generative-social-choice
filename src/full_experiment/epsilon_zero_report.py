"""
Generate markdown reports showing the proportion of zero epsilon values.

Creates reports for each epsilon stripplot, broken down by method, ablation, and topic.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import (
    VOTING_METHODS,
    OUTPUT_DIR,
    ALL_TOPICS,
    TOPIC_DISPLAY_NAMES,
    TOPIC_SHORT_NAMES,
    ABLATIONS,
)
from .visualizer import (
    collect_results_for_topic,
    collect_all_results,
    METHOD_NAMES,
    BARPLOT_METHOD_ORDER,
)
from .epsilon_100 import (
    collect_epsilon_100_for_topic,
    collect_all_epsilon_100,
)

logger = logging.getLogger(__name__)


def compute_zero_proportion(values: List[float]) -> Tuple[int, int, float]:
    """
    Compute the proportion of zero values in a list.
    
    Args:
        values: List of epsilon values
    
    Returns:
        Tuple of (n_zeros, n_total, proportion)
    """
    if not values:
        return 0, 0, 0.0
    
    # Filter out None and negative sentinel values
    valid_values = [v for v in values if v is not None and v >= 0]
    if not valid_values:
        return 0, 0, 0.0
    
    n_zeros = sum(1 for v in valid_values if v == 0.0)
    n_total = len(valid_values)
    proportion = n_zeros / n_total if n_total > 0 else 0.0
    
    return n_zeros, n_total, proportion


def collect_zero_proportions_by_ablation(
    output_dir: Path = OUTPUT_DIR,
    topics: Optional[List[str]] = None
) -> Dict[str, Dict[str, Tuple[int, int, float]]]:
    """
    Collect zero proportions for all methods across all ablations (aggregate).
    
    Returns:
        Dict mapping ablation -> {method: (n_zeros, n_total, proportion)}
    """
    if topics is None:
        topics = ALL_TOPICS
    
    results = {}
    for ablation in ABLATIONS:
        all_results = collect_all_results(output_dir, ablation, topics)
        results[ablation] = {}
        for method in VOTING_METHODS:
            values = all_results.get(method, [])
            results[ablation][method] = compute_zero_proportion(values)
    
    return results


def collect_zero_proportions_by_topic_and_ablation(
    output_dir: Path = OUTPUT_DIR,
    topics: Optional[List[str]] = None
) -> Dict[str, Dict[str, Dict[str, Tuple[int, int, float]]]]:
    """
    Collect zero proportions for all methods, by topic and ablation.
    
    Returns:
        Dict mapping topic -> ablation -> {method: (n_zeros, n_total, proportion)}
    """
    if topics is None:
        topics = ALL_TOPICS
    
    results = {}
    for topic in topics:
        results[topic] = {}
        for ablation in ABLATIONS:
            topic_results = collect_results_for_topic(topic, output_dir, ablation)
            results[topic][ablation] = {}
            for method in VOTING_METHODS:
                values = topic_results.get(method, [])
                results[topic][ablation][method] = compute_zero_proportion(values)
    
    return results


def collect_epsilon_100_zero_proportions_by_ablation(
    output_dir: Path = OUTPUT_DIR,
    topics: Optional[List[str]] = None
) -> Dict[str, Dict[str, Tuple[int, int, float]]]:
    """
    Collect epsilon-100 zero proportions for all methods across all ablations.
    
    Returns:
        Dict mapping ablation -> {method: (n_zeros, n_total, proportion)}
    """
    if topics is None:
        topics = ALL_TOPICS
    
    results = {}
    for ablation in ABLATIONS:
        all_results = collect_all_epsilon_100(output_dir, ablation, topics)
        results[ablation] = {}
        for method in VOTING_METHODS:
            values = all_results.get(method, [])
            results[ablation][method] = compute_zero_proportion(values)
    
    return results


def collect_epsilon_100_zero_proportions_by_topic(
    output_dir: Path = OUTPUT_DIR,
    topics: Optional[List[str]] = None
) -> Dict[str, Dict[str, Dict[str, Tuple[int, int, float]]]]:
    """
    Collect epsilon-100 zero proportions by topic and ablation.
    
    Returns:
        Dict mapping topic -> ablation -> {method: (n_zeros, n_total, proportion)}
    """
    if topics is None:
        topics = ALL_TOPICS
    
    results = {}
    for topic in topics:
        results[topic] = {}
        for ablation in ABLATIONS:
            topic_results = collect_epsilon_100_for_topic(topic, output_dir, ablation)
            results[topic][ablation] = {}
            for method in VOTING_METHODS:
                values = topic_results.get(method, [])
                results[topic][ablation][method] = compute_zero_proportion(values)
    
    return results


def format_proportion(n_zeros: int, n_total: int, proportion: float) -> str:
    """Format proportion as percentage with count."""
    if n_total == 0:
        return "N/A"
    return f"{proportion:.1%} ({n_zeros}/{n_total})"


def generate_aggregate_report(
    output_dir: Path = OUTPUT_DIR,
    topics: Optional[List[str]] = None
) -> str:
    """
    Generate the main aggregate epsilon stripplot report.
    
    Returns:
        Markdown string
    """
    if topics is None:
        topics = ALL_TOPICS
    
    # Collect aggregate data
    aggregate_data = collect_zero_proportions_by_ablation(output_dir, topics)
    topic_data = collect_zero_proportions_by_topic_and_ablation(output_dir, topics)
    
    lines = [
        "# Epsilon Zero Proportion Report",
        "",
        "This report shows the proportion of epsilon values that equal zero for each voting method.",
        "",
        "## Aggregate Results (All Topics)",
        "",
        "| Method | Full | No Filtering | No Bridging |",
        "|--------|------|--------------|-------------|",
    ]
    
    # Add rows for each method in order
    for method in BARPLOT_METHOD_ORDER:
        method_name = METHOD_NAMES.get(method, method)
        row = [f"| {method_name}"]
        for ablation in ABLATIONS:
            n_zeros, n_total, prop = aggregate_data.get(ablation, {}).get(method, (0, 0, 0.0))
            row.append(format_proportion(n_zeros, n_total, prop))
        row.append("|")
        lines.append(" | ".join(row))
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Per-Topic Breakdown")
    lines.append("")
    
    # Per-topic tables
    for topic in sorted(topics, key=lambda t: TOPIC_DISPLAY_NAMES.get(t, t)):
        display_name = TOPIC_DISPLAY_NAMES.get(topic, topic)
        lines.append(f"### {display_name}")
        lines.append("")
        lines.append("| Method | Full | No Filtering | No Bridging |")
        lines.append("|--------|------|--------------|-------------|")
        
        for method in BARPLOT_METHOD_ORDER:
            method_name = METHOD_NAMES.get(method, method)
            row = [f"| {method_name}"]
            for ablation in ABLATIONS:
                data = topic_data.get(topic, {}).get(ablation, {}).get(method, (0, 0, 0.0))
                n_zeros, n_total, prop = data
                row.append(format_proportion(n_zeros, n_total, prop))
            row.append("|")
            lines.append(" | ".join(row))
        
        lines.append("")
    
    return "\n".join(lines)


def generate_per_method_report(
    method: str,
    output_dir: Path = OUTPUT_DIR,
    topics: Optional[List[str]] = None
) -> str:
    """
    Generate a report for a specific method (showing topics as rows).
    
    Args:
        method: Voting method key
        output_dir: Output directory
        topics: List of topics to include
    
    Returns:
        Markdown string
    """
    if topics is None:
        topics = ALL_TOPICS
    
    method_name = METHOD_NAMES.get(method, method)
    topic_data = collect_zero_proportions_by_topic_and_ablation(output_dir, topics)
    
    lines = [
        f"# Epsilon Zero Proportion Report: {method_name}",
        "",
        f"This report shows the proportion of epsilon values that equal zero for the **{method_name}** method.",
        "",
        "## Results by Topic",
        "",
        "| Topic | Full | No Filtering | No Bridging |",
        "|-------|------|--------------|-------------|",
    ]
    
    for topic in sorted(topics, key=lambda t: TOPIC_DISPLAY_NAMES.get(t, t)):
        display_name = TOPIC_DISPLAY_NAMES.get(topic, topic)
        row = [f"| {display_name}"]
        for ablation in ABLATIONS:
            data = topic_data.get(topic, {}).get(ablation, {}).get(method, (0, 0, 0.0))
            n_zeros, n_total, prop = data
            row.append(format_proportion(n_zeros, n_total, prop))
        row.append("|")
        lines.append(" | ".join(row))
    
    # Add aggregate row
    lines.append("")
    lines.append("### Aggregate (All Topics)")
    lines.append("")
    
    aggregate_data = collect_zero_proportions_by_ablation(output_dir, topics)
    lines.append("| Ablation | Zero Proportion |")
    lines.append("|----------|-----------------|")
    
    for ablation in ABLATIONS:
        ablation_display = {
            "full": "Full",
            "no_filtering": "No Filtering",
            "no_bridging": "No Bridging"
        }.get(ablation, ablation)
        n_zeros, n_total, prop = aggregate_data.get(ablation, {}).get(method, (0, 0, 0.0))
        lines.append(f"| {ablation_display} | {format_proportion(n_zeros, n_total, prop)} |")
    
    return "\n".join(lines)


def generate_epsilon_100_report(
    output_dir: Path = OUTPUT_DIR,
    topics: Optional[List[str]] = None
) -> str:
    """
    Generate the epsilon-100 stripplot report.
    
    Returns:
        Markdown string
    """
    if topics is None:
        topics = ALL_TOPICS
    
    # Collect data
    aggregate_data = collect_epsilon_100_zero_proportions_by_ablation(output_dir, topics)
    topic_data = collect_epsilon_100_zero_proportions_by_topic(output_dir, topics)
    
    lines = [
        "# Epsilon-100 Zero Proportion Report",
        "",
        "This report shows the proportion of epsilon-100 values (computed against all 100 personas) that equal zero.",
        "",
        "## Aggregate Results (All Topics)",
        "",
        "| Method | Full | No Filtering | No Bridging |",
        "|--------|------|--------------|-------------|",
    ]
    
    for method in BARPLOT_METHOD_ORDER:
        method_name = METHOD_NAMES.get(method, method)
        row = [f"| {method_name}"]
        for ablation in ABLATIONS:
            n_zeros, n_total, prop = aggregate_data.get(ablation, {}).get(method, (0, 0, 0.0))
            row.append(format_proportion(n_zeros, n_total, prop))
        row.append("|")
        lines.append(" | ".join(row))
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Per-Topic Breakdown")
    lines.append("")
    
    for topic in sorted(topics, key=lambda t: TOPIC_DISPLAY_NAMES.get(t, t)):
        display_name = TOPIC_DISPLAY_NAMES.get(topic, topic)
        lines.append(f"### {display_name}")
        lines.append("")
        lines.append("| Method | Full | No Filtering | No Bridging |")
        lines.append("|--------|------|--------------|-------------|")
        
        for method in BARPLOT_METHOD_ORDER:
            method_name = METHOD_NAMES.get(method, method)
            row = [f"| {method_name}"]
            for ablation in ABLATIONS:
                data = topic_data.get(topic, {}).get(ablation, {}).get(method, (0, 0, 0.0))
                n_zeros, n_total, prop = data
                row.append(format_proportion(n_zeros, n_total, prop))
            row.append("|")
            lines.append(" | ".join(row))
        
        lines.append("")
    
    return "\n".join(lines)


def generate_all_reports(
    output_dir: Path = OUTPUT_DIR,
    topics: Optional[List[str]] = None
) -> None:
    """
    Generate all epsilon zero proportion reports and save to reports directory.
    
    Args:
        output_dir: Output directory containing data/
        topics: List of topics to include (None = all)
    """
    if topics is None:
        # Auto-discover topics
        data_dir = output_dir / "data"
        if data_dir.exists():
            topics = [d.name for d in sorted(data_dir.iterdir()) 
                     if d.is_dir() and not d.name.startswith("_")]
        else:
            topics = ALL_TOPICS
    
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating epsilon zero proportion reports to {reports_dir}")
    
    # 1. Main aggregate report
    logger.info("Generating aggregate epsilon stripplot report...")
    report = generate_aggregate_report(output_dir, topics)
    report_path = reports_dir / "epsilon_stripplot_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"  Saved: {report_path}")
    
    # 2. Per-method reports
    for method in VOTING_METHODS:
        logger.info(f"Generating report for {method}...")
        report = generate_per_method_report(method, output_dir, topics)
        report_path = reports_dir / f"epsilon_stripplot_{method}_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"  Saved: {report_path}")
    
    # 3. Epsilon-100 report
    logger.info("Generating epsilon-100 stripplot report...")
    report = generate_epsilon_100_report(output_dir, topics)
    report_path = reports_dir / "epsilon_100_stripplot_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"  Saved: {report_path}")
    
    logger.info("All reports generated successfully!")


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    output_dir = Path("outputs/full_experiment")
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    
    generate_all_reports(output_dir)



