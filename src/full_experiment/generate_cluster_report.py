"""
Generate a markdown report showing cluster sizes for each topic and rep.

Outputs a report with:
- H1: Report title
- H2: Topic sections
- H3: Rep subsections with tables showing cluster_id and statement count
"""

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from .config import OUTPUT_DIR, TOPIC_DISPLAY_NAMES

logger = logging.getLogger(__name__)


def get_topic_display_name(topic_slug: str) -> str:
    """Get display name for a topic slug."""
    return TOPIC_DISPLAY_NAMES.get(topic_slug, topic_slug)


def count_cluster_sizes(filter_assignments_path: Path) -> dict:
    """
    Count the number of statements in each cluster.
    
    Args:
        filter_assignments_path: Path to filter_assignments.json
        
    Returns:
        Dict mapping cluster_id to statement count
    """
    with open(filter_assignments_path, "r") as f:
        assignments = json.load(f)
    
    cluster_counts = Counter()
    for assignment in assignments:
        cluster_id = assignment["cluster_id"]
        cluster_counts[cluster_id] += 1
    
    return dict(cluster_counts)


def plot_cluster_count_stripplot(
    cluster_counts_by_topic: Dict[str, List[int]],
    output_path: Path,
) -> None:
    """
    Plot a strip plot showing the number of clusters per topic.
    
    Args:
        cluster_counts_by_topic: Dict mapping topic name to list of cluster counts (one per rep)
        output_path: Path to save the plot
    """
    # Prepare data for seaborn
    data = []
    for topic_name, counts in cluster_counts_by_topic.items():
        for count in counts:
            data.append({
                "Topic": topic_name,
                "Number of Clusters": count,
            })
    
    if not data:
        logger.warning("No cluster count data to plot")
        return
    
    df = pd.DataFrame(data)
    
    # Get topic order (same as in the data)
    topic_order = list(cluster_counts_by_topic.keys())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, len(topic_order) * 0.5)))
    
    # Plot strip plot with jitter
    sns.stripplot(
        data=df,
        x="Number of Clusters",
        y="Topic",
        order=topic_order,
        color="steelblue",
        alpha=0.6,
        jitter=0.25,
        size=8,
        ax=ax,
    )
    
    ax.set_xlabel("Number of Clusters", fontsize=12)
    ax.set_ylabel("Topic", fontsize=12)
    ax.set_title("Number of Clusters Across Topics and Reps", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")
    
    # Add horizontal lines between topics for visual separation
    for i in range(len(topic_order) - 1):
        ax.axhline(y=i + 0.5, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    
    # Add note about sample size
    n_topics = len(topic_order)
    n_reps = len(data) // n_topics if n_topics > 0 else 0
    ax.text(
        0.98, 0.02,
        f"{n_topics} topics, {n_reps} reps each",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        color="gray",
    )
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved cluster count strip plot to {output_path}")
    plt.close()


def plot_cluster_count_histogram(
    cluster_counts_by_topic: Dict[str, List[int]],
    output_path: Path,
) -> None:
    """
    Plot a histogram of number of clusters aggregated across all topics.
    Each bar represents an integer count of clusters.
    
    Args:
        cluster_counts_by_topic: Dict mapping topic name to list of cluster counts (one per rep)
        output_path: Path to save the plot
    """
    # Flatten all cluster counts
    all_counts = []
    for counts in cluster_counts_by_topic.values():
        all_counts.extend(counts)
    
    if not all_counts:
        logger.warning("No cluster count data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get the range of cluster counts for discrete bins
    min_count = min(all_counts)
    max_count = max(all_counts)
    
    # Create bins centered on integers
    bins = [i - 0.5 for i in range(min_count, max_count + 2)]
    
    ax.hist(
        all_counts,
        bins=bins,
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
    )
    
    # Set x-ticks: every 2 up to 10, then every 5 after that
    ticks = []
    for i in range(min_count, max_count + 1):
        if i <= 10 and i % 2 == 0:
            ticks.append(i)
        elif i > 10 and i % 5 == 0:
            ticks.append(i)
    ax.set_xticks(ticks)
    
    ax.set_xlabel("Number of Clusters", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Number of Clusters (All Topics)", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add summary statistics
    mean_count = np.mean(all_counts)
    median_count = np.median(all_counts)
    ax.axvline(mean_count, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_count:.1f}")
    ax.axvline(median_count, color="orange", linestyle="-.", linewidth=1.5, label=f"Median: {median_count:.1f}")
    ax.legend()
    
    # Add sample size note
    ax.text(
        0.98, 0.98,
        f"n={len(all_counts)} reps across {len(cluster_counts_by_topic)} topics",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        color="gray",
    )
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved cluster count histogram to {output_path}")
    plt.close()


def generate_cluster_report(
    output_dir: Path = OUTPUT_DIR,
    report_path: Path = None,
) -> None:
    """
    Generate a markdown report showing cluster sizes for each topic and rep.
    
    Args:
        output_dir: Base output directory containing data/
        report_path: Path to write the markdown report
    """
    data_dir = output_dir / "data"
    
    if report_path is None:
        report_path = output_dir / "reports" / "cluster_size_report.md"
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return
    
    # Get all topic directories sorted
    topic_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    lines = []
    lines.append("# Cluster Size Report\n")
    lines.append("This report shows the number of statements in each cluster for every topic and rep.\n")
    
    # Collect cluster counts for strip plot
    cluster_counts_by_topic: Dict[str, List[int]] = {}
    
    for topic_dir in topic_dirs:
        topic_slug = topic_dir.name
        topic_name = get_topic_display_name(topic_slug)
        
        lines.append(f"## Topic: {topic_name}\n")
        
        # Initialize list for this topic's cluster counts
        cluster_counts_by_topic[topic_name] = []
        
        # Get all rep directories (skip ones with _removed suffix)
        rep_dirs = sorted([
            d for d in topic_dir.iterdir()
            if d.is_dir() and d.name.startswith("rep") and "_removed" not in d.name
        ], key=lambda x: int(re.search(r'\d+', x.name).group()) if re.search(r'\d+', x.name) else 0)
        
        for rep_dir in rep_dirs:
            rep_name = rep_dir.name.replace("rep", "Rep ")
            
            filter_assignments_path = rep_dir / "filter_assignments.json"
            
            if not filter_assignments_path.exists():
                lines.append(f"### {rep_name}\n")
                lines.append("*No filter_assignments.json found*\n")
                continue
            
            cluster_counts = count_cluster_sizes(filter_assignments_path)
            
            if not cluster_counts:
                lines.append(f"### {rep_name}\n")
                lines.append("*No clusters found*\n")
                continue
            
            # Record number of clusters for this rep
            n_clusters = len(cluster_counts)
            cluster_counts_by_topic[topic_name].append(n_clusters)
            
            lines.append(f"### {rep_name}\n")
            lines.append("| Cluster ID | Statement Count |")
            lines.append("|------------|-----------------|")
            
            # Sort by cluster ID
            for cluster_id in sorted(cluster_counts.keys()):
                count = cluster_counts[cluster_id]
                lines.append(f"| {cluster_id} | {count} |")
            
            # Add summary row
            total = sum(cluster_counts.values())
            lines.append(f"| **Total** | **{total}** |")
            lines.append(f"\n*{n_clusters} clusters*\n")
    
    # Write report
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Saved cluster size report to {report_path}")
    
    # Generate strip plot
    plot_path = report_path.parent / "cluster_count_stripplot.png"
    plot_cluster_count_stripplot(cluster_counts_by_topic, plot_path)
    
    # Generate histogram
    hist_path = report_path.parent / "cluster_count_histogram.png"
    plot_cluster_count_histogram(cluster_counts_by_topic, hist_path)


def main():
    """Main entry point for generating cluster size report."""
    parser = argparse.ArgumentParser(
        description="Generate markdown report of cluster sizes"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: outputs/full_experiment)",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Path for output report (default: outputs/full_experiment/reports/cluster_size_report.md)",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    generate_cluster_report(
        output_dir=args.output_dir,
        report_path=args.report_path,
    )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

