"""
Generate cluster size histogram plots.

This script creates:
1. Combined histogram of all cluster sizes (ignoring winner status)
2. Per-method histograms
3. Normalized histogram comparing winner vs non-winner clusters
4. Biggest cluster size per rep histogram
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .config import OUTPUT_DIR
from .visualizer import (
    collect_all_cluster_sizes,
    METHOD_NAMES,
    BARPLOT_METHOD_ORDER,
)

logger = logging.getLogger(__name__)


def plot_cluster_size_histogram(
    results: Dict[str, List[tuple]],
    title: str = "Cluster Size Distribution (All Clusters)",
    output_path: Optional[Path] = None,
    bins: int = 30,
) -> None:
    """
    Plot histogram of all cluster sizes across all methods.
    Ignores winner/non-winner distinction.

    Args:
        results: Dict mapping method to list of tuples: (cluster_size, is_winner_cluster)
        title: Plot title
        output_path: Path to save figure (None = show)
        bins: Number of histogram bins
    """
    # Collect all cluster sizes (ignoring winner status)
    all_sizes = []
    for method, values in results.items():
        for cluster_size, _ in values:
            all_sizes.append(cluster_size)

    if not all_sizes:
        logger.warning("No cluster size values to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(all_sizes, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")

    ax.set_xlabel("Cluster Size", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # Add summary statistics
    mean_size = np.mean(all_sizes)
    median_size = np.median(all_sizes)
    ax.axvline(mean_size, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_size:.1f}")
    ax.axvline(median_size, color="orange", linestyle="-.", linewidth=1.5, label=f"Median: {median_size:.1f}")
    ax.legend()

    # Add sample size note
    ax.text(
        0.98, 0.98,
        f"n={len(all_sizes)} clusters",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        color="gray",
    )

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved cluster size histogram to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_cluster_size_histogram_per_method(
    results: Dict[str, List[tuple]],
    output_dir: Path,
    bins: int = 30,
) -> None:
    """
    Plot separate histograms for each voting method.
    Ignores winner/non-winner distinction.

    Args:
        results: Dict mapping method to list of tuples: (cluster_size, is_winner_cluster)
        output_dir: Directory to save figures
        bins: Number of histogram bins
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for method in BARPLOT_METHOD_ORDER:
        values = results.get(method, [])
        if not values:
            continue

        sizes = [cluster_size for cluster_size, _ in values]
        display_name = METHOD_NAMES.get(method, method)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(sizes, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")

        ax.set_xlabel("Cluster Size", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"Cluster Size Distribution: {display_name}", fontsize=14)
        ax.grid(True, alpha=0.3, axis="y")

        # Add summary statistics
        mean_size = np.mean(sizes)
        median_size = np.median(sizes)
        ax.axvline(mean_size, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_size:.1f}")
        ax.axvline(median_size, color="orange", linestyle="-.", linewidth=1.5, label=f"Median: {median_size:.1f}")
        ax.legend()

        # Add sample size note
        ax.text(
            0.98, 0.98,
            f"n={len(sizes)} clusters",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            color="gray",
        )

        plt.tight_layout()

        output_path = output_dir / f"cluster_size_histogram_{method}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved {display_name} histogram to {output_path}")
        plt.close()


def plot_cluster_size_histogram_normalized(
    results: Dict[str, List[tuple]],
    title: str = "Cluster Size Distribution: Winner vs Non-Winner Clusters",
    output_path: Optional[Path] = None,
    bins: int = 30,
) -> None:
    """
    Plot normalized histograms comparing winner vs non-winner clusters.
    Creates a subplot grid with one subplot per voting method.
    Uses density normalization to allow comparison despite different sample sizes.

    Args:
        results: Dict mapping method to list of tuples: (cluster_size, is_winner_cluster)
        title: Plot title
        output_path: Path to save figure (None = show)
        bins: Number of histogram bins
    """
    # Get methods that have data
    methods_with_data = [m for m in BARPLOT_METHOD_ORDER if results.get(m)]
    n_methods = len(methods_with_data)

    if n_methods == 0:
        logger.warning("No cluster size values to plot")
        return

    # Calculate global bin edges across all data for consistent x-axis
    all_sizes = []
    for method in methods_with_data:
        for cluster_size, _ in results[method]:
            all_sizes.append(cluster_size)
    bin_edges = np.histogram_bin_edges(all_sizes, bins=bins)

    # Create subplot grid (2 rows for 7 methods)
    n_cols = 4
    n_rows = (n_methods + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for idx, method in enumerate(methods_with_data):
        ax = axes[idx]
        display_name = METHOD_NAMES.get(method, method)
        values = results[method]

        # Separate winner and non-winner cluster sizes for this method
        winner_sizes = [cs for cs, is_winner in values if is_winner]
        non_winner_sizes = [cs for cs, is_winner in values if not is_winner]

        # Plot non-winner clusters (blue)
        if non_winner_sizes:
            ax.hist(
                non_winner_sizes,
                bins=bin_edges,
                density=True,
                edgecolor="black",
                alpha=0.5,
                color="blue",
                label=f"Non-winner (n={len(non_winner_sizes)})",
            )

        # Plot winner clusters (red)
        if winner_sizes:
            ax.hist(
                winner_sizes,
                bins=bin_edges,
                density=True,
                edgecolor="black",
                alpha=0.5,
                color="red",
                label=f"Winner (n={len(winner_sizes)})",
            )

        ax.set_title(display_name, fontsize=11)
        ax.set_xlabel("Cluster Size", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

        # Add mean lines
        if winner_sizes:
            winner_mean = np.mean(winner_sizes)
            ax.axvline(winner_mean, color="darkred", linestyle="--", linewidth=1)
        if non_winner_sizes:
            non_winner_mean = np.mean(non_winner_sizes)
            ax.axvline(non_winner_mean, color="darkblue", linestyle="--", linewidth=1)

    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved normalized histogram to {output_path}")
        plt.close()
    else:
        plt.show()


def collect_biggest_cluster_sizes(output_dir: Path) -> List[int]:
    """
    Collect the biggest cluster size for each rep across all topics.

    Args:
        output_dir: Output directory containing data/

    Returns:
        List of biggest cluster sizes (one per rep)
    """
    data_dir = output_dir / "data"
    biggest_cluster_sizes = []

    for topic_dir in sorted(data_dir.iterdir()):
        if not topic_dir.is_dir():
            continue

        for rep_dir in sorted(topic_dir.glob("rep*")):
            if "_removed" in rep_dir.name:
                continue

            filter_file = rep_dir / "filter_assignments.json"
            if not filter_file.exists():
                continue

            with open(filter_file) as f:
                assignments = json.load(f)

            # Count cluster sizes
            cluster_sizes = {}
            for a in assignments:
                cid = a["cluster_id"]
                cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1

            if cluster_sizes:
                max_size = max(cluster_sizes.values())
                biggest_cluster_sizes.append(max_size)

    return biggest_cluster_sizes


def plot_biggest_cluster_size_histogram(
    output_dir: Path,
    output_path: Optional[Path] = None,
) -> None:
    """
    Plot histogram of the biggest cluster size per rep.
    Each integer has its own bar (no grouping).

    Args:
        output_dir: Output directory containing data/
        output_path: Path to save figure (None = show)
    """
    biggest_cluster_sizes = collect_biggest_cluster_sizes(output_dir)

    if not biggest_cluster_sizes:
        logger.warning("No biggest cluster sizes to plot")
        return

    logger.info(f"Found {len(biggest_cluster_sizes)} reps")

    # Plot histogram with each integer as its own bar
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get unique values and their counts
    counts = Counter(biggest_cluster_sizes)
    values = sorted(counts.keys())
    frequencies = [counts[v] for v in values]

    # Create bar plot
    ax.bar(values, frequencies, width=0.8, edgecolor="black", color="steelblue", alpha=0.8)

    ax.set_xlabel("Biggest Cluster Size", fontsize=12)
    ax.set_ylabel("Frequency (# of Reps)", fontsize=12)
    ax.set_title("Distribution of Biggest Cluster Size per Rep", fontsize=14)

    # Y-axis: increments of 1
    max_freq = max(frequencies)
    ax.set_yticks(range(0, max_freq + 2))

    # X-axis: increments of 5, plus include 1
    max_val = max(values)
    x_ticks = [1]  # Always include 1
    x_ticks.extend(range(5, max_val + 5, 5))  # 5, 10, 15, ..., up to max
    x_ticks = sorted(set(x_ticks))  # Remove duplicates and sort
    ax.set_xticks(x_ticks)

    ax.grid(True, alpha=0.3, axis="y")

    # Add count labels on bars
    for v, f in zip(values, frequencies):
        ax.text(v, f + 0.1, str(f), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved biggest cluster size histogram to {output_path}")
        plt.close()
    else:
        plt.show()


def main():
    """Main entry point for generating cluster size histograms."""
    parser = argparse.ArgumentParser(
        description="Generate cluster size histogram plots"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: outputs/full_experiment)",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        help="Ablation type (default: full)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of histogram bins (default: 30)",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output_dir = args.output_dir
    ablation = args.ablation

    # Auto-discover topics
    data_dir = output_dir / "data"
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    topics = [d.name for d in sorted(data_dir.iterdir()) if d.is_dir()]
    logger.info(f"Found {len(topics)} topics")

    # Collect cluster size data
    all_cluster_sizes = collect_all_cluster_sizes(output_dir, ablation, topics)

    # Set output directory for figures
    figures_dir = output_dir / "figures" / ablation / "aggregate"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate combined histogram
    plot_cluster_size_histogram(
        all_cluster_sizes,
        title="Cluster Size Distribution (All Clusters)",
        output_path=figures_dir / "cluster_size_histogram.png",
        bins=args.bins,
    )

    # Generate per-method histograms
    per_method_dir = figures_dir / "cluster_size_histograms_per_method"
    plot_cluster_size_histogram_per_method(
        all_cluster_sizes,
        output_dir=per_method_dir,
        bins=args.bins,
    )

    # Generate normalized winner vs non-winner histogram
    plot_cluster_size_histogram_normalized(
        all_cluster_sizes,
        title="Cluster Size Distribution: Winner vs Non-Winner Clusters",
        output_path=figures_dir / "cluster_size_histogram_normalized.png",
        bins=args.bins,
    )

    # Generate biggest cluster size per rep histogram
    plot_biggest_cluster_size_histogram(
        output_dir,
        output_path=figures_dir / "biggest_cluster_size_histogram.png",
    )

    logger.info("Done generating cluster size histograms!")


if __name__ == "__main__":
    main()

