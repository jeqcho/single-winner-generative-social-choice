"""
Generate winner-in-biggest-cluster table visualizations.

This script creates heatmap-style tables showing what percentage of samples
have their winning statement from the largest cluster, organized by voting
method (rows) and topic (columns).

Generates separate tables for "full" and "no_filtering" ablations, each with:
1. Strict version (ties don't count): Winner's cluster must be the unique largest
2. Inclusive version (ties count): Winner's cluster can be tied for largest
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import OUTPUT_DIR, VOTING_METHODS, TOPIC_DISPLAY_NAMES
from .statement_filter import load_filter_assignments
from .visualizer import BARPLOT_METHOD_ORDER, METHOD_NAMES

logger = logging.getLogger(__name__)


def collect_winner_in_biggest_cluster_data(
    output_dir: Path,
    ablation: str,
    include_ties: bool,
) -> Dict[str, Dict[str, float]]:
    """
    Collect data on what percentage of samples have winners from the biggest cluster.

    Args:
        output_dir: Output directory containing data/
        ablation: Ablation type ("full" or "no_filtering")
        include_ties: If True, count samples where winner's cluster is tied for biggest.
                      If False, only count samples where winner's cluster is uniquely biggest.

    Returns:
        Dict mapping method -> topic -> percentage (0-100)
    """
    data_dir = output_dir / "data"
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return {}

    # Discover topics
    topics = [d.name for d in sorted(data_dir.iterdir()) if d.is_dir()]

    # Initialize counters: {method: {topic: {"total": N, "in_biggest": N}}}
    counters = {
        method: {topic: {"total": 0, "in_biggest": 0} for topic in topics}
        for method in VOTING_METHODS
    }

    for topic_slug in topics:
        topic_dir = data_dir / topic_slug
        if not topic_dir.exists():
            continue

        # Iterate through all rep directories
        for rep_dir in sorted(topic_dir.glob("rep*")):
            # Handle ablation subdirectory
            if ablation == "no_filtering":
                data_subdir = rep_dir / "ablation_no_filtering"
            else:
                data_subdir = rep_dir

            if not data_subdir.exists():
                continue

            # Load filter assignments (clustering data)
            # For no_filtering, we still need cluster info from the main rep dir
            # since no_filtering doesn't have its own filter_assignments
            if ablation == "no_filtering":
                assignments_file = rep_dir / "filter_assignments.json"
            else:
                assignments_file = data_subdir / "filter_assignments.json"

            if not assignments_file.exists():
                continue

            try:
                assignments = load_filter_assignments(assignments_file.parent)
            except Exception as e:
                logger.warning(f"Failed to load filter assignments from {assignments_file}: {e}")
                continue

            # Compute cluster sizes: count statements per cluster_id
            cluster_sizes = {}
            for assignment in assignments:
                cluster_id = assignment["cluster_id"]
                cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

            if not cluster_sizes:
                continue

            # Find max cluster size and count how many clusters have that size
            max_size = max(cluster_sizes.values())
            num_clusters_at_max = sum(1 for size in cluster_sizes.values() if size == max_size)

            # Create mapping from statement_idx to cluster_id
            stmt_to_cluster = {a["statement_idx"]: a["cluster_id"] for a in assignments}

            # For no_filtering, winner index IS the original statement index
            # For full, we need to map through kept_indices
            if ablation == "no_filtering":
                # All statements are used, winner index = original index
                index_mapping = None
            else:
                # Get kept_indices: sorted list of original statement indices where keep=1
                index_mapping = sorted([
                    a["statement_idx"]
                    for a in assignments
                    if a["keep"] == 1
                ])

            # Iterate through all sample directories
            for sample_dir in sorted(data_subdir.glob("sample*")):
                results_file = sample_dir / "results.json"
                if not results_file.exists():
                    continue

                try:
                    with open(results_file, 'r') as f:
                        sample_results = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load results from {results_file}: {e}")
                    continue

                # For each method, check if winner is from biggest cluster
                for method in VOTING_METHODS:
                    if method not in sample_results:
                        continue

                    winner = sample_results[method].get("winner")
                    if winner is None:
                        continue

                    try:
                        winner_idx = int(winner)

                        # Map winner index to original statement index
                        if index_mapping is None:
                            # no_filtering: winner index IS the original index
                            original_idx = winner_idx
                        else:
                            # full: map through kept_indices
                            if winner_idx >= len(index_mapping):
                                logger.warning(
                                    f"Winner index {winner_idx} out of range for {method} in {sample_dir}"
                                )
                                continue
                            original_idx = index_mapping[winner_idx]

                        winner_cluster_id = stmt_to_cluster.get(original_idx)

                        if winner_cluster_id is None:
                            logger.warning(
                                f"Could not find cluster for statement {original_idx} in {sample_dir}"
                            )
                            continue

                        winner_cluster_size = cluster_sizes[winner_cluster_id]

                        # Check if winner is from biggest cluster
                        counters[method][topic_slug]["total"] += 1

                        if include_ties:
                            # Winner's cluster can be tied for biggest
                            if winner_cluster_size == max_size:
                                counters[method][topic_slug]["in_biggest"] += 1
                        else:
                            # Winner's cluster must be uniquely biggest
                            if winner_cluster_size == max_size and num_clusters_at_max == 1:
                                counters[method][topic_slug]["in_biggest"] += 1

                    except (ValueError, KeyError, IndexError) as e:
                        logger.warning(f"Error processing winner for {method} in {sample_dir}: {e}")
                        continue

    # Convert counts to percentages
    result = {method: {} for method in VOTING_METHODS}
    for method in VOTING_METHODS:
        for topic_slug in topics:
            total = counters[method][topic_slug]["total"]
            in_biggest = counters[method][topic_slug]["in_biggest"]
            if total > 0:
                result[method][topic_slug] = (in_biggest / total) * 100
            else:
                result[method][topic_slug] = float('nan')

    return result


def plot_winner_biggest_cluster_table(
    data: Dict[str, Dict[str, float]],
    title: str,
    output_path: Path,
) -> None:
    """
    Plot a heatmap table of winner-in-biggest-cluster percentages.

    Args:
        data: Dict mapping method -> topic -> percentage (0-100)
        title: Plot title
        output_path: Path to save figure
    """
    # Get topics in a consistent order
    all_topics = set()
    for method_data in data.values():
        all_topics.update(method_data.keys())
    topics = sorted(all_topics)

    if not topics:
        logger.warning("No data to plot")
        return

    # Build matrix: rows = methods (in BARPLOT_METHOD_ORDER), cols = topics
    methods = [m for m in BARPLOT_METHOD_ORDER if m in data]
    matrix = np.zeros((len(methods), len(topics)))

    for i, method in enumerate(methods):
        for j, topic in enumerate(topics):
            val = data.get(method, {}).get(topic, float('nan'))
            matrix[i, j] = val

    # Create display labels
    method_labels = [METHOD_NAMES.get(m, m) for m in methods]
    topic_labels = [TOPIC_DISPLAY_NAMES.get(t, t) for t in topics]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Create heatmap
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        xticklabels=topic_labels,
        yticklabels=method_labels,
        ax=ax,
        cbar_kws={"label": "% Winners from Biggest Cluster"},
        linewidths=0.5,
        linecolor='white',
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Topic", fontsize=12)
    ax.set_ylabel("Voting Method", fontsize=12)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved heatmap to {output_path}")
    plt.close()


def main():
    """Main entry point for generating winner-in-biggest-cluster tables."""
    parser = argparse.ArgumentParser(
        description="Generate winner-in-biggest-cluster table plots"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: outputs/full_experiment)",
    )
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    output_dir = args.output_dir

    # Generate tables for both ablations
    ablations = ["full", "no_filtering"]

    for ablation in ablations:
        logger.info(f"Processing ablation: {ablation}")

        figures_dir = output_dir / "figures" / ablation / "aggregate"

        # Generate strict version (ties don't count)
        logger.info(f"  Collecting data (strict, ties don't count)...")
        data_strict = collect_winner_in_biggest_cluster_data(
            output_dir, ablation, include_ties=False
        )
        plot_winner_biggest_cluster_table(
            data_strict,
            title=f"Winners from Biggest Cluster (Strict, No Ties) - {ablation}",
            output_path=figures_dir / "winner_biggest_cluster_table_strict.png",
        )

        # Generate inclusive version (ties count)
        logger.info(f"  Collecting data (inclusive, ties count)...")
        data_inclusive = collect_winner_in_biggest_cluster_data(
            output_dir, ablation, include_ties=True
        )
        plot_winner_biggest_cluster_table(
            data_inclusive,
            title=f"Winners from Biggest Cluster (Ties Included) - {ablation}",
            output_path=figures_dir / "winner_biggest_cluster_table_inclusive.png",
        )

    logger.info("Done generating winner-in-biggest-cluster tables!")


if __name__ == "__main__":
    main()

