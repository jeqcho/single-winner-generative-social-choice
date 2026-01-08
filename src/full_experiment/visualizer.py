"""
Visualization functions for experiment results.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from .config import VOTING_METHODS, OUTPUT_DIR, TOPIC_SHORT_NAMES, TOPIC_DISPLAY_NAMES
from .statement_filter import load_filter_assignments

logger = logging.getLogger(__name__)

# Color palette for voting methods
METHOD_COLORS = {
    "schulze": "#1f77b4",
    "veto_by_consumption": "#ff7f0e",
    "borda": "#2ca02c",
    "irv": "#d62728",
    "plurality": "#9467bd",
    "chatgpt": "#8c564b",
    "chatgpt_with_rankings": "#e377c2",
    "chatgpt_with_personas": "#17becf",
}

# Display names for methods
METHOD_NAMES = {
    "schulze": "Schulze",
    "veto_by_consumption": "Veto by Consumption",
    "borda": "Borda",
    "irv": "IRV",
    "plurality": "Plurality",
    "chatgpt": "ChatGPT",
    "chatgpt_with_rankings": "ChatGPT + Rankings",
    "chatgpt_with_personas": "ChatGPT + Personas",
}

# Custom order for bar plots: veto, borda, schulze, irv, plurality, gpt, gpt+rankings, gpt+personas
BARPLOT_METHOD_ORDER = [
    "veto_by_consumption",
    "borda",
    "schulze",
    "irv",
    "plurality",
    "chatgpt",
    "chatgpt_with_rankings",
    "chatgpt_with_personas",
]


def collect_results_for_topic(
    topic_slug: str,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full"
) -> Dict[str, List[float]]:
    """
    Collect all epsilon results for a topic across all repetitions and samples.
    
    Args:
        topic_slug: Topic slug
        output_dir: Output directory
        ablation: Ablation type
    
    Returns:
        Dict mapping method name to list of epsilon values
    """
    results = {method: [] for method in VOTING_METHODS}
    
    topic_dir = output_dir / "data" / topic_slug
    
    if not topic_dir.exists():
        logger.warning(f"Topic directory not found: {topic_dir}")
        return results
    
    # Iterate through all rep directories
    for rep_dir in sorted(topic_dir.glob("rep*")):
        # Handle ablation subdirectory
        if ablation != "full":
            data_dir = rep_dir / f"ablation_{ablation}"
        else:
            data_dir = rep_dir
        
        if not data_dir.exists():
            continue
        
        # Iterate through all sample directories
        for sample_dir in sorted(data_dir.glob("sample*")):
            results_file = sample_dir / "results.json"
            if not results_file.exists():
                continue
            
            with open(results_file, 'r') as f:
                sample_results = json.load(f)
            
            for method in VOTING_METHODS:
                if method in sample_results:
                    epsilon = sample_results[method].get("epsilon")
                    if epsilon is not None:
                        results[method].append(epsilon)
    
    return results


def collect_results_clustered_for_topic(
    topic_slug: str,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full"
) -> Dict[str, List[List[float]]]:
    """
    Collect epsilon results preserving the hierarchical structure (outer reps → inner samples).
    
    Args:
        topic_slug: Topic slug
        output_dir: Output directory
        ablation: Ablation type
    
    Returns:
        Dict mapping method name to list of lists: outer list = outer reps, inner list = samples within rep
    """
    # Initialize with empty lists for each method
    results = {method: [] for method in VOTING_METHODS}
    
    topic_dir = output_dir / "data" / topic_slug
    
    if not topic_dir.exists():
        logger.warning(f"Topic directory not found: {topic_dir}")
        return results
    
    # Iterate through all rep directories (outer loop)
    for rep_dir in sorted(topic_dir.glob("rep*")):
        # Handle ablation subdirectory
        if ablation != "full":
            data_dir = rep_dir / f"ablation_{ablation}"
        else:
            data_dir = rep_dir
        
        if not data_dir.exists():
            continue
        
        # Collect all samples for this rep (inner loop)
        rep_results = {method: [] for method in VOTING_METHODS}
        
        for sample_dir in sorted(data_dir.glob("sample*")):
            results_file = sample_dir / "results.json"
            if not results_file.exists():
                continue
            
            with open(results_file, 'r') as f:
                sample_results = json.load(f)
            
            for method in VOTING_METHODS:
                if method in sample_results:
                    epsilon = sample_results[method].get("epsilon")
                    if epsilon is not None:
                        rep_results[method].append(epsilon)
        
        # Add this rep's results to the main results
        for method in VOTING_METHODS:
            if rep_results[method]:  # Only add if we have samples
                results[method].append(rep_results[method])
    
    return results


def collect_all_results_clustered(
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full",
    topics: Optional[List[str]] = None
) -> Dict[str, List[List[float]]]:
    """
    Collect all epsilon results across topics, preserving cluster structure.
    
    Each outer rep across all topics is treated as one cluster.
    
    Args:
        output_dir: Output directory
        ablation: Ablation type
        topics: List of topics to include (None = all)
    
    Returns:
        Dict mapping method name to list of lists (clusters)
    """
    all_results = {method: [] for method in VOTING_METHODS}
    
    data_dir = output_dir / "data"
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return all_results
    
    # Get all topic directories
    if topics is None:
        topic_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    else:
        topic_dirs = [data_dir / t for t in topics if (data_dir / t).exists()]
    
    for topic_dir in topic_dirs:
        topic_results = collect_results_clustered_for_topic(
            topic_dir.name, output_dir, ablation
        )
        for method in VOTING_METHODS:
            all_results[method].extend(topic_results[method])
    
    return all_results


def compute_cluster_ci(clustered_values: List[List[float]], confidence: float = 0.95):
    """
    Compute cluster-aware confidence interval.
    
    1. Average the inner samples within each cluster (outer rep)
    2. Compute CI on the cluster means using t-distribution
    
    Args:
        clustered_values: List of lists, where each inner list is samples from one cluster
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (grand_mean, ci_half_width, n_clusters)
    """
    if not clustered_values:
        return None, None, 0
    
    # Step 1: Compute mean within each cluster
    cluster_means = []
    for cluster in clustered_values:
        valid = [v for v in cluster if v is not None]
        if valid:
            cluster_means.append(np.mean(valid))
    
    if not cluster_means:
        return None, None, 0
    
    n = len(cluster_means)
    
    if n < 2:
        # Not enough clusters for CI, return mean with no CI
        return np.mean(cluster_means), None, n
    
    # Step 2: Compute CI on cluster means
    grand_mean = np.mean(cluster_means)
    sem = np.std(cluster_means, ddof=1) / np.sqrt(n)
    
    # t-value for confidence interval
    alpha = 1 - confidence
    t_val = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_half_width = t_val * sem
    
    return grand_mean, ci_half_width, n


def collect_all_results(
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full",
    topics: Optional[List[str]] = None
) -> Dict[str, List[float]]:
    """
    Collect all epsilon results across all topics.
    
    Args:
        output_dir: Output directory
        ablation: Ablation type
        topics: List of topics to include (None = all)
    
    Returns:
        Dict mapping method name to list of epsilon values
    """
    all_results = {method: [] for method in VOTING_METHODS}
    
    data_dir = output_dir / "data"
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return all_results
    
    # Get all topic directories
    if topics is None:
        topic_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    else:
        topic_dirs = [data_dir / t for t in topics if (data_dir / t).exists()]
    
    for topic_dir in topic_dirs:
        topic_results = collect_results_for_topic(
            topic_dir.name, output_dir, ablation
        )
        for method in VOTING_METHODS:
            all_results[method].extend(topic_results[method])
    
    return all_results


def collect_epsilon_by_topic(
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full",
    topics: Optional[List[str]] = None
) -> Dict[str, Dict[str, List[float]]]:
    """
    Collect epsilon results organized by topic, then by method.
    
    Args:
        output_dir: Output directory
        ablation: Ablation type
        topics: List of topics to include (None = all)
    
    Returns:
        Dict mapping topic_slug to {method: [epsilon values]}
    """
    results_by_topic = {}
    
    data_dir = output_dir / "data"
    if not data_dir.exists():
        return results_by_topic
    
    # Get all topic directories
    if topics is None:
        topic_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    else:
        topic_dirs = [data_dir / t for t in topics if (data_dir / t).exists()]
    
    for topic_dir in topic_dirs:
        topic_slug = topic_dir.name
        topic_results = collect_results_for_topic(topic_slug, output_dir, ablation)
        results_by_topic[topic_slug] = topic_results
    
    return results_by_topic


def collect_likert_by_topic(
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full",
    topics: Optional[List[str]] = None
) -> Dict[str, Dict[str, List[float]]]:
    """
    Collect Likert results organized by topic, then by method.
    
    Args:
        output_dir: Output directory
        ablation: Ablation type
        topics: List of topics to include (None = all)
    
    Returns:
        Dict mapping topic_slug to {method: [Likert values]}
    """
    results_by_topic = {}
    
    data_dir = output_dir / "data"
    if not data_dir.exists():
        return results_by_topic
    
    # Get all topic directories
    if topics is None:
        topic_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    else:
        topic_dirs = [data_dir / t for t in topics if (data_dir / t).exists()]
    
    for topic_dir in topic_dirs:
        topic_slug = topic_dir.name
        topic_results = collect_likert_for_topic(topic_slug, output_dir, ablation)
        results_by_topic[topic_slug] = topic_results
    
    return results_by_topic


def collect_cluster_sizes_for_topic(
    topic_slug: str,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full"
) -> Dict[str, List[tuple]]:
    """
    Collect cluster size data for each voting method, with flags indicating winner clusters.
    
    Args:
        topic_slug: Topic slug
        output_dir: Output directory
        ablation: Ablation type
    
    Returns:
        Dict mapping method name to list of tuples: (cluster_size, is_winner_cluster)
    """
    results = {method: [] for method in VOTING_METHODS}
    
    topic_dir = output_dir / "data" / topic_slug
    
    if not topic_dir.exists():
        logger.warning(f"Topic directory not found: {topic_dir}")
        return results
    
    # Iterate through all rep directories
    for rep_dir in sorted(topic_dir.glob("rep*")):
        # Handle ablation subdirectory
        if ablation != "full":
            data_dir = rep_dir / f"ablation_{ablation}"
        else:
            data_dir = rep_dir
        
        if not data_dir.exists():
            continue
        
        # Load filter assignments (clustering data)
        assignments_file = data_dir / "filter_assignments.json"
        if not assignments_file.exists():
            # Skip if no clustering data (e.g., for no_filtering ablation)
            continue
        
        try:
            assignments = load_filter_assignments(data_dir)
        except Exception as e:
            logger.warning(f"Failed to load filter assignments from {data_dir}: {e}")
            continue
        
        # Compute cluster sizes: count statements per cluster_id
        cluster_sizes = {}
        for assignment in assignments:
            cluster_id = assignment["cluster_id"]
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
        
        # Get kept_indices: sorted list of original statement indices where keep=1
        kept_indices = sorted([
            a["statement_idx"]
            for a in assignments
            if a["keep"] == 1
        ])
        
        # Create mapping from statement_idx to cluster_id
        stmt_to_cluster = {a["statement_idx"]: a["cluster_id"] for a in assignments}
        
        # Iterate through all sample directories
        for sample_dir in sorted(data_dir.glob("sample*")):
            results_file = sample_dir / "results.json"
            if not results_file.exists():
                continue
            
            try:
                with open(results_file, 'r') as f:
                    sample_results = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load results from {results_file}: {e}")
                continue
            
            # For each method, find which cluster contains the winner
            for method in VOTING_METHODS:
                if method not in sample_results:
                    continue
                
                winner = sample_results[method].get("winner")
                if winner is None:
                    continue
                
                try:
                    winner_idx = int(winner)
                    # Map winner index (filtered) → original statement index
                    if winner_idx >= len(kept_indices):
                        logger.warning(f"Winner index {winner_idx} out of range for {method} in {sample_dir}")
                        continue
                    
                    original_idx = kept_indices[winner_idx]
                    winner_cluster_id = stmt_to_cluster.get(original_idx)
                    
                    if winner_cluster_id is None:
                        logger.warning(f"Could not find cluster for statement {original_idx} in {sample_dir}")
                        continue
                    
                    # Add all cluster sizes with flags
                    for cluster_id, size in cluster_sizes.items():
                        is_winner = (cluster_id == winner_cluster_id)
                        results[method].append((size, is_winner))
                
                except (ValueError, KeyError, IndexError) as e:
                    logger.warning(f"Error processing winner for {method} in {sample_dir}: {e}")
                    continue
    
    return results


def collect_winner_cluster_percentile_clustered_for_topic(
    topic_slug: str,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full"
) -> Dict[str, List[List[float]]]:
    """
    Collect percentile rankings of winner cluster sizes, preserving hierarchical structure.
    
    For each sample, compute the percentile ranking of the winner cluster size
    among all cluster sizes in that sample.
    
    Args:
        topic_slug: Topic slug
        output_dir: Output directory
        ablation: Ablation type
    
    Returns:
        Dict mapping method name to list of lists: outer list = outer reps, inner list = percentile rankings
    """
    results = {method: [] for method in VOTING_METHODS}
    
    topic_dir = output_dir / "data" / topic_slug
    
    if not topic_dir.exists():
        logger.warning(f"Topic directory not found: {topic_dir}")
        return results
    
    # Iterate through all rep directories (outer loop)
    for rep_dir in sorted(topic_dir.glob("rep*")):
        # Handle ablation subdirectory
        if ablation != "full":
            data_dir = rep_dir / f"ablation_{ablation}"
        else:
            data_dir = rep_dir
        
        if not data_dir.exists():
            continue
        
        # Load filter assignments (clustering data)
        assignments_file = data_dir / "filter_assignments.json"
        if not assignments_file.exists():
            continue
        
        try:
            assignments = load_filter_assignments(data_dir)
        except Exception as e:
            logger.warning(f"Failed to load filter assignments from {data_dir}: {e}")
            continue
        
        # Compute cluster sizes: count statements per cluster_id
        cluster_sizes = {}
        for assignment in assignments:
            cluster_id = assignment["cluster_id"]
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
        
        # Get kept_indices: sorted list of original statement indices where keep=1
        kept_indices = sorted([
            a["statement_idx"]
            for a in assignments
            if a["keep"] == 1
        ])
        
        # Create mapping from statement_idx to cluster_id
        stmt_to_cluster = {a["statement_idx"]: a["cluster_id"] for a in assignments}
        
        # Collect all samples for this rep
        rep_results = {method: [] for method in VOTING_METHODS}
        
        # Iterate through all sample directories
        for sample_dir in sorted(data_dir.glob("sample*")):
            results_file = sample_dir / "results.json"
            if not results_file.exists():
                continue
            
            try:
                with open(results_file, 'r') as f:
                    sample_results = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load results from {results_file}: {e}")
                continue
            
            # Get all cluster sizes for this sample (sorted)
            all_sizes = sorted(cluster_sizes.values())
            
            # For each method, find percentile of winner cluster
            for method in VOTING_METHODS:
                if method not in sample_results:
                    continue
                
                winner = sample_results[method].get("winner")
                if winner is None:
                    continue
                
                try:
                    winner_idx = int(winner)
                    # Map winner index (filtered) → original statement index
                    if winner_idx >= len(kept_indices):
                        logger.warning(f"Winner index {winner_idx} out of range for {method} in {sample_dir}")
                        continue
                    
                    original_idx = kept_indices[winner_idx]
                    winner_cluster_id = stmt_to_cluster.get(original_idx)
                    
                    if winner_cluster_id is None:
                        logger.warning(f"Could not find cluster for statement {original_idx} in {sample_dir}")
                        continue
                    
                    winner_cluster_size = cluster_sizes[winner_cluster_id]
                    
                    # Compute percentile ranking (0-100)
                    # percentileofscore gives the percentile where the score would fall
                    percentile = stats.percentileofscore(all_sizes, winner_cluster_size, kind='rank')
                    rep_results[method].append(percentile)
                
                except (ValueError, KeyError, IndexError) as e:
                    logger.warning(f"Error processing winner for {method} in {sample_dir}: {e}")
                    continue
        
        # Add this rep's results to the main results
        for method in VOTING_METHODS:
            if rep_results[method]:  # Only add if we have samples
                results[method].append(rep_results[method])
    
    return results


def collect_all_winner_cluster_percentile_clustered(
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full",
    topics: Optional[List[str]] = None
) -> Dict[str, List[List[float]]]:
    """
    Collect all winner cluster percentile rankings across topics, preserving cluster structure.
    
    Args:
        output_dir: Output directory
        ablation: Ablation type
        topics: List of topics to include (None = all)
    
    Returns:
        Dict mapping method name to list of lists (clusters)
    """
    all_results = {method: [] for method in VOTING_METHODS}
    
    data_dir = output_dir / "data"
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return all_results
    
    # Get all topic directories
    if topics is None:
        topic_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    else:
        topic_dirs = [data_dir / t for t in topics if (data_dir / t).exists()]
    
    for topic_dir in topic_dirs:
        topic_results = collect_winner_cluster_percentile_clustered_for_topic(
            topic_dir.name, output_dir, ablation
        )
        for method in VOTING_METHODS:
            all_results[method].extend(topic_results[method])
    
    return all_results


def plot_winner_cluster_percentile_barplot(
    clustered_results: Dict[str, List[List[float]]],
    title: str = "Winner Cluster Size Percentile Ranking by Voting Method",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot bar chart of average percentile ranking of winner cluster sizes with 95% CI error bars.
    
    Args:
        clustered_results: Dict mapping method to list of lists (outer reps → inner samples)
        title: Plot title
        output_path: Path to save figure (None = show)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    means = []
    cis = []
    colors = []
    n_clusters_list = []
    
    # Use custom order for bar plots
    for method in BARPLOT_METHOD_ORDER:
        clusters = clustered_results.get(method, [])
        if clusters:
            mean, ci, n_clusters = compute_cluster_ci(clusters)
            if mean is not None:
                methods.append(METHOD_NAMES.get(method, method))
                means.append(mean)
                cis.append(ci if ci is not None else 0)
                colors.append(METHOD_COLORS.get(method, "#333333"))
                n_clusters_list.append(n_clusters)
    
    if not methods:
        logger.warning("No data to plot")
        return
    
    x = np.arange(len(methods))
    
    # Asymmetric error bars: clip to [0, 100] (percentile ∈ [0, 100])
    lower_errors = [min(ci, mean) for mean, ci in zip(means, cis)]  # Can't go below 0
    upper_errors = [min(ci, 100 - mean) for mean, ci in zip(means, cis)]  # Can't go above 100
    yerr = [lower_errors, upper_errors]
    
    bars = ax.bar(x, means, yerr=yerr, capsize=5, color=colors, alpha=0.8)
    
    ax.set_xlabel("Voting Method", fontsize=12)
    ax.set_ylabel("Percentile Ranking", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, ci in zip(bars, means, cis):
        height = bar.get_height()
        ax.annotate(
            f'{mean:.1f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=9
        )
    
    # Add note about error bars
    n_clusters = n_clusters_list[0] if n_clusters_list else 0
    ax.text(0.02, 0.98, f"Error bars: 95% CI (n={n_clusters} outer reps)",
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            color='gray')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved winner cluster percentile bar plot to {output_path}")
        plt.close()
    else:
        plt.show()


def collect_all_cluster_sizes(
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full",
    topics: Optional[List[str]] = None
) -> Dict[str, List[tuple]]:
    """
    Collect all cluster size data across topics.
    
    Args:
        output_dir: Output directory
        ablation: Ablation type
        topics: List of topics to include (None = all)
    
    Returns:
        Dict mapping method name to list of tuples: (cluster_size, is_winner_cluster)
    """
    all_results = {method: [] for method in VOTING_METHODS}
    
    data_dir = output_dir / "data"
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return all_results
    
    # Get all topic directories
    if topics is None:
        topic_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    else:
        topic_dirs = [data_dir / t for t in topics if (data_dir / t).exists()]
    
    for topic_dir in topic_dirs:
        topic_results = collect_cluster_sizes_for_topic(
            topic_dir.name, output_dir, ablation
        )
        for method in VOTING_METHODS:
            all_results[method].extend(topic_results[method])
    
    return all_results


def plot_epsilon_histogram(
    results: Dict[str, List[float]],
    title: str = "Epsilon Distribution by Voting Method",
    output_path: Optional[Path] = None,
    bins: int = 20
) -> None:
    """
    Plot histogram of epsilon values for each voting method.
    
    Args:
        results: Dict mapping method to list of epsilon values
        title: Plot title
        output_path: Path to save figure (None = show)
        bins: Number of histogram bins
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get all epsilon values to determine range
    all_values = []
    for values in results.values():
        all_values.extend([v for v in values if v is not None])
    
    if not all_values:
        logger.warning("No epsilon values to plot")
        return
    
    # Fixed range 0 to 1 for epsilon
    bin_edges = np.linspace(0, 1, bins + 1)
    
    # Collect data for stacked histogram
    data_to_plot = []
    labels = []
    colors = []
    for method in VOTING_METHODS:
        values = [v for v in results.get(method, []) if v is not None]
        if values:
            data_to_plot.append(values)
            labels.append(METHOD_NAMES.get(method, method))
            colors.append(METHOD_COLORS.get(method, None))
    
    # Plot stacked histogram
    if data_to_plot:
        ax.hist(
            data_to_plot,
            bins=bin_edges,
            label=labels,
            color=colors,
            stacked=True,
            edgecolor='white',
            linewidth=0.5
        )
    
    ax.set_xlabel("Epsilon (ε)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved histogram to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_single_method_histogram(
    values: List[float],
    method: str,
    title: str = None,
    output_path: Optional[Path] = None,
    bins: int = 20
) -> None:
    """
    Plot histogram of epsilon values for a single voting method.
    
    Args:
        values: List of epsilon values
        method: Method name (for color and default title)
        title: Plot title (optional)
        output_path: Path to save figure (None = show)
        bins: Number of histogram bins
    """
    clean_values = [v for v in values if v is not None]
    
    if not clean_values:
        logger.warning(f"No epsilon values for {method}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    color = METHOD_COLORS.get(method, "#1f77b4")
    display_name = METHOD_NAMES.get(method, method)
    
    # Fixed range 0 to 1 for epsilon
    bin_edges = np.linspace(0, 1, bins + 1)
    
    ax.hist(
        clean_values,
        bins=bin_edges,
        color=color,
        edgecolor='white',
        linewidth=0.5,
        alpha=0.8
    )
    
    ax.set_xlabel("Epsilon (ε)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title or f"Epsilon Distribution: {display_name}", fontsize=14)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Add stats annotation
    mean_val = np.mean(clean_values)
    std_val = np.std(clean_values)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_epsilon_barplot(
    clustered_results: Dict[str, List[List[float]]],
    title: str = "Average Epsilon by Voting Method",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot bar chart of average epsilon with 95% CI error bars (cluster-aware).
    
    Args:
        clustered_results: Dict mapping method to list of lists (outer reps → inner samples)
        title: Plot title
        output_path: Path to save figure (None = show)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    means = []
    cis = []
    colors = []
    n_clusters_list = []
    
    # Use custom order for bar plots
    for method in BARPLOT_METHOD_ORDER:
        clusters = clustered_results.get(method, [])
        if clusters:
            mean, ci, n_clusters = compute_cluster_ci(clusters)
            if mean is not None:
                methods.append(METHOD_NAMES.get(method, method))
                means.append(mean)
                cis.append(ci if ci is not None else 0)
                colors.append(METHOD_COLORS.get(method, "#333333"))
                n_clusters_list.append(n_clusters)
    
    if not methods:
        logger.warning("No data to plot")
        return
    
    x = np.arange(len(methods))
    
    # Asymmetric error bars: clip to [0, 1] (epsilon ∈ [0, 1])
    lower_errors = [min(ci, mean) for mean, ci in zip(means, cis)]  # Can't go below 0
    upper_errors = [min(ci, 1 - mean) for mean, ci in zip(means, cis)]  # Can't go above 1
    yerr = [lower_errors, upper_errors]
    
    bars = ax.bar(x, means, yerr=yerr, capsize=5, color=colors, alpha=0.8)
    
    ax.set_xlabel("Voting Method", fontsize=12)
    ax.set_ylabel("Average Epsilon (ε)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, ci in zip(bars, means, cis):
        height = bar.get_height()
        ax.annotate(
            f'{mean:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=9
        )
    
    # Add note about error bars
    n_clusters = n_clusters_list[0] if n_clusters_list else 0
    ax.text(0.02, 0.98, f"Error bars: 95% CI (n={n_clusters} outer reps)",
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            color='gray')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved bar plot to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_epsilon_stripplot(
    results: Dict[str, List[float]],
    title: str = "Epsilon Distribution by Voting Method",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot horizontal strip plot of epsilon values with methods as rows.
    
    Args:
        results: Dict mapping method to list of epsilon values
        title: Plot title
        output_path: Path to save figure (None = show)
    """
    # Prepare data for seaborn
    data = []
    for method in BARPLOT_METHOD_ORDER:
        values = results.get(method, [])
        display_name = METHOD_NAMES.get(method, method)
        for v in values:
            if v is not None and v >= 0:  # Filter out sentinel values
                data.append({"Method": display_name, "Epsilon": v, "method_key": method})
    
    if not data:
        logger.warning("No epsilon values to plot")
        return
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create color palette based on method order (use custom order, but only include methods that have data)
    method_order = [METHOD_NAMES.get(m, m) for m in BARPLOT_METHOD_ORDER if METHOD_NAMES.get(m, m) in df["Method"].values]
    palette = {METHOD_NAMES.get(m, m): METHOD_COLORS.get(m, "#333333") for m in BARPLOT_METHOD_ORDER}
    
    # Plot strip plot with jitter
    sns.stripplot(
        data=df,
        x="Epsilon",
        y="Method",
        hue="Method",
        order=method_order,
        hue_order=method_order,
        palette=palette,
        jitter=0.3,
        alpha=0.6,
        size=4,
        ax=ax,
        legend=False
    )
    
    # Add mean markers
    for i, method in enumerate(method_order):
        method_data = df[df["Method"] == method]["Epsilon"]
        if len(method_data) > 0:
            mean_val = method_data.mean()
            ax.scatter([mean_val], [i], color='black', s=100, marker='|', linewidths=2, zorder=10)
    
    ax.set_xlabel("Epsilon (ε)", fontsize=12)
    ax.set_ylabel("Voting Method", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add note about sample size
    n_samples = len(df)
    ax.text(0.98, 0.02, f"n={n_samples} samples | bars = means",
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            horizontalalignment='right', color='gray')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved strip plot to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_epsilon_stripplot_by_topic(
    results_by_topic: Dict[str, List[float]],
    method: str,
    title: str = None,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot horizontal strip plot of epsilon values with topics as rows (for a single method).
    
    Args:
        results_by_topic: Dict mapping topic_slug to list of epsilon values
        method: Voting method name (for title)
        title: Plot title (optional)
        output_path: Path to save figure (None = show)
    """
    # Prepare data for seaborn
    data = []
    for topic_slug, values in results_by_topic.items():
        display_name = TOPIC_DISPLAY_NAMES.get(topic_slug, topic_slug[:30])
        for v in values:
            if v is not None and v >= 0:  # Filter out sentinel values
                data.append({"Topic": display_name, "Epsilon": v})
    
    if not data:
        logger.warning(f"No epsilon values to plot for {method}")
        return
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Create figure - adjust height based on number of topics
    n_topics = df["Topic"].nunique()
    fig_height = max(4, n_topics * 0.6)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Get topic order (sorted alphabetically)
    topic_order = sorted(df["Topic"].unique())
    
    method_display = METHOD_NAMES.get(method, method)
    method_color = METHOD_COLORS.get(method, "#333333")
    
    # Plot strip plot with jitter
    sns.stripplot(
        data=df,
        x="Epsilon",
        y="Topic",
        order=topic_order,
        color=method_color,
        jitter=0.3,
        alpha=0.6,
        size=4,
        ax=ax
    )
    
    # Add mean markers
    for i, topic in enumerate(topic_order):
        topic_data = df[df["Topic"] == topic]["Epsilon"]
        if len(topic_data) > 0:
            mean_val = topic_data.mean()
            ax.scatter([mean_val], [i], color='black', s=100, marker='|', linewidths=2, zorder=10)
    
    ax.set_xlabel("Epsilon (ε)", fontsize=12)
    ax.set_ylabel("Topic", fontsize=12)
    ax.set_title(title or f"Epsilon by Topic: {method_display}", fontsize=14)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add note about sample size
    n_samples = len(df)
    ax.text(0.98, 0.02, f"n={n_samples} samples | bars = means",
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            horizontalalignment='right', color='gray')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved topic strip plot to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_epsilon_stripplot_comparison(
    output_dir: Path = OUTPUT_DIR,
    method: str = "chatgpt",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot horizontal strip plot comparing filtering vs no_filtering for selected topics.
    
    Creates a plot with 4 topics (Campus Speech, Free Speech, Littering, Environment),
    each with 2 mini-rows showing no_filtering (top) and filtering (bottom).
    Campus Speech and Free Speech are colored blue, Littering and Environment are red.
    
    Args:
        output_dir: Output directory containing experiment data
        method: Voting method to display (default: "chatgpt")
        output_path: Path to save figure (None = show)
    """
    import pandas as pd
    
    # Define topics with their slugs, display names, and colors
    topics_config = [
        ("what-are-your-thoughts-on-the-way-university-campu", "Campus Speech", "#2563eb"),  # blue
        ("what-limits-if-any-should-exist-on-free-speech-reg", "Free Speech", "#2563eb"),     # blue
        ("what-are-the-best-policies-to-prevent-littering-in", "Littering", "#dc2626"),       # red
    ]
    
    # Collect data for each topic and ablation
    data = []
    row_order = []
    row_colors = {}
    
    for topic_slug, display_name, color in topics_config:
        for ablation, ablation_label in [("no_filtering", "no filtering"), ("full", "filtering")]:
            row_label = f"{display_name} ({ablation_label})"
            row_order.append(row_label)
            row_colors[row_label] = color
            
            # Collect results for this topic and ablation
            topic_results = collect_results_for_topic(topic_slug, output_dir, ablation)
            method_values = topic_results.get(method, [])
            
            for v in method_values:
                if v is not None and v >= 0:
                    data.append({
                        "Row": row_label,
                        "Epsilon": v,
                        "Color": color
                    })
    
    if not data:
        logger.warning(f"No epsilon values to plot for comparison")
        return
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each row separately to control colors
    for i, row_label in enumerate(row_order):
        row_data = df[df["Row"] == row_label]
        if len(row_data) > 0:
            color = row_colors[row_label]
            ax.scatter(
                row_data["Epsilon"],
                [i] * len(row_data) + np.random.uniform(-0.2, 0.2, len(row_data)),
                color=color,
                alpha=0.6,
                s=20,
                zorder=5
            )
            # Add mean marker
            mean_val = row_data["Epsilon"].mean()
            ax.scatter([mean_val], [i], color='black', s=100, marker='|', linewidths=2, zorder=10)
    
    # Set y-axis labels with colored text
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order)
    
    # Color the y-axis tick labels
    for i, label in enumerate(ax.get_yticklabels()):
        label.set_color(row_colors[row_order[i]])
    
    # Add horizontal lines to separate topic groups
    n_topics = len(topics_config)
    for i in range(1, n_topics):
        ax.axhline(y=i * 2 - 0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    method_display = METHOD_NAMES.get(method, method)
    ax.set_xlabel("Epsilon (ε)", fontsize=12)
    ax.set_ylabel("")
    ax.set_title(f"Epsilon Comparison: Filtering vs No Filtering ({method_display})", fontsize=14)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Invert y-axis so first topic is at top
    ax.invert_yaxis()
    
    # Add note about sample size
    n_samples = len(df)
    ax.text(0.98, 0.02, f"n={n_samples} samples | bars = means",
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            horizontalalignment='right', color='gray')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison strip plot to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_likert_stripplot_by_topic(
    results_by_topic: Dict[str, List[float]],
    method: str,
    title: str = None,
    output_path: Optional[Path] = None
) -> None:
    """
    Plot horizontal strip plot of Likert ratings with topics as rows (for a single method).
    
    Args:
        results_by_topic: Dict mapping topic_slug to list of Likert ratings
        method: Voting method name (for title)
        title: Plot title (optional)
        output_path: Path to save figure (None = show)
    """
    # Prepare data for seaborn
    data = []
    for topic_slug, values in results_by_topic.items():
        display_name = TOPIC_DISPLAY_NAMES.get(topic_slug, topic_slug[:30])
        for v in values:
            if v is not None:
                data.append({"Topic": display_name, "Likert": v})
    
    if not data:
        logger.warning(f"No Likert values to plot for {method}")
        return
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Create figure - adjust height based on number of topics
    n_topics = df["Topic"].nunique()
    fig_height = max(4, n_topics * 0.6)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Get topic order (sorted alphabetically)
    topic_order = sorted(df["Topic"].unique())
    
    method_display = METHOD_NAMES.get(method, method)
    method_color = METHOD_COLORS.get(method, "#333333")
    
    # Plot strip plot with jitter
    sns.stripplot(
        data=df,
        x="Likert",
        y="Topic",
        order=topic_order,
        color=method_color,
        jitter=0.3,
        alpha=0.6,
        size=4,
        ax=ax
    )
    
    # Add mean markers
    for i, topic in enumerate(topic_order):
        topic_data = df[df["Topic"] == topic]["Likert"]
        if len(topic_data) > 0:
            mean_val = topic_data.mean()
            ax.scatter([mean_val], [i], color='black', s=100, marker='|', linewidths=2, zorder=10)
    
    ax.set_xlabel("Likert Rating (1-5)", fontsize=12)
    ax.set_ylabel("Topic", fontsize=12)
    ax.set_title(title or f"Likert by Topic: {method_display}", fontsize=14)
    ax.set_xlim(1, 5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add note about sample size
    n_samples = len(df)
    ax.text(0.98, 0.02, f"n={n_samples} samples | bars = means",
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            horizontalalignment='right', color='gray')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved Likert topic strip plot to {output_path}")
        plt.close()
    else:
        plt.show()


def collect_likert_for_topic(
    topic_slug: str,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full"
) -> Dict[str, List[float]]:
    """
    Collect average Likert ratings for winners of each voting method.
    
    Args:
        topic_slug: Topic slug
        output_dir: Output directory
        ablation: Ablation type
    
    Returns:
        Dict mapping method name to list of average Likert ratings
    """
    results = {method: [] for method in VOTING_METHODS}
    
    topic_dir = output_dir / "data" / topic_slug
    
    if not topic_dir.exists():
        return results
    
    for rep_dir in sorted(topic_dir.glob("rep*")):
        if ablation != "full":
            data_dir = rep_dir / f"ablation_{ablation}"
        else:
            data_dir = rep_dir
        
        if not data_dir.exists():
            continue
        
        # Load Likert ratings - path depends on ablation type
        if ablation == "full":
            likert_file = data_dir / "filtered_likert.json"
        elif ablation == "no_bridging":
            likert_file = data_dir / "full_likert.json"
        elif ablation == "no_filtering":
            likert_file = rep_dir / "full_likert.json"
        else:
            likert_file = data_dir / "filtered_likert.json"
        
        if not likert_file.exists():
            continue
        
        with open(likert_file, 'r') as f:
            likert = json.load(f)
        
        # Iterate through samples
        for sample_dir in sorted(data_dir.glob("sample*")):
            results_file = sample_dir / "results.json"
            if not results_file.exists():
                continue
            
            with open(results_file, 'r') as f:
                sample_results = json.load(f)
            
            # Load sampled persona indices
            persona_file = sample_dir / "persona_indices.json"
            if not persona_file.exists():
                continue
            
            with open(persona_file, 'r') as f:
                persona_indices = json.load(f)
            
            for method in VOTING_METHODS:
                if method in sample_results:
                    winner = sample_results[method].get("winner")
                    if winner is not None:
                        winner_idx = int(winner)
                        # #region agent log
                        _log_path = Path("/home/ec2-user/single-winner-generative-social-choice/.cursor/debug.log")
                        _likert_rows = len(likert)
                        _likert_cols = len(likert[0]) if likert else 0
                        _max_p_idx = max(persona_indices) if persona_indices else -1
                        _debug_data = {"topic": topic_slug, "ablation": ablation, "rep": rep_dir.name, "sample": sample_dir.name, "method": method, "winner_idx": winner_idx, "likert_rows": _likert_rows, "likert_cols": _likert_cols, "max_persona_idx": _max_p_idx, "likert_file": str(likert_file)}
                        if winner_idx >= _likert_cols or _max_p_idx >= _likert_rows:
                            with open(_log_path, 'a') as _f: _f.write(json.dumps({"hypothesisId": "A-E", "location": "visualizer.py:920", "message": "INDEX_OUT_OF_BOUNDS_SKIPPED", "data": _debug_data, "timestamp": int(__import__('time').time()*1000)}) + "\n")
                            continue  # Skip this entry - data mismatch
                        # #endregion
                        # Average Likert rating for winner across sampled personas
                        ratings = [likert[p_idx][winner_idx] for p_idx in persona_indices]
                        avg_rating = np.mean(ratings)
                        results[method].append(avg_rating)
    
    return results


def collect_likert_clustered_for_topic(
    topic_slug: str,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full"
) -> Dict[str, List[List[float]]]:
    """
    Collect Likert ratings preserving the hierarchical structure (outer reps → inner samples).
    
    Args:
        topic_slug: Topic slug
        output_dir: Output directory
        ablation: Ablation type
    
    Returns:
        Dict mapping method name to list of lists: outer list = outer reps, inner list = samples within rep
    """
    results = {method: [] for method in VOTING_METHODS}
    
    topic_dir = output_dir / "data" / topic_slug
    
    if not topic_dir.exists():
        return results
    
    for rep_dir in sorted(topic_dir.glob("rep*")):
        if ablation != "full":
            data_dir = rep_dir / f"ablation_{ablation}"
        else:
            data_dir = rep_dir
        
        if not data_dir.exists():
            continue
        
        # Load Likert ratings - path depends on ablation type
        if ablation == "full":
            likert_file = data_dir / "filtered_likert.json"
        elif ablation == "no_bridging":
            likert_file = data_dir / "full_likert.json"
        elif ablation == "no_filtering":
            likert_file = rep_dir / "full_likert.json"
        else:
            likert_file = data_dir / "filtered_likert.json"
        
        if not likert_file.exists():
            continue
        
        with open(likert_file, 'r') as f:
            likert = json.load(f)
        
        # Collect all samples for this rep
        rep_results = {method: [] for method in VOTING_METHODS}
        
        for sample_dir in sorted(data_dir.glob("sample*")):
            results_file = sample_dir / "results.json"
            if not results_file.exists():
                continue
            
            with open(results_file, 'r') as f:
                sample_results = json.load(f)
            
            # Load sampled persona indices
            persona_file = sample_dir / "persona_indices.json"
            if not persona_file.exists():
                continue
            
            with open(persona_file, 'r') as f:
                persona_indices = json.load(f)
            
            for method in VOTING_METHODS:
                if method in sample_results:
                    winner = sample_results[method].get("winner")
                    if winner is not None:
                        winner_idx = int(winner)
                        # #region agent log
                        _log_path = Path("/home/ec2-user/single-winner-generative-social-choice/.cursor/debug.log")
                        _likert_rows = len(likert)
                        _likert_cols = len(likert[0]) if likert else 0
                        _max_p_idx = max(persona_indices) if persona_indices else -1
                        _debug_data = {"topic": topic_slug, "ablation": ablation, "rep": rep_dir.name, "sample": sample_dir.name, "method": method, "winner_idx": winner_idx, "likert_rows": _likert_rows, "likert_cols": _likert_cols, "max_persona_idx": _max_p_idx, "likert_file": str(likert_file)}
                        if winner_idx >= _likert_cols or _max_p_idx >= _likert_rows:
                            with open(_log_path, 'a') as _f: _f.write(json.dumps({"hypothesisId": "A-E", "location": "visualizer.py:clustered", "message": "INDEX_OUT_OF_BOUNDS_SKIPPED", "data": _debug_data, "timestamp": int(__import__('time').time()*1000)}) + "\n")
                            continue  # Skip this entry - data mismatch
                        # #endregion
                        ratings = [likert[p_idx][winner_idx] for p_idx in persona_indices]
                        avg_rating = np.mean(ratings)
                        rep_results[method].append(avg_rating)
        
        # Add this rep's results to the main results
        for method in VOTING_METHODS:
            if rep_results[method]:
                results[method].append(rep_results[method])
    
    return results


def collect_all_likert_clustered(
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full",
    topics: Optional[List[str]] = None
) -> Dict[str, List[List[float]]]:
    """
    Collect all Likert results across topics, preserving cluster structure.
    
    Each outer rep across all topics is treated as one cluster.
    
    Args:
        output_dir: Output directory
        ablation: Ablation type
        topics: List of topics to include (None = all)
    
    Returns:
        Dict mapping method name to list of lists (clusters)
    """
    all_results = {method: [] for method in VOTING_METHODS}
    
    data_dir = output_dir / "data"
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return all_results
    
    # Get all topic directories
    if topics is None:
        topic_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    else:
        topic_dirs = [data_dir / t for t in topics if (data_dir / t).exists()]
    
    for topic_dir in topic_dirs:
        topic_results = collect_likert_clustered_for_topic(
            topic_dir.name, output_dir, ablation
        )
        for method in VOTING_METHODS:
            all_results[method].extend(topic_results[method])
    
    return all_results


def plot_likert_barplot(
    clustered_results: Dict[str, List[List[float]]],
    title: str = "Average Likert Rating by Voting Method",
    output_path: Optional[Path] = None,
    highlight_vbc_lower_ci: bool = False
) -> None:
    """
    Plot bar chart of average Likert ratings with 95% CI error bars (cluster-aware).
    
    Args:
        clustered_results: Dict mapping method to list of lists (outer reps → inner samples)
        title: Plot title
        output_path: Path to save figure (None = show)
        highlight_vbc_lower_ci: If True, draw a red horizontal line at the lower CI of Veto by Consumption
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    means = []
    cis = []
    colors = []
    n_clusters_list = []
    method_keys = []  # Store method keys to find VBC later
    
    # Use custom order for bar plots
    for method in BARPLOT_METHOD_ORDER:
        clusters = clustered_results.get(method, [])
        if clusters:
            mean, ci, n_clusters = compute_cluster_ci(clusters)
            if mean is not None:
                methods.append(METHOD_NAMES.get(method, method))
                means.append(mean)
                cis.append(ci if ci is not None else 0)
                colors.append(METHOD_COLORS.get(method, "#333333"))
                n_clusters_list.append(n_clusters)
                method_keys.append(method)
    
    if not methods:
        logger.warning("No data to plot")
        return
    
    x = np.arange(len(methods))
    
    # Asymmetric error bars: clip lower bound at 1 (Likert scale is 1-5)
    lower_errors = [min(ci, mean - 1) for mean, ci in zip(means, cis)]  # Can't go below 1
    upper_errors = [min(ci, 5 - mean) for mean, ci in zip(means, cis)]  # Can't go above 5
    yerr = [lower_errors, upper_errors]
    
    bars = ax.bar(x, means, yerr=yerr, capsize=5, color=colors, alpha=0.8)
    
    # Add red line at lower CI of Veto by Consumption if requested
    if highlight_vbc_lower_ci:
        vbc_method = "veto_by_consumption"
        if vbc_method in method_keys:
            vbc_idx = method_keys.index(vbc_method)
            vbc_mean = means[vbc_idx]
            vbc_ci = cis[vbc_idx]
            vbc_lower_ci = max(1, vbc_mean - vbc_ci)  # Clip at 1 (minimum Likert value)
            ax.axhline(y=vbc_lower_ci, color='red', linestyle='--', linewidth=2, 
                      label=f'VBC Lower CI: {vbc_lower_ci:.2f}', zorder=10)
            ax.legend(loc='best')
    
    ax.set_xlabel("Voting Method", fontsize=12)
    ax.set_ylabel("Average Likert Rating (1-5)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylim(1, 5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.annotate(
            f'{mean:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=9
        )
    
    # Add note about error bars
    n_clusters = n_clusters_list[0] if n_clusters_list else 0
    ax.text(0.02, 0.98, f"Error bars: 95% CI (n={n_clusters} outer reps)",
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            color='gray')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved Likert bar plot to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_likert_stripplot(
    results: Dict[str, List[float]],
    title: str = "Likert Rating Distribution by Voting Method",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot horizontal strip plot of Likert ratings with methods as rows.
    
    Args:
        results: Dict mapping method to list of Likert ratings
        title: Plot title
        output_path: Path to save figure (None = show)
    """
    # Prepare data for seaborn
    data = []
    for method in BARPLOT_METHOD_ORDER:
        values = results.get(method, [])
        display_name = METHOD_NAMES.get(method, method)
        for v in values:
            if v is not None:
                data.append({"Method": display_name, "Likert": v, "method_key": method})
    
    if not data:
        logger.warning("No Likert values to plot")
        return
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create color palette based on method order (use custom order, but only include methods that have data)
    method_order = [METHOD_NAMES.get(m, m) for m in BARPLOT_METHOD_ORDER if METHOD_NAMES.get(m, m) in df["Method"].values]
    palette = {METHOD_NAMES.get(m, m): METHOD_COLORS.get(m, "#333333") for m in BARPLOT_METHOD_ORDER}
    
    # Plot strip plot with jitter
    sns.stripplot(
        data=df,
        x="Likert",
        y="Method",
        hue="Method",
        order=method_order,
        hue_order=method_order,
        palette=palette,
        jitter=0.3,
        alpha=0.6,
        size=4,
        ax=ax,
        legend=False
    )
    
    # Add mean markers
    for i, method in enumerate(method_order):
        method_data = df[df["Method"] == method]["Likert"]
        if len(method_data) > 0:
            mean_val = method_data.mean()
            ax.scatter([mean_val], [i], color='black', s=100, marker='|', linewidths=2, zorder=10)
    
    ax.set_xlabel("Likert Rating (1-5)", fontsize=12)
    ax.set_ylabel("Voting Method", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(1, 5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add note about sample size
    n_samples = len(df)
    ax.text(0.98, 0.02, f"n={n_samples} samples | bars = means",
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            horizontalalignment='right', color='gray')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved Likert strip plot to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_likert_histogram(
    results: Dict[str, List[float]],
    title: str = "Likert Rating Distribution by Voting Method",
    output_path: Optional[Path] = None,
    bins: int = 20
) -> None:
    """
    Plot stacked histogram of Likert ratings for each voting method.
    
    Args:
        results: Dict mapping method to list of Likert ratings
        title: Plot title
        output_path: Path to save figure (None = show)
        bins: Number of histogram bins
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get all values to determine range
    all_values = []
    for values in results.values():
        all_values.extend([v for v in values if v is not None])
    
    if not all_values:
        logger.warning("No Likert values to plot")
        return
    
    # Fixed range 1 to 5 for Likert scale
    bin_edges = np.linspace(1, 5, bins + 1)
    
    # Collect data for stacked histogram
    data_to_plot = []
    labels = []
    colors = []
    for method in VOTING_METHODS:
        values = [v for v in results.get(method, []) if v is not None]
        if values:
            data_to_plot.append(values)
            labels.append(METHOD_NAMES.get(method, method))
            colors.append(METHOD_COLORS.get(method, None))
    
    # Plot stacked histogram
    if data_to_plot:
        ax.hist(
            data_to_plot,
            bins=bin_edges,
            label=labels,
            color=colors,
            stacked=True,
            edgecolor='white',
            linewidth=0.5
        )
    
    ax.set_xlabel("Likert Rating", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(1, 5)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved Likert histogram to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_single_method_likert_histogram(
    values: List[float],
    method: str,
    title: str = None,
    output_path: Optional[Path] = None,
    bins: int = 20
) -> None:
    """
    Plot histogram of Likert ratings for a single voting method.
    
    Args:
        values: List of Likert ratings
        method: Method name (for color and default title)
        title: Plot title (optional)
        output_path: Path to save figure (None = show)
        bins: Number of histogram bins
    """
    clean_values = [v for v in values if v is not None]
    
    if not clean_values:
        logger.warning(f"No Likert values for {method}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    color = METHOD_COLORS.get(method, "#1f77b4")
    display_name = METHOD_NAMES.get(method, method)
    
    # Fixed range 1 to 5 for Likert scale
    bin_edges = np.linspace(1, 5, bins + 1)
    
    ax.hist(
        clean_values,
        bins=bin_edges,
        color=color,
        edgecolor='white',
        linewidth=0.5,
        alpha=0.8
    )
    
    ax.set_xlabel("Likert Rating", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title or f"Likert Distribution: {display_name}", fontsize=14)
    ax.set_xlim(1, 5)
    ax.grid(True, alpha=0.3)
    
    # Add stats annotation
    mean_val = np.mean(clean_values)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_cluster_size_stripplot(
    results: Dict[str, List[tuple]],
    title: str = "Cluster Size Distribution by Voting Method",
    output_path: Optional[Path] = None,
    use_log_scale: bool = False
) -> None:
    """
    Plot horizontal strip plot of cluster sizes with methods as rows.
    Red dots indicate clusters containing the winner, blue dots indicate other clusters.
    
    Args:
        results: Dict mapping method to list of tuples: (cluster_size, is_winner_cluster)
        title: Plot title
        output_path: Path to save figure (None = show)
        use_log_scale: If True, use logarithmic x-axis
    """
    # Prepare data for seaborn
    data = []
    for method in BARPLOT_METHOD_ORDER:
        values = results.get(method, [])
        display_name = METHOD_NAMES.get(method, method)
        for cluster_size, is_winner in values:
            data.append({
                "Method": display_name,
                "ClusterSize": cluster_size,
                "IsWinner": is_winner,
                "method_key": method
            })
    
    if not data:
        logger.warning("No cluster size values to plot")
        return
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get method order
    method_order = [METHOD_NAMES.get(m, m) for m in BARPLOT_METHOD_ORDER if METHOD_NAMES.get(m, m) in df["Method"].values]
    
    # Plot strip plot with jitter
    # We'll plot winner and non-winner clusters separately to control colors
    winner_data = df[df["IsWinner"] == True]
    other_data = df[df["IsWinner"] == False]
    
    if len(other_data) > 0:
        sns.stripplot(
            data=other_data,
            x="ClusterSize",
            y="Method",
            order=method_order,
            color='blue',
            alpha=0.6,
            jitter=0.3,
            size=4,
            ax=ax,
            label='Other clusters'
        )
    
    if len(winner_data) > 0:
        sns.stripplot(
            data=winner_data,
            x="ClusterSize",
            y="Method",
            order=method_order,
            color='red',
            alpha=0.6,
            jitter=0.3,
            size=4,
            ax=ax,
            label='Winner cluster'
        )
    
    ax.set_xlabel("Cluster Size", fontsize=12)
    ax.set_ylabel("Voting Method", fontsize=12)
    ax.set_title(title, fontsize=14)
    if use_log_scale:
        ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend with only one blue and one red entry
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.6, label='Other clusters'),
        Patch(facecolor='red', alpha=0.6, label='Winner cluster')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    # Add note about sample size
    n_samples = len(df)
    n_winner = len(winner_data)
    ax.text(0.98, 0.02, f"n={n_samples} clusters (n={n_winner} winner clusters)",
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            horizontalalignment='right', color='gray')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved cluster size strip plot to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_cluster_size_violinplot(
    results: Dict[str, List[tuple]],
    title: str = "Cluster Size Distribution by Voting Method",
    output_path: Optional[Path] = None,
    use_log_scale: bool = False
) -> None:
    """
    Plot violin plot of cluster sizes with methods as rows.
    For each method, two violins side by side: blue for other clusters, red for winner clusters.
    
    Args:
        results: Dict mapping method to list of tuples: (cluster_size, is_winner_cluster)
        title: Plot title
        output_path: Path to save figure (None = show)
        use_log_scale: If True, use logarithmic x-axis
    """
    # Prepare data for seaborn
    data = []
    for method in BARPLOT_METHOD_ORDER:
        values = results.get(method, [])
        display_name = METHOD_NAMES.get(method, method)
        for cluster_size, is_winner in values:
            data.append({
                "Method": display_name,
                "ClusterSize": cluster_size,
                "IsWinner": is_winner,
                "ClusterType": "Winner cluster" if is_winner else "Other clusters",
                "method_key": method
            })
    
    if not data:
        logger.warning("No cluster size values to plot")
        return
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Get method order
    method_order = [METHOD_NAMES.get(m, m) for m in BARPLOT_METHOD_ORDER if METHOD_NAMES.get(m, m) in df["Method"].values]
    
    # Create figure with more width to accommodate side-by-side violins
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot violin plot with hue to create side-by-side violins
    sns.violinplot(
        data=df,
        x="ClusterSize",
        y="Method",
        hue="ClusterType",
        order=method_order,
        hue_order=["Other clusters", "Winner cluster"],
        palette={"Other clusters": "blue", "Winner cluster": "red"},
        alpha=0.6,
        ax=ax,
        inner="quart",  # Show quartiles inside violins
        cut=0  # Clip KDE at data bounds (prevents extending below 0)
    )
    
    ax.set_xlabel("Cluster Size", fontsize=12)
    ax.set_ylabel("Voting Method", fontsize=12)
    ax.set_title(title, fontsize=14)
    if use_log_scale:
        ax.set_xscale('log')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend outside the plot with only one blue and one red entry
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.6, label='Other clusters'),
        Patch(facecolor='red', alpha=0.6, label='Winner cluster')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add note about sample size
    n_samples = len(df)
    n_winner = len(df[df["IsWinner"] == True])
    ax.text(0.98, 0.02, f"n={n_samples} clusters (n={n_winner} winner clusters)",
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            horizontalalignment='right', color='gray')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved cluster size violin plot to {output_path}")
        plt.close()
    else:
        plt.show()


# =============================================================================
# Multi-Persona Collection and Plotting Functions
# =============================================================================

# Standard persona counts to support
PERSONA_COUNTS = [5, 10, 20]


def get_sample_results_dir_for_n_personas(
    rep_dir: Path,
    ablation: str,
    n_personas: int
) -> Path:
    """
    Get the directory containing sample results for a given ablation and persona count.
    
    Args:
        rep_dir: Path to the rep directory (e.g., data/topic/rep0)
        ablation: Ablation type ('full', 'no_filtering', 'no_bridging')
        n_personas: Number of personas (5, 10, or 20)
    
    Returns:
        Path to the directory containing sample subdirectories
    """
    if ablation == "full":
        base_dir = rep_dir
    elif ablation == "no_filtering":
        base_dir = rep_dir / "ablation_no_filtering"
    elif ablation == "no_bridging":
        base_dir = rep_dir / "ablation_no_bridging"
    else:
        raise ValueError(f"Unknown ablation: {ablation}")
    
    if n_personas == 20:
        # Standard 20 personas go directly in base_dir
        return base_dir
    else:
        # Other counts go in subdirectories
        return base_dir / f"{n_personas}-personas"


def collect_results_for_n_personas_topic(
    topic_slug: str,
    n_personas: int,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full"
) -> Dict[str, List[float]]:
    """
    Collect epsilon results for a specific persona count for a topic.
    
    Args:
        topic_slug: Topic slug
        n_personas: Number of personas (5, 10, or 20)
        output_dir: Output directory
        ablation: Ablation type
    
    Returns:
        Dict mapping method name to list of epsilon values
    """
    results = {method: [] for method in VOTING_METHODS}
    
    topic_dir = output_dir / "data" / topic_slug
    
    if not topic_dir.exists():
        logger.warning(f"Topic directory not found: {topic_dir}")
        return results
    
    # Iterate through all rep directories
    for rep_dir in sorted(topic_dir.glob("rep*")):
        # Get sample results directory for this persona count
        sample_base = get_sample_results_dir_for_n_personas(rep_dir, ablation, n_personas)
        
        if not sample_base.exists():
            continue
        
        # Iterate through all sample directories
        for sample_dir in sorted(sample_base.glob("sample*")):
            results_file = sample_dir / "results.json"
            if not results_file.exists():
                continue
            
            with open(results_file, 'r') as f:
                sample_results = json.load(f)
            
            for method in VOTING_METHODS:
                if method in sample_results:
                    epsilon = sample_results[method].get("epsilon")
                    if epsilon is not None:
                        results[method].append(epsilon)
    
    return results


def collect_results_for_n_personas_clustered_topic(
    topic_slug: str,
    n_personas: int,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full"
) -> Dict[str, List[List[float]]]:
    """
    Collect epsilon results clustered by repetition for a specific persona count.
    
    Args:
        topic_slug: Topic slug
        n_personas: Number of personas (5, 10, or 20)
        output_dir: Output directory
        ablation: Ablation type
    
    Returns:
        Dict mapping method name to list of lists (outer: reps, inner: samples)
    """
    results = {method: [] for method in VOTING_METHODS}
    
    topic_dir = output_dir / "data" / topic_slug
    
    if not topic_dir.exists():
        logger.warning(f"Topic directory not found: {topic_dir}")
        return results
    
    # Iterate through all rep directories
    for rep_dir in sorted(topic_dir.glob("rep*")):
        # Get sample results directory for this persona count
        sample_base = get_sample_results_dir_for_n_personas(rep_dir, ablation, n_personas)
        
        if not sample_base.exists():
            continue
        
        # Collect all samples for this rep
        rep_results = {method: [] for method in VOTING_METHODS}
        
        for sample_dir in sorted(sample_base.glob("sample*")):
            results_file = sample_dir / "results.json"
            if not results_file.exists():
                continue
            
            with open(results_file, 'r') as f:
                sample_results = json.load(f)
            
            for method in VOTING_METHODS:
                if method in sample_results:
                    epsilon = sample_results[method].get("epsilon")
                    if epsilon is not None:
                        rep_results[method].append(epsilon)
        
        # Add this rep's results to the clustered results
        for method in VOTING_METHODS:
            if rep_results[method]:
                results[method].append(rep_results[method])
    
    return results


def collect_all_results_for_n_personas(
    n_personas: int,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full",
    topics: Optional[List[str]] = None
) -> Dict[str, List[float]]:
    """
    Collect all epsilon results for a specific persona count across all topics.
    
    Args:
        n_personas: Number of personas (5, 10, or 20)
        output_dir: Output directory
        ablation: Ablation type
        topics: List of topics to include (None = all)
    
    Returns:
        Dict mapping method name to list of epsilon values
    """
    all_results = {method: [] for method in VOTING_METHODS}
    
    data_dir = output_dir / "data"
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return all_results
    
    # Get all topic directories
    if topics is None:
        topic_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    else:
        topic_dirs = [data_dir / t for t in topics if (data_dir / t).exists()]
    
    for topic_dir in topic_dirs:
        topic_results = collect_results_for_n_personas_topic(
            topic_dir.name, n_personas, output_dir, ablation
        )
        for method in VOTING_METHODS:
            all_results[method].extend(topic_results[method])
    
    return all_results


def collect_all_results_for_n_personas_clustered(
    n_personas: int,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full",
    topics: Optional[List[str]] = None
) -> Dict[str, List[List[float]]]:
    """
    Collect all epsilon results clustered by repetition for a specific persona count.
    
    Args:
        n_personas: Number of personas (5, 10, or 20)
        output_dir: Output directory
        ablation: Ablation type
        topics: List of topics to include (None = all)
    
    Returns:
        Dict mapping method name to list of lists (outer: reps, inner: samples)
    """
    all_results = {method: [] for method in VOTING_METHODS}
    
    data_dir = output_dir / "data"
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return all_results
    
    # Get all topic directories
    if topics is None:
        topic_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    else:
        topic_dirs = [data_dir / t for t in topics if (data_dir / t).exists()]
    
    for topic_dir in topic_dirs:
        topic_results = collect_results_for_n_personas_clustered_topic(
            topic_dir.name, n_personas, output_dir, ablation
        )
        for method in VOTING_METHODS:
            all_results[method].extend(topic_results[method])
    
    return all_results


def plot_epsilon_multi_persona_barplot(
    results_by_n_personas: Dict[int, Dict[str, List[List[float]]]],
    title: str = "Average Epsilon by Persona Count",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot bar chart comparing average epsilon across different persona counts.
    
    Args:
        results_by_n_personas: Dict mapping n_personas to clustered results
        title: Plot title
        output_path: Path to save figure (None = show)
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Get persona counts and methods
    persona_counts = sorted(results_by_n_personas.keys())
    n_groups = len(BARPLOT_METHOD_ORDER)
    n_bars = len(persona_counts)
    
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)
    
    # Colors for different persona counts
    persona_colors = {
        5: '#e74c3c',   # red
        10: '#f39c12',  # orange
        20: '#27ae60',  # green
    }
    
    for i, n_personas in enumerate(persona_counts):
        clustered_results = results_by_n_personas[n_personas]
        
        means = []
        cis = []
        
        for method in BARPLOT_METHOD_ORDER:
            clusters = clustered_results.get(method, [])
            if clusters:
                mean, ci, _ = compute_cluster_ci(clusters)
                means.append(mean if mean is not None else 0)
                cis.append(ci if ci is not None else 0)
            else:
                means.append(0)
                cis.append(0)
        
        # Asymmetric error bars
        lower_errors = [min(ci, mean) for mean, ci in zip(means, cis)]
        upper_errors = [min(ci, 1 - mean) for mean, ci in zip(means, cis)]
        yerr = [lower_errors, upper_errors]
        
        offset = (i - (n_bars - 1) / 2) * bar_width
        color = persona_colors.get(n_personas, '#333333')
        
        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=yerr,
            capsize=3,
            color=color,
            alpha=0.8,
            label=f'{n_personas} personas'
        )
    
    ax.set_xlabel("Voting Method", fontsize=12)
    ax.set_ylabel("Average Epsilon (ε)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_NAMES.get(m, m) for m in BARPLOT_METHOD_ORDER], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved multi-persona barplot to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_epsilon_multi_persona_stripplot(
    results_by_n_personas: Dict[int, Dict[str, List[float]]],
    title: str = "Epsilon Distribution by Persona Count",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot strip plot comparing epsilon distributions across different persona counts.
    
    Args:
        results_by_n_personas: Dict mapping n_personas to flat results
        title: Plot title
        output_path: Path to save figure (None = show)
    """
    import pandas as pd
    
    # Prepare data for seaborn
    data = []
    for n_personas, results in results_by_n_personas.items():
        for method in BARPLOT_METHOD_ORDER:
            values = results.get(method, [])
            display_name = METHOD_NAMES.get(method, method)
            for v in values:
                if v is not None and v >= 0:
                    data.append({
                        "Method": display_name,
                        "Epsilon": v,
                        "Personas": f"{n_personas}",
                        "method_key": method
                    })
    
    if not data:
        logger.warning("No epsilon values to plot")
        return
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get method order
    method_order = [METHOD_NAMES.get(m, m) for m in BARPLOT_METHOD_ORDER if METHOD_NAMES.get(m, m) in df["Method"].values]
    
    # Colors for different persona counts
    persona_palette = {
        '5': '#e74c3c',   # red
        '10': '#f39c12',  # orange
        '20': '#27ae60',  # green
    }
    
    # Plot strip plot with hue for persona counts
    sns.stripplot(
        data=df,
        x="Epsilon",
        y="Method",
        hue="Personas",
        order=method_order,
        hue_order=['5', '10', '20'],
        palette=persona_palette,
        jitter=0.3,
        alpha=0.6,
        size=4,
        ax=ax,
        dodge=True
    )
    
    ax.set_xlabel("Epsilon (ε)", fontsize=12)
    ax.set_ylabel("Voting Method", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(title="Personas", loc='lower right')
    
    # Add note about sample size
    n_samples = len(df)
    ax.text(0.98, 0.02, f"n={n_samples} total samples",
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            horizontalalignment='right', color='gray')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved multi-persona stripplot to {output_path}")
        plt.close()
    else:
        plt.show()


def generate_all_plots(
    output_dir: Path = OUTPUT_DIR,
    topics: Optional[List[str]] = None,
    ablations: Optional[List[str]] = None
) -> None:
    """
    Generate all plots for the experiment.
    
    Args:
        output_dir: Output directory
        topics: List of topics (None = auto-detect)
        ablations: List of ablations to plot
    """
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    if ablations is None:
        ablations = ["full"]
    
    for ablation in ablations:
        # Create subfolder for this ablation
        ablation_dir = figures_dir / ablation
        ablation_dir.mkdir(parents=True, exist_ok=True)
        
        ablation_label = f" ({ablation.replace('_', ' ')})" if ablation != "full" else ""
        
        # Aggregate plots in ablation/aggregate/
        aggregate_dir = ablation_dir / "aggregate"
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        
        # Flat results for histograms
        all_results = collect_all_results(output_dir, ablation, topics)
        # Clustered results for barplots (95% CI)
        all_results_clustered = collect_all_results_clustered(output_dir, ablation, topics)
        
        plot_epsilon_histogram(
            all_results,
            title=f"Epsilon Distribution by Voting Method{ablation_label}",
            output_path=aggregate_dir / "epsilon_histogram.png"
        )
        
        plot_epsilon_barplot(
            all_results_clustered,
            title=f"Average Epsilon by Voting Method{ablation_label}",
            output_path=aggregate_dir / "epsilon_barplot.png"
        )
        
        # Aggregate strip plot (methods as rows)
        plot_epsilon_stripplot(
            all_results,
            title=f"Epsilon Distribution by Voting Method{ablation_label}",
            output_path=aggregate_dir / "epsilon_stripplot.png"
        )
        
        # Collect epsilon by topic for per-method strip plots
        epsilon_by_topic = collect_epsilon_by_topic(output_dir, ablation, topics)
        
        # Per-method epsilon strip plots (topics as rows)
        for method in VOTING_METHODS:
            # Collect this method's values organized by topic
            method_by_topic = {
                topic_slug: topic_results.get(method, [])
                for topic_slug, topic_results in epsilon_by_topic.items()
            }
            # Filter out empty topics
            method_by_topic = {k: v for k, v in method_by_topic.items() if v}
            if method_by_topic:
                method_display = METHOD_NAMES.get(method, method)
                plot_epsilon_stripplot_by_topic(
                    method_by_topic,
                    method,
                    title=f"Epsilon by Topic: {method_display}{ablation_label}",
                    output_path=aggregate_dir / f"epsilon_stripplot_{method}.png"
                )
        
        # Per-method epsilon histograms for aggregate
        for method in VOTING_METHODS:
            method_values = all_results.get(method, [])
            if method_values:
                method_display = METHOD_NAMES.get(method, method)
                plot_single_method_histogram(
                    method_values,
                    method,
                    title=f"Epsilon: All Topics - {method_display}{ablation_label}",
                    output_path=aggregate_dir / f"epsilon_histogram_{method}.png"
                )
        
        # Collect all Likert results for aggregate plots
        all_likert_results = {}
        for method in VOTING_METHODS:
            all_likert_results[method] = []
        
        # Collect Likert by topic for per-method strip plots
        likert_by_topic = collect_likert_by_topic(output_dir, ablation, topics)
        
        # Aggregate Likert results from all topics
        for topic_slug, topic_results in likert_by_topic.items():
            for method in VOTING_METHODS:
                all_likert_results[method].extend(topic_results.get(method, []))
        
        # Clustered results for barplots (95% CI)
        all_likert_results_clustered = collect_all_likert_clustered(output_dir, ablation, topics)
        
        # Aggregate Likert bar plot
        plot_likert_barplot(
            all_likert_results_clustered,
            title=f"Average Likert Rating by Voting Method{ablation_label}",
            output_path=aggregate_dir / "likert_barplot.png"
        )
        
        # Special plot for no_bridging with red line at VBC lower CI
        if ablation == "no_bridging":
            plot_likert_barplot(
                all_likert_results_clustered,
                title=f"Average Likert Rating by Voting Method{ablation_label}",
                output_path=aggregate_dir / "likert_barplot_vbc_highlight.png",
                highlight_vbc_lower_ci=True
            )
        
        # Aggregate Likert strip plot (methods as rows)
        plot_likert_stripplot(
            all_likert_results,
            title=f"Likert Rating Distribution by Voting Method{ablation_label}",
            output_path=aggregate_dir / "likert_stripplot.png"
        )
        
        # Cluster size strip plot (only for full ablation)
        if ablation == "full":
            all_cluster_sizes = collect_all_cluster_sizes(output_dir, ablation, topics)
            plot_cluster_size_stripplot(
                all_cluster_sizes,
                title=f"Cluster Size Distribution by Voting Method{ablation_label}",
                output_path=aggregate_dir / "cluster_size_stripplot.png",
                use_log_scale=False
            )
            plot_cluster_size_stripplot(
                all_cluster_sizes,
                title=f"Cluster Size Distribution by Voting Method (Log Scale){ablation_label}",
                output_path=aggregate_dir / "cluster_size_stripplot_log.png",
                use_log_scale=True
            )
            plot_cluster_size_violinplot(
                all_cluster_sizes,
                title=f"Cluster Size Distribution by Voting Method{ablation_label}",
                output_path=aggregate_dir / "cluster_size_violinplot.png",
                use_log_scale=False
            )
            plot_cluster_size_violinplot(
                all_cluster_sizes,
                title=f"Cluster Size Distribution by Voting Method (Log Scale){ablation_label}",
                output_path=aggregate_dir / "cluster_size_violinplot_log.png",
                use_log_scale=True
            )
            
            # Winner cluster percentile ranking bar plot
            all_percentile_clustered = collect_all_winner_cluster_percentile_clustered(output_dir, ablation, topics)
            plot_winner_cluster_percentile_barplot(
                all_percentile_clustered,
                title=f"Winner Cluster Size Percentile Ranking by Voting Method{ablation_label}",
                output_path=aggregate_dir / "winner_cluster_percentile_barplot.png"
            )
        
        # Per-method Likert strip plots (topics as rows)
        for method in VOTING_METHODS:
            # Collect this method's values organized by topic
            method_by_topic = {
                topic_slug: topic_results.get(method, [])
                for topic_slug, topic_results in likert_by_topic.items()
            }
            # Filter out empty topics
            method_by_topic = {k: v for k, v in method_by_topic.items() if v}
            if method_by_topic:
                method_display = METHOD_NAMES.get(method, method)
                plot_likert_stripplot_by_topic(
                    method_by_topic,
                    method,
                    title=f"Likert by Topic: {method_display}{ablation_label}",
                    output_path=aggregate_dir / f"likert_stripplot_{method}.png"
                )
        
        # Per-topic plots in ablation/topic_short_name/
        data_dir = output_dir / "data"
        if topics is None and data_dir.exists():
            topics_to_plot = [d.name for d in data_dir.iterdir() if d.is_dir()]
        else:
            topics_to_plot = topics or []
        
        for topic in topics_to_plot:
            # Use short name for folder, display name for title
            short_name = TOPIC_SHORT_NAMES.get(topic, topic[:20])
            display_name = TOPIC_DISPLAY_NAMES.get(topic, topic[:50])
            
            # Create subfolder for this topic
            topic_dir = ablation_dir / short_name
            topic_dir.mkdir(parents=True, exist_ok=True)
            
            # Flat results for histograms
            topic_results = collect_results_for_topic(topic, output_dir, ablation)
            # Clustered results for barplots (95% CI)
            topic_results_clustered = collect_results_clustered_for_topic(topic, output_dir, ablation)
            
            plot_epsilon_histogram(
                topic_results,
                title=f"Epsilon Distribution: {display_name}{ablation_label}",
                output_path=topic_dir / "epsilon_histogram.png"
            )
            
            plot_epsilon_barplot(
                topic_results_clustered,
                title=f"Average Epsilon: {display_name}{ablation_label}",
                output_path=topic_dir / "epsilon_barplot.png"
            )
            
            # Per-topic epsilon strip plot (methods as rows)
            plot_epsilon_stripplot(
                topic_results,
                title=f"Epsilon Distribution: {display_name}{ablation_label}",
                output_path=topic_dir / "epsilon_stripplot.png"
            )
            
            # Per-method epsilon histograms
            for method in VOTING_METHODS:
                method_values = topic_results.get(method, [])
                if method_values:
                    method_display = METHOD_NAMES.get(method, method)
                    plot_single_method_histogram(
                        method_values,
                        method,
                        title=f"Epsilon: {display_name} - {method_display}{ablation_label}",
                        output_path=topic_dir / f"epsilon_histogram_{method}.png"
                    )
            
            # Likert plots - flat for histograms, clustered for barplot
            likert_results = collect_likert_for_topic(topic, output_dir, ablation)
            likert_results_clustered = collect_likert_clustered_for_topic(topic, output_dir, ablation)
            
            plot_likert_barplot(
                likert_results_clustered,
                title=f"Average Likert Rating: {display_name}{ablation_label}",
                output_path=topic_dir / "likert_barplot.png"
            )
            
            # Per-topic Likert strip plot (methods as rows)
            plot_likert_stripplot(
                likert_results,
                title=f"Likert Distribution: {display_name}{ablation_label}",
                output_path=topic_dir / "likert_stripplot.png"
            )
            
            # Cluster size strip plot (only for full ablation)
            if ablation == "full":
                topic_cluster_sizes = collect_cluster_sizes_for_topic(topic, output_dir, ablation)
                plot_cluster_size_stripplot(
                    topic_cluster_sizes,
                    title=f"Cluster Size Distribution: {display_name}{ablation_label}",
                    output_path=topic_dir / "cluster_size_stripplot.png",
                    use_log_scale=False
                )
                plot_cluster_size_stripplot(
                    topic_cluster_sizes,
                    title=f"Cluster Size Distribution: {display_name} (Log Scale){ablation_label}",
                    output_path=topic_dir / "cluster_size_stripplot_log.png",
                    use_log_scale=True
                )
                plot_cluster_size_violinplot(
                    topic_cluster_sizes,
                    title=f"Cluster Size Distribution: {display_name}{ablation_label}",
                    output_path=topic_dir / "cluster_size_violinplot.png",
                    use_log_scale=False
                )
                plot_cluster_size_violinplot(
                    topic_cluster_sizes,
                    title=f"Cluster Size Distribution: {display_name} (Log Scale){ablation_label}",
                    output_path=topic_dir / "cluster_size_violinplot_log.png",
                    use_log_scale=True
                )
                
                # Winner cluster percentile ranking bar plot
                topic_percentile_clustered = collect_winner_cluster_percentile_clustered_for_topic(topic, output_dir, ablation)
                plot_winner_cluster_percentile_barplot(
                    topic_percentile_clustered,
                    title=f"Winner Cluster Size Percentile Ranking: {display_name}{ablation_label}",
                    output_path=topic_dir / "winner_cluster_percentile_barplot.png"
                )
            
            plot_likert_histogram(
                likert_results,
                title=f"Likert Distribution: {display_name}{ablation_label}",
                output_path=topic_dir / "likert_histogram.png"
            )
            
            # Per-method Likert histograms
            for method in VOTING_METHODS:
                method_values = likert_results.get(method, [])
                if method_values:
                    method_display = METHOD_NAMES.get(method, method)
                    plot_single_method_likert_histogram(
                        method_values,
                        method,
                        title=f"Likert: {display_name} - {method_display}{ablation_label}",
                        output_path=topic_dir / f"likert_histogram_{method}.png"
                    )
        
        # Generate multi-persona plots for this ablation
        logger.info(f"Generating multi-persona plots for ablation: {ablation}")
        _generate_multi_persona_plots_for_ablation(output_dir, topics, ablation)
    
    logger.info(f"Generated all plots in {figures_dir}")


def _generate_multi_persona_plots_for_ablation(
    output_dir: Path,
    topics: Optional[List[str]],
    ablation: str
) -> None:
    """
    Generate multi-persona comparison plots for a specific ablation.
    
    Args:
        output_dir: Output directory
        topics: List of topics
        ablation: Ablation type
    """
    figures_dir = output_dir / "figures" / ablation / "aggregate"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    ablation_label = f" ({ablation.replace('_', ' ')})" if ablation != "full" else ""
    
    # Collect results for each persona count
    flat_results_by_n = {}
    clustered_results_by_n = {}
    
    for n_personas in PERSONA_COUNTS:
        flat_results = collect_all_results_for_n_personas(
            n_personas, output_dir, ablation, topics
        )
        clustered_results = collect_all_results_for_n_personas_clustered(
            n_personas, output_dir, ablation, topics
        )
        
        # Only include if we have data
        total_samples = sum(len(v) for v in flat_results.values())
        if total_samples > 0:
            flat_results_by_n[n_personas] = flat_results
            clustered_results_by_n[n_personas] = clustered_results
    
    if not flat_results_by_n:
        logger.warning(f"No multi-persona data found for ablation {ablation}")
        return
    
    # Generate barplot
    plot_epsilon_multi_persona_barplot(
        clustered_results_by_n,
        title=f"Average Epsilon by Persona Count{ablation_label}",
        output_path=figures_dir / "epsilon_multi_persona_barplot.png"
    )
    
    # Generate stripplot
    plot_epsilon_multi_persona_stripplot(
        flat_results_by_n,
        title=f"Epsilon Distribution by Persona Count{ablation_label}",
        output_path=figures_dir / "epsilon_multi_persona_stripplot.png"
    )

