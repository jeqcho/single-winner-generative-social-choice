"""
Scatter plot of rank vs epsilon for precomputed and random insertion statements.

X-axis: Average rank position (0-100)
Y-axis: Epsilon (0-1, lower = better consensus)

Two distributions:
- Blue: Precomputed statements (original pool)
- Red: Random insertion statements
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import PROJECT_ROOT, PHASE2_DATA_DIR

logger = logging.getLogger(__name__)

# Set style for slide-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Slide-quality figure sizes
FIGURE_SIZE_WIDE = (14, 6)

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "random_insertion_test" / "scatter"

# Topics to process (short names used in directory structure)
TOPICS = ["abortion", "electoral", "environment", "healthcare", "policing", "trust"]

# Number of reps and mini_reps
N_REPS = 10
N_MINI_REPS = 4


def compute_average_ranks(preferences: List[List[str]]) -> Dict[str, float]:
    """
    Compute average rank for each statement across all voters.
    
    Args:
        preferences: 2D list where preferences[rank][voter] = statement_id
        
    Returns:
        Dict mapping statement_id to average rank
    """
    # Initialize accumulator: statement_id -> list of ranks
    statement_ranks: Dict[str, List[int]] = {}
    
    for rank_idx, rank_row in enumerate(preferences):
        for statement_id in rank_row:
            if statement_id not in statement_ranks:
                statement_ranks[statement_id] = []
            statement_ranks[statement_id].append(rank_idx)
    
    # Compute average for each statement
    return {sid: np.mean(ranks) for sid, ranks in statement_ranks.items()}


def load_precomputed_data(topic: str) -> List[Tuple[float, float]]:
    """
    Load precomputed statements with their average ranks and epsilons.
    
    Args:
        topic: Short topic name (e.g., "environment")
        
    Returns:
        List of (average_rank, epsilon) tuples
    """
    data_points = []
    
    for rep_idx in range(N_REPS):
        base_path = PHASE2_DATA_DIR / topic / "uniform" / "persona_no_context" / f"rep{rep_idx}"
        
        # Load preferences
        preferences_path = base_path / "preferences.json"
        if not preferences_path.exists():
            logger.warning(f"Missing preferences file: {preferences_path}")
            continue
            
        with open(preferences_path, 'r') as f:
            preferences = json.load(f)
        
        # Compute average ranks
        avg_ranks = compute_average_ranks(preferences)
        
        # Load precomputed epsilons
        epsilons_path = base_path / "precomputed_epsilons.json"
        if not epsilons_path.exists():
            logger.warning(f"Missing epsilons file: {epsilons_path}")
            continue
            
        with open(epsilons_path, 'r') as f:
            epsilons = json.load(f)
        
        # Combine: (avg_rank, epsilon) for each statement
        for statement_id, epsilon in epsilons.items():
            if epsilon is not None and statement_id in avg_ranks:
                data_points.append((avg_ranks[statement_id], epsilon))
    
    logger.info(f"Loaded {len(data_points)} precomputed data points for topic '{topic}'")
    return data_points


def load_random_insertion_data(topic: str) -> List[Tuple[float, float]]:
    """
    Load random insertion statements with their average insertion positions and epsilons.
    
    Args:
        topic: Short topic name (e.g., "environment")
        
    Returns:
        List of (average_position, epsilon) tuples
    """
    data_points = []
    
    for rep_idx in range(N_REPS):
        for mini_rep_idx in range(N_MINI_REPS):
            results_path = (
                PHASE2_DATA_DIR / topic / "uniform" / "persona_no_context"
                / f"rep{rep_idx}" / f"mini_rep{mini_rep_idx}" / "results.json"
            )
            
            if not results_path.exists():
                logger.warning(f"Missing results file: {results_path}")
                continue
                
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Get random_insertion data
            if "results" in results and "random_insertion" in results["results"]:
                ri = results["results"]["random_insertion"]
                epsilon = ri.get("epsilon")
                positions = ri.get("insertion_positions", [])
                
                # Filter None values and compute average
                valid_positions = [p for p in positions if p is not None]
                
                if epsilon is not None and valid_positions:
                    avg_position = np.mean(valid_positions)
                    data_points.append((avg_position, epsilon))
    
    logger.info(f"Loaded {len(data_points)} random insertion data points for topic '{topic}'")
    return data_points


def plot_scatter(
    precomputed: List[Tuple[float, float]],
    random_insertion: List[Tuple[float, float]],
    topic: str,
    output_path: Path
) -> None:
    """
    Create scatter plot of rank vs epsilon.
    
    Args:
        precomputed: List of (avg_rank, epsilon) for precomputed statements
        random_insertion: List of (avg_position, epsilon) for random insertion
        topic: Topic name for title
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    
    # Extract x, y for precomputed
    if precomputed:
        pre_x, pre_y = zip(*precomputed)
        ax.scatter(pre_x, pre_y, alpha=0.5, label=f"Precomputed (n={len(precomputed):,})", s=20)
    
    # Extract x, y for random insertion
    if random_insertion:
        ri_x, ri_y = zip(*random_insertion)
        ax.scatter(ri_x, ri_y, alpha=0.7, label=f"Random Insertion (n={len(random_insertion):,})", s=50, marker='x')
    
    # Formatting
    ax.set_xlabel("Average Rank Position", fontsize=12)
    ax.set_ylabel("Epsilon (lower = better consensus)", fontsize=12)
    ax.set_title(f"Rank vs Epsilon - {topic.title()}", fontsize=14)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.02, 1.02)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved scatter plot to {output_path}")


def plot_combined_scatter(
    all_precomputed: List[Tuple[float, float]],
    all_random_insertion: List[Tuple[float, float]],
    output_path: Path
) -> None:
    """
    Create combined scatter plot with data from all topics.
    
    Args:
        all_precomputed: Combined list of (avg_rank, epsilon) from all topics
        all_random_insertion: Combined list of (avg_position, epsilon) from all topics
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    
    # Extract x, y for precomputed
    if all_precomputed:
        pre_x, pre_y = zip(*all_precomputed)
        ax.scatter(pre_x, pre_y, alpha=0.3, label=f"Precomputed (n={len(all_precomputed):,})", s=15)
    
    # Extract x, y for random insertion
    if all_random_insertion:
        ri_x, ri_y = zip(*all_random_insertion)
        ax.scatter(ri_x, ri_y, alpha=0.7, label=f"Random Insertion (n={len(all_random_insertion):,})", s=50, marker='x')
    
    # Formatting
    ax.set_xlabel("Average Rank Position", fontsize=12)
    ax.set_ylabel("Epsilon (lower = better consensus)", fontsize=12)
    ax.set_title("Rank vs Epsilon - All Topics Combined", fontsize=14)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.02, 1.02)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved combined scatter plot to {output_path}")


def main():
    """Generate all scatter plots."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Collect data for combined plot
    combined_precomputed = []
    combined_random_insertion = []
    
    # Generate per-topic scatter plots
    for topic in TOPICS:
        logger.info(f"Processing topic: {topic}")
        
        # Load data
        precomputed = load_precomputed_data(topic)
        random_insertion = load_random_insertion_data(topic)
        
        if not precomputed and not random_insertion:
            logger.warning(f"Skipping topic '{topic}' due to missing data")
            continue
        
        # Plot individual scatter
        output_path = OUTPUT_DIR / f"scatter_{topic}.png"
        plot_scatter(precomputed, random_insertion, topic, output_path)
        
        # Accumulate for combined plot
        combined_precomputed.extend(precomputed)
        combined_random_insertion.extend(random_insertion)
    
    # Generate combined scatter plot
    if combined_precomputed or combined_random_insertion:
        output_path = OUTPUT_DIR / "scatter_all_topics.png"
        plot_combined_scatter(combined_precomputed, combined_random_insertion, output_path)
    
    logger.info("Done generating all scatter plots")


if __name__ == "__main__":
    main()
