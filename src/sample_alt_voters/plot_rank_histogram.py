"""
Plot histogram comparing statement rank distribution vs insertion positions.

This script generates histograms comparing two distributions for uniform voters:
1. Statement Ranks: Distribution of all rank positions (0-99) from voter preferences
2. Insertion Positions: Distribution of insertion positions from random_insertion method (0-100)

The comparison helps validate whether insertion sort positions follow the expected
distribution given the underlying preference rankings.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "random_insertion_test" / "ranks"

# Topics to process (short names used in directory structure)
TOPICS = ["abortion", "electoral", "environment", "healthcare", "policing", "trust"]

# Number of reps and mini_reps
N_REPS = 10
N_MINI_REPS = 4


def load_rank_data(topic: str) -> List[int]:
    """
    Load all rank positions from preferences.json files for a topic.
    
    The preferences matrix has shape [100 ranks][100 voters], where
    preferences[rank][voter] = statement index at that rank for that voter.
    
    We collect all rank indices (0-99) to see their distribution.
    Since each voter ranks all 100 statements, we expect a uniform distribution.
    
    Args:
        topic: Short topic name (e.g., "environment")
        
    Returns:
        List of all rank positions (0-99) across all voters and reps
    """
    all_ranks = []
    
    for rep_idx in range(N_REPS):
        preferences_path = (
            PHASE2_DATA_DIR / topic / "uniform" / "persona_no_context" 
            / f"rep{rep_idx}" / "preferences.json"
        )
        
        if not preferences_path.exists():
            logger.warning(f"Missing preferences file: {preferences_path}")
            continue
            
        with open(preferences_path, 'r') as f:
            preferences = json.load(f)
        
        # preferences[rank][voter] = statement index
        # We want to collect all rank indices
        for rank_idx, rank_row in enumerate(preferences):
            # Each rank position appears once per voter
            # Add this rank index once for each voter
            all_ranks.extend([rank_idx] * len(rank_row))
    
    logger.info(f"Loaded {len(all_ranks)} rank positions for topic '{topic}'")
    return all_ranks


def load_insertion_positions(topic: str) -> List[int]:
    """
    Load all insertion positions from random_insertion results.
    
    Args:
        topic: Short topic name (e.g., "environment")
        
    Returns:
        List of all insertion positions (0-100) across all mini_reps and reps
    """
    all_positions = []
    
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
            
            # Get random_insertion positions
            if "results" in results and "random_insertion" in results["results"]:
                positions = results["results"]["random_insertion"].get("insertion_positions", [])
                # Filter out None values
                valid_positions = [p for p in positions if p is not None]
                all_positions.extend(valid_positions)
    
    logger.info(f"Loaded {len(all_positions)} insertion positions for topic '{topic}'")
    return all_positions


def plot_histogram(
    ranks: List[int],
    insertions: List[int],
    topic: str,
    output_path: Path
) -> None:
    """
    Create overlaid histogram comparing rank and insertion position distributions.
    
    Args:
        ranks: List of rank positions (0-99)
        insertions: List of insertion positions (0-100)
        topic: Topic name for title
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    
    # Define bins: 0-100 (101 bins to cover both distributions)
    bins = np.arange(0, 102) - 0.5  # Center bins on integers
    
    # Plot both distributions
    ax.hist(
        ranks, 
        bins=bins, 
        alpha=0.6, 
        label=f"Statement Ranks (n={len(ranks):,})",
        color="#3498db",  # Blue
        density=True,
        edgecolor='white',
        linewidth=0.5
    )
    
    ax.hist(
        insertions, 
        bins=bins, 
        alpha=0.6, 
        label=f"Insertion Positions (n={len(insertions):,})",
        color="#e74c3c",  # Red
        density=True,
        edgecolor='white',
        linewidth=0.5
    )
    
    # Formatting
    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Rank vs Insertion Position Distribution - {topic.title()}", fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(-1, 101)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved histogram to {output_path}")


def plot_combined_histogram(
    all_ranks: List[int],
    all_insertions: List[int],
    output_path: Path
) -> None:
    """
    Create combined histogram with data from all topics.
    
    Args:
        all_ranks: Combined list of rank positions from all topics
        all_insertions: Combined list of insertion positions from all topics
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    
    # Define bins: 0-100 (101 bins to cover both distributions)
    bins = np.arange(0, 102) - 0.5  # Center bins on integers
    
    # Plot both distributions
    ax.hist(
        all_ranks, 
        bins=bins, 
        alpha=0.6, 
        label=f"Statement Ranks (n={len(all_ranks):,})",
        color="#3498db",  # Blue
        density=True,
        edgecolor='white',
        linewidth=0.5
    )
    
    ax.hist(
        all_insertions, 
        bins=bins, 
        alpha=0.6, 
        label=f"Insertion Positions (n={len(all_insertions):,})",
        color="#e74c3c",  # Red
        density=True,
        edgecolor='white',
        linewidth=0.5
    )
    
    # Formatting
    ax.set_xlabel("Position", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Rank vs Insertion Position Distribution - All Topics Combined", fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(-1, 101)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved combined histogram to {output_path}")


def main():
    """Generate all histogram plots."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Collect data for combined plot
    combined_ranks = []
    combined_insertions = []
    
    # Generate per-topic histograms
    for topic in TOPICS:
        logger.info(f"Processing topic: {topic}")
        
        # Load data
        ranks = load_rank_data(topic)
        insertions = load_insertion_positions(topic)
        
        if not ranks or not insertions:
            logger.warning(f"Skipping topic '{topic}' due to missing data")
            continue
        
        # Plot individual histogram
        output_path = OUTPUT_DIR / f"histogram_{topic}.png"
        plot_histogram(ranks, insertions, topic, output_path)
        
        # Accumulate for combined plot
        combined_ranks.extend(ranks)
        combined_insertions.extend(insertions)
    
    # Generate combined histogram
    if combined_ranks and combined_insertions:
        output_path = OUTPUT_DIR / "histogram_all_topics.png"
        plot_combined_histogram(combined_ranks, combined_insertions, output_path)
    
    logger.info("Done generating all histograms")


if __name__ == "__main__":
    main()
