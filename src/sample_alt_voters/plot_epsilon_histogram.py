"""
Plot histogram comparing precomputed epsilon distribution vs random insertion epsilons.

This script generates histograms comparing two distributions for uniform voters:
1. Precomputed Epsilons: Distribution of epsilon values for all 100 original statements
2. Random Insertion Epsilons: Distribution of epsilon values for randomly inserted statements

The comparison helps understand how random insertion statements perform compared
to the original statement pool in terms of consensus quality (lower epsilon = better).
"""

import json
import logging
from pathlib import Path
from typing import List

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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "random_insertion_test" / "epsilons"

# Topics to process (short names used in directory structure)
TOPICS = ["abortion", "electoral", "environment", "healthcare", "policing", "trust"]

# Number of reps and mini_reps
N_REPS = 10
N_MINI_REPS = 4


def load_precomputed_epsilons(topic: str) -> List[float]:
    """
    Load all precomputed epsilon values for a topic.
    
    Args:
        topic: Short topic name (e.g., "environment")
        
    Returns:
        List of all epsilon values from precomputed_epsilons.json across all reps
    """
    all_epsilons = []
    
    for rep_idx in range(N_REPS):
        epsilons_path = (
            PHASE2_DATA_DIR / topic / "uniform" / "persona_no_context" 
            / f"rep{rep_idx}" / "precomputed_epsilons.json"
        )
        
        if not epsilons_path.exists():
            logger.warning(f"Missing precomputed_epsilons file: {epsilons_path}")
            continue
            
        with open(epsilons_path, 'r') as f:
            epsilons = json.load(f)
        
        # epsilons is a dict: {statement_id: epsilon}
        # Filter out None values
        valid_epsilons = [e for e in epsilons.values() if e is not None]
        all_epsilons.extend(valid_epsilons)
    
    logger.info(f"Loaded {len(all_epsilons)} precomputed epsilons for topic '{topic}'")
    return all_epsilons


def load_random_insertion_epsilons(topic: str) -> List[float]:
    """
    Load all epsilon values from random_insertion results.
    
    Args:
        topic: Short topic name (e.g., "environment")
        
    Returns:
        List of all epsilon values from random_insertion across all mini_reps and reps
    """
    all_epsilons = []
    
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
            
            # Get random_insertion epsilon
            if "results" in results and "random_insertion" in results["results"]:
                epsilon = results["results"]["random_insertion"].get("epsilon")
                if epsilon is not None:
                    all_epsilons.append(epsilon)
    
    logger.info(f"Loaded {len(all_epsilons)} random insertion epsilons for topic '{topic}'")
    return all_epsilons


def plot_epsilon_histogram(
    precomputed: List[float],
    random_insertion: List[float],
    topic: str,
    output_path: Path
) -> None:
    """
    Create overlaid histogram comparing precomputed and random insertion epsilon distributions.
    
    Args:
        precomputed: List of precomputed epsilon values
        random_insertion: List of random insertion epsilon values
        topic: Topic name for title
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    
    # Define bins: 0 to 1 with appropriate resolution
    bins = np.linspace(0, 1, 51)  # 50 bins from 0 to 1
    
    # Plot both distributions
    ax.hist(
        precomputed, 
        bins=bins, 
        alpha=0.6, 
        label=f"Precomputed Epsilons (n={len(precomputed):,})",
        color="#3498db",  # Blue
        density=True,
        edgecolor='white',
        linewidth=0.5
    )
    
    ax.hist(
        random_insertion, 
        bins=bins, 
        alpha=0.6, 
        label=f"Random Insertion Epsilons (n={len(random_insertion):,})",
        color="#e74c3c",  # Red
        density=True,
        edgecolor='white',
        linewidth=0.5
    )
    
    # Formatting
    ax.set_xlabel("Epsilon (lower = better consensus)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Epsilon Distribution: Precomputed vs Random Insertion - {topic.title()}", fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(-0.02, 1.02)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved epsilon histogram to {output_path}")


def plot_combined_epsilon_histogram(
    all_precomputed: List[float],
    all_random_insertion: List[float],
    output_path: Path
) -> None:
    """
    Create combined histogram with data from all topics.
    
    Args:
        all_precomputed: Combined list of precomputed epsilon values from all topics
        all_random_insertion: Combined list of random insertion epsilon values from all topics
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    
    # Define bins: 0 to 1 with appropriate resolution
    bins = np.linspace(0, 1, 51)  # 50 bins from 0 to 1
    
    # Plot both distributions
    ax.hist(
        all_precomputed, 
        bins=bins, 
        alpha=0.6, 
        label=f"Precomputed Epsilons (n={len(all_precomputed):,})",
        color="#3498db",  # Blue
        density=True,
        edgecolor='white',
        linewidth=0.5
    )
    
    ax.hist(
        all_random_insertion, 
        bins=bins, 
        alpha=0.6, 
        label=f"Random Insertion Epsilons (n={len(all_random_insertion):,})",
        color="#e74c3c",  # Red
        density=True,
        edgecolor='white',
        linewidth=0.5
    )
    
    # Formatting
    ax.set_xlabel("Epsilon (lower = better consensus)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Epsilon Distribution: Precomputed vs Random Insertion - All Topics Combined", fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(-0.02, 1.02)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved combined epsilon histogram to {output_path}")


def main():
    """Generate all epsilon histogram plots."""
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
    
    # Generate per-topic histograms
    for topic in TOPICS:
        logger.info(f"Processing topic: {topic}")
        
        # Load data
        precomputed = load_precomputed_epsilons(topic)
        random_insertion = load_random_insertion_epsilons(topic)
        
        if not precomputed or not random_insertion:
            logger.warning(f"Skipping topic '{topic}' due to missing data")
            continue
        
        # Plot individual histogram
        output_path = OUTPUT_DIR / f"epsilon_histogram_{topic}.png"
        plot_epsilon_histogram(precomputed, random_insertion, topic, output_path)
        
        # Accumulate for combined plot
        combined_precomputed.extend(precomputed)
        combined_random_insertion.extend(random_insertion)
    
    # Generate combined histogram
    if combined_precomputed and combined_random_insertion:
        output_path = OUTPUT_DIR / "epsilon_histogram_all_topics.png"
        plot_combined_epsilon_histogram(combined_precomputed, combined_random_insertion, output_path)
    
    logger.info("Done generating all epsilon histograms")


if __name__ == "__main__":
    main()
