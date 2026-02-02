"""
Plot histogram comparing precomputed epsilon (m=100) vs random insertion with m=101.

This script generates histograms comparing two distributions for uniform voters:
1. Precomputed Epsilons (m=100): Epsilon values for original 100 statements
2. Random Insertion Epsilons (m=101): Recomputed epsilon for random insertion
   by constructing 101-alternative profile and computing epsilon naturally

The m=101 approach treats the new statement as a full participant in the
alternative pool, rather than using m_override=100 which prevents extra veto power.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from pvc_toolbox import compute_critical_epsilon

from .config import PROJECT_ROOT, PHASE2_DATA_DIR

logger = logging.getLogger(__name__)

# Set style for slide-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Slide-quality figure sizes
FIGURE_SIZE_WIDE = (14, 6)

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "random_insertion_test" / "epsilons_what_if"

# Topics to process (short names used in directory structure)
TOPICS = ["abortion", "electoral", "environment", "healthcare", "policing", "trust"]

# Number of reps and mini_reps
N_REPS = 10
N_MINI_REPS = 4


def construct_101_preferences(
    preferences: List[List[str]], 
    insertion_positions: List[int]
) -> List[List[str]]:
    """
    Construct 101-alternative preference profile by inserting new statement.
    
    Args:
        preferences: Original 100x100 preference matrix [rank][voter]
        insertion_positions: List of positions where "100" should be inserted for each voter
        
    Returns:
        New preference matrix with 101 alternatives [rank][voter]
    """
    n_ranks = len(preferences)
    n_voters = len(preferences[0]) if preferences else 0
    
    # Convert to voter-centric format, insert "100", then convert back
    voter_rankings = []
    for voter_idx in range(n_voters):
        ranking = [preferences[rank][voter_idx] for rank in range(n_ranks)]
        voter_rankings.append(ranking)
    
    # Insert "100" at specified position for each voter
    for voter_idx, pos in enumerate(insertion_positions):
        if pos is not None and voter_idx < len(voter_rankings):
            pos = max(0, min(pos, len(voter_rankings[voter_idx])))
            voter_rankings[voter_idx].insert(pos, "100")
    
    # Convert back to [rank][voter] format
    n_new_ranks = 101
    new_preferences = []
    for rank in range(n_new_ranks):
        rank_row = []
        for voter_idx in range(n_voters):
            if rank < len(voter_rankings[voter_idx]):
                rank_row.append(voter_rankings[voter_idx][rank])
            else:
                rank_row.append("100")
        new_preferences.append(rank_row)
    
    return new_preferences


def compute_epsilon_m101(preferences_101: List[List[str]]) -> Optional[float]:
    """
    Compute epsilon for the new statement "100" with m=101.
    """
    alternatives = [str(i) for i in range(101)]
    winner = "100"
    
    try:
        epsilon = compute_critical_epsilon(preferences_101, alternatives, winner)
        return epsilon
    except Exception as e:
        logger.error(f"Epsilon computation failed: {e}")
        return None


def load_precomputed_epsilons(topic: str) -> List[float]:
    """
    Load all precomputed epsilon values for a topic.
    
    Args:
        topic: Short topic name
        
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
        
        # Filter out None values
        valid_epsilons = [e for e in epsilons.values() if e is not None]
        all_epsilons.extend(valid_epsilons)
    
    logger.info(f"Loaded {len(all_epsilons)} precomputed epsilons for topic '{topic}'")
    return all_epsilons


def load_random_insertion_epsilons_m101(topic: str) -> List[float]:
    """
    Load random_insertion data and recompute epsilons with m=101.
    
    Args:
        topic: Short topic name
        
    Returns:
        List of recomputed epsilon values using m=101
    """
    all_epsilons = []
    
    for rep_idx in range(N_REPS):
        base_path = PHASE2_DATA_DIR / topic / "uniform" / "persona_no_context" / f"rep{rep_idx}"
        
        # Load preferences
        preferences_path = base_path / "preferences.json"
        if not preferences_path.exists():
            continue
            
        with open(preferences_path, 'r') as f:
            preferences = json.load(f)
        
        for mini_rep_idx in range(N_MINI_REPS):
            results_path = base_path / f"mini_rep{mini_rep_idx}" / "results.json"
            
            if not results_path.exists():
                continue
                
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Get random_insertion data
            if "results" not in results or "random_insertion" not in results["results"]:
                continue
                
            ri = results["results"]["random_insertion"]
            insertion_positions = ri.get("insertion_positions", [])
            
            if not insertion_positions:
                continue
            
            # Handle None values
            valid_positions = [p if p is not None else 50 for p in insertion_positions]
            
            # Construct 101-alternative preferences
            preferences_101 = construct_101_preferences(preferences, valid_positions)
            
            # Compute epsilon with m=101
            epsilon = compute_epsilon_m101(preferences_101)
            
            if epsilon is not None:
                all_epsilons.append(epsilon)
    
    logger.info(f"Computed {len(all_epsilons)} random insertion epsilons (m=101) for topic '{topic}'")
    return all_epsilons


def plot_epsilon_histogram(
    precomputed: List[float],
    random_insertion: List[float],
    topic: str,
    output_path: Path
) -> None:
    """
    Create overlaid histogram comparing precomputed and random insertion epsilon distributions.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    
    # Define bins: 0 to 1 with appropriate resolution
    bins = np.linspace(0, 1, 51)
    
    # Plot both distributions
    ax.hist(
        precomputed, 
        bins=bins, 
        alpha=0.6, 
        label=f"Precomputed m=100 (n={len(precomputed):,})",
        color="#3498db",  # Blue
        density=True,
        edgecolor='white',
        linewidth=0.5
    )
    
    ax.hist(
        random_insertion, 
        bins=bins, 
        alpha=0.6, 
        label=f"Random Insertion m=101 (n={len(random_insertion):,})",
        color="#e74c3c",  # Red
        density=True,
        edgecolor='white',
        linewidth=0.5
    )
    
    # Formatting
    ax.set_xlabel("Epsilon (lower = better consensus)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Epsilon Distribution: Precomputed vs Random Insertion (m=101) - {topic.title()}", fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(-0.02, 1.02)
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
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    
    bins = np.linspace(0, 1, 51)
    
    ax.hist(
        all_precomputed, 
        bins=bins, 
        alpha=0.6, 
        label=f"Precomputed m=100 (n={len(all_precomputed):,})",
        color="#3498db",
        density=True,
        edgecolor='white',
        linewidth=0.5
    )
    
    ax.hist(
        all_random_insertion, 
        bins=bins, 
        alpha=0.6, 
        label=f"Random Insertion m=101 (n={len(all_random_insertion):,})",
        color="#e74c3c",
        density=True,
        edgecolor='white',
        linewidth=0.5
    )
    
    ax.set_xlabel("Epsilon (lower = better consensus)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Epsilon Distribution: Precomputed vs Random Insertion (m=101) - All Topics Combined", fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved combined epsilon histogram to {output_path}")


def main():
    """Generate all epsilon histogram plots with m=101 recomputation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    combined_precomputed = []
    combined_random_insertion = []
    
    for topic in tqdm(TOPICS, desc="Processing topics"):
        logger.info(f"Processing topic: {topic}")
        
        # Load precomputed epsilons
        precomputed = load_precomputed_epsilons(topic)
        
        # Compute random insertion epsilons with m=101
        random_insertion = load_random_insertion_epsilons_m101(topic)
        
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
    
    # Log summary
    logger.info(f"\nSummary:")
    logger.info(f"  Precomputed epsilons: {len(combined_precomputed)} points")
    logger.info(f"  Random insertion (m=101): {len(combined_random_insertion)} points")
    
    logger.info("Done generating all epsilon histograms")


if __name__ == "__main__":
    main()
