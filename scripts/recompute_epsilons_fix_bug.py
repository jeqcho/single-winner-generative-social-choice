#!/usr/bin/env python3
"""
Recompute epsilons to fix the bug where all voters were assumed to have identical preferences.

The original computation in compute_epsilon_from_positions incorrectly used:
    ranking = list(range(n_originals))  # Same for ALL voters!

This script fixes that by using actual voter preferences from preferences.json.

Output schema for each method:
- epsilon: Same as epsilon_m100 (for backward compatibility)
- epsilon_m100: Computed with actual preferences, m_override=100
- epsilon_m101: Computed with actual preferences, natural m=101
- epsilon_buggy: Original buggy value (backup)

Usage:
    uv run python scripts/recompute_epsilons_fix_bug.py
    uv run python scripts/recompute_epsilons_fix_bug.py --dry-run
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import logging
import argparse
from typing import List, Dict, Optional, Tuple

from tqdm import tqdm
from pvc_toolbox import compute_critical_epsilon
from src.experiment_utils.epsilon_calculator import compute_critical_epsilon_custom

# Configuration
DATA_DIR = project_root / "outputs" / "sample_alt_voters" / "data"
TOPICS = ["abortion", "healthcare", "electoral", "policing", "trust", "environment"]
ALT_DIST = "persona_no_context"
N_REPS = 10
N_MINI_REPS = 4

# Voter distributions to process
VOTER_CONFIGS = [
    {"path": "{topic}/uniform/{alt_dist}", "name": "uniform"},
    {"path": "{topic}/clustered/conservative_traditional/{alt_dist}", "name": "conservative"},
    {"path": "{topic}/clustered/progressive_liberal/{alt_dist}", "name": "progressive"},
]

# Methods that have insertion_positions and need recomputation
METHODS_WITH_INSERTION = [
    "chatgpt_double_star",
    "chatgpt_double_star_rankings",
    "chatgpt_double_star_personas",
    "chatgpt_triple_star",
    "random_insertion",
]

logger = logging.getLogger(__name__)


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
    
    # Convert to voter-centric format
    voter_rankings = []
    for voter_idx in range(n_voters):
        ranking = [preferences[rank][voter_idx] for rank in range(n_ranks)]
        voter_rankings.append(ranking)
    
    # Insert "100" at specified position for each voter
    for voter_idx, pos in enumerate(insertion_positions):
        if pos is not None and voter_idx < len(voter_rankings):
            # Clamp position to valid range
            pos = max(0, min(pos, len(voter_rankings[voter_idx])))
            voter_rankings[voter_idx].insert(pos, "100")
        elif voter_idx < len(voter_rankings):
            # None position - insert at bottom
            voter_rankings[voter_idx].append("100")
    
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


def compute_epsilons(preferences_101: List[List[str]]) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute both epsilon values for the new statement "100".
    
    Returns:
        Tuple of (epsilon_m101, epsilon_m100)
    """
    alternatives = [str(i) for i in range(101)]
    winner = "100"
    
    epsilon_m101 = None
    epsilon_m100 = None
    
    try:
        epsilon_m101 = compute_critical_epsilon(preferences_101, alternatives, winner)
    except Exception as e:
        logger.error(f"Epsilon m101 computation failed: {e}")
    
    try:
        epsilon_m100 = compute_critical_epsilon_custom(
            preferences_101, alternatives, winner, m_override=100
        )
    except Exception as e:
        logger.error(f"Epsilon m100 computation failed: {e}")
    
    return epsilon_m101, epsilon_m100


def process_results_file(
    results_path: Path,
    preferences: List[List[str]],
    dry_run: bool = False
) -> Dict[str, Dict]:
    """Process a results.json file and recompute epsilons."""
    changes = {}
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    modified = False
    
    for method in METHODS_WITH_INSERTION:
        if method not in results.get("results", {}):
            continue
            
        method_data = results["results"][method]
        
        if "insertion_positions" not in method_data:
            continue
        
        insertion_positions = method_data["insertion_positions"]
        
        # Get the original (buggy) epsilon - check various field names
        old_epsilon = method_data.get("epsilon")
        if old_epsilon is None:
            old_epsilon = method_data.get("epsilon_original")
        
        # Construct 101-alternative profile with actual preferences
        preferences_101 = construct_101_preferences(preferences, insertion_positions)
        
        # Compute correct epsilons
        epsilon_m101, epsilon_m100 = compute_epsilons(preferences_101)
        
        if epsilon_m101 is not None or epsilon_m100 is not None:
            changes[method] = {
                "epsilon_buggy": old_epsilon,
                "epsilon_m101": epsilon_m101,
                "epsilon_m100": epsilon_m100,
            }
            
            # Update the results
            if "epsilon_buggy" not in method_data and old_epsilon is not None:
                method_data["epsilon_buggy"] = old_epsilon
            
            if epsilon_m101 is not None:
                method_data["epsilon_m101"] = epsilon_m101
            
            if epsilon_m100 is not None:
                method_data["epsilon_m100"] = epsilon_m100
                method_data["epsilon"] = epsilon_m100  # epsilon = epsilon_m100
            
            modified = True
    
    if modified and not dry_run:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return changes


def main():
    """Recompute epsilons with correct voter preferences."""
    parser = argparse.ArgumentParser(description="Recompute epsilons to fix the bug")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    if args.dry_run:
        logger.info("DRY RUN MODE - no files will be modified")
    
    total_updates = 0
    stats = {
        "epsilon_m101": [],
        "epsilon_m100": [],
        "epsilon_buggy": [],
    }
    
    for voter_config in VOTER_CONFIGS:
        voter_name = voter_config["name"]
        logger.info(f"\nProcessing {voter_name} voters...")
        
        for topic in TOPICS:
            path_template = voter_config["path"].format(topic=topic, alt_dist=ALT_DIST)
            topic_dir = DATA_DIR / path_template
            
            if not topic_dir.exists():
                logger.warning(f"Directory not found: {topic_dir}")
                continue
            
            logger.info(f"  Topic: {topic}")
            
            for rep_idx in tqdm(range(N_REPS), desc=f"    {topic}", leave=False):
                rep_dir = topic_dir / f"rep{rep_idx}"
                
                if not rep_dir.exists():
                    continue
                
                # Load preferences
                preferences_path = rep_dir / "preferences.json"
                if not preferences_path.exists():
                    logger.warning(f"Missing preferences: {preferences_path}")
                    continue
                
                with open(preferences_path, 'r') as f:
                    preferences = json.load(f)
                
                # Process each mini-rep
                for mini_rep_idx in range(N_MINI_REPS):
                    results_path = rep_dir / f"mini_rep{mini_rep_idx}" / "results.json"
                    
                    if not results_path.exists():
                        continue
                    
                    changes = process_results_file(results_path, preferences, args.dry_run)
                    
                    if changes:
                        total_updates += len(changes)
                        for method, change in changes.items():
                            if change["epsilon_m101"] is not None:
                                stats["epsilon_m101"].append(change["epsilon_m101"])
                            if change["epsilon_m100"] is not None:
                                stats["epsilon_m100"].append(change["epsilon_m100"])
                            if change["epsilon_buggy"] is not None:
                                stats["epsilon_buggy"].append(change["epsilon_buggy"])
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Total epsilon values updated: {total_updates}")
    
    if stats["epsilon_m101"]:
        import numpy as np
        logger.info(f"\nEpsilon m101 stats:")
        logger.info(f"  mean: {np.mean(stats['epsilon_m101']):.6f}")
        logger.info(f"  min:  {np.min(stats['epsilon_m101']):.6f}")
        logger.info(f"  max:  {np.max(stats['epsilon_m101']):.6f}")
    
    if stats["epsilon_m100"]:
        import numpy as np
        logger.info(f"\nEpsilon m100 stats:")
        logger.info(f"  mean: {np.mean(stats['epsilon_m100']):.6f}")
        logger.info(f"  min:  {np.min(stats['epsilon_m100']):.6f}")
        logger.info(f"  max:  {np.max(stats['epsilon_m100']):.6f}")
    
    if stats["epsilon_buggy"]:
        import numpy as np
        logger.info(f"\nOriginal buggy epsilon stats:")
        logger.info(f"  mean: {np.mean(stats['epsilon_buggy']):.6f}")
        logger.info(f"  min:  {np.min(stats['epsilon_buggy']):.6f}")
        logger.info(f"  max:  {np.max(stats['epsilon_buggy']):.6f}")
    
    if args.dry_run:
        logger.info("\nDRY RUN complete - no files were modified")
    else:
        logger.info("\nAll epsilon values have been updated!")
        logger.info("Run 'uv run python scripts/generate_slide_plots.py' to regenerate plots")


if __name__ == "__main__":
    main()
