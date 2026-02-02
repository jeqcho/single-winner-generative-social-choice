#!/usr/bin/env python3
"""
Recompute epsilons for conservative_traditional voters on abortion topic using m=101.

This script updates epsilon values for methods that generate new statements
by constructing a 101-alternative preference profile and computing epsilon naturally.

Scope: conservative_traditional (clustered) voters, abortion topic, persona_no_context only

Usage:
    uv run python scripts/recompute_epsilons_conservative_abortion.py
    uv run python scripts/recompute_epsilons_conservative_abortion.py --dry-run
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import logging
import argparse
from typing import List, Dict, Optional

from tqdm import tqdm
from pvc_toolbox import compute_critical_epsilon

# Configuration
DATA_DIR = project_root / "outputs" / "sample_alt_voters" / "data"
TOPIC = "abortion"
VOTER_DIST = "conservative_traditional"
ALT_DIST = "persona_no_context"
N_REPS = 10
N_MINI_REPS = 4

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
    """
    n_ranks = len(preferences)
    n_voters = len(preferences[0]) if preferences else 0
    
    voter_rankings = []
    for voter_idx in range(n_voters):
        ranking = [preferences[rank][voter_idx] for rank in range(n_ranks)]
        voter_rankings.append(ranking)
    
    for voter_idx, pos in enumerate(insertion_positions):
        if pos is not None and voter_idx < len(voter_rankings):
            pos = max(0, min(pos, len(voter_rankings[voter_idx])))
            voter_rankings[voter_idx].insert(pos, "100")
    
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
    """Compute epsilon for the new statement '100' with m=101."""
    alternatives = [str(i) for i in range(101)]
    winner = "100"
    
    try:
        epsilon = compute_critical_epsilon(preferences_101, alternatives, winner)
        return epsilon
    except Exception as e:
        logger.error(f"Epsilon computation failed: {e}")
        return None


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
        old_epsilon = method_data.get("epsilon")
        
        valid_positions = [p if p is not None else 50 for p in insertion_positions]
        
        preferences_101 = construct_101_preferences(preferences, valid_positions)
        new_epsilon = compute_epsilon_m101(preferences_101)
        
        if new_epsilon is not None:
            changes[method] = {
                "old_epsilon": old_epsilon,
                "new_epsilon": new_epsilon,
                "diff": new_epsilon - old_epsilon if old_epsilon is not None else None,
            }
            
            if "epsilon_original" not in method_data:
                method_data["epsilon_original"] = old_epsilon
            method_data["epsilon"] = new_epsilon
            modified = True
    
    if modified and not dry_run:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return changes


def main():
    """Recompute epsilons for conservative abortion."""
    parser = argparse.ArgumentParser(description="Recompute epsilons with m=101 for conservative abortion")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    if args.dry_run:
        logger.info("DRY RUN MODE - no files will be modified")
    
    logger.info(f"Processing: {TOPIC} / clustered / {VOTER_DIST} / {ALT_DIST}")
    
    total_changes = 0
    all_diffs = []
    
    # Data path for conservative abortion
    topic_dir = DATA_DIR / TOPIC / "clustered" / VOTER_DIST / ALT_DIST
    
    if not topic_dir.exists():
        logger.error(f"Directory not found: {topic_dir}")
        return
    
    for rep_idx in tqdm(range(N_REPS), desc="Processing reps"):
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
                total_changes += len(changes)
                for method, change in changes.items():
                    if change["diff"] is not None:
                        all_diffs.append(change["diff"])
                    logger.debug(
                        f"rep{rep_idx}/mini_rep{mini_rep_idx} - {method}: "
                        f"{change['old_epsilon']:.4f} -> {change['new_epsilon']:.4f}"
                    )
    
    # Summary
    logger.info(f"\nTotal epsilon values updated: {total_changes}")
    if all_diffs:
        import numpy as np
        logger.info(f"Epsilon changes - mean: {np.mean(all_diffs):.6f}, "
                   f"std: {np.std(all_diffs):.6f}, "
                   f"min: {np.min(all_diffs):.6f}, "
                   f"max: {np.max(all_diffs):.6f}")
    
    if args.dry_run:
        logger.info("\nDRY RUN complete - no files were modified")
    else:
        logger.info("\nAll epsilon values have been updated!")
        logger.info("Run 'uv run python scripts/generate_slide_plots.py' to regenerate plots")


if __name__ == "__main__":
    main()
