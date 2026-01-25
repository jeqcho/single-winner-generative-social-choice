"""
Re-run voting methods for failed mini-reps.

After fixing invalid voters, re-runs voting methods for mini-reps
that originally failed due to -1 values in rankings.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from .config import PHASE2_DATA_DIR
from .preference_builder_iterative import subsample_preferences
from src.sampling_experiment.epsilon_calculator import load_precomputed_epsilons
from src.sampling_experiment.voting_methods import (
    run_schulze,
    run_borda,
    run_irv,
    run_plurality,
    run_veto_by_consumption,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Traditional voting methods (no API calls needed)
VOTING_METHODS = {
    'schulze': run_schulze,
    'borda': run_borda,
    'irv': run_irv,
    'plurality': run_plurality,
    'veto_by_consumption': run_veto_by_consumption,
}


def find_failed_minireps() -> List[Tuple[Path, int]]:
    """
    Find all mini-reps with errors in their results.
    
    Returns:
        List of (rep_dir, mini_rep_id) tuples
    """
    failed = []
    
    for results_file in PHASE2_DATA_DIR.glob("**/mini_rep*/results.json"):
        with open(results_file) as f:
            results = json.load(f)
        
        # Check if any traditional method has an error
        for method in VOTING_METHODS.keys():
            if method in results.get("results", {}):
                if results["results"][method].get("error"):
                    mini_rep_dir = results_file.parent
                    rep_dir = mini_rep_dir.parent
                    mini_rep_id = int(mini_rep_dir.name.replace("mini_rep", ""))
                    failed.append((rep_dir, mini_rep_id))
                    break
    
    return failed


def rerun_minirep(rep_dir: Path, mini_rep_id: int) -> bool:
    """
    Re-run voting methods for a single mini-rep.
    
    Returns:
        True if successful, False if error.
    """
    mini_rep_dir = rep_dir / f"mini_rep{mini_rep_id}"
    results_path = mini_rep_dir / "results.json"
    
    # Load existing results to get voter/alt indices
    with open(results_path) as f:
        results = json.load(f)
    
    voter_indices = results["voter_indices"]
    alt_indices = results["alt_indices"]
    
    # Load full preferences
    with open(rep_dir / "preferences.json") as f:
        full_preferences = json.load(f)
    
    # Subsample preferences for this mini-rep using the saved indices
    mini_preferences, _, _ = subsample_preferences(
        full_preferences,
        voter_indices=voter_indices,
        alt_indices=alt_indices
    )
    
    # Load precomputed epsilons
    epsilons = load_precomputed_epsilons(rep_dir)
    
    # Run each voting method
    for method_name, method_fn in VOTING_METHODS.items():
        try:
            result = method_fn(mini_preferences)
            winner = result['winner'] if isinstance(result, dict) else result
            
            # Map mini-rep winner back to full index
            full_winner_idx = alt_indices[int(winner)]
            
            # Look up epsilon
            eps = epsilons.get(str(full_winner_idx))
            
            results["results"][method_name] = {
                "winner": str(winner),
                "epsilon": eps,
                "full_winner_idx": str(full_winner_idx)
            }
        except Exception as e:
            logger.error(f"Method {method_name} failed for {mini_rep_dir}: {e}")
            results["results"][method_name] = {
                "winner": None,
                "error": str(e)
            }
            return False
    
    # Save updated results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return True


def main():
    """Re-run all failed mini-reps."""
    logger.info("Finding failed mini-reps...")
    failed = find_failed_minireps()
    logger.info(f"Found {len(failed)} failed mini-reps")
    
    success = 0
    errors = 0
    
    for rep_dir, mini_rep_id in failed:
        logger.info(f"Re-running {rep_dir.name}/mini_rep{mini_rep_id}...")
        if rerun_minirep(rep_dir, mini_rep_id):
            success += 1
        else:
            errors += 1
    
    logger.info(f"Done! Success: {success}, Errors: {errors}")
    
    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
