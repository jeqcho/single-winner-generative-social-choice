"""
Re-evaluate mini-reps with updated precomputed epsilons.

After fixing invalid voters and recomputing epsilons, this script
updates the results.json files in each mini-rep with the correct
epsilon values looked up from precomputed_epsilons.json.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .config import PHASE2_DATA_DIR
from src.sampling_experiment.epsilon_calculator import load_precomputed_epsilons, lookup_epsilon

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def reeval_mini_rep(mini_rep_dir: Path, epsilons: Dict[int, float]) -> bool:
    """
    Re-evaluate a single mini-rep by looking up epsilon values.
    
    Returns:
        True if successful, False if error.
    """
    results_path = mini_rep_dir / "results.json"
    if not results_path.exists():
        return False
    
    with open(results_path) as f:
        results = json.load(f)
    
    updated = False
    for method, method_result in results.get("results", {}).items():
        if "full_winner_idx" in method_result and method_result.get("winner") is not None:
            full_idx = str(method_result["full_winner_idx"])
            # Keys in epsilons dict are strings
            eps = epsilons.get(full_idx)
            if eps != method_result.get("epsilon"):
                method_result["epsilon"] = eps
                updated = True
    
    if updated:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return True


def reeval_rep(rep_dir: Path) -> int:
    """
    Re-evaluate all mini-reps in a rep directory.
    
    Returns:
        Number of mini-reps updated.
    """
    # Load precomputed epsilons for this rep
    epsilons = load_precomputed_epsilons(rep_dir)
    if not epsilons:
        logger.warning(f"No precomputed epsilons found for {rep_dir}")
        return 0
    
    # Find all mini-rep directories
    mini_rep_dirs = sorted(rep_dir.glob("mini_rep*"))
    updated = 0
    
    for mini_rep_dir in mini_rep_dirs:
        if reeval_mini_rep(mini_rep_dir, epsilons):
            updated += 1
    
    return updated


def main():
    """Re-evaluate all mini-reps across all reps."""
    logger.info("Finding all rep directories...")
    
    # Find all directories with precomputed_epsilons.json
    epsilon_files = list(PHASE2_DATA_DIR.glob("**/precomputed_epsilons.json"))
    logger.info(f"Found {len(epsilon_files)} rep directories with epsilons")
    
    total_updated = 0
    for eps_file in tqdm(epsilon_files, desc="Re-evaluating reps"):
        rep_dir = eps_file.parent
        updated = reeval_rep(rep_dir)
        total_updated += updated
    
    logger.info(f"Updated {total_updated} mini-rep results files")
    
    # Verify by checking one result
    sample_results = list(PHASE2_DATA_DIR.glob("**/mini_rep0/results.json"))
    if sample_results:
        with open(sample_results[0]) as f:
            sample = json.load(f)
        schulze_eps = sample.get("results", {}).get("schulze", {}).get("epsilon")
        logger.info(f"Sample epsilon (schulze): {schulze_eps}")
        if schulze_eps is None:
            logger.error("Epsilon still null after re-evaluation!")
            sys.exit(1)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
