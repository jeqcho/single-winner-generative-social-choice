"""
Targeted re-run for invalid voters.

Re-generates rankings only for voters with invalid preferences (duplicates/-1 values).
Each invalid voter gets up to 10 attempts to produce a valid ranking.

Fails loudly if any voter cannot produce a valid ranking after 10 attempts.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable

from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from .config import (
    PHASE2_DATA_DIR,
    PERSONAS_PATH,
    TOPIC_QUESTIONS,
    REASONING_EFFORT,
)
from .preference_builder_iterative import validate_preferences
from .run_experiment import load_statements_for_rep
from src.degeneracy_mitigation.iterative_ranking_star import rank_voter
from src.degeneracy_mitigation.config import HASH_SEED
from src.sampling_experiment.epsilon_calculator import precompute_all_epsilons, save_precomputed_epsilons

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

MAX_ATTEMPTS_PER_VOTER = 10


def load_personas() -> List[str]:
    """Load all adult personas."""
    with open(PERSONAS_PATH) as f:
        return json.load(f)


def is_valid_ranking(ranking: List, n_alts: int) -> bool:
    """Check if a ranking is valid (no duplicates, no -1s, correct length)."""
    if len(ranking) != n_alts:
        return False
    if len(set(ranking)) != n_alts:
        return False
    if -1 in ranking or "-1" in ranking:
        return False
    return True


def parse_rep_path(rep_dir: Path) -> Tuple[str, str, str, int]:
    """
    Parse rep directory path to extract topic, voter_dist, alt_dist, rep_id.
    
    Path format: .../data/{topic}/{voter_dist}/{alt_dist}/rep{id}[_cluster]
    """
    parts = rep_dir.parts
    # Find 'data' in path
    data_idx = parts.index('data')
    topic = parts[data_idx + 1]
    voter_dist = parts[data_idx + 2]
    alt_dist = parts[data_idx + 3]
    rep_name = parts[data_idx + 4]
    
    # Extract rep_id from rep name (e.g., "rep0" or "rep0_progressive_liberal")
    rep_id = int(rep_name.split('_')[0].replace('rep', ''))
    
    return topic, voter_dist, alt_dist, rep_id


def rerun_invalid_voter(
    client: OpenAI,
    voter_idx: int,
    persona: str,
    statements: List[Dict],
    topic_question: str,
    n_alts: int,
    rep_path: str
) -> Optional[List[int]]:
    """
    Re-run ranking for a single invalid voter up to MAX_ATTEMPTS_PER_VOTER times.
    
    Returns:
        Valid ranking list if successful, None if failed after all attempts.
    """
    for attempt in range(1, MAX_ATTEMPTS_PER_VOTER + 1):
        logger.info(f"  Attempt {attempt}/{MAX_ATTEMPTS_PER_VOTER} for voter {voter_idx}")
        
        try:
            result = rank_voter(
                client=client,
                voter_idx=voter_idx,
                persona=persona,
                statements=statements,
                topic=topic_question,
                reasoning_effort=REASONING_EFFORT,
                hash_seed=HASH_SEED + attempt * 1000  # Different seed per attempt
            )
            
            ranking = result.get('ranking', [])
            
            if is_valid_ranking(ranking, n_alts):
                logger.info(f"  SUCCESS: Voter {voter_idx} produced valid ranking on attempt {attempt}")
                return ranking
            else:
                unique = len(set(ranking))
                logger.warning(f"  Attempt {attempt} failed: {unique}/{n_alts} unique values")
                
        except Exception as e:
            logger.error(f"  Attempt {attempt} error: {e}")
    
    # All attempts failed
    logger.error(f"FAILURE: Voter {voter_idx} in {rep_path} failed after {MAX_ATTEMPTS_PER_VOTER} attempts")
    return None


def process_rep_directory(
    rep_dir: Path,
    all_personas: List[str],
    client: OpenAI,
    failures: List[Dict]
) -> bool:
    """
    Process a single rep directory - fix invalid voters and recompute epsilons.
    
    Returns:
        True if all voters valid and epsilons computed, False if any failures.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {rep_dir}")
    logger.info(f"{'='*60}")
    
    # Load preferences
    pref_path = rep_dir / "preferences.json"
    with open(pref_path) as f:
        preferences = json.load(f)
    
    n_alts = len(preferences)
    n_voters = len(preferences[0]) if preferences else 0
    
    # Identify invalid voters
    invalid_voters, validation_info = validate_preferences(preferences)
    
    if not invalid_voters:
        logger.info("No invalid voters found - computing epsilons")
        epsilons = precompute_all_epsilons(preferences, max_workers=10)
        save_precomputed_epsilons(epsilons, rep_dir)
        return True
    
    logger.info(f"Found {len(invalid_voters)} invalid voters: {invalid_voters}")
    
    # Parse path to get topic, alt_dist, rep_id
    topic, voter_dist, alt_dist, rep_id = parse_rep_path(rep_dir)
    
    # Load voter info
    voters_path = rep_dir / "voters.json"
    with open(voters_path) as f:
        voters_data = json.load(f)
    voter_indices = voters_data["voter_indices"]
    
    # Load statements
    topic_slug_map = {
        "abortion": "what-should-guide-laws-concerning-abortion",
        "electoral": "what-reforms-if-any-should-replace-or-modify-the-e"
    }
    topic_slug = topic_slug_map.get(topic, topic)
    statements = load_statements_for_rep(topic_slug, alt_dist, rep_id)
    topic_question = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    
    # Re-run invalid voters in parallel
    rep_failures = []
    results = {}  # voter_idx -> new_ranking or None
    
    def rerun_voter_task(local_voter_idx: int) -> Tuple[int, Optional[List[int]]]:
        """Task to rerun a single voter."""
        global_persona_idx = voter_indices[local_voter_idx]
        persona = all_personas[global_persona_idx]
        logger.info(f"Re-running voter {local_voter_idx} (persona {global_persona_idx})...")
        new_ranking = rerun_invalid_voter(
            client=client,
            voter_idx=local_voter_idx,
            persona=persona,
            statements=statements,
            topic_question=topic_question,
            n_alts=n_alts,
            rep_path=str(rep_dir)
        )
        return local_voter_idx, new_ranking
    
    # Run all invalid voters in parallel (up to 10 concurrent)
    with ThreadPoolExecutor(max_workers=min(10, len(invalid_voters))) as executor:
        futures = {executor.submit(rerun_voter_task, idx): idx for idx in invalid_voters}
        for future in as_completed(futures):
            local_voter_idx, new_ranking = future.result()
            results[local_voter_idx] = new_ranking
    
    # Process results and update preferences
    for local_voter_idx, new_ranking in results.items():
        if new_ranking is not None:
            for rank, alt in enumerate(new_ranking):
                preferences[rank][local_voter_idx] = str(alt)
            logger.info(f"Updated voter {local_voter_idx} in preferences")
        else:
            global_persona_idx = voter_indices[local_voter_idx]
            rep_failures.append({
                "rep_dir": str(rep_dir),
                "voter_idx": local_voter_idx,
                "persona_idx": global_persona_idx
            })
    
    if rep_failures:
        # Don't save/compute anything if there are failures
        failures.extend(rep_failures)
        logger.error(f"SKIPPING epsilon computation for {rep_dir} due to {len(rep_failures)} failed voters")
        return False
    
    # All voters fixed - save preferences and compute epsilons
    with open(pref_path, 'w') as f:
        json.dump(preferences, f)
    logger.info(f"Saved updated preferences to {pref_path}")
    
    # Validate again to make sure
    invalid_after, _ = validate_preferences(preferences)
    if invalid_after:
        logger.error(f"UNEXPECTED: Still have {len(invalid_after)} invalid voters after fixing!")
        failures.append({
            "rep_dir": str(rep_dir),
            "voter_idx": invalid_after,
            "error": "Still invalid after fixing"
        })
        return False
    
    # Compute epsilons
    logger.info("Computing epsilons...")
    epsilons = precompute_all_epsilons(preferences, max_workers=10)
    save_precomputed_epsilons(epsilons, rep_dir)
    
    # Verify epsilons are valid
    valid_epsilons = [e for e in epsilons.values() if e is not None]
    if not valid_epsilons:
        logger.error(f"All epsilons are null for {rep_dir}!")
        failures.append({
            "rep_dir": str(rep_dir),
            "error": "All epsilons null"
        })
        return False
    
    logger.info(f"Computed {len(valid_epsilons)} valid epsilons, mean={sum(valid_epsilons)/len(valid_epsilons):.4f}")
    return True


def rerun_invalid_voters():
    """Main function to re-run all invalid voters across all reps."""
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)
    client = OpenAI(api_key=api_key)
    
    # Load personas
    logger.info("Loading personas...")
    all_personas = load_personas()
    logger.info(f"Loaded {len(all_personas)} personas")
    
    # Find all rep directories
    if not PHASE2_DATA_DIR.exists():
        logger.error(f"Data directory not found: {PHASE2_DATA_DIR}")
        sys.exit(1)
    
    pref_files = list(PHASE2_DATA_DIR.glob("**/preferences.json"))
    logger.info(f"Found {len(pref_files)} preference files")
    
    # Track failures
    failures = []
    success_count = 0
    skip_count = 0
    
    for pref_path in tqdm(pref_files, desc="Processing reps"):
        rep_dir = pref_path.parent
        
        try:
            success = process_rep_directory(rep_dir, all_personas, client, failures)
            if success:
                success_count += 1
            else:
                skip_count += 1
        except Exception as e:
            logger.error(f"Error processing {rep_dir}: {e}")
            failures.append({
                "rep_dir": str(rep_dir),
                "error": str(e)
            })
            skip_count += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total reps processed: {len(pref_files)}")
    print(f"Successful: {success_count}")
    print(f"Skipped/Failed: {skip_count}")
    
    if failures:
        print("\n" + "=" * 60)
        print("FAILURE SUMMARY")
        print("=" * 60)
        for f in failures:
            print(f"  - {f}")
        print(f"\nTotal failures: {len(failures)}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("\nAll reps processed successfully!")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    rerun_invalid_voters()
