"""
Test script for the original (single-call) insertion algorithm.

This script validates the standard insertion approach by:
1. Taking known alternatives from existing preference rankings
2. Removing them from the rankings
3. Using the original insertion method to predict their position
4. Comparing predicted vs original position

Metrics collected:
- position_error: predicted - original (negative = too preferred)
- absolute_position_error
- epsilon comparison

Usage:
    # Run test for a single topic
    uv run python -m src.experiment_utils.test_original_insertion --topic abortion
    uv run python -m src.experiment_utils.test_original_insertion --topic environment
    
    # Run both topics
    uv run python -m src.experiment_utils.test_original_insertion --all-topics
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from .statement_insertion import insert_statement_into_ranking
from src.sample_alt_voters.config import (
    PERSONAS_PATH,
    SAMPLED_STATEMENTS_DIR,
    SAMPLED_CONTEXT_DIR,
    PHASE2_DATA_DIR,
    TOPIC_QUESTIONS,
    TOPIC_SHORT_NAMES,
    BASE_SEED,
)
from src.experiment_utils.epsilon_calculator import (
    compute_epsilon_for_new_statement,
    load_precomputed_epsilons,
)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "original_insertion_test"

# Topics to test
TEST_TOPICS = {
    "abortion": "what-should-guide-laws-concerning-abortion",
    "environment": "what-balance-should-be-struck-between-environmenta",
}


def load_personas() -> List[str]:
    """Load all adult personas."""
    with open(PERSONAS_PATH) as f:
        return json.load(f)


def load_statements_for_rep(topic_short: str, rep_id: int) -> List[Dict]:
    """
    Load statements for a specific rep.
    
    Uses persona_no_context (Alt1) distribution.
    """
    pool_path = SAMPLED_STATEMENTS_DIR / "persona_no_context" / f"{topic_short}.json"
    with open(pool_path) as f:
        pool_data = json.load(f)
    
    context_path = SAMPLED_CONTEXT_DIR / topic_short / f"rep{rep_id}.json"
    with open(context_path) as f:
        context_data = json.load(f)
    
    context_ids = context_data["context_persona_ids"]
    statements = []
    for pid in context_ids:
        if pid in pool_data["statements"]:
            statements.append({
                "id": pid,
                "statement": pool_data["statements"][pid]
            })
    
    return statements


def load_preferences_for_rep(topic_short: str, rep_id: int) -> List[List[str]]:
    """Load preferences matrix for a rep."""
    prefs_path = (PHASE2_DATA_DIR / topic_short / "uniform" / "persona_no_context" / 
                  f"rep{rep_id}" / "preferences.json")
    with open(prefs_path) as f:
        return json.load(f)


def load_voters_for_rep(topic_short: str, rep_id: int) -> Dict:
    """Load voter info for a rep."""
    voters_path = (PHASE2_DATA_DIR / topic_short / "uniform" / "persona_no_context" /
                   f"rep{rep_id}" / "voters.json")
    with open(voters_path) as f:
        return json.load(f)


def load_epsilons_for_rep(topic_short: str, rep_id: int) -> Dict[str, float]:
    """Load precomputed epsilons for a rep."""
    eps_path = (PHASE2_DATA_DIR / topic_short / "uniform" / "persona_no_context" /
                f"rep{rep_id}" / "precomputed_epsilons.json")
    with open(eps_path) as f:
        return json.load(f)


def select_test_alternatives(n_alts: int, n_test: int = 10, seed: int = 42) -> List[int]:
    """
    Select test alternatives with stratified sampling.
    
    Strategy: 3 from ranks 1-33 (top), 4 from 34-66 (middle), 3 from 67-100 (bottom)
    
    Returns list of alternative indices to test.
    """
    rng = random.Random(seed)
    
    # Stratified sampling
    top_range = list(range(0, n_alts // 3))        # Ranks 1-33 (indices 0-32)
    mid_range = list(range(n_alts // 3, 2 * n_alts // 3))  # Ranks 34-66
    bot_range = list(range(2 * n_alts // 3, n_alts))       # Ranks 67-100
    
    selected = []
    selected.extend(rng.sample(top_range, min(3, len(top_range))))
    selected.extend(rng.sample(mid_range, min(4, len(mid_range))))
    selected.extend(rng.sample(bot_range, min(3, len(bot_range))))
    
    return sorted(selected)


def get_original_position_for_voter(
    preferences: List[List[str]],
    voter_idx: int,
    alt_idx: int
) -> int:
    """
    Get the original position of an alternative in a voter's ranking.
    
    Returns 0-indexed position (0 = most preferred).
    """
    n_alts = len(preferences)
    for rank in range(n_alts):
        if preferences[rank][voter_idx] == str(alt_idx):
            return rank
    return -1  # Not found


def remove_alt_from_ranking(ranking: List[int], alt_to_remove: int) -> List[int]:
    """Remove an alternative from a ranking."""
    return [alt for alt in ranking if alt != alt_to_remove]


def test_original_insertion_for_voter(
    voter_idx: int,
    persona: str,
    original_ranking: List[int],
    statements: List[Dict],
    test_alt_idx: int,
    original_position: int,
    topic: str,
    openai_client: OpenAI,
    topic_short: str,
    rep_id: int,
) -> Dict:
    """
    Test original insertion for a single voter and alternative.
    
    Returns dict with test results.
    """
    # Get the statement text for the test alternative
    test_statement = statements[test_alt_idx]["statement"]
    
    # Remove the test alternative from the ranking
    reduced_ranking = remove_alt_from_ranking(original_ranking, test_alt_idx)
    
    # Create reduced statements list (excluding test alt)
    reduced_statements = [s for i, s in enumerate(statements) if i != test_alt_idx]
    
    # Map old indices to new indices
    old_to_new = {}
    new_idx = 0
    for old_idx in range(len(statements)):
        if old_idx != test_alt_idx:
            old_to_new[old_idx] = new_idx
            new_idx += 1
    
    # Convert ranking to new indices
    reduced_ranking_new = [old_to_new[alt] for alt in reduced_ranking]
    
    # Call original insertion method
    try:
        new_ranking = insert_statement_into_ranking(
            persona=persona,
            current_ranking=reduced_ranking_new,
            statements=reduced_statements,
            new_statement=test_statement,
            topic=topic,
            openai_client=openai_client,
            voter_dist="uniform",
            alt_dist="persona_no_context",
            method="original_insertion_test",
            rep=rep_id,
            voter_idx=voter_idx,
        )
        
        # The new statement gets index = len(reduced_statements)
        new_stmt_idx = len(reduced_statements)
        predicted_position = new_ranking.index(new_stmt_idx)
        
        # Calculate position error
        position_error = predicted_position - original_position
        
        return {
            "voter_idx": voter_idx,
            "test_alt_idx": test_alt_idx,
            "original_position": original_position,
            "predicted_position": predicted_position,
            "position_error": position_error,
            "absolute_error": abs(position_error),
            "success": True,
        }
    
    except Exception as e:
        logger.error(f"Error for voter {voter_idx}, alt {test_alt_idx}: {e}")
        return {
            "voter_idx": voter_idx,
            "test_alt_idx": test_alt_idx,
            "original_position": original_position,
            "predicted_position": None,
            "position_error": None,
            "absolute_error": None,
            "success": False,
            "error": str(e),
        }


def run_test_for_topic(
    topic_short: str,
    rep_id: int = 0,
    n_test_alts: int = 10,
    max_workers: int = 50,
) -> Dict:
    """
    Run original insertion test for a topic.
    
    Args:
        topic_short: Short topic name (e.g., "abortion")
        rep_id: Rep ID to test (default 0)
        n_test_alts: Number of alternatives to test per rep
        max_workers: Max parallel workers for API calls
    
    Returns:
        Dict with test results and summary statistics
    """
    logger.info(f"Running test for topic={topic_short}, rep={rep_id}")
    
    # Initialize OpenAI client
    openai_client = OpenAI()
    
    # Load data
    all_personas = load_personas()
    statements = load_statements_for_rep(topic_short, rep_id)
    preferences = load_preferences_for_rep(topic_short, rep_id)
    voters_info = load_voters_for_rep(topic_short, rep_id)
    precomputed_epsilons = load_epsilons_for_rep(topic_short, rep_id)
    
    topic_slug = TEST_TOPICS[topic_short]
    topic_question = TOPIC_QUESTIONS[topic_slug]
    
    voter_indices = voters_info["voter_indices"]
    n_voters = len(voter_indices)
    n_alts = len(statements)
    
    # Select test alternatives (same seed as chunked test for comparability)
    test_alt_indices = select_test_alternatives(n_alts, n_test_alts, BASE_SEED + rep_id)
    logger.info(f"Selected test alternatives: {test_alt_indices}")
    
    # Prepare all test tasks
    all_results = []
    tasks = []
    
    for test_alt_idx in test_alt_indices:
        original_epsilon = precomputed_epsilons.get(str(test_alt_idx), None)
        
        for voter_idx in range(n_voters):
            # Get voter's original ranking
            voter_ranking = [int(preferences[rank][voter_idx]) for rank in range(n_alts)]
            
            # Get original position of test alt in this voter's ranking
            original_pos = get_original_position_for_voter(preferences, voter_idx, test_alt_idx)
            
            # Get persona
            persona_idx = voter_indices[voter_idx]
            persona = all_personas[persona_idx]
            
            tasks.append({
                "voter_idx": voter_idx,
                "persona": persona,
                "original_ranking": voter_ranking,
                "test_alt_idx": test_alt_idx,
                "original_position": original_pos,
                "original_epsilon": original_epsilon,
            })
    
    logger.info(f"Total tasks: {len(tasks)} ({n_test_alts} alts × {n_voters} voters)")
    
    # Run tests in parallel
    start_time = time.time()
    
    def process_task(task):
        return test_original_insertion_for_voter(
            voter_idx=task["voter_idx"],
            persona=task["persona"],
            original_ranking=task["original_ranking"],
            statements=statements,
            test_alt_idx=task["test_alt_idx"],
            original_position=task["original_position"],
            topic=topic_question,
            openai_client=openai_client,
            topic_short=topic_short,
            rep_id=rep_id,
        )
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_task, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures),
                          desc=f"Testing {topic_short}", unit="voter"):
            task = futures[future]
            try:
                result = future.result()
                result["original_epsilon"] = task["original_epsilon"]
                all_results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {e}")
                all_results.append({
                    "voter_idx": task["voter_idx"],
                    "test_alt_idx": task["test_alt_idx"],
                    "success": False,
                    "error": str(e),
                })
    
    elapsed_time = time.time() - start_time
    
    # Compute summary statistics
    successful_results = [r for r in all_results if r.get("success", False)]
    
    if successful_results:
        position_errors = [r["position_error"] for r in successful_results]
        absolute_errors = [r["absolute_error"] for r in successful_results]
        
        summary = {
            "topic": topic_short,
            "rep_id": rep_id,
            "n_test_alts": n_test_alts,
            "n_voters": n_voters,
            "total_tests": len(all_results),
            "successful_tests": len(successful_results),
            "failed_tests": len(all_results) - len(successful_results),
            "elapsed_time": elapsed_time,
            "position_error_mean": np.mean(position_errors),
            "position_error_std": np.std(position_errors),
            "position_error_median": np.median(position_errors),
            "absolute_error_mean": np.mean(absolute_errors),
            "absolute_error_std": np.std(absolute_errors),
            "absolute_error_median": np.median(absolute_errors),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Per-alternative summary
        alt_summaries = {}
        for test_alt_idx in test_alt_indices:
            alt_results = [r for r in successful_results if r["test_alt_idx"] == test_alt_idx]
            if alt_results:
                alt_errors = [r["position_error"] for r in alt_results]
                original_pos = alt_results[0]["original_position"] if alt_results else None
                alt_summaries[str(test_alt_idx)] = {
                    "original_epsilon": precomputed_epsilons.get(str(test_alt_idx)),
                    "mean_original_position": np.mean([r["original_position"] for r in alt_results]),
                    "mean_predicted_position": np.mean([r["predicted_position"] for r in alt_results]),
                    "mean_position_error": np.mean(alt_errors),
                    "std_position_error": np.std(alt_errors),
                    "n_voters": len(alt_results),
                }
        
        summary["per_alternative"] = alt_summaries
    else:
        summary = {
            "topic": topic_short,
            "rep_id": rep_id,
            "total_tests": len(all_results),
            "successful_tests": 0,
            "failed_tests": len(all_results),
            "error": "No successful tests",
        }
    
    return {
        "summary": summary,
        "results": all_results,
    }


def save_results(results: Dict, topic_short: str, rep_id: int) -> None:
    """Save test results to disk."""
    output_path = OUTPUT_DIR / topic_short / f"rep{rep_id}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(output_path / "test_results.json", "w") as f:
        json.dump(results["results"], f, indent=2)
    
    # Save summary
    with open(output_path / "summary.json", "w") as f:
        json.dump(results["summary"], f, indent=2)
    
    logger.info(f"Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test original insertion algorithm")
    parser.add_argument("--topic", type=str, choices=["abortion", "environment"],
                       help="Topic to test")
    parser.add_argument("--all-topics", action="store_true",
                       help="Test all topics")
    parser.add_argument("--rep", type=int, default=0,
                       help="Rep ID to test (default: 0)")
    parser.add_argument("--n-alts", type=int, default=10,
                       help="Number of alternatives to test (default: 10)")
    parser.add_argument("--max-workers", type=int, default=50,
                       help="Max parallel workers (default: 50)")
    
    args = parser.parse_args()
    
    topics_to_test = []
    if args.all_topics:
        topics_to_test = list(TEST_TOPICS.keys())
    elif args.topic:
        topics_to_test = [args.topic]
    else:
        parser.error("Must specify --topic or --all-topics")
    
    for topic_short in topics_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing topic: {topic_short}")
        logger.info(f"{'='*60}")
        
        results = run_test_for_topic(
            topic_short=topic_short,
            rep_id=args.rep,
            n_test_alts=args.n_alts,
            max_workers=args.max_workers,
        )
        
        save_results(results, topic_short, args.rep)
        
        # Print summary
        summary = results["summary"]
        logger.info(f"\nSummary for {topic_short}:")
        logger.info(f"  Successful tests: {summary.get('successful_tests', 0)}/{summary.get('total_tests', 0)}")
        if summary.get("successful_tests", 0) > 0:
            logger.info(f"  Position error: mean={summary['position_error_mean']:.2f}, std={summary['position_error_std']:.2f}")
            logger.info(f"  Absolute error: mean={summary['absolute_error_mean']:.2f}, std={summary['absolute_error_std']:.2f}")
    
    logger.info("\nAll tests complete!")


if __name__ == "__main__":
    main()
