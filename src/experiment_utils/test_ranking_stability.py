"""
Test ranking stability by running iterative rankings multiple times.

This script tests whether the iterative ranking process produces stable 
preference profiles by running the same ranking multiple times for the 
same voters and comparing results.

Usage:
    # Run for a single topic
    uv run python -m src.experiment_utils.test_ranking_stability --topic abortion
    
    # Run for all topics
    uv run python -m src.experiment_utils.test_ranking_stability --all-topics
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.degeneracy_mitigation.iterative_ranking import rank_voter
from src.degeneracy_mitigation.config import HASH_SEED
from src.experiment_utils.config import RANKING_REASONING
from src.sample_alt_voters.config import (
    PERSONAS_PATH,
    SAMPLED_STATEMENTS_DIR,
    SAMPLED_CONTEXT_DIR,
    PHASE2_DATA_DIR,
    TOPIC_QUESTIONS,
)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ranking_stability_test"

# Topics to test
TEST_TOPICS = {
    "abortion": "what-should-guide-laws-concerning-abortion",
    "environment": "what-balance-should-be-struck-between-environmenta",
}


def load_personas():
    """Load all adult personas."""
    with open(PERSONAS_PATH) as f:
        return json.load(f)


def load_statements_for_rep(topic_short, rep_id):
    """Load statements for a specific rep."""
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


def load_voters_for_rep(topic_short, rep_id):
    """Load voter info for a rep."""
    voters_path = (PHASE2_DATA_DIR / topic_short / "uniform" / "persona_no_context" /
                   f"rep{rep_id}" / "voters.json")
    with open(voters_path) as f:
        return json.load(f)


def run_single_ranking(
    openai_client,
    voter_idx,
    persona,
    statements,
    topic,
    iteration_num,
):
    """Run a single ranking for one voter."""
    # Use a different hash seed for each iteration to avoid cached results
    # The voter_seed inside rank_voter controls shuffling per round
    result = rank_voter(
        client=openai_client,
        voter_idx=voter_idx,
        persona=persona,
        statements=statements,
        topic=topic,
        reasoning_effort=RANKING_REASONING,
        hash_seed=HASH_SEED,
        voter_dist="uniform",
        alt_dist="persona_no_context",
        rep=0,
    )
    
    return {
        "voter_idx": voter_idx,
        "iteration": iteration_num,
        "ranking": result.get("ranking", []),
        "all_valid": result.get("all_valid", False),
        "total_retries": result.get("total_retries", 0),
    }


def run_stability_test(
    topic_short,
    n_voters=10,
    n_iterations=10,
    rep_id=0,
    max_workers=50,
):
    """
    Run stability test for a topic.
    
    Args:
        topic_short: Short topic name (e.g., "abortion")
        n_voters: Number of voters to test
        n_iterations: Number of times to run ranking per voter
        rep_id: Replication ID to use for voters/statements
        max_workers: Max parallel workers
    
    Returns:
        Dict with test results
    """
    logger.info(f"Running stability test for {topic_short}")
    logger.info(f"  Voters: {n_voters}, Iterations: {n_iterations}")
    
    # Initialize OpenAI client
    openai_client = OpenAI()
    
    # Load data
    all_personas = load_personas()
    statements = load_statements_for_rep(topic_short, rep_id)
    voters_info = load_voters_for_rep(topic_short, rep_id)
    
    topic_slug = TEST_TOPICS[topic_short]
    topic_question = TOPIC_QUESTIONS[topic_slug]
    
    # Get first n_voters
    voter_indices = voters_info["voter_indices"][:n_voters]
    
    logger.info(f"Testing {n_voters} voters with {n_iterations} iterations each")
    logger.info(f"Total tasks: {n_voters * n_iterations}")
    logger.info(f"Total API calls: {n_voters * n_iterations * 5}")
    
    # Prepare all tasks
    tasks = []
    for local_idx, global_idx in enumerate(voter_indices):
        persona = all_personas[global_idx]
        for iteration in range(n_iterations):
            tasks.append({
                "local_idx": local_idx,
                "global_idx": global_idx,
                "persona": persona,
                "iteration": iteration,
            })
    
    # Run tasks in parallel
    all_results = []
    start_time = time.time()
    
    def process_task(task):
        return run_single_ranking(
            openai_client=openai_client,
            voter_idx=task["local_idx"],
            persona=task["persona"],
            statements=statements,
            topic=topic_question,
            iteration_num=task["iteration"],
        )
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_task, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures),
                          desc=f"Testing {topic_short}", unit="ranking"):
            task = futures[future]
            try:
                result = future.result()
                result["global_voter_idx"] = task["global_idx"]
                all_results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {e}")
                all_results.append({
                    "voter_idx": task["local_idx"],
                    "global_voter_idx": task["global_idx"],
                    "iteration": task["iteration"],
                    "error": str(e),
                })
    
    elapsed_time = time.time() - start_time
    
    # Organize results by voter
    results_by_voter = {}
    for result in all_results:
        voter_idx = result["voter_idx"]
        if voter_idx not in results_by_voter:
            results_by_voter[voter_idx] = {
                "global_voter_idx": result.get("global_voter_idx"),
                "rankings": [],
            }
        if "ranking" in result:
            results_by_voter[voter_idx]["rankings"].append({
                "iteration": result["iteration"],
                "ranking": result["ranking"],
                "all_valid": result.get("all_valid", False),
                "total_retries": result.get("total_retries", 0),
            })
    
    # Compute basic stats
    n_successful = sum(1 for r in all_results if "ranking" in r and r.get("all_valid", False))
    n_failed = len(all_results) - n_successful
    
    summary = {
        "topic": topic_short,
        "n_voters": n_voters,
        "n_iterations": n_iterations,
        "total_tasks": len(tasks),
        "successful_tasks": n_successful,
        "failed_tasks": n_failed,
        "elapsed_time": elapsed_time,
        "timestamp": datetime.now().isoformat(),
    }
    
    return {
        "summary": summary,
        "results_by_voter": results_by_voter,
    }


def save_results(results, topic_short):
    """Save test results to disk."""
    output_path = OUTPUT_DIR / topic_short
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save raw rankings
    with open(output_path / "raw_rankings.json", "w") as f:
        json.dump(results["results_by_voter"], f, indent=2)
    
    # Save summary
    with open(output_path / "summary.json", "w") as f:
        json.dump(results["summary"], f, indent=2)
    
    logger.info(f"Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Test ranking stability")
    parser.add_argument("--topic", type=str, choices=["abortion", "environment"],
                       help="Topic to test")
    parser.add_argument("--all-topics", action="store_true",
                       help="Test all topics")
    parser.add_argument("--n-voters", type=int, default=10,
                       help="Number of voters to test (default: 10)")
    parser.add_argument("--n-iterations", type=int, default=10,
                       help="Number of iterations per voter (default: 10)")
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
        
        results = run_stability_test(
            topic_short=topic_short,
            n_voters=args.n_voters,
            n_iterations=args.n_iterations,
            max_workers=args.max_workers,
        )
        
        save_results(results, topic_short)
        
        # Print summary
        summary = results["summary"]
        logger.info(f"\nSummary for {topic_short}:")
        logger.info(f"  Successful tasks: {summary['successful_tasks']}/{summary['total_tasks']}")
        logger.info(f"  Elapsed time: {summary['elapsed_time']:.1f}s")
    
    logger.info("\nAll tests complete!")


if __name__ == "__main__":
    main()
