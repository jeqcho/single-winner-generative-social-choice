#!/usr/bin/env python3
"""
Run GPT**, GPT***, and Random Insertion methods using batched iterative ranking.

This is Stage 2 of the pipeline. It uses accurate batched iterative ranking
instead of single-call insertion. For each replication:
1. Generates 17 new statements per rep (1 GPT*** + 15 GPT** + 1 Random)
2. Runs ONE iterative ranking per voter with all 117 statements
3. Extracts positions relative to the 100 original statements
4. Computes epsilon for each method

Usage:
    uv run python scripts/run_gpt_star_batched.py
    uv run python scripts/run_gpt_star_batched.py --topic abortion
    uv run python scripts/run_gpt_star_batched.py --dry-run
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sample_alt_voters.config import (
    PHASE2_DATA_DIR,
    PERSONAS_PATH,
    TOPIC_QUESTIONS,
    TOPIC_SHORT_NAMES,
    BASE_SEED,
    K_SAMPLE,
    P_SAMPLE,
    N_SAMPLES_PER_REP,
)
from src.sample_alt_voters.run_experiment import load_statements_for_rep
from src.sample_alt_voters.preference_builder_iterative import subsample_preferences
from src.experiment_utils.voting_methods import (
    generate_new_statement,
    generate_new_statement_with_rankings,
    generate_new_statement_with_personas,
    generate_bridging_statement_no_context,
)
from src.experiment_utils.batched_iterative_insertion import (
    run_batched_ranking_for_rep,
    compute_epsilon_from_positions,
)

load_dotenv()

# Setup logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"run_gpt_star_batched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# Topics to process (short names)
TOPICS = ["abortion", "healthcare", "electoral", "policing", "trust", "environment"]

# Topic short name to slug mapping
TOPIC_SLUG_MAP = {v: k for k, v in TOPIC_SHORT_NAMES.items()}


def load_personas() -> List[str]:
    """Load all adult personas."""
    with open(PERSONAS_PATH) as f:
        return json.load(f)


def parse_rep_path(rep_dir: Path) -> Tuple[str, str, str, int, str]:
    """Parse rep directory path to extract topic, voter_dist, alt_dist, rep_id.
    
    Handles two structures:
    - Uniform: data/{topic}/uniform/{alt_dist}/rep{id}
    - Clustered: data/{topic}/clustered/{cluster}/{alt_dist}/rep{id}
    """
    parts = rep_dir.parts
    data_idx = parts.index('data')
    topic = parts[data_idx + 1]
    voter_type = parts[data_idx + 2]  # "uniform" or "clustered"
    
    if voter_type == "uniform":
        alt_dist = parts[data_idx + 3]
        rep_name = parts[data_idx + 4]
        voter_dist = "uniform"
    else:  # clustered
        voter_dist = parts[data_idx + 3]  # progressive_liberal or conservative_traditional
        alt_dist = parts[data_idx + 4]
        rep_name = parts[data_idx + 5]
    
    rep_id = int(rep_name.replace('rep', ''))
    return topic, voter_dist, alt_dist, rep_id, rep_name


def generate_all_statements_for_minireps(
    rep_dir: Path,
    topic_question: str,
    topic_short: str,
    all_statements: List[Dict],
    all_personas: List[str],
    global_voter_indices: List[int],
    full_preferences: List[List[str]],
    openai_client: OpenAI,
    voter_dist: str = None,
    alt_dist: str = None,
    rep_id: int = None,
) -> List[Dict]:
    """Generate all statements (GPT**, GPT***, random_insertion) for all mini-reps.
    
    Per mini-rep: 3 GPT** + 1 GPT*** + 1 random_insertion = 5 methods
    Total: N_SAMPLES_PER_REP mini-reps × 5 methods = 20 statements
    """
    new_statements = []
    all_rep_personas = [all_personas[idx] for idx in global_voter_indices]
    
    for mini_rep_id in range(N_SAMPLES_PER_REP):
        mini_rep_dir = rep_dir / f"mini_rep{mini_rep_id}"
        results_path = mini_rep_dir / "results.json"
        
        if not results_path.exists():
            logger.warning(f"Results file not found: {results_path}")
            continue
        
        with open(results_path) as f:
            data = json.load(f)
        
        voter_indices = data['voter_indices']
        alt_indices = data['alt_indices']
        
        sample_statements = [all_statements[i] for i in alt_indices]
        sample_personas = [all_rep_personas[i] for i in voter_indices]
        sample_prefs, _, _ = subsample_preferences(
            full_preferences, k_voters=K_SAMPLE, p_alts=P_SAMPLE,
            seed=BASE_SEED + mini_rep_id * 100
        )
        
        # GPT** base
        logger.info(f"    Generating GPT** base for mini_rep{mini_rep_id}...")
        stmt = generate_new_statement(
            sample_statements, topic_question, openai_client,
            voter_dist=voter_dist, alt_dist=alt_dist, rep=rep_id, mini_rep=mini_rep_id
        )
        if stmt:
            new_statements.append({"statement": stmt, "method": "chatgpt_double_star", "mini_rep": mini_rep_id})
        
        # GPT** + Rankings
        logger.info(f"    Generating GPT**+Rank for mini_rep{mini_rep_id}...")
        stmt = generate_new_statement_with_rankings(
            sample_statements, sample_prefs, topic_question, openai_client,
            voter_dist=voter_dist, alt_dist=alt_dist, rep=rep_id, mini_rep=mini_rep_id
        )
        if stmt:
            new_statements.append({"statement": stmt, "method": "chatgpt_double_star_rankings", "mini_rep": mini_rep_id})
        
        # GPT** + Personas
        logger.info(f"    Generating GPT**+Pers for mini_rep{mini_rep_id}...")
        stmt = generate_new_statement_with_personas(
            sample_statements, sample_personas, topic_question, openai_client,
            voter_dist=voter_dist, alt_dist=alt_dist, rep=rep_id, mini_rep=mini_rep_id
        )
        if stmt:
            new_statements.append({"statement": stmt, "method": "chatgpt_double_star_personas", "mini_rep": mini_rep_id})
        
        # GPT*** (one per mini-rep)
        triple_star = generate_triple_star_statement(
            topic_question, openai_client,
            voter_dist=voter_dist, alt_dist=alt_dist, rep_id=rep_id, mini_rep_id=mini_rep_id
        )
        if triple_star:
            new_statements.append(triple_star)
        
        # Random insertion (one per mini-rep, different seed per mini-rep)
        random_stmt = sample_random_statement(
            topic_short=topic_short, rep_id=rep_id, mini_rep_id=mini_rep_id
        )
        if random_stmt:
            new_statements.append(random_stmt)
    
    return new_statements


def generate_triple_star_statement(
    topic_question: str, openai_client: OpenAI,
    voter_dist: str = None, alt_dist: str = None, rep_id: int = None, mini_rep_id: int = None,
) -> Optional[Dict]:
    """Generate GPT*** statement for a specific mini-rep."""
    logger.info(f"    Generating GPT*** for mini_rep{mini_rep_id}...")
    stmt = generate_bridging_statement_no_context(
        topic_question, openai_client,
        voter_dist=voter_dist, alt_dist=alt_dist, rep=rep_id
    )
    if stmt:
        return {"statement": stmt, "method": "chatgpt_triple_star", "mini_rep": mini_rep_id}
    return None


def sample_random_statement(topic_short: str, rep_id: int, mini_rep_id: int, seed: int = None) -> Optional[Dict]:
    """
    Sample a random statement from the global pool that is NOT in the current rep.
    
    This serves as a baseline - random statements should have higher epsilon than
    GPT-generated bridging statements.
    
    Args:
        topic_short: Short topic name (e.g., "abortion", "healthcare")
        rep_id: Replication ID
        mini_rep_id: Mini-replication ID
        seed: Random seed for reproducibility (default: BASE_SEED + rep_id * 1000 + mini_rep_id)
    
    Returns:
        Dict with statement and method, or None if sampling fails
    """
    if seed is None:
        seed = BASE_SEED + rep_id * 1000 + mini_rep_id
    project_root = Path(__file__).parent.parent
    pool_path = project_root / "data" / "sample-alt-voters" / "sampled-statements" / "persona_no_context" / f"{topic_short}.json"
    context_path = project_root / "data" / "sample-alt-voters" / "sampled-context" / topic_short / f"rep{rep_id}.json"
    
    if not pool_path.exists():
        logger.error(f"Global pool not found: {pool_path}")
        return None
    
    if not context_path.exists():
        logger.error(f"Context file not found: {context_path}")
        return None
    
    # Load global pool
    with open(pool_path) as f:
        pool_data = json.load(f)
    
    # Load context to get which 100 IDs are used in this rep
    with open(context_path) as f:
        context_data = json.load(f)
    
    used_ids = set(context_data.get("context_persona_ids", []))
    pool_statements = pool_data.get("statements", {})
    
    # Find IDs NOT in the current rep
    available_ids = [pid for pid in pool_statements.keys() if pid not in used_ids]
    
    if not available_ids:
        logger.error("No available statements outside current rep")
        return None
    
    # Sample a random statement
    rng = random.Random(seed)
    sampled_id = rng.choice(available_ids)
    sampled_statement = pool_statements[sampled_id]
    
    logger.info(f"  Random insertion: Sampled statement ID {sampled_id} from {len(available_ids)} available")
    logger.info(f"    Statement: {sampled_statement[:100]}...")
    
    return {
        "statement": sampled_statement,
        "method": "random_insertion",
        "mini_rep": mini_rep_id,
        "sampled_id": sampled_id,
    }


def save_results(
    rep_dir: Path,
    all_positions: Dict[str, List[Optional[int]]],
    new_statements: List[Dict],
    original_statements: List[Dict],
):
    """Save all results to mini-rep results.json files.
    
    All methods (GPT**, GPT***, random_insertion) are saved to their respective
    mini-rep results.json files based on the mini_rep field.
    """
    n_originals = len(original_statements)
    results_by_mini_rep: Dict[int, Dict[str, Dict]] = {}
    
    for stmt_data in new_statements:
        method = stmt_data["method"]
        mini_rep = stmt_data.get("mini_rep")
        statement = stmt_data["statement"]
        
        if mini_rep is None:
            logger.warning(f"Statement for method {method} has no mini_rep, skipping")
            continue
        
        key = f"{method}_mr{mini_rep}"
        positions = all_positions.get(key, [])
        epsilon = compute_epsilon_from_positions(positions, original_statements)
        
        if mini_rep not in results_by_mini_rep:
            results_by_mini_rep[mini_rep] = {}
        
        result_data = {
            "winner": str(n_originals),
            "new_statement": statement,
            "is_new": True,
            "insertion_positions": positions,
            "epsilon": epsilon,
        }
        
        # Add sampled_id for random_insertion
        if method == "random_insertion" and "sampled_id" in stmt_data:
            result_data["sampled_id"] = stmt_data["sampled_id"]
        
        results_by_mini_rep[mini_rep][method] = result_data
    
    # Save all results to mini-rep results.json files
    for mini_rep_id, methods in results_by_mini_rep.items():
        results_path = rep_dir / f"mini_rep{mini_rep_id}" / "results.json"
        if not results_path.exists():
            logger.warning(f"Results file not found: {results_path}")
            continue
        with open(results_path) as f:
            data = json.load(f)
        for method, result in methods.items():
            data['results'][method] = result
            logger.info(f"  Saved {method} mini_rep{mini_rep_id}: epsilon={result['epsilon']}")
        with open(results_path, 'w') as f:
            json.dump(data, f, indent=2)


def process_rep(
    rep_dir: Path, topic_short: str, rep_id: int,
    all_personas: List[str], openai_client: OpenAI,
) -> Dict:
    """Process a single rep: generate statements and run batched ranking."""
    topic_slug = TOPIC_SLUG_MAP.get(topic_short, topic_short)
    topic_question = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    _, voter_dist, alt_dist, _, _ = parse_rep_path(rep_dir)
    
    statements = load_statements_for_rep(topic_slug, "persona_no_context", rep_id)
    with open(rep_dir / "voters.json") as f:
        voters_data = json.load(f)
    global_voter_indices = voters_data["voter_indices"]
    with open(rep_dir / "preferences.json") as f:
        full_preferences = json.load(f)
    
    voter_personas = [all_personas[i] for i in global_voter_indices]
    stats = {"statements_generated": 0, "ranking_success": False}
    
    # Phase 1: Generate all statements (GPT**, GPT***, random_insertion) for all mini-reps
    logger.info(f"  Phase 1: Generating new statements for {N_SAMPLES_PER_REP} mini-reps...")
    new_statements = generate_all_statements_for_minireps(
        rep_dir, topic_question, topic_short, statements, all_personas,
        global_voter_indices, full_preferences, openai_client,
        voter_dist=voter_dist, alt_dist=alt_dist, rep_id=rep_id
    )
    stats["statements_generated"] = len(new_statements)
    
    logger.info(f"  Generated {len(new_statements)} new statements (target: {N_SAMPLES_PER_REP * 5})")
    if not new_statements:
        logger.error(f"  No statements generated, skipping ranking")
        return stats
    
    # Phase 2: Batched ranking
    logger.info(f"  Phase 2: Running batched iterative ranking for 100 voters...")
    all_positions = run_batched_ranking_for_rep(
        original_statements=statements,
        new_statements=new_statements,
        voter_personas=voter_personas,
        topic=topic_question,
        openai_client=openai_client,
        voter_dist=voter_dist,
        alt_dist=alt_dist,
        rep=rep_id,
        max_workers=50,
    )
    stats["ranking_success"] = True
    
    # Phase 3: Save results
    logger.info(f"  Phase 3: Saving results...")
    save_results(rep_dir, all_positions, new_statements, statements)
    
    return stats


def rep_already_processed(rep_dir: Path) -> bool:
    """Check if a rep already has GPT*** results in mini_rep0."""
    results_path = rep_dir / "mini_rep0" / "results.json"
    if not results_path.exists():
        return False
    try:
        with open(results_path) as f:
            data = json.load(f)
        return "chatgpt_triple_star" in data.get("results", {})
    except Exception:
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run GPT**/GPT***/Random with batched iterative ranking")
    parser.add_argument("--topic", type=str, choices=TOPICS, help="Run for a specific topic only")
    parser.add_argument("--dry-run", action="store_true", help="List reps without running")
    parser.add_argument("--skip-completed", action="store_true", help="Skip reps that already have GPT*** results")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Running GPT**/GPT***/Random with Batched Iterative Ranking")
    logger.info("=" * 60)
    
    topics_to_run = [args.topic] if args.topic else TOPICS
    logger.info(f"Topics: {topics_to_run}")
    logger.info(f"Log file: {LOG_FILE}")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)
    openai_client = OpenAI(api_key=api_key)
    
    logger.info("\nLoading personas...")
    all_personas = load_personas()
    logger.info(f"Loaded {len(all_personas)} personas")
    
    all_reps = []
    for topic in topics_to_run:
        topic_dir = PHASE2_DATA_DIR / topic
        if not topic_dir.exists():
            continue
        for pref_file in topic_dir.glob("**/persona_no_context/**/preferences.json"):
            rep_dir = pref_file.parent
            try:
                parsed = parse_rep_path(rep_dir)
                all_reps.append((rep_dir, parsed))
            except Exception as e:
                logger.warning(f"Failed to parse {rep_dir}: {e}")
    
    logger.info(f"\nFound {len(all_reps)} reps to process")
    
    # Filter out already-completed reps if --skip-completed
    if args.skip_completed:
        original_count = len(all_reps)
        all_reps = [(rep_dir, parsed) for rep_dir, parsed in all_reps if not rep_already_processed(rep_dir)]
        skipped = original_count - len(all_reps)
        logger.info(f"Skipping {skipped} already-completed reps, {len(all_reps)} remaining")
    
    if args.dry_run:
        logger.info("\nDry run - listing reps:")
        for rep_dir, (topic, voter_dist, alt_dist, rep_id, rep_name) in all_reps:
            logger.info(f"  {topic}/{voter_dist}/{alt_dist}/{rep_name}")
        logger.info(f"\nTotal: {len(all_reps)} reps")
        logger.info(f"Est. API calls: {len(all_reps) * 100 * 6} ranking (6 rounds for 117 stmts) + {len(all_reps) * 16} generation")
        return
    
    total_stats = {"reps_processed": 0, "statements_generated": 0, "ranking_success": 0}
    
    for rep_dir, (topic, voter_dist, alt_dist, rep_id, rep_name) in tqdm(all_reps, desc="Processing reps"):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {topic}/{voter_dist}/{alt_dist}/{rep_name}")
        logger.info(f"{'=' * 60}")
        
        try:
            stats = process_rep(rep_dir, topic, rep_id, all_personas, openai_client)
            total_stats["reps_processed"] += 1
            total_stats["statements_generated"] += stats["statements_generated"]
            if stats["ranking_success"]:
                total_stats["ranking_success"] += 1
        except Exception as e:
            logger.error(f"Error processing {rep_dir}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Reps processed: {total_stats['reps_processed']}")
    logger.info(f"Statements generated: {total_stats['statements_generated']}")
    logger.info(f"Rankings successful: {total_stats['ranking_success']}")
    logger.info(f"Log file: {LOG_FILE}")


if __name__ == "__main__":
    main()
