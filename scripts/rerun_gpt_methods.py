#!/usr/bin/env python3
"""
Re-run GPT** (3 variants), GPT***, and New Random methods for Alt1.

This script:
- Processes only persona_no_context (Alt1) directories
- Runs for 6 selected topics across all 3 voter distributions
- Re-runs GPT** (double star) methods at mini-rep level
- Re-runs GPT*** (triple star) at rep level
- Runs New Random (sanity check) at mini-rep level
- Stores insertion_positions for all methods

Usage:
    uv run python scripts/rerun_gpt_methods.py
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

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
)
from src.sample_alt_voters.run_experiment import load_statements_for_rep
from src.sample_alt_voters.preference_builder_iterative import subsample_preferences
from src.experiment_utils.voting_methods import (
    run_chatgpt_double_star,
    run_chatgpt_double_star_with_rankings,
    run_chatgpt_double_star_with_personas,
    run_chatgpt_triple_star,
    run_new_random,
)

load_dotenv()

# Setup logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"rerun_gpt_methods_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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


def parse_rep_path(rep_dir: Path) -> tuple:
    """Parse rep directory path to extract topic, voter_dist, alt_dist, rep_id."""
    parts = rep_dir.parts
    data_idx = parts.index('data')
    topic = parts[data_idx + 1]
    voter_dist = parts[data_idx + 2]
    alt_dist = parts[data_idx + 3]
    rep_name = parts[data_idx + 4]
    # Extract rep_id from e.g. "rep0" or "rep1_conservative_traditional"
    rep_id = int(rep_name.split('_')[0].replace('rep', ''))
    return topic, voter_dist, alt_dist, rep_id, rep_name


def run_triple_star_for_rep(
    rep_dir: Path,
    all_statements: List[Dict],
    all_personas: List[str],
    global_voter_indices: List[int],
    topic_question: str,
    openai_client: OpenAI,
) -> Dict:
    """Run GPT*** for a rep and save results."""
    output_path = rep_dir / "chatgpt_triple_star.json"
    
    # Load full preferences
    with open(rep_dir / "preferences.json") as f:
        full_preferences = json.load(f)
    
    # Get the personas for the 100 sampled voters
    voter_personas = [all_personas[i] for i in global_voter_indices]
    
    logger.info(f"  Running GPT*** for {rep_dir.name}...")
    
    try:
        result = run_chatgpt_triple_star(
            topic=topic_question,
            all_statements=all_statements,
            voter_personas=voter_personas,
            full_preferences=full_preferences,
            openai_client=openai_client,
            n_generations=1,
        )
        
        # Save result
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"  GPT*** epsilon: {result.get('epsilon', 'N/A')}")
        return result
        
    except Exception as e:
        logger.error(f"  GPT*** failed: {e}")
        return {"error": str(e)}


def run_double_star_and_random_for_mini_rep(
    rep_dir: Path,
    mini_rep_id: int,
    topic_short: str,
    topic_question: str,
    rep_id: int,
    all_statements: List[Dict],
    all_personas: List[str],
    global_voter_indices: List[int],
    openai_client: OpenAI,
) -> int:
    """
    Run GPT** (3 variants) and New Random for a mini-rep.
    
    Returns number of methods successfully run.
    """
    mini_rep_dir = rep_dir / f"mini_rep{mini_rep_id}"
    results_path = mini_rep_dir / "results.json"
    
    if not results_path.exists():
        logger.warning(f"    Results file not found: {results_path}")
        return 0
    
    # Load existing results
    with open(results_path) as f:
        data = json.load(f)
    
    results = data['results']
    voter_indices = data['voter_indices']
    alt_indices = data['alt_indices']
    
    # Load full preferences
    with open(rep_dir / "preferences.json") as f:
        full_preferences = json.load(f)
    
    # Get the 100 voter personas for this rep
    all_rep_personas = [all_personas[idx] for idx in global_voter_indices]
    
    # Get sampled statements and personas for mini-rep
    sample_statements = [all_statements[i] for i in alt_indices]
    sample_personas = [all_rep_personas[i] for i in voter_indices]
    
    # Subsample preferences for mini-rep
    sample_prefs, _, _ = subsample_preferences(
        full_preferences,
        k_voters=K_SAMPLE,
        p_alts=P_SAMPLE,
        seed=BASE_SEED + mini_rep_id * 100
    )
    
    methods_run = 0
    
    # Run GPT** base
    logger.info(f"    Running chatgpt_double_star...")
    try:
        result = run_chatgpt_double_star(
            sample_statements=sample_statements,
            all_statements=all_statements,
            sample_personas=sample_personas,
            full_preferences=full_preferences,
            voter_indices=voter_indices,
            topic=topic_question,
            openai_client=openai_client,
            voter_personas=all_rep_personas,
        )
        results["chatgpt_double_star"] = result
        methods_run += 1
        logger.info(f"      epsilon: {result.get('epsilon', 'N/A')}")
    except Exception as e:
        logger.error(f"      Failed: {e}")
        results["chatgpt_double_star"] = {"winner": None, "error": str(e)}
    
    # Run GPT** with rankings
    logger.info(f"    Running chatgpt_double_star_rankings...")
    try:
        result = run_chatgpt_double_star_with_rankings(
            sample_statements=sample_statements,
            sample_preferences=sample_prefs,
            all_statements=all_statements,
            sample_personas=sample_personas,
            full_preferences=full_preferences,
            voter_indices=voter_indices,
            topic=topic_question,
            openai_client=openai_client,
            voter_personas=all_rep_personas,
        )
        results["chatgpt_double_star_rankings"] = result
        methods_run += 1
        logger.info(f"      epsilon: {result.get('epsilon', 'N/A')}")
    except Exception as e:
        logger.error(f"      Failed: {e}")
        results["chatgpt_double_star_rankings"] = {"winner": None, "error": str(e)}
    
    # Run GPT** with personas
    logger.info(f"    Running chatgpt_double_star_personas...")
    try:
        result = run_chatgpt_double_star_with_personas(
            sample_statements=sample_statements,
            sample_personas=sample_personas,
            all_statements=all_statements,
            full_preferences=full_preferences,
            voter_indices=voter_indices,
            topic=topic_question,
            openai_client=openai_client,
            voter_personas=all_rep_personas,
        )
        results["chatgpt_double_star_personas"] = result
        methods_run += 1
        logger.info(f"      epsilon: {result.get('epsilon', 'N/A')}")
    except Exception as e:
        logger.error(f"      Failed: {e}")
        results["chatgpt_double_star_personas"] = {"winner": None, "error": str(e)}
    
    # Run New Random
    logger.info(f"    Running new_random...")
    try:
        # Use a seed based on rep_id and mini_rep_id for reproducibility
        seed = BASE_SEED + rep_id * 1000 + mini_rep_id
        result = run_new_random(
            topic_short=topic_short,
            rep_id=rep_id,
            all_statements=all_statements,
            voter_personas=all_rep_personas,
            full_preferences=full_preferences,
            openai_client=openai_client,
            seed=seed,
        )
        results["new_random"] = result
        methods_run += 1
        logger.info(f"      epsilon: {result.get('epsilon', 'N/A')}")
    except Exception as e:
        logger.error(f"      Failed: {e}")
        results["new_random"] = {"winner": None, "error": str(e)}
    
    # Save updated results
    with open(results_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return methods_run


def process_rep(
    rep_dir: Path,
    topic_short: str,
    rep_id: int,
    all_personas: List[str],
    openai_client: OpenAI,
) -> Dict:
    """Process a single rep: run GPT*** and GPT**/New Random for all mini-reps."""
    
    topic_slug = TOPIC_SLUG_MAP.get(topic_short, topic_short)
    topic_question = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    
    # Load statements
    # For clustered reps, we need to use rep_id 0 or 1 depending on cluster
    # But load_statements_for_rep handles this correctly
    statements = load_statements_for_rep(topic_slug, "persona_no_context", rep_id)
    
    # Load voters.json to get global persona indices
    with open(rep_dir / "voters.json") as f:
        voters_data = json.load(f)
    global_voter_indices = voters_data["voter_indices"]
    
    stats = {
        "triple_star_success": False,
        "mini_reps_processed": 0,
        "methods_run": 0,
    }
    
    # Run GPT***
    result = run_triple_star_for_rep(
        rep_dir=rep_dir,
        all_statements=statements,
        all_personas=all_personas,
        global_voter_indices=global_voter_indices,
        topic_question=topic_question,
        openai_client=openai_client,
    )
    if "error" not in result:
        stats["triple_star_success"] = True
    
    # Run GPT** and New Random for each mini-rep
    for mini_rep_id in range(5):
        logger.info(f"  Processing mini_rep{mini_rep_id}...")
        methods = run_double_star_and_random_for_mini_rep(
            rep_dir=rep_dir,
            mini_rep_id=mini_rep_id,
            topic_short=topic_short,
            topic_question=topic_question,
            rep_id=rep_id,
            all_statements=statements,
            all_personas=all_personas,
            global_voter_indices=global_voter_indices,
            openai_client=openai_client,
        )
        stats["mini_reps_processed"] += 1
        stats["methods_run"] += methods
    
    return stats


def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("Re-running GPT**/GPT***/New Random for Alt1")
    logger.info("="*60)
    logger.info(f"Topics: {TOPICS}")
    logger.info(f"Alt distribution: persona_no_context only")
    logger.info(f"Voter distributions: uniform, progressive_liberal, conservative_traditional")
    logger.info(f"Log file: {LOG_FILE}")
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)
    openai_client = OpenAI(api_key=api_key)
    
    # Load personas
    logger.info("\nLoading personas...")
    all_personas = load_personas()
    logger.info(f"Loaded {len(all_personas)} personas")
    
    # Find all matching rep directories
    all_reps = []
    for topic in TOPICS:
        topic_dir = PHASE2_DATA_DIR / topic
        if not topic_dir.exists():
            logger.warning(f"Topic directory not found: {topic_dir}")
            continue
        
        # Find persona_no_context reps
        for pref_file in topic_dir.glob("**/persona_no_context/**/preferences.json"):
            rep_dir = pref_file.parent
            try:
                parsed = parse_rep_path(rep_dir)
                all_reps.append((rep_dir, parsed))
            except Exception as e:
                logger.warning(f"Failed to parse {rep_dir}: {e}")
    
    logger.info(f"\nFound {len(all_reps)} reps to process")
    
    # Process each rep
    total_stats = {
        "reps_processed": 0,
        "triple_star_success": 0,
        "mini_reps_processed": 0,
        "methods_run": 0,
    }
    
    for rep_dir, (topic, voter_dist, alt_dist, rep_id, rep_name) in tqdm(all_reps, desc="Processing reps"):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {topic}/{voter_dist}/{alt_dist}/{rep_name}")
        logger.info(f"{'='*60}")
        
        try:
            stats = process_rep(
                rep_dir=rep_dir,
                topic_short=topic,
                rep_id=rep_id,
                all_personas=all_personas,
                openai_client=openai_client,
            )
            
            total_stats["reps_processed"] += 1
            if stats["triple_star_success"]:
                total_stats["triple_star_success"] += 1
            total_stats["mini_reps_processed"] += stats["mini_reps_processed"]
            total_stats["methods_run"] += stats["methods_run"]
            
        except Exception as e:
            logger.error(f"Error processing {rep_dir}: {e}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Reps processed: {total_stats['reps_processed']}")
    logger.info(f"GPT*** successful: {total_stats['triple_star_success']}")
    logger.info(f"Mini-reps processed: {total_stats['mini_reps_processed']}")
    logger.info(f"Methods run: {total_stats['methods_run']}")
    logger.info(f"Log file: {LOG_FILE}")


if __name__ == "__main__":
    main()
