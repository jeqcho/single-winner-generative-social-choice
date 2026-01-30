#!/usr/bin/env python3
"""
Fix GPT** epsilon values for multiple topics.

The bug: GPT** epsilon was computed using gpt-5.2 for insertion instead of gpt-5-mini.
The fix: Re-run insertions with the correct model and recompute epsilon.

Topics to fix: policing, trust, environment, abortion
(electoral already fixed by previous script, healthcare ran after bug fix)

Usage:
    uv run python scripts/fix_double_star_epsilons.py
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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
)
from src.sample_alt_voters.run_experiment import load_statements_for_rep
from src.experiment_utils.voting_methods import insert_new_statement_into_rankings
from src.experiment_utils.epsilon_calculator import compute_epsilon_for_new_statement

load_dotenv()

# Setup logging
LOG_FILE = Path(__file__).parent.parent / "logs" / f"fix_double_star_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# GPT** methods to fix
DOUBLE_STAR_METHODS = [
    "chatgpt_double_star",
    "chatgpt_double_star_rankings",
    "chatgpt_double_star_personas",
]

# Topics to fix (short name -> slug)
TOPICS_TO_FIX = {
    "policing": "what-strategies-should-guide-policing-to-address-b",
    "trust": "how-should-we-increase-the-general-publics-trust-i",
    "environment": "what-balance-should-be-struck-between-environmenta",
    "abortion": "what-should-guide-laws-concerning-abortion",
}


def load_personas() -> List[str]:
    """Load all adult personas."""
    with open(PERSONAS_PATH) as f:
        return json.load(f)


def parse_rep_path(rep_dir: Path) -> tuple:
    """Parse rep directory path to extract voter_dist, alt_dist, rep_id."""
    parts = rep_dir.parts
    data_idx = parts.index('data')
    voter_dist = parts[data_idx + 2]
    alt_dist = parts[data_idx + 3]
    rep_name = parts[data_idx + 4]
    rep_id = int(rep_name.split('_')[0].replace('rep', ''))
    return voter_dist, alt_dist, rep_id


def fix_mini_rep(
    rep_dir: Path,
    mini_rep_id: int,
    all_statements: List[Dict],
    all_personas: List[str],
    global_voter_indices: List[int],
    topic_question: str,
    openai_client: OpenAI,
) -> int:
    """
    Fix GPT** epsilon values for a single mini-rep.
    
    Returns number of fixes made.
    """
    mini_rep_dir = rep_dir / f"mini_rep{mini_rep_id}"
    results_path = mini_rep_dir / "results.json"
    
    if not results_path.exists():
        return 0
    
    # Load existing results
    with open(results_path) as f:
        data = json.load(f)
    
    results = data['results']
    
    # Load full preferences
    with open(rep_dir / "preferences.json") as f:
        full_preferences = json.load(f)
    
    # Get the 100 voter personas for this rep
    all_rep_personas = [all_personas[idx] for idx in global_voter_indices]
    all_voter_indices = list(range(len(global_voter_indices)))
    
    fixes = 0
    
    for method in DOUBLE_STAR_METHODS:
        if method not in results:
            continue
        
        result = results[method]
        new_statement = result.get('new_statement')
        
        # Skip if no new statement
        if not new_statement:
            continue
        
        logger.info(f"    Fixing {method}...")
        
        try:
            # Re-run insertion with correct model (RANKING_MODEL)
            updated_prefs, insertion_positions = insert_new_statement_into_rankings(
                new_statement=new_statement,
                all_statements=all_statements,
                voter_personas=all_rep_personas,
                voter_indices=all_voter_indices,
                full_preferences=full_preferences,
                topic=topic_question,
                openai_client=openai_client,
            )
            
            # Compute epsilon with m=100 (no veto power for new statement)
            epsilon = compute_epsilon_for_new_statement(updated_prefs, len(all_statements))
            
            if epsilon is not None:
                old_epsilon = result.get('epsilon')
                result['epsilon'] = epsilon
                fixes += 1
                logger.info(f"      {method}: epsilon {old_epsilon} -> {epsilon}")
            else:
                logger.warning(f"      {method}: epsilon computation returned None")
                
        except Exception as e:
            logger.error(f"      {method} failed: {e}")
    
    # Save updated results if any fixes were made
    if fixes > 0:
        with open(results_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    return fixes


def fix_topic(
    topic_short: str,
    topic_slug: str,
    all_personas: List[str],
    openai_client: OpenAI,
) -> int:
    """Fix all reps for a single topic. Returns total fixes made."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing topic: {topic_short}")
    logger.info(f"{'='*60}")
    
    topic_dir = PHASE2_DATA_DIR / topic_short
    if not topic_dir.exists():
        logger.error(f"Topic directory not found: {topic_dir}")
        return 0
    
    pref_files = list(topic_dir.glob("**/preferences.json"))
    logger.info(f"Found {len(pref_files)} reps for {topic_short}")
    
    topic_fixes = 0
    topic_question = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    
    for pref_path in tqdm(pref_files, desc=f"Processing {topic_short}"):
        rep_dir = pref_path.parent
        
        try:
            voter_dist, alt_dist, rep_id = parse_rep_path(rep_dir)
            
            logger.info(f"\n  Processing {voter_dist}/{alt_dist}/rep{rep_id}")
            
            # Load statements
            statements = load_statements_for_rep(topic_slug, alt_dist, rep_id)
            
            # Load voters.json to get global persona indices
            with open(rep_dir / "voters.json") as f:
                voters_data = json.load(f)
            global_voter_indices = voters_data["voter_indices"]
            
            # Process each mini-rep
            for mini_rep_id in range(5):
                logger.info(f"    Mini-rep {mini_rep_id}...")
                fixes = fix_mini_rep(
                    rep_dir=rep_dir,
                    mini_rep_id=mini_rep_id,
                    all_statements=statements,
                    all_personas=all_personas,
                    global_voter_indices=global_voter_indices,
                    topic_question=topic_question,
                    openai_client=openai_client,
                )
                topic_fixes += fixes
                
        except Exception as e:
            logger.error(f"Error processing {rep_dir}: {e}")
    
    logger.info(f"\nTopic {topic_short} complete: {topic_fixes} fixes")
    return topic_fixes


def main():
    """Fix GPT** epsilon values for all specified topics."""
    logger.info("="*60)
    logger.info("Fixing GPT** epsilon values for multiple topics")
    logger.info("="*60)
    logger.info(f"Topics to fix: {list(TOPICS_TO_FIX.keys())}")
    logger.info(f"Log file: {LOG_FILE}")
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)
    openai_client = OpenAI(api_key=api_key)
    
    # Load personas
    logger.info("Loading personas...")
    all_personas = load_personas()
    logger.info(f"Loaded {len(all_personas)} personas")
    
    # Process each topic
    total_fixes = 0
    topic_results = {}
    
    for topic_short, topic_slug in TOPICS_TO_FIX.items():
        fixes = fix_topic(
            topic_short=topic_short,
            topic_slug=topic_slug,
            all_personas=all_personas,
            openai_client=openai_client,
        )
        topic_results[topic_short] = fixes
        total_fixes += fixes
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")
    for topic, fixes in topic_results.items():
        logger.info(f"  {topic}: {fixes} fixes")
    logger.info(f"Total GPT** epsilon fixes: {total_fixes}")
    logger.info(f"Log file: {LOG_FILE}")


if __name__ == "__main__":
    main()
