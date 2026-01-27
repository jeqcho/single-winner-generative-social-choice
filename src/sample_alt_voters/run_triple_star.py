"""
Run ChatGPT*** (triple star) on existing sample_alt_voters results.

GPT*** generates blind bridging statements without context about existing statements.
It runs on the FULL 100x100 preference matrix (not mini-rep subsamples).
Results are stored at the rep level: rep{N}/chatgpt_triple_star.json
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from .config import (
    PHASE2_DATA_DIR,
    PERSONAS_PATH,
    TOPIC_QUESTIONS,
    TOPIC_SHORT_NAMES,
)
from .run_experiment import load_statements_for_rep
from src.experiment_utils.voting_methods import run_chatgpt_triple_star

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


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
    rep_id = int(rep_name.split('_')[0].replace('rep', ''))
    return topic, voter_dist, alt_dist, rep_id


def run_triple_star_for_rep(
    rep_dir: Path,
    all_statements: List[Dict],
    all_personas: List[str],
    topic_question: str,
    openai_client: OpenAI,
    n_generations: int = 1,
    force: bool = False,
) -> Dict:
    """Run GPT*** for a single rep using the full 100x100 matrix."""
    
    output_path = rep_dir / "chatgpt_triple_star.json"
    
    # Skip if already exists and not forcing
    if output_path.exists() and not force:
        logger.info(f"  Skipping {rep_dir.name} (already exists)")
        with open(output_path) as f:
            return json.load(f)
    
    # Load full preferences
    with open(rep_dir / "preferences.json") as f:
        full_preferences = json.load(f)
    
    # Load voters.json to get persona indices
    with open(rep_dir / "voters.json") as f:
        voters_data = json.load(f)
    voter_indices = voters_data["voter_indices"]
    
    # Get the personas for the 100 sampled voters
    voter_personas = [all_personas[i] for i in voter_indices]
    
    logger.info(f"  Running GPT*** for {rep_dir.name}...")
    
    try:
        result = run_chatgpt_triple_star(
            topic=topic_question,
            all_statements=all_statements,
            voter_personas=voter_personas,
            full_preferences=full_preferences,
            openai_client=openai_client,
            n_generations=n_generations,
        )
        
        # Save result
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"  GPT*** epsilon: {result.get('epsilon', 'N/A')}")
        return result
        
    except Exception as e:
        logger.error(f"  GPT*** failed: {e}")
        return {"error": str(e)}


def main():
    """Run GPT*** on all existing reps."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GPT*** on sample_alt_voters results")
    parser.add_argument('--n-generations', type=int, default=1,
                       help="Number of blind statements to generate (default: 1)")
    parser.add_argument('--force', action='store_true',
                       help="Re-run even if results exist")
    args = parser.parse_args()
    
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
    
    # Topic mapping: short name (folder) -> full slug
    topic_slug_map = {v: k for k, v in TOPIC_SHORT_NAMES.items()}
    
    # Find all rep directories
    pref_files = list(PHASE2_DATA_DIR.glob("**/preferences.json"))
    logger.info(f"Found {len(pref_files)} rep directories")
    
    completed = 0
    skipped = 0
    failed = 0
    
    for pref_path in tqdm(pref_files, desc="Processing reps"):
        rep_dir = pref_path.parent
        
        try:
            topic, voter_dist, alt_dist, rep_id = parse_rep_path(rep_dir)
            topic_slug = topic_slug_map.get(topic, topic)
            topic_question = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
            
            # Load statements
            statements = load_statements_for_rep(topic_slug, alt_dist, rep_id)
            
            # Check if already exists
            output_path = rep_dir / "chatgpt_triple_star.json"
            if output_path.exists() and not args.force:
                skipped += 1
                continue
            
            # Run triple star
            result = run_triple_star_for_rep(
                rep_dir=rep_dir,
                all_statements=statements,
                all_personas=all_personas,
                topic_question=topic_question,
                openai_client=openai_client,
                n_generations=args.n_generations,
                force=args.force,
            )
            
            if "error" not in result:
                completed += 1
            else:
                failed += 1
                
        except Exception as e:
            logger.error(f"Error processing {rep_dir}: {e}")
            failed += 1
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Skipped (already exists): {skipped}")
    logger.info(f"Failed: {failed}")


if __name__ == "__main__":
    main()
