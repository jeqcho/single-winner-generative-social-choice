"""
Fix epsilon values for GPT* and GPT** methods.

GPT* methods select from all 100 statements, so we can look up epsilon directly
from precomputed_epsilons.json using the winner index.

GPT** methods generate new statements, so we need to:
1. Insert the new statement into voter rankings (API calls)
2. Compute epsilon for the new statement
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from .config import (
    PHASE2_DATA_DIR,
    PERSONAS_PATH,
    TOPIC_QUESTIONS,
)
from .run_experiment import load_statements_for_rep
from src.experiment_utils.epsilon_calculator import (
    load_precomputed_epsilons,
    compute_epsilon_for_new_statement,
)
from src.experiment_utils.voting_methods import insert_new_statement_into_rankings

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


def fix_gpt_star_epsilons(results: Dict, precomputed_epsilons: Dict[str, float]) -> int:
    """
    Fix GPT* epsilon values by looking up from precomputed_epsilons.
    
    Returns number of fixes made.
    """
    fixes = 0
    star_methods = ['chatgpt_star', 'chatgpt_star_rankings', 'chatgpt_star_personas']
    
    for method in star_methods:
        if method not in results:
            continue
        
        result = results[method]
        winner = result.get('winner')
        
        # Skip if already has epsilon or no winner
        if result.get('epsilon') is not None or winner is None:
            continue
        
        # Look up epsilon directly from precomputed
        epsilon = precomputed_epsilons.get(str(winner))
        if epsilon is not None:
            result['epsilon'] = epsilon
            result['full_winner_idx'] = str(winner)
            fixes += 1
    
    return fixes


def fix_gpt_double_star_epsilons(
    results: Dict,
    all_statements: List[Dict],
    full_preferences: List[List[str]],
    voter_indices: List[int],
    all_personas: List[str],
    global_voter_indices: List[int],
    topic_question: str,
    openai_client: OpenAI,
) -> int:
    """
    Fix GPT** epsilon values by inserting new statements and computing epsilon.
    
    Returns number of fixes made.
    """
    fixes = 0
    double_star_methods = [
        'chatgpt_double_star',
        'chatgpt_double_star_rankings',
        'chatgpt_double_star_personas'
    ]
    
    for method in double_star_methods:
        if method not in results:
            continue
        
        result = results[method]
        new_statement = result.get('new_statement')
        
        # Skip if already has epsilon or no new statement
        if result.get('epsilon') is not None or not new_statement:
            continue
        
        # Get the personas for the sampled voters
        # voter_indices are local (0-19), global_voter_indices maps to persona indices
        selected_personas = [all_personas[global_voter_indices[i]] for i in voter_indices]
        
        try:
            logger.info(f"    Inserting new statement for {method}...")
            
            # Insert new statement into rankings
            updated_prefs = insert_new_statement_into_rankings(
                new_statement=new_statement,
                all_statements=all_statements,
                voter_personas=selected_personas,
                voter_indices=voter_indices,
                full_preferences=full_preferences,
                topic=topic_question,
                openai_client=openai_client,
            )
            
            # Compute epsilon with m=100 (no veto power for new statement)
            epsilon = compute_epsilon_for_new_statement(updated_prefs, len(all_statements))
            
            if epsilon is not None:
                result['epsilon'] = epsilon
                fixes += 1
                logger.info(f"    {method}: epsilon = {epsilon:.4f}")
            else:
                logger.warning(f"    {method}: epsilon computation returned None")
                
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
    
    return fixes


def process_mini_rep(
    rep_dir: Path,
    mini_rep_id: int,
    all_statements: List[Dict],
    all_personas: List[str],
    topic_question: str,
    openai_client: Optional[OpenAI],
    fix_double_star: bool = True,
) -> Dict[str, int]:
    """Process a single mini-rep and fix epsilon values."""
    
    mini_rep_dir = rep_dir / f"mini_rep{mini_rep_id}"
    results_path = mini_rep_dir / "results.json"
    
    if not results_path.exists():
        return {'star_fixes': 0, 'double_star_fixes': 0}
    
    # Load existing results
    with open(results_path) as f:
        data = json.load(f)
    
    results = data['results']
    voter_indices = data['voter_indices']
    
    # Load precomputed epsilons
    precomputed = load_precomputed_epsilons(rep_dir)
    
    # Load voters.json to get global persona indices
    with open(rep_dir / "voters.json") as f:
        voters_data = json.load(f)
    global_voter_indices = voters_data["voter_indices"]
    
    # Load full preferences
    with open(rep_dir / "preferences.json") as f:
        full_preferences = json.load(f)
    
    # Fix GPT* epsilons (no API calls needed)
    star_fixes = fix_gpt_star_epsilons(results, precomputed)
    
    # Fix GPT** epsilons (requires API calls)
    double_star_fixes = 0
    if fix_double_star and openai_client is not None:
        double_star_fixes = fix_gpt_double_star_epsilons(
            results=results,
            all_statements=all_statements,
            full_preferences=full_preferences,
            voter_indices=voter_indices,
            all_personas=all_personas,
            global_voter_indices=global_voter_indices,
            topic_question=topic_question,
            openai_client=openai_client,
        )
    
    # Save updated results if any fixes were made
    if star_fixes > 0 or double_star_fixes > 0:
        with open(results_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    return {'star_fixes': star_fixes, 'double_star_fixes': double_star_fixes}


def main():
    """Fix epsilon values for all GPT* and GPT** results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix GPT* and GPT** epsilon values")
    parser.add_argument('--star-only', action='store_true',
                       help="Only fix GPT* epsilons (no API calls)")
    parser.add_argument('--double-star-only', action='store_true',
                       help="Only fix GPT** epsilons")
    args = parser.parse_args()
    
    fix_star = not args.double_star_only
    fix_double_star = not args.star_only
    
    # Initialize OpenAI client if needed for GPT**
    openai_client = None
    if fix_double_star:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not set (required for GPT** fixes)")
            if not fix_star:
                sys.exit(1)
            logger.warning("Skipping GPT** fixes")
            fix_double_star = False
        else:
            openai_client = OpenAI(api_key=api_key)
    
    # Load personas
    logger.info("Loading personas...")
    all_personas = load_personas()
    logger.info(f"Loaded {len(all_personas)} personas")
    
    # Topic mapping
    topic_slug_map = {
        "abortion": "what-should-guide-laws-concerning-abortion",
        "electoral": "what-reforms-if-any-should-replace-or-modify-the-e",
    }
    
    # Find all rep directories
    pref_files = list(PHASE2_DATA_DIR.glob("**/preferences.json"))
    logger.info(f"Found {len(pref_files)} rep directories")
    
    total_star_fixes = 0
    total_double_star_fixes = 0
    
    for pref_path in tqdm(pref_files, desc="Processing reps"):
        rep_dir = pref_path.parent
        
        try:
            topic, voter_dist, alt_dist, rep_id = parse_rep_path(rep_dir)
            topic_slug = topic_slug_map.get(topic, topic)
            topic_question = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
            
            # Load statements
            statements = load_statements_for_rep(topic_slug, alt_dist, rep_id)
            
            # Process each mini-rep
            for mini_rep_id in range(5):
                fixes = process_mini_rep(
                    rep_dir=rep_dir,
                    mini_rep_id=mini_rep_id,
                    all_statements=statements,
                    all_personas=all_personas,
                    topic_question=topic_question,
                    openai_client=openai_client,
                    fix_double_star=fix_double_star,
                )
                
                if fix_star:
                    total_star_fixes += fixes['star_fixes']
                if fix_double_star:
                    total_double_star_fixes += fixes['double_star_fixes']
                    
        except Exception as e:
            logger.error(f"Error processing {rep_dir}: {e}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"GPT* epsilon fixes: {total_star_fixes}")
    logger.info(f"GPT** epsilon fixes: {total_double_star_fixes}")
    logger.info(f"Total fixes: {total_star_fixes + total_double_star_fixes}")


if __name__ == "__main__":
    main()
