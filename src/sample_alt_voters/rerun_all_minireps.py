"""
Re-run all mini-reps with ChatGPT* and ChatGPT** voting methods.

This script adds the new ChatGPT* and ChatGPT** variants to existing results
without re-running traditional methods or base ChatGPT methods.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from .config import (
    PHASE2_DATA_DIR,
    PERSONAS_PATH,
    SAMPLED_STATEMENTS_DIR,
    SAMPLED_CONTEXT_DIR,
    TOPIC_SHORT_NAMES,
    TOPIC_QUESTIONS,
    BASE_SEED,
    N_ALTERNATIVES,
)
from .preference_builder_iterative import subsample_preferences
from .run_experiment import load_statements_for_rep
from src.sampling_experiment.epsilon_calculator import load_precomputed_epsilons
from src.sampling_experiment.voting_methods import (
    run_chatgpt_star,
    run_chatgpt_star_with_rankings,
    run_chatgpt_star_with_personas,
    run_chatgpt_double_star,
    run_chatgpt_double_star_with_rankings,
    run_chatgpt_double_star_with_personas,
)

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


def run_chatgpt_star_methods(
    sample_statements: List[Dict],
    sample_preferences: List[List[str]],
    sample_personas: List[str],
    all_statements: List[Dict],
    full_preferences: List[List[str]],
    voter_indices: List[int],
    topic: str,
    openai_client: OpenAI,
) -> Dict[str, Dict]:
    """Run only the ChatGPT* and ChatGPT** voting methods."""
    results = {}
    
    # ChatGPT* variants
    try:
        results["chatgpt_star"] = run_chatgpt_star(
            all_statements, sample_statements, openai_client
        )
    except Exception as e:
        logger.error(f"Error running chatgpt_star: {e}")
        results["chatgpt_star"] = {"winner": None, "error": str(e)}
    
    try:
        results["chatgpt_star_rankings"] = run_chatgpt_star_with_rankings(
            all_statements, sample_statements, sample_preferences, openai_client
        )
    except Exception as e:
        logger.error(f"Error running chatgpt_star_rankings: {e}")
        results["chatgpt_star_rankings"] = {"winner": None, "error": str(e)}
    
    try:
        results["chatgpt_star_personas"] = run_chatgpt_star_with_personas(
            all_statements, sample_personas, openai_client
        )
    except Exception as e:
        logger.error(f"Error running chatgpt_star_personas: {e}")
        results["chatgpt_star_personas"] = {"winner": None, "error": str(e)}
    
    # ChatGPT** variants
    try:
        results["chatgpt_double_star"] = run_chatgpt_double_star(
            sample_statements, all_statements, sample_personas,
            full_preferences, voter_indices, topic, openai_client
        )
    except Exception as e:
        logger.error(f"Error running chatgpt_double_star: {e}")
        results["chatgpt_double_star"] = {"winner": None, "error": str(e)}
    
    try:
        results["chatgpt_double_star_rankings"] = run_chatgpt_double_star_with_rankings(
            sample_statements, sample_preferences, all_statements,
            sample_personas, full_preferences, voter_indices, topic, openai_client
        )
    except Exception as e:
        logger.error(f"Error running chatgpt_double_star_rankings: {e}")
        results["chatgpt_double_star_rankings"] = {"winner": None, "error": str(e)}
    
    try:
        results["chatgpt_double_star_personas"] = run_chatgpt_double_star_with_personas(
            sample_statements, sample_personas, all_statements,
            full_preferences, voter_indices, topic, openai_client
        )
    except Exception as e:
        logger.error(f"Error running chatgpt_double_star_personas: {e}")
        results["chatgpt_double_star_personas"] = {"winner": None, "error": str(e)}
    
    return results


def rerun_minirep(
    rep_dir: Path,
    mini_rep_id: int,
    all_statements: List[Dict],
    all_personas: List[str],
    topic: str,
    openai_client: OpenAI,
) -> bool:
    """
    Re-run ChatGPT* and ChatGPT** methods for a single mini-rep.
    
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
    
    # Load voters.json to get global persona indices
    with open(rep_dir / "voters.json") as f:
        voters_data = json.load(f)
    global_voter_indices = voters_data["voter_indices"]
    
    # Load full preferences
    with open(rep_dir / "preferences.json") as f:
        full_preferences = json.load(f)
    
    # Subsample preferences for this mini-rep using the saved indices
    sample_preferences, _, _ = subsample_preferences(
        full_preferences,
        voter_indices=voter_indices,
        alt_indices=alt_indices
    )
    
    # Get sampled statements and personas
    sample_statements = [all_statements[i] for i in alt_indices]
    # Map local voter indices to global persona indices
    sample_personas = [all_personas[global_voter_indices[i]] for i in voter_indices]
    
    # Load precomputed epsilons
    epsilons = load_precomputed_epsilons(rep_dir)
    
    # Run ChatGPT* and ChatGPT** methods
    new_results = run_chatgpt_star_methods(
        sample_statements=sample_statements,
        sample_preferences=sample_preferences,
        sample_personas=sample_personas,
        all_statements=all_statements,
        full_preferences=full_preferences,
        voter_indices=voter_indices,
        topic=topic,
        openai_client=openai_client,
    )
    
    # Add epsilon values for winners
    alt_mapping = {str(i): str(alt_indices[i]) for i in range(len(alt_indices))}
    
    for method_name, result in new_results.items():
        winner = result.get("winner")
        if winner is not None and winner in alt_mapping:
            full_winner_idx = alt_mapping[winner]
            result["epsilon"] = epsilons.get(full_winner_idx)
            result["full_winner_idx"] = full_winner_idx
        
        # Merge into existing results
        results["results"][method_name] = result
    
    # Save updated results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return True


def main():
    """Re-run ChatGPT* and ChatGPT** methods for all mini-reps."""
    
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
    pref_files = list(PHASE2_DATA_DIR.glob("**/preferences.json"))
    logger.info(f"Found {len(pref_files)} rep directories")
    
    # Topic slug mapping
    topic_slug_map = {
        "abortion": "what-should-guide-laws-concerning-abortion",
        "electoral": "what-reforms-if-any-should-replace-or-modify-the-e",
    }
    
    # Process each rep directory
    total_minireps = 0
    success_count = 0
    error_count = 0
    
    for pref_path in tqdm(pref_files, desc="Processing reps"):
        rep_dir = pref_path.parent
        
        try:
            topic, voter_dist, alt_dist, rep_id = parse_rep_path(rep_dir)
            topic_slug = topic_slug_map.get(topic, topic)
            topic_question = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
            
            # Load statements for this rep
            statements = load_statements_for_rep(topic_slug, alt_dist, rep_id)
            
            # Process each mini-rep
            for mini_rep_id in range(5):
                mini_rep_dir = rep_dir / f"mini_rep{mini_rep_id}"
                if not mini_rep_dir.exists():
                    continue
                
                total_minireps += 1
                
                try:
                    success = rerun_minirep(
                        rep_dir=rep_dir,
                        mini_rep_id=mini_rep_id,
                        all_statements=statements,
                        all_personas=all_personas,
                        topic=topic_question,
                        openai_client=client,
                    )
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    logger.error(f"Error processing {rep_dir.name}/mini_rep{mini_rep_id}: {e}")
                    error_count += 1
                    
        except Exception as e:
            logger.error(f"Error processing {rep_dir}: {e}")
            error_count += 5  # Assume all mini-reps failed
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total mini-reps processed: {total_minireps}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Errors: {error_count}")
    
    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
