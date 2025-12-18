"""
Main orchestration script for large-scale social choice experiment pipeline.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
import logging

# Create timestamped log directory and file
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f'logs/{timestamp}'
os.makedirs(log_dir, exist_ok=True)
log_file = f'{log_dir}/experiment.log'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger.info(f"Logging to {log_file}")

from src.large_scale.persona_loader import (
    load_personas_from_huggingface,
    split_personas,
    save_persona_splits,
    load_persona_splits
)
from src.large_scale.generate_statements import generate_all_statements, save_statements, load_statements
from src.large_scale.discriminative_ranking import get_discriminative_rankings, save_preferences, load_preferences
from src.large_scale.evaluative_scoring import get_all_ratings, save_evaluations, load_evaluations
from src.large_scale.voting_methods import evaluate_all_methods
from src.large_scale.biclique import compute_proportional_veto_core
from src.compute_pvc import compute_pvc  # For successive veto


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text[:50]  # Limit length


def load_topics(filepath: str = "data/topics.txt") -> List[str]:
    """Load topics from file."""
    with open(filepath, 'r') as f:
        topics = [line.strip() for line in f if line.strip()]
    return topics


def run_experiment(
    topic: str,
    generative_personas: List[str],
    discriminative_personas: List[str],
    evaluative_personas: List[str],
    openai_client: OpenAI,
    output_dir: str = "data/large_scale/results",
    skip_if_exists: bool = True,
    test_mode: bool = False
) -> Dict:
    """
    Run a single experiment for a topic.
    
    Args:
        topic: The topic/question to run the experiment on
        generative_personas: List of generative persona strings
        discriminative_personas: List of discriminative persona strings
        evaluative_personas: List of evaluative persona strings
        openai_client: OpenAI client instance
        output_dir: Directory to save results
        skip_if_exists: Whether to skip if result file already exists
    
    Returns:
        Dictionary with all experiment results
    """
    logger.info("="*80)
    logger.info(f"Running experiment for topic: {topic}")
    logger.info("="*80)
    
    start_time = time.time()
    
    topic_slug = slugify(topic)
    output_path = os.path.join(output_dir, f"{topic_slug}.json")
    
    # Check if file exists and skip if requested
    if skip_if_exists and os.path.exists(output_path):
        logger.info(f"⏭️  Skipping {topic_slug} (result file already exists)")
        with open(output_path, 'r') as f:
            return json.load(f)
    
    # Determine base directory based on test_mode
    base_dir = "data/large_scale/test" if test_mode else "data/large_scale/prod"
    
    # Step 1: Generate statements
    logger.info(f"\n📝 Step 1: Generating statements from {len(generative_personas)} generative personas...")
    step_start = time.time()
    statements_path = f"{base_dir}/statements/{topic_slug}.json"
    if os.path.exists(statements_path):
        logger.info(f"  ✓ Loading existing statements from {statements_path}")
        statements = load_statements(topic_slug, input_dir=f"{base_dir}/statements")
    else:
        statements = generate_all_statements(topic, generative_personas, openai_client)
        save_statements(statements, topic_slug, output_dir=f"{base_dir}/statements")
    logger.info(f"  ⏱️  Step 1 completed in {time.time() - step_start:.1f}s")
    
    # Step 2: Get discriminative rankings
    logger.info(f"\n🗳️  Step 2: Getting preference rankings from {len(discriminative_personas)} discriminative personas...")
    step_start = time.time()
    preferences_path = f"{base_dir}/preferences/{topic_slug}.json"
    if os.path.exists(preferences_path):
        logger.info(f"  ✓ Loading existing preferences from {preferences_path}")
        preference_matrix = load_preferences(topic_slug, input_dir=f"{base_dir}/preferences")
    else:
        preference_matrix = get_discriminative_rankings(
            discriminative_personas, statements, topic, openai_client
        )
        save_preferences(preference_matrix, topic_slug, output_dir=f"{base_dir}/preferences")
    logger.info(f"  ⏱️  Step 2 completed in {time.time() - step_start:.1f}s")
    
    # Step 3: Get evaluative ratings
    logger.info(f"\n⭐ Step 3: Getting Likert ratings from {len(evaluative_personas)} evaluative personas...")
    step_start = time.time()
    evaluations_path = f"{base_dir}/evaluations/{topic_slug}.json"
    if os.path.exists(evaluations_path):
        logger.info(f"  ✓ Loading existing evaluations from {evaluations_path}")
        evaluations = load_evaluations(topic_slug, input_dir=f"{base_dir}/evaluations")
    else:
        evaluations = get_all_ratings(evaluative_personas, statements, topic, openai_client)
        save_evaluations(evaluations, topic_slug, output_dir=f"{base_dir}/evaluations")
    logger.info(f"  ⏱️  Step 3 completed in {time.time() - step_start:.1f}s")
    
    # Step 4: Compute PVC using biclique algorithm
    logger.info(f"\n🎯 Step 4: Computing PVC...")
    step_start = time.time()
    
    # Convert preference matrix to profile format for biclique algorithm
    # preference_matrix[rank][voter] -> profile[voter][rank]
    n_statements = len(preference_matrix)
    n_voters = len(preference_matrix[0]) if preference_matrix else 0
    
    profile = []
    for voter_idx in range(n_voters):
        voter_ranking = []
        for rank in range(n_statements):
            statement_idx = preference_matrix[rank][voter_idx]
            voter_ranking.append(statement_idx)
        profile.append(voter_ranking)
    
    # Compute PVC using the biclique flow algorithm
    pvc_result_obj = compute_proportional_veto_core(profile)
    pvc_result = sorted(pvc_result_obj.core)  # Convert set to sorted list
    pvc_size = len(pvc_result)
    pvc_percentage = (pvc_size / len(statements) * 100) if statements else 0
    
    logger.info(f"  PVC result: {pvc_result}")
    logger.info(f"  PVC size: {pvc_size} / {len(statements)} ({pvc_percentage:.1f}%)")
    logger.info(f"  PVC parameters: r={pvc_result_obj.r}, t={pvc_result_obj.t}, alpha={pvc_result_obj.alpha}")
    logger.info(f"  ⏱️  Step 4 completed in {time.time() - step_start:.1f}s")
    
    # Step 5: Evaluate voting methods
    logger.info(f"\n🏆 Step 5: Evaluating voting methods...")
    step_start = time.time()
    method_results = evaluate_all_methods(
        preference_matrix, statements, discriminative_personas, openai_client, pvc=pvc_result
    )
    
    # Log results
    for method_name, method_result in method_results.items():
        winner = method_result.get("winner")
        if winner is not None:
            in_pvc = method_result.get("in_pvc", False)
            in_pvc_symbol = "✓" if in_pvc else "✗"
            logger.info(f"  {in_pvc_symbol} {method_name}: winner={winner}, in_pvc={in_pvc}")
        else:
            logger.error(f"  ✗ {method_name}: no winner (error: {method_result.get('error', 'unknown')})")
    logger.info(f"  ⏱️  Step 5 completed in {time.time() - step_start:.1f}s")
    
    # Compile results
    results = {
        "topic": topic,
        "n_generative_personas": len(generative_personas),
        "n_discriminative_personas": len(discriminative_personas),
        "n_evaluative_personas": len(evaluative_personas),
        "n_statements": len(statements),
        "statements": statements,
        "preference_matrix": preference_matrix,
        "evaluations": evaluations,
        "pvc": pvc_result,
        "pvc_size": pvc_size,
        "pvc_percentage": pvc_percentage,
        "method_results": method_results
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time
    logger.info(f"\n✅ Experiment completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info(f"   Results saved to: {output_path}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run large-scale social choice experiment pipeline")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with 20/5/5 personas (default: production mode with 900/50/50)"
    )
    parser.add_argument(
        "--topic-index",
        type=int,
        help="Run only topic at index N (0-indexed)"
    )
    parser.add_argument(
        "--load-personas",
        action="store_true",
        help="Load existing persona splits instead of generating new ones"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/large_scale/results",
        help="Output directory for results (default: data/large_scale/results)"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=api_key)
    
    # Determine persona counts based on mode
    if args.test_mode:
        n_generative = 20
        n_discriminative = 5
        n_evaluative = 5
        print("Running in TEST MODE with 20/5/5 personas")
    else:
        n_generative = 900
        n_discriminative = 50
        n_evaluative = 50
        print("Running in PRODUCTION MODE with 900/50/50 personas")
    
    # Load or generate personas
    if args.load_personas:
        print("\nLoading existing persona splits...")
        generative_personas, discriminative_personas, evaluative_personas = load_persona_splits(test_mode=args.test_mode)
        
        # Verify counts match expected
        if len(generative_personas) != n_generative:
            print(f"WARNING: Expected {n_generative} generative personas, got {len(generative_personas)}")
        if len(discriminative_personas) != n_discriminative:
            print(f"WARNING: Expected {n_discriminative} discriminative personas, got {len(discriminative_personas)}")
        if len(evaluative_personas) != n_evaluative:
            print(f"WARNING: Expected {n_evaluative} evaluative personas, got {len(evaluative_personas)}")
    else:
        print("\nLoading and splitting personas from HuggingFace...")
        all_personas = load_personas_from_huggingface()
        generative_personas, discriminative_personas, evaluative_personas = split_personas(
            all_personas, n_generative, n_discriminative, n_evaluative
        )
        save_persona_splits(generative_personas, discriminative_personas, evaluative_personas, test_mode=args.test_mode)
    
    # Load topics
    topics = load_topics("data/topics.txt")
    print(f"\nLoaded {len(topics)} topics")
    
    # Filter topics if index specified
    if args.topic_index is not None:
        if args.topic_index < 0 or args.topic_index >= len(topics):
            raise ValueError(f"Topic index {args.topic_index} out of range (0-{len(topics)-1})")
        topics = [topics[args.topic_index]]
        print(f"Running only topic at index {args.topic_index}")
    
    # Run experiments
    all_results = []
    for i, topic in enumerate(topics):
        try:
            result = run_experiment(
                topic,
                generative_personas,
                discriminative_personas,
                evaluative_personas,
                openai_client,
                output_dir=args.output_dir,
                test_mode=args.test_mode
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\nERROR processing topic {i} ({topic}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"Completed {len(all_results)}/{len(topics)} experiments")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

