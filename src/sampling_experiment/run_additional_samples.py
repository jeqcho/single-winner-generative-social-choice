"""
Run additional samples for existing experiment data.

This script runs sample1 and sample2 (skipping sample0 which contains migrated data)
for all reps and (K, P) configurations.

Usage:
    uv run python -m src.sampling_experiment.run_additional_samples
    uv run python -m src.sampling_experiment.run_additional_samples --start-sample 1 --end-sample 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

from .config import (
    OUTPUT_DIR,
    TEST_TOPIC,
    N_REPS,
    N_VOTER_POOL,
    N_ALT_POOL,
    K_VALUES,
    P_VALUES,
    BASE_SEED,
    N_SAMPLES_PER_KP,
    TOPIC_QUESTIONS,
)
from .data_loader import (
    load_pool_data,
    load_preferences,
    sample_kp,
    extract_subprofile,
    check_cache_exists,
)
from .epsilon_calculator import (
    load_precomputed_epsilons,
    lookup_epsilon,
    compute_epsilon_for_new_statement,
)
from .run_experiment import (
    run_traditional_methods,
    run_chatgpt_methods,
    run_chatgpt_star_methods,
    run_chatgpt_double_star_methods,
)


def setup_logging(output_dir: Path) -> None:
    """Set up logging to file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"additional_samples_{timestamp}.log"
    
    # Create handlers
    file_handler = logging.FileHandler(log_file, mode='a')
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Set format
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Suppress verbose logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    logging.info(f"Logging to {log_file}")


def run_single_sample(
    rep_dir: Path,
    k: int,
    p: int,
    sample_idx: int,
    seed: int,
    full_preferences: List[List[str]],
    precomputed_epsilons: Dict[str, float],
    voter_personas: List[str],
    alt_statements: List[Dict],
    topic: str,
    openai_client: OpenAI
) -> None:
    """Run a single sample for a given (K, P) configuration."""
    logger = logging.getLogger(__name__)
    
    sample_dir = rep_dir / f"k{k}_p{p}" / f"sample{sample_idx}"
    
    # Check if already done
    if check_cache_exists(sample_dir, "results.json"):
        logger.info(f"    Sample {sample_idx}: cached, skipping")
        return
    
    logger.info(f"    Sample {sample_idx}: Running...")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute seed for this sample
    # Use sample_idx * 10000 to ensure different samples get different seeds
    sample_seed = seed * 1000 + k * 100 + p + sample_idx * 10000
    
    # Sample K voters and P alternatives
    voter_sample, alt_sample = sample_kp(N_VOTER_POOL, N_ALT_POOL, k, p, sample_seed)
    
    # Extract subprofile
    sample_prefs, alt_mapping = extract_subprofile(
        full_preferences, voter_sample, alt_sample
    )
    
    # Get sample statements and personas
    sample_statements = [alt_statements[i] for i in alt_sample]
    sample_personas = [voter_personas[i] for i in voter_sample]
    
    # Save sample info
    with open(sample_dir / "sample_info.json", 'w') as f:
        json.dump({
            "k": k, "p": p,
            "sample_idx": sample_idx,
            "voter_sample": voter_sample,
            "alt_sample": alt_sample,
            "alt_mapping": {str(k): v for k, v in alt_mapping.items()},
        }, f, indent=2)
    
    # Run all methods
    results = {}
    
    # Traditional methods
    trad_results = run_traditional_methods(
        sample_prefs, alt_mapping, precomputed_epsilons
    )
    results.update(trad_results)
    
    # ChatGPT methods
    chatgpt_results = run_chatgpt_methods(
        sample_statements, sample_prefs, sample_personas,
        alt_mapping, precomputed_epsilons, openai_client
    )
    results.update(chatgpt_results)
    
    # ChatGPT* methods
    star_results = run_chatgpt_star_methods(
        alt_statements, sample_statements, sample_prefs, sample_personas,
        precomputed_epsilons, openai_client
    )
    results.update(star_results)
    
    # ChatGPT** methods
    double_star_results = run_chatgpt_double_star_methods(
        alt_statements, sample_statements, sample_prefs, sample_personas,
        full_preferences, voter_sample, voter_personas, topic,
        precomputed_epsilons, openai_client
    )
    results.update(double_star_results)
    
    # Save results
    with open(sample_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log summary
    logger.info(f"      Results: " + ", ".join(
        f"{m}={r.get('epsilon', 'N/A'):.3f}" if r.get('epsilon') else f"{m}=N/A"
        for m, r in list(results.items())[:5]
    ))


def run_additional_samples(
    topic_slug: str,
    output_dir: Path,
    openai_client: OpenAI,
    start_sample: int = 1,
    end_sample: int = N_SAMPLES_PER_KP,
    n_reps: int = N_REPS
) -> None:
    """Run additional samples for all reps."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info(f"RUNNING ADDITIONAL SAMPLES")
    logger.info(f"Topic: {topic_slug}")
    logger.info(f"Reps: {n_reps}")
    logger.info(f"Samples: {start_sample} to {end_sample - 1}")
    logger.info(f"K values: {K_VALUES}")
    logger.info(f"P values: {P_VALUES}")
    logger.info("=" * 80)
    
    topic = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    
    for rep_idx in range(n_reps):
        rep_dir = output_dir / "data" / topic_slug / f"rep{rep_idx}"
        
        if not rep_dir.exists():
            logger.warning(f"Rep {rep_idx}: directory not found, skipping")
            continue
        
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Rep {rep_idx}")
        logger.info(f"{'=' * 60}")
        
        # Load cached data
        try:
            voter_indices, alt_indices, voter_personas, alt_statements = load_pool_data(rep_dir)
            full_preferences = load_preferences(rep_dir)
            precomputed_epsilons = load_precomputed_epsilons(rep_dir)
        except Exception as e:
            logger.error(f"Rep {rep_idx}: failed to load cached data: {e}")
            continue
        
        logger.info(f"  Loaded: {len(voter_personas)} voters, {len(alt_statements)} alternatives")
        
        # Compute base seed for this rep
        seed = BASE_SEED + rep_idx
        
        # Run samples for each (K, P) configuration
        for k in K_VALUES:
            for p in P_VALUES:
                logger.info(f"  K={k}, P={p}:")
                
                for sample_idx in range(start_sample, end_sample):
                    try:
                        run_single_sample(
                            rep_dir, k, p, sample_idx, seed,
                            full_preferences, precomputed_epsilons,
                            voter_personas, alt_statements, topic,
                            openai_client
                        )
                    except Exception as e:
                        logger.error(f"    Sample {sample_idx} failed: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
    
    logger.info("\nADDITIONAL SAMPLES COMPLETE")


def main():
    parser = argparse.ArgumentParser(
        description="Run additional samples for existing experiment data"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=TEST_TOPIC,
        help="Topic slug (default: public trust topic)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--start-sample",
        type=int,
        default=1,
        help="First sample index to run (default: 1, skips sample0)"
    )
    parser.add_argument(
        "--end-sample",
        type=int,
        default=N_SAMPLES_PER_KP,
        help=f"Last sample index + 1 (default: {N_SAMPLES_PER_KP})"
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=N_REPS,
        help=f"Number of reps to process (default: {N_REPS})"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.output_dir)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting at {datetime.now().isoformat()}")
    
    # Create OpenAI client
    openai_client = OpenAI(timeout=120.0)
    
    # Run additional samples
    run_additional_samples(
        args.topic,
        args.output_dir,
        openai_client,
        args.start_sample,
        args.end_sample,
        args.n_reps
    )


if __name__ == "__main__":
    main()
