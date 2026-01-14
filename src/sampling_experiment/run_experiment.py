"""
Test experiment runner for the sampling experiment.

Runs the experiment on the public trust topic only.

Usage:
    uv run python -m src.sampling_experiment.run_experiment
    uv run python -m src.sampling_experiment.run_experiment --test
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

from .config import (
    OUTPUT_DIR,
    TEST_TOPIC,
    N_REPS,
    N_REPS_TEST,
    N_VOTER_POOL,
    N_ALT_POOL,
    K_VALUES,
    P_VALUES,
    N_SAMPLES_PER_KP,
    BASE_SEED,
    MAX_WORKERS,
    TOPIC_QUESTIONS,
    TRADITIONAL_METHODS,
    CHATGPT_METHODS,
    CHATGPT_STAR_METHODS,
    CHATGPT_DOUBLE_STAR_METHODS,
)
from .data_loader import (
    load_all_entries,
    sample_pools,
    sample_kp,
    extract_subprofile,
    save_pool_data,
    load_pool_data,
    save_preferences,
    load_preferences,
    check_cache_exists,
)
from .preference_builder import build_full_preferences
from .epsilon_calculator import (
    precompute_all_epsilons,
    lookup_epsilon,
    compute_epsilon_for_new_statement,
    save_precomputed_epsilons,
    load_precomputed_epsilons,
    get_mean_epsilon,
)
from .voting_methods import (
    run_schulze,
    run_borda,
    run_irv,
    run_plurality,
    run_veto_by_consumption,
    run_chatgpt,
    run_chatgpt_with_rankings,
    run_chatgpt_with_personas,
    run_chatgpt_star,
    run_chatgpt_star_with_rankings,
    run_chatgpt_star_with_personas,
    run_chatgpt_double_star,
    run_chatgpt_double_star_with_rankings,
    run_chatgpt_double_star_with_personas,
    insert_new_statement_into_rankings,
)
from .visualizer import generate_all_visualizations


def setup_logging(output_dir: Path, test_mode: bool = False) -> None:
    """Set up logging to file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix = "test_" if test_mode else "experiment_"
    log_file = log_dir / f"{prefix}{timestamp}.log"
    
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


def run_traditional_methods(
    sample_preferences: List[List[str]],
    alt_mapping: Dict[int, int],
    precomputed_epsilons: Dict[str, float]
) -> Dict[str, Dict]:
    """Run traditional voting methods and look up epsilons."""
    logger = logging.getLogger(__name__)
    results = {}
    
    # Schulze
    logger.info("  Running Schulze...")
    result = run_schulze(sample_preferences)
    if result["winner"] is not None:
        full_winner = str(alt_mapping[int(result["winner"])])
        result["winner_full"] = full_winner
        result["epsilon"] = lookup_epsilon(precomputed_epsilons, full_winner)
    results["schulze"] = result
    
    # Borda
    logger.info("  Running Borda...")
    result = run_borda(sample_preferences)
    if result["winner"] is not None:
        full_winner = str(alt_mapping[int(result["winner"])])
        result["winner_full"] = full_winner
        result["epsilon"] = lookup_epsilon(precomputed_epsilons, full_winner)
    results["borda"] = result
    
    # IRV
    logger.info("  Running IRV...")
    result = run_irv(sample_preferences)
    if result["winner"] is not None:
        full_winner = str(alt_mapping[int(result["winner"])])
        result["winner_full"] = full_winner
        result["epsilon"] = lookup_epsilon(precomputed_epsilons, full_winner)
    results["irv"] = result
    
    # Plurality
    logger.info("  Running Plurality...")
    result = run_plurality(sample_preferences)
    if result["winner"] is not None:
        full_winner = str(alt_mapping[int(result["winner"])])
        result["winner_full"] = full_winner
        result["epsilon"] = lookup_epsilon(precomputed_epsilons, full_winner)
    results["plurality"] = result
    
    # Veto by Consumption
    logger.info("  Running Veto by Consumption...")
    result = run_veto_by_consumption(sample_preferences)
    if result["winner"] is not None:
        full_winner = str(alt_mapping[int(result["winner"])])
        result["winner_full"] = full_winner
        result["epsilon"] = lookup_epsilon(precomputed_epsilons, full_winner)
    results["veto_by_consumption"] = result
    
    return results


def run_chatgpt_methods(
    sample_statements: List[Dict],
    sample_preferences: List[List[str]],
    sample_personas: List[str],
    alt_mapping: Dict[int, int],
    precomputed_epsilons: Dict[str, float],
    openai_client: OpenAI
) -> Dict[str, Dict]:
    """Run ChatGPT methods (select from P alternatives)."""
    logger = logging.getLogger(__name__)
    results = {}
    
    # ChatGPT baseline
    logger.info("  Running ChatGPT...")
    result = run_chatgpt(sample_statements, openai_client)
    if result["winner"] is not None:
        full_winner = str(alt_mapping[int(result["winner"])])
        result["winner_full"] = full_winner
        result["epsilon"] = lookup_epsilon(precomputed_epsilons, full_winner)
    results["chatgpt"] = result
    
    # ChatGPT with rankings
    logger.info("  Running ChatGPT+Rankings...")
    result = run_chatgpt_with_rankings(sample_statements, sample_preferences, openai_client)
    if result["winner"] is not None:
        full_winner = str(alt_mapping[int(result["winner"])])
        result["winner_full"] = full_winner
        result["epsilon"] = lookup_epsilon(precomputed_epsilons, full_winner)
    results["chatgpt_rankings"] = result
    
    # ChatGPT with personas
    logger.info("  Running ChatGPT+Personas...")
    result = run_chatgpt_with_personas(sample_statements, sample_personas, openai_client)
    if result["winner"] is not None:
        full_winner = str(alt_mapping[int(result["winner"])])
        result["winner_full"] = full_winner
        result["epsilon"] = lookup_epsilon(precomputed_epsilons, full_winner)
    results["chatgpt_personas"] = result
    
    return results


def run_chatgpt_star_methods(
    all_statements: List[Dict],
    sample_statements: List[Dict],
    sample_preferences: List[List[str]],
    sample_personas: List[str],
    precomputed_epsilons: Dict[str, float],
    openai_client: OpenAI
) -> Dict[str, Dict]:
    """Run ChatGPT* methods (select from all 100 alternatives)."""
    logger = logging.getLogger(__name__)
    results = {}
    
    # ChatGPT*
    logger.info("  Running ChatGPT*...")
    result = run_chatgpt_star(all_statements, sample_statements, openai_client)
    if result["winner"] is not None:
        result["epsilon"] = lookup_epsilon(precomputed_epsilons, result["winner"])
    results["chatgpt_star"] = result
    
    # ChatGPT* with rankings
    logger.info("  Running ChatGPT*+Rankings...")
    result = run_chatgpt_star_with_rankings(
        all_statements, sample_statements, sample_preferences, openai_client
    )
    if result["winner"] is not None:
        result["epsilon"] = lookup_epsilon(precomputed_epsilons, result["winner"])
    results["chatgpt_star_rankings"] = result
    
    # ChatGPT* with personas
    logger.info("  Running ChatGPT*+Personas...")
    result = run_chatgpt_star_with_personas(all_statements, sample_personas, openai_client)
    if result["winner"] is not None:
        result["epsilon"] = lookup_epsilon(precomputed_epsilons, result["winner"])
    results["chatgpt_star_personas"] = result
    
    return results


def run_chatgpt_double_star_methods(
    all_statements: List[Dict],
    sample_statements: List[Dict],
    sample_preferences: List[List[str]],
    sample_personas: List[str],
    full_preferences: List[List[str]],
    voter_sample_indices: List[int],
    voter_personas: List[str],
    topic: str,
    precomputed_epsilons: Dict[str, float],
    openai_client: OpenAI
) -> Dict[str, Dict]:
    """Run ChatGPT** methods (generate new statement)."""
    logger = logging.getLogger(__name__)
    results = {}
    
    # ChatGPT**
    logger.info("  Running ChatGPT**...")
    result = run_chatgpt_double_star(
        sample_statements, all_statements, sample_personas,
        full_preferences, voter_sample_indices, topic, openai_client
    )
    if result.get("new_statement"):
        # Re-query voters to insert new statement
        logger.info("    Inserting new statement into rankings...")
        selected_personas = [voter_personas[i] for i in voter_sample_indices]
        updated_prefs = insert_new_statement_into_rankings(
            result["new_statement"], all_statements, selected_personas,
            voter_sample_indices, full_preferences, topic, openai_client
        )
        # Compute epsilon with m=100 (no veto power for new alt)
        epsilon = compute_epsilon_for_new_statement(updated_prefs, len(all_statements))
        result["epsilon"] = epsilon
    results["chatgpt_double_star"] = result
    
    # ChatGPT** with rankings
    logger.info("  Running ChatGPT**+Rankings...")
    result = run_chatgpt_double_star_with_rankings(
        sample_statements, sample_preferences, all_statements, sample_personas,
        full_preferences, voter_sample_indices, topic, openai_client
    )
    if result.get("new_statement"):
        logger.info("    Inserting new statement into rankings...")
        selected_personas = [voter_personas[i] for i in voter_sample_indices]
        updated_prefs = insert_new_statement_into_rankings(
            result["new_statement"], all_statements, selected_personas,
            voter_sample_indices, full_preferences, topic, openai_client
        )
        epsilon = compute_epsilon_for_new_statement(updated_prefs, len(all_statements))
        result["epsilon"] = epsilon
    results["chatgpt_double_star_rankings"] = result
    
    # ChatGPT** with personas
    logger.info("  Running ChatGPT**+Personas...")
    result = run_chatgpt_double_star_with_personas(
        sample_statements, sample_personas, all_statements,
        full_preferences, voter_sample_indices, topic, openai_client
    )
    if result.get("new_statement"):
        logger.info("    Inserting new statement into rankings...")
        selected_personas = [voter_personas[i] for i in voter_sample_indices]
        updated_prefs = insert_new_statement_into_rankings(
            result["new_statement"], all_statements, selected_personas,
            voter_sample_indices, full_preferences, topic, openai_client
        )
        epsilon = compute_epsilon_for_new_statement(updated_prefs, len(all_statements))
        result["epsilon"] = epsilon
    results["chatgpt_double_star_personas"] = result
    
    return results


def run_single_rep(
    topic_slug: str,
    rep_idx: int,
    all_entries: List[Dict],
    openai_client: OpenAI,
    output_dir: Path
) -> None:
    """Run a single replication of the experiment."""
    logger = logging.getLogger(__name__)
    
    rep_dir = output_dir / "data" / topic_slug / f"rep{rep_idx}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=" * 60)
    logger.info(f"Rep {rep_idx}")
    logger.info(f"Output: {rep_dir}")
    logger.info(f"=" * 60)
    
    # Compute seed for this rep
    seed = BASE_SEED + rep_idx
    
    # Step 1: Sample pools
    logger.info("Step 1: Sampling voter and alternative pools...")
    
    if check_cache_exists(rep_dir, "pool_data.json"):
        logger.info("  Loading cached pool data...")
        voter_indices, alt_indices, voter_personas, alt_statements = load_pool_data(rep_dir)
    else:
        voter_indices, alt_indices, voter_personas, alt_statements = sample_pools(
            all_entries, N_VOTER_POOL, N_ALT_POOL, seed
        )
        save_pool_data(voter_indices, alt_indices, voter_personas, alt_statements, rep_dir)
    
    logger.info(f"  Voters: {len(voter_personas)}, Alternatives: {len(alt_statements)}")
    
    # Step 2: Build full preference profile
    logger.info("Step 2: Building full 100x100 preference profile...")
    
    topic = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    
    if check_cache_exists(rep_dir, "full_preferences.json"):
        logger.info("  Loading cached preferences...")
        full_preferences = load_preferences(rep_dir)
    else:
        full_preferences = build_full_preferences(
            voter_personas, alt_statements, topic_slug, openai_client
        )
        save_preferences(full_preferences, rep_dir)
    
    logger.info(f"  Preferences: {len(full_preferences)} x {len(full_preferences[0])}")
    
    # Step 3: Precompute epsilons
    logger.info("Step 3: Precomputing epsilons for all 100 alternatives...")
    
    if check_cache_exists(rep_dir, "precomputed_epsilons.json"):
        logger.info("  Loading cached epsilons...")
        precomputed_epsilons = load_precomputed_epsilons(rep_dir)
    else:
        precomputed_epsilons = precompute_all_epsilons(full_preferences)
        save_precomputed_epsilons(precomputed_epsilons, rep_dir)
    
    mean_eps = get_mean_epsilon(precomputed_epsilons)
    logger.info(f"  Mean epsilon: {mean_eps:.4f}")
    
    # Step 4: Run samples
    logger.info("Step 4: Running (K, P) samples...")
    
    for k in K_VALUES:
        for p in P_VALUES:
            for sample_idx in range(N_SAMPLES_PER_KP):
                sample_dir = rep_dir / f"k{k}_p{p}" / f"sample{sample_idx}"
                
                if check_cache_exists(sample_dir, "results.json"):
                    logger.info(f"  K={k}, P={p}, Sample {sample_idx}: cached, skipping")
                    continue
                
                logger.info(f"  K={k}, P={p}, Sample {sample_idx}: Running...")
                sample_dir.mkdir(parents=True, exist_ok=True)
                
                # Sample K voters and P alternatives
                # Use sample_idx * 10000 to ensure different samples get different seeds
                sample_seed = seed * 1000 + k * 100 + p + sample_idx * 10000
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
                logger.info(f"    Results: " + ", ".join(
                    f"{m}={r.get('epsilon', 'N/A'):.3f}" if r.get('epsilon') else f"{m}=N/A"
                    for m, r in list(results.items())[:5]
                ))
    
    logger.info(f"Completed rep {rep_idx}")


def run_experiment(
    topic_slug: str,
    output_dir: Path,
    openai_client: OpenAI,
    n_reps: int = N_REPS
) -> None:
    """Run the full experiment for a single topic."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info(f"STARTING EXPERIMENT")
    logger.info(f"Topic: {topic_slug}")
    logger.info(f"Reps: {n_reps}")
    logger.info(f"K values: {K_VALUES}")
    logger.info(f"P values: {P_VALUES}")
    logger.info("=" * 80)
    
    # Load all entries
    all_entries = load_all_entries(topic_slug)
    
    # Run each rep
    for rep_idx in range(n_reps):
        try:
            run_single_rep(topic_slug, rep_idx, all_entries, openai_client, output_dir)
        except Exception as e:
            logger.error(f"Rep {rep_idx} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    try:
        generate_all_visualizations(output_dir, topic_slug, n_reps)
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
    
    logger.info("\nEXPERIMENT COMPLETE")


def main():
    parser = argparse.ArgumentParser(description="Run sampling experiment (public trust topic)")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with fewer reps"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Determine number of reps
    n_reps = N_REPS_TEST if args.test else N_REPS
    
    # Setup
    setup_logging(args.output_dir, test_mode=args.test)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting at {datetime.now().isoformat()}")
    logger.info(f"Test mode: {args.test}")
    logger.info(f"Number of reps: {n_reps}")
    
    # Create OpenAI client
    openai_client = OpenAI(timeout=120.0)
    
    # Run experiment on public trust topic
    run_experiment(TEST_TOPIC, args.output_dir, openai_client, n_reps)


if __name__ == "__main__":
    main()
