"""
Full sampling experiment runner.

Runs comprehensive voting experiments across all 13 topics with:
- Fixed K=20 voters, P=20 alternatives
- 10 replications x 5 samples = 50 samples per topic
- GPT-5-mini for preference ranking, GPT-5.2 for voting methods
- ChatGPT*** method (blind bridging statement generation)
- Likert experiment on all reps
- Mini variant for immigration topic with persona bridging statements

Usage:
    uv run python -m src.full_sampling_experiment.run_experiment
    uv run python -m src.full_sampling_experiment.run_experiment --topics trust immigration
    uv run python -m src.full_sampling_experiment.run_experiment --skip-mini-variant
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

from .config import (
    OUTPUT_DIR,
    ALL_TOPICS,
    MINI_VARIANT_TOPIC,
    N_REPS,
    N_VOTER_POOL,
    N_ALT_POOL,
    K_VALUE,
    P_VALUE,
    N_SAMPLES_PER_REP,
    BASE_SEED,
    MAX_WORKERS,
    MODEL_RANKING,
    MODEL_VOTING,
    TOPIC_QUESTIONS,
    TOPIC_SHORT_NAMES,
    TOPIC_DISPLAY_NAMES,
)
from src.sampling_experiment.data_loader import (
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
from src.sampling_experiment.preference_builder import build_full_preferences
from src.sampling_experiment.epsilon_calculator import (
    precompute_all_epsilons,
    lookup_epsilon,
    compute_epsilon_for_new_statement,
    save_precomputed_epsilons,
    load_precomputed_epsilons,
    get_mean_epsilon,
)
from src.sampling_experiment.voting_methods import (
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
    run_chatgpt_triple_star,
)
from .likert_experiment import (
    collect_likert_scores,
    save_likert_scores,
    load_likert_scores,
    plot_likert_histograms,
)
from .mini_variant import run_mini_variant_rep


def setup_logging(output_dir: Path) -> None:
    """Set up logging to file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"full_experiment_{timestamp}.log"
    
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
    
    return log_file


def run_traditional_methods(
    sample_preferences: List[List[str]],
    alt_mapping: Dict[int, int],
    precomputed_epsilons: Dict[str, float]
) -> Dict[str, Dict]:
    """Run traditional voting methods and look up epsilons."""
    logger = logging.getLogger(__name__)
    results = {}
    
    for method_name, method_fn in [
        ("schulze", run_schulze),
        ("borda", run_borda),
        ("irv", run_irv),
        ("plurality", run_plurality),
        ("veto_by_consumption", run_veto_by_consumption),
    ]:
        logger.info(f"    Running {method_name}...")
        result = method_fn(sample_preferences)
        if result["winner"] is not None:
            full_winner = str(alt_mapping[int(result["winner"])])
            result["winner_full"] = full_winner
            result["epsilon"] = lookup_epsilon(precomputed_epsilons, full_winner)
        results[method_name] = result
    
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
    logger.info("    Running ChatGPT...")
    result = run_chatgpt(sample_statements, openai_client, MODEL_VOTING)
    if result["winner"] is not None:
        full_winner = str(alt_mapping[int(result["winner"])])
        result["winner_full"] = full_winner
        result["epsilon"] = lookup_epsilon(precomputed_epsilons, full_winner)
    results["chatgpt"] = result
    
    # ChatGPT with rankings
    logger.info("    Running ChatGPT+Rankings...")
    result = run_chatgpt_with_rankings(sample_statements, sample_preferences, openai_client, MODEL_VOTING)
    if result["winner"] is not None:
        full_winner = str(alt_mapping[int(result["winner"])])
        result["winner_full"] = full_winner
        result["epsilon"] = lookup_epsilon(precomputed_epsilons, full_winner)
    results["chatgpt_rankings"] = result
    
    # ChatGPT with personas
    logger.info("    Running ChatGPT+Personas...")
    result = run_chatgpt_with_personas(sample_statements, sample_personas, openai_client, MODEL_VOTING)
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
    logger.info("    Running ChatGPT*...")
    result = run_chatgpt_star(all_statements, sample_statements, openai_client, MODEL_VOTING)
    if result["winner"] is not None:
        result["epsilon"] = lookup_epsilon(precomputed_epsilons, result["winner"])
    results["chatgpt_star"] = result
    
    # ChatGPT* with rankings
    logger.info("    Running ChatGPT*+Rankings...")
    result = run_chatgpt_star_with_rankings(
        all_statements, sample_statements, sample_preferences, openai_client, MODEL_VOTING
    )
    if result["winner"] is not None:
        result["epsilon"] = lookup_epsilon(precomputed_epsilons, result["winner"])
    results["chatgpt_star_rankings"] = result
    
    # ChatGPT* with personas
    logger.info("    Running ChatGPT*+Personas...")
    result = run_chatgpt_star_with_personas(all_statements, sample_personas, openai_client, MODEL_VOTING)
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
    logger.info("    Running ChatGPT**...")
    result = run_chatgpt_double_star(
        sample_statements, all_statements, sample_personas,
        full_preferences, voter_sample_indices, topic, openai_client, MODEL_VOTING
    )
    if result.get("new_statement"):
        logger.info("      Inserting new statement into rankings...")
        selected_personas = [voter_personas[i] for i in voter_sample_indices]
        updated_prefs = insert_new_statement_into_rankings(
            result["new_statement"], all_statements, selected_personas,
            voter_sample_indices, full_preferences, topic, openai_client
        )
        epsilon = compute_epsilon_for_new_statement(updated_prefs, len(all_statements))
        result["epsilon"] = epsilon
    results["chatgpt_double_star"] = result
    
    # ChatGPT** with rankings
    logger.info("    Running ChatGPT**+Rankings...")
    result = run_chatgpt_double_star_with_rankings(
        sample_statements, sample_preferences, all_statements, sample_personas,
        full_preferences, voter_sample_indices, topic, openai_client, MODEL_VOTING
    )
    if result.get("new_statement"):
        logger.info("      Inserting new statement into rankings...")
        selected_personas = [voter_personas[i] for i in voter_sample_indices]
        updated_prefs = insert_new_statement_into_rankings(
            result["new_statement"], all_statements, selected_personas,
            voter_sample_indices, full_preferences, topic, openai_client
        )
        epsilon = compute_epsilon_for_new_statement(updated_prefs, len(all_statements))
        result["epsilon"] = epsilon
    results["chatgpt_double_star_rankings"] = result
    
    # ChatGPT** with personas
    logger.info("    Running ChatGPT**+Personas...")
    result = run_chatgpt_double_star_with_personas(
        sample_statements, sample_personas, all_statements,
        full_preferences, voter_sample_indices, topic, openai_client, MODEL_VOTING
    )
    if result.get("new_statement"):
        logger.info("      Inserting new statement into rankings...")
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
    
    # Step 2: Build full preference profile (using GPT-5-mini for ranking)
    logger.info(f"Step 2: Building full {N_VOTER_POOL}x{N_ALT_POOL} preference profile (model={MODEL_RANKING})...")
    
    topic = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    
    if check_cache_exists(rep_dir, "full_preferences.json"):
        logger.info("  Loading cached preferences...")
        full_preferences = load_preferences(rep_dir)
    else:
        # Use MODEL_RANKING for preference construction
        full_preferences = build_full_preferences(
            voter_personas, alt_statements, topic_slug, openai_client,
            model=MODEL_RANKING  # GPT-5-mini
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
    
    # Step 4: Collect Likert scores (only rep 0)
    if rep_idx == 0:
        logger.info("Step 4: Collecting Likert scores (rep 0 only)...")
        
        likert_path = rep_dir / "likert_scores.json"
        if likert_path.exists():
            logger.info("  Loading cached Likert scores...")
            likert_scores = load_likert_scores(likert_path)
        else:
            likert_scores = collect_likert_scores(
                voter_personas, alt_statements, topic, openai_client
            )
            save_likert_scores(likert_scores, likert_path)
        
        logger.info(f"  Likert scores shape: {likert_scores.shape}")
    else:
        logger.info("Step 4: Skipping Likert scores (only collected for rep 0)")
    
    # Step 5: Generate ChatGPT*** result (once per rep, reused for all samples)
    logger.info("Step 5: Running ChatGPT*** (blind bridging, once per rep)...")
    
    triple_star_cache_path = rep_dir / "chatgpt_triple_star.json"
    if triple_star_cache_path.exists():
        logger.info("  Loading cached ChatGPT*** result...")
        with open(triple_star_cache_path) as f:
            triple_star_result = json.load(f)
    else:
        triple_star_result = run_chatgpt_triple_star(
            topic, alt_statements, voter_personas, full_preferences,
            openai_client, model=MODEL_VOTING
        )
        with open(triple_star_cache_path, 'w') as f:
            json.dump(triple_star_result, f, indent=2)
    
    logger.info(f"  ChatGPT*** epsilon: {triple_star_result.get('epsilon', 'N/A')}")
    
    # Step 6: Run samples
    logger.info(f"Step 6: Running {N_SAMPLES_PER_REP} samples (K={K_VALUE}, P={P_VALUE})...")
    
    for sample_idx in range(N_SAMPLES_PER_REP):
        sample_dir = rep_dir / f"sample{sample_idx}"
        
        if check_cache_exists(sample_dir, "results.json"):
            logger.info(f"  Sample {sample_idx}: cached, skipping")
            continue
        
        logger.info(f"  Sample {sample_idx}: Running...")
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample K voters and P alternatives
        sample_seed = seed * 1000 + K_VALUE * 100 + P_VALUE + sample_idx * 10000
        voter_sample, alt_sample = sample_kp(N_VOTER_POOL, N_ALT_POOL, K_VALUE, P_VALUE, sample_seed)
        
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
                "k": K_VALUE, "p": P_VALUE,
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
        
        # ChatGPT*** method (reuse pre-computed result from Step 5)
        results["chatgpt_triple_star"] = triple_star_result
        
        # Save results
        with open(sample_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log summary
        logger.info(f"    Results: " + ", ".join(
            f"{m}={r.get('epsilon', 'N/A'):.3f}" if r.get('epsilon') is not None else f"{m}=N/A"
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
    logger.info(f"Topic: {topic_slug} ({TOPIC_DISPLAY_NAMES.get(topic_slug, topic_slug)})")
    logger.info(f"Reps: {n_reps}")
    logger.info(f"K={K_VALUE}, P={P_VALUE}")
    logger.info(f"Samples per rep: {N_SAMPLES_PER_REP}")
    logger.info(f"Model (ranking): {MODEL_RANKING}")
    logger.info(f"Model (voting): {MODEL_VOTING}")
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
    
    logger.info(f"\nCompleted topic: {topic_slug}")


def run_mini_variant(
    output_dir: Path,
    openai_client: OpenAI,
    n_reps: int = N_REPS
) -> None:
    """Run the mini variant experiment for immigration topic."""
    logger = logging.getLogger(__name__)
    
    topic_slug = MINI_VARIANT_TOPIC
    
    logger.info("=" * 80)
    logger.info("STARTING MINI VARIANT EXPERIMENT")
    logger.info(f"Topic: {topic_slug} ({TOPIC_DISPLAY_NAMES.get(topic_slug, topic_slug)})")
    logger.info(f"Reps: {n_reps}")
    logger.info("=" * 80)
    
    # Load original statements from main experiment
    all_entries = load_all_entries(topic_slug)
    
    mini_output_dir = output_dir / "mini_variant" / "immigration"
    
    for rep_idx in range(n_reps):
        try:
            # Load pool data from main experiment
            main_rep_dir = output_dir / "data" / topic_slug / f"rep{rep_idx}"
            
            if not (main_rep_dir / "pool_data.json").exists():
                logger.warning(f"Main experiment rep {rep_idx} not found, skipping mini variant")
                continue
            
            voter_indices, alt_indices, voter_personas, alt_statements = load_pool_data(main_rep_dir)
            
            # Reconstruct statements with personas from all_entries
            # (alt_statements from pool_data doesn't have 'persona' field)
            statements_with_personas = [
                {"statement": all_entries[idx]["statement"], 
                 "persona": all_entries[idx]["persona"],
                 "original_idx": idx}
                for idx in alt_indices
            ]
            
            # Run mini variant rep
            run_mini_variant_rep(
                rep_idx, statements_with_personas, voter_personas, topic_slug,
                mini_output_dir, openai_client
            )
            
        except Exception as e:
            logger.error(f"Mini variant rep {rep_idx} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("\nCompleted mini variant experiment")


def generate_visualizations(output_dir: Path, topics: List[str], n_reps: int) -> None:
    """Generate all visualizations after experiment completion."""
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 80)
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect Likert scores across all topics (only rep 0)
    all_topics_scores = {}
    
    for topic_slug in topics:
        likert_path = output_dir / "data" / topic_slug / "rep0" / "likert_scores.json"
        
        if likert_path.exists():
            scores = load_likert_scores(likert_path)
            all_topics_scores[topic_slug] = scores.flatten()
            logger.info(f"  {topic_slug}: {len(all_topics_scores[topic_slug]):,} Likert scores")
    
    # Plot Likert histograms
    if all_topics_scores:
        logger.info("Plotting Likert histograms...")
        plot_likert_histograms(
            all_topics_scores,
            figures_dir / "likert_histograms_all_topics.png"
        )
    
    # Mini variant comparison (if available)
    mini_variant_dir = output_dir / "mini_variant" / "immigration"
    
    if mini_variant_dir.exists():
        logger.info("Plotting mini variant comparison...")
        
        # Collect mini variant Likert scores (only rep 0)
        mini_likert_path = mini_variant_dir / "rep0" / "likert_scores.json"
        mini_scores = None
        if mini_likert_path.exists():
            mini_scores = load_likert_scores(mini_likert_path)
        
        if mini_scores is not None and MINI_VARIANT_TOPIC in all_topics_scores:
            from .likert_experiment import plot_likert_comparison
            
            plot_likert_comparison(
                {MINI_VARIANT_TOPIC: all_topics_scores[MINI_VARIANT_TOPIC]},
                {MINI_VARIANT_TOPIC: mini_scores.flatten()},
                figures_dir / "mini_variant_comparison.png",
                main_label="Original Statements",
                variant_label="Bridging Statements"
            )
    
    logger.info("Visualizations complete")


def main():
    parser = argparse.ArgumentParser(description="Run full sampling experiment")
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="Specific topics to run (use short names from TOPIC_SHORT_NAMES)"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=N_REPS,
        help="Number of replications"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--skip-mini-variant",
        action="store_true",
        help="Skip the mini variant experiment"
    )
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip visualization generation"
    )
    
    args = parser.parse_args()
    
    # Determine which topics to run
    if args.topics:
        # Map short names back to full slugs
        short_to_full = {v: k for k, v in TOPIC_SHORT_NAMES.items()}
        topics = []
        for t in args.topics:
            if t in short_to_full:
                topics.append(short_to_full[t])
            elif t in ALL_TOPICS:
                topics.append(t)
            else:
                print(f"Unknown topic: {t}")
                print(f"Valid short names: {list(TOPIC_SHORT_NAMES.values())}")
                sys.exit(1)
    else:
        topics = ALL_TOPICS
    
    # Setup
    log_file = setup_logging(args.output_dir)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting full experiment at {datetime.now().isoformat()}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Topics: {len(topics)}")
    logger.info(f"Reps per topic: {args.reps}")
    logger.info(f"K={K_VALUE}, P={P_VALUE}")
    logger.info(f"Samples per rep: {N_SAMPLES_PER_REP}")
    logger.info(f"Total samples per topic: {args.reps * N_SAMPLES_PER_REP}")
    logger.info(f"Model (ranking): {MODEL_RANKING}")
    logger.info(f"Model (voting): {MODEL_VOTING}")
    
    # Create OpenAI client
    openai_client = OpenAI(timeout=120.0)
    
    # Run experiment on each topic
    for idx, topic_slug in enumerate(topics):
        logger.info(f"\n{'#' * 80}")
        logger.info(f"TOPIC {idx + 1}/{len(topics)}: {TOPIC_SHORT_NAMES.get(topic_slug, topic_slug)}")
        logger.info(f"{'#' * 80}")
        
        try:
            run_experiment(topic_slug, args.output_dir, openai_client, args.reps)
        except Exception as e:
            logger.error(f"Topic {topic_slug} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Run mini variant (immigration topic only)
    if not args.skip_mini_variant and MINI_VARIANT_TOPIC in topics:
        logger.info("\n" + "#" * 80)
        logger.info("MINI VARIANT EXPERIMENT")
        logger.info("#" * 80)
        
        try:
            run_mini_variant(args.output_dir, openai_client, args.reps)
        except Exception as e:
            logger.error(f"Mini variant failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Generate visualizations
    if not args.skip_visualizations:
        try:
            generate_visualizations(args.output_dir, topics, args.reps)
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info("\n" + "=" * 80)
    logger.info("FULL EXPERIMENT COMPLETE")
    logger.info(f"Finished at {datetime.now().isoformat()}")
    logger.info(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
