"""
Main experiment orchestration script.

Usage:
    uv run python -m src.full_experiment.run_experiment --test
    uv run python -m src.full_experiment.run_experiment
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

from .config import (
    OUTPUT_DIR,
    ALL_TOPICS,
    TEST_TOPICS,
    N_STATEMENT_REPS,
    N_PERSONA_SAMPLES,
    N_STATEMENT_REPS_TEST,
    N_PERSONA_SAMPLES_TEST,
    N_STATEMENTS,
    N_SAMPLE_PERSONAS,
    N_PERSONAS,
    BASE_SEED,
    ABLATION_FULL,
    ABLATION_NO_BRIDGING,
    ABLATION_NO_FILTERING,
    ABLATIONS,
)
from .data_loader import (
    load_all_statements,
    sample_entries,
    save_sampled_data,
    load_sampled_data,
    check_cache_exists,
    load_json_cache,
    save_json_cache,
)
from .bridging_generator import (
    generate_bridging_statements,
    save_bridging_statements,
    load_bridging_statements,
)
from .preference_builder import (
    build_full_preferences,
    build_full_likert,
    save_preferences,
    load_preferences,
    save_likert,
    load_likert,
)
from .statement_filter import (
    cluster_statements,
    apply_filter_to_preferences,
    apply_filter_to_likert,
    create_no_filter_assignments,
    save_filter_assignments,
    load_filter_assignments,
    save_filtered_preferences,
    load_filtered_preferences,
    save_filtered_likert,
    load_filtered_likert,
)
from .voting_runner import (
    sample_personas_for_voting,
    extract_sampled_preferences,
    run_all_voting_methods,
    save_voting_results,
    save_sampled_persona_indices,
    save_sampled_preferences,
)
from .visualizer import generate_all_plots


def setup_logging(output_dir: Path, test_mode: bool = False) -> None:
    """Set up logging to file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if test_mode:
        log_file = log_dir / f"test_{timestamp}.log"
    else:
        log_file = log_dir / f"experiment_{timestamp}.log"
    
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
    
    # Suppress verbose httpx logs (HTTP Request: POST ... "HTTP/1.1 200 OK")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    logging.info(f"Logging to {log_file}")


def run_single_experiment(
    topic_slug: str,
    rep_idx: int,
    ablation: str,
    all_entries: list,
    openai_client: OpenAI,
    output_dir: Path,
    n_persona_samples: int = N_PERSONA_SAMPLES
) -> None:
    """
    Run a single experiment for one topic, one repetition, one ablation.
    
    Args:
        topic_slug: Topic slug
        rep_idx: Repetition index (0-4)
        ablation: Ablation type
        all_entries: All entries for this topic (each has 'persona' and 'statement')
        openai_client: OpenAI client
        output_dir: Output directory
        n_persona_samples: Number of persona samples (inner loop)
    """
    logger = logging.getLogger(__name__)
    
    # Determine output directory
    rep_dir = output_dir / "data" / topic_slug / f"rep{rep_idx}"
    if ablation != ABLATION_FULL:
        data_dir = rep_dir / f"ablation_{ablation}"
    else:
        data_dir = rep_dir
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=" * 60)
    logger.info(f"Topic: {topic_slug}")
    logger.info(f"Rep: {rep_idx}, Ablation: {ablation}")
    logger.info(f"Output: {data_dir}")
    logger.info(f"=" * 60)
    
    # Compute seed for this repetition
    seed = BASE_SEED + rep_idx
    
    # =========================================================================
    # Step 1: Sample entries (persona + statement bundled together)
    # =========================================================================
    logger.info("Step 1: Sampling entries (persona + statement pairs)...")
    
    if check_cache_exists(rep_dir, "sampled_indices.json"):
        logger.info("  Loading cached sampled data...")
        sampled_indices, statements, personas = load_sampled_data(rep_dir, all_entries)
    else:
        sampled_indices, statements, personas = sample_entries(all_entries, N_STATEMENTS, seed)
        save_sampled_data(rep_dir, sampled_indices)
    
    logger.info(f"  Sampled {len(statements)} entries (same personas and statements)")
    
    # =========================================================================
    # Step 2: Generate bridging statements (skip for no_bridging ablation)
    # =========================================================================
    # For no_bridging: use original statements, cache in data_dir (ablation-specific)
    # For full/no_filtering: use bridging statements, cache in rep_dir (shared)
    # This ensures no_filtering only differs from full in the filtering step
    if ablation == ABLATION_NO_BRIDGING:
        logger.info("Step 2: SKIPPED (no bridging ablation)")
        # Use original statements as "bridging statements"
        bridging_statements = [
            {"persona_idx": i, "persona": personas[i], "statement": stmt["statement"]}
            for i, stmt in enumerate(statements)
        ]
        # no_bridging uses its own cache directory since statements differ
        shared_cache_dir = data_dir
    else:
        logger.info("Step 2: Generating bridging statements...")
        
        # full and no_filtering share bridging statements from rep_dir
        shared_cache_dir = rep_dir
        
        if check_cache_exists(shared_cache_dir, "bridging_statements.json"):
            logger.info("  Loading cached bridging statements...")
            bridging_statements = load_bridging_statements(shared_cache_dir)
        else:
            bridging_statements = generate_bridging_statements(
                personas, statements, topic_slug, openai_client
            )
            save_bridging_statements(bridging_statements, shared_cache_dir)
        
        logger.info(f"  Generated {len(bridging_statements)} bridging statements")
    
    # =========================================================================
    # Step 3: Build full preference profile (100x100)
    # =========================================================================
    logger.info("Step 3: Building full preference profile...")
    
    # Use bridging statements for preference building
    stmt_dicts = [{"statement": s["statement"]} for s in bridging_statements]
    
    # Use shared_cache_dir: rep_dir for full/no_filtering, data_dir for no_bridging
    # This ensures no_filtering reuses preferences from full experiment
    if check_cache_exists(shared_cache_dir, "full_preferences.json"):
        logger.info("  Loading cached preferences...")
        preferences = load_preferences(shared_cache_dir)
    else:
        preferences = build_full_preferences(
            personas, stmt_dicts, topic_slug, openai_client
        )
        save_preferences(preferences, shared_cache_dir)
    
    logger.info(f"  Built preferences: {len(preferences)} x {len(preferences[0])}")
    
    # Build Likert ratings
    if check_cache_exists(shared_cache_dir, "full_likert.json"):
        logger.info("  Loading cached Likert ratings...")
        likert = load_likert(shared_cache_dir)
    else:
        likert = build_full_likert(
            personas, stmt_dicts, topic_slug, openai_client
        )
        save_likert(likert, shared_cache_dir)
    
    logger.info(f"  Built Likert: {len(likert)} x {len(likert[0])}")
    
    # =========================================================================
    # Step 4: Filter similar statements (skip for no_filtering and no_bridging)
    # =========================================================================
    if ablation in (ABLATION_NO_FILTERING, ABLATION_NO_BRIDGING):
        logger.info(f"Step 4: SKIPPED ({ablation} ablation)")
        # No filtering - keep all statements
        assignments = create_no_filter_assignments(len(bridging_statements))
        filtered_prefs = preferences
        filtered_likert = likert
        kept_indices = list(range(len(bridging_statements)))
    else:
        logger.info("Step 4: Filtering similar statements...")
        
        if check_cache_exists(data_dir, "filter_assignments.json"):
            logger.info("  Loading cached filter assignments...")
            assignments = load_filter_assignments(data_dir)
        else:
            assignments = cluster_statements(
                stmt_dicts, topic_slug, openai_client
            )
            save_filter_assignments(assignments, data_dir)
        
        # Apply filter
        filtered_prefs, kept_indices = apply_filter_to_preferences(
            preferences, assignments
        )
        filtered_likert, _ = apply_filter_to_likert(likert, assignments)
        
        save_filtered_preferences(filtered_prefs, data_dir)
        save_filtered_likert(filtered_likert, data_dir)
    
    n_kept = len(kept_indices)
    logger.info(f"  Kept {n_kept} unique statements")
    
    # Get filtered statements for ChatGPT methods
    filtered_stmt_dicts = [stmt_dicts[i] for i in kept_indices]
    
    # =========================================================================
    # Steps 5-7: Sample personas, run voting, compute epsilon
    # =========================================================================
    logger.info("Steps 5-7: Running voting methods...")
    
    for sample_idx in range(n_persona_samples):
        sample_dir = data_dir / f"sample{sample_idx}"
        
        if check_cache_exists(sample_dir, "results.json"):
            logger.info(f"  Sample {sample_idx}: cached, skipping")
            continue
        
        logger.info(f"  Sample {sample_idx}: Running...")
        
        # Sample personas
        sample_seed = seed * 100 + sample_idx
        sampled_persona_indices = sample_personas_for_voting(
            N_PERSONAS, N_SAMPLE_PERSONAS, sample_seed
        )
        
        # Extract preferences for sampled personas
        sampled_prefs = extract_sampled_preferences(
            filtered_prefs, sampled_persona_indices
        )
        
        # Save sampled data
        save_sampled_persona_indices(sampled_persona_indices, sample_dir)
        save_sampled_preferences(sampled_prefs, sample_dir)
        
        # Run voting methods
        results = run_all_voting_methods(
            sampled_prefs, filtered_stmt_dicts, openai_client
        )
        
        # Save results
        save_voting_results(results, sample_dir)
        
        logger.info(f"    Winners: " + ", ".join(
            f"{m}={r.get('winner')}" for m, r in results.items()
        ))
    
    logger.info(f"Completed: {topic_slug} rep{rep_idx} {ablation}")


def run_full_experiment(
    topics: list,
    ablations: list,
    output_dir: Path,
    openai_client: OpenAI,
    test_mode: bool = False
) -> None:
    """
    Run the full experiment across all topics, repetitions, and ablations.
    """
    logger = logging.getLogger(__name__)
    
    # Use reduced parameters in test mode
    n_reps = N_STATEMENT_REPS_TEST if test_mode else N_STATEMENT_REPS
    n_samples = N_PERSONA_SAMPLES_TEST if test_mode else N_PERSONA_SAMPLES
    
    logger.info("=" * 80)
    logger.info(f"STARTING {'TEST ' if test_mode else ''}EXPERIMENT")
    logger.info(f"Topics: {len(topics)}")
    logger.info(f"Repetitions: {n_reps}")
    logger.info(f"Persona samples: {n_samples}")
    logger.info(f"Ablations: {ablations}")
    logger.info("=" * 80)
    
    for topic_idx, topic_slug in enumerate(topics):
        logger.info(f"\n{'#' * 80}")
        logger.info(f"TOPIC {topic_idx + 1}/{len(topics)}: {topic_slug}")
        logger.info(f"{'#' * 80}")
        
        # Load all entries for this topic (each has persona + statement bundled)
        all_entries = load_all_statements(topic_slug)
        
        for rep_idx in range(n_reps):
            for ablation in ablations:
                try:
                    run_single_experiment(
                        topic_slug,
                        rep_idx,
                        ablation,
                        all_entries,
                        openai_client,
                        output_dir,
                        n_persona_samples=n_samples
                    )
                except Exception as e:
                    logger.error(f"Failed: {topic_slug} rep{rep_idx} {ablation}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
    
    # Generate plots
    logger.info("\nGenerating plots...")
    try:
        generate_all_plots(output_dir, topics, ablations)
    except Exception as e:
        logger.error(f"Plot generation failed: {e}")
    
    logger.info("\nEXPERIMENT COMPLETE")


def main():
    parser = argparse.ArgumentParser(description="Run PVC bridging experiment")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test mode with 2 topics only"
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="Specific topics to run (overrides --test)"
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=None,
        choices=ABLATIONS,
        help="Ablations to run"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Determine topics
    if args.topics:
        topics = args.topics
    elif args.test:
        topics = TEST_TOPICS
    else:
        topics = ALL_TOPICS
    
    # Determine ablations
    ablations = args.ablations or [ABLATION_FULL]
    
    # Setup logging
    setup_logging(args.output_dir, test_mode=args.test)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting at {datetime.now().isoformat()}")
    logger.info(f"Topics: {topics}")
    logger.info(f"Ablations: {ablations}")
    
    # Create OpenAI client with 60s read timeout
    openai_client = OpenAI(timeout=60.0)
    
    # Run experiment
    run_full_experiment(topics, ablations, args.output_dir, openai_client, test_mode=args.test)


if __name__ == "__main__":
    main()

