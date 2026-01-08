"""
Run voting experiments with different persona counts (5, 10 personas).

This script reuses existing preference profiles and runs voting methods
with fewer personas than the standard 20, storing results in subdirectories.

Usage:
    uv run python -m src.full_experiment.run_multi_persona
    uv run python -m src.full_experiment.run_multi_persona --persona-counts 5 10
    uv run python -m src.full_experiment.run_multi_persona --test
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
    N_PERSONA_SAMPLES,
    N_PERSONA_SAMPLES_TEST,
    N_PERSONAS,
    BASE_SEED,
    ABLATION_FULL,
    ABLATIONS,
)
from .voting_runner import (
    sample_personas_for_voting,
    extract_sampled_preferences,
    run_all_voting_methods,
    save_voting_results,
    save_sampled_persona_indices,
    save_sampled_preferences,
)
from .data_loader import check_cache_exists

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path, test_mode: bool = False) -> None:
    """Set up logging to file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if test_mode:
        log_file = log_dir / f"multi_persona_test_{timestamp}.log"
    else:
        log_file = log_dir / f"multi_persona_{timestamp}.log"
    
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
    
    # Suppress verbose httpx logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    logging.info(f"Logging to {log_file}")


def get_filtered_preferences_path(rep_dir: Path, ablation: str) -> Path:
    """
    Get the path to the filtered preferences file for a given ablation.
    
    Args:
        rep_dir: Path to the rep directory (e.g., data/topic/rep0)
        ablation: Ablation type ('full', 'no_filtering', 'no_bridging')
    
    Returns:
        Path to the preferences JSON file
    """
    if ablation == "full":
        return rep_dir / "filtered_preferences.json"
    elif ablation == "no_filtering":
        return rep_dir / "full_preferences.json"
    elif ablation == "no_bridging":
        return rep_dir / "ablation_no_bridging" / "full_preferences.json"
    else:
        raise ValueError(f"Unknown ablation: {ablation}")


def get_filtered_statements_path(rep_dir: Path, ablation: str) -> Path:
    """
    Get the path to the bridging statements file for a given ablation.
    
    Args:
        rep_dir: Path to the rep directory
        ablation: Ablation type
    
    Returns:
        Path to the bridging statements JSON file
    """
    if ablation == "full" or ablation == "no_filtering":
        return rep_dir / "bridging_statements.json"
    elif ablation == "no_bridging":
        # For no_bridging, we need to construct statements from the original data
        # But for voting, we just need the statement texts from filter_assignments
        return rep_dir / "ablation_no_bridging" / "bridging_statements.json"
    else:
        raise ValueError(f"Unknown ablation: {ablation}")


def get_output_dir_for_n_personas(
    rep_dir: Path,
    ablation: str,
    n_personas: int
) -> Path:
    """
    Get the output directory for a specific persona count.
    
    Args:
        rep_dir: Path to the rep directory
        ablation: Ablation type
        n_personas: Number of personas (5, 10, or 20)
    
    Returns:
        Path to the output directory
    """
    if ablation == "full":
        base_dir = rep_dir
    else:
        base_dir = rep_dir / f"ablation_{ablation}"
    
    if n_personas == 20:
        # Standard 20 personas go directly in base_dir
        return base_dir
    else:
        # Other counts go in subdirectories
        return base_dir / f"{n_personas}-personas"


def run_multi_persona_voting(
    topic_slug: str,
    rep_idx: int,
    ablation: str,
    n_sample_personas: int,
    output_dir: Path,
    openai_client: OpenAI,
    n_persona_samples: int = N_PERSONA_SAMPLES
) -> None:
    """
    Run voting experiments with a specific number of personas.
    
    Args:
        topic_slug: Topic slug
        rep_idx: Repetition index (0-4)
        ablation: Ablation type
        n_sample_personas: Number of personas to sample (5, 10, or 20)
        output_dir: Base output directory
        openai_client: OpenAI client
        n_persona_samples: Number of persona sampling repetitions (inner loop)
    """
    rep_dir = output_dir / "data" / topic_slug / f"rep{rep_idx}"
    
    # Get the filtered preferences
    prefs_path = get_filtered_preferences_path(rep_dir, ablation)
    if not prefs_path.exists():
        logger.warning(f"Preferences file not found: {prefs_path}")
        return
    
    with open(prefs_path, 'r') as f:
        full_preferences = json.load(f)
    
    # Get statements and personas for ChatGPT methods
    statements_path = get_filtered_statements_path(rep_dir, ablation)
    all_personas = None
    if statements_path.exists():
        with open(statements_path, 'r') as f:
            bridging_statements = json.load(f)
        stmt_dicts = [{"statement": s["statement"]} for s in bridging_statements]
        # Extract all personas (each bridging statement has a persona)
        all_personas = [s.get("persona", f"Voter {i}") for i, s in enumerate(bridging_statements)]
    else:
        # Create dummy statements if not available
        n_statements = len(full_preferences)
        stmt_dicts = [{"statement": f"Statement {i}"} for i in range(n_statements)]
        all_personas = [f"Voter {i}" for i in range(n_statements)]
        logger.warning(f"Bridging statements not found, using placeholders")
    
    # For filtered preferences, we need to filter statements too
    if ablation == "full":
        # Load filter assignments to get kept indices
        filter_path = rep_dir / "filter_assignments.json"
        if filter_path.exists():
            with open(filter_path, 'r') as f:
                assignments = json.load(f)
            kept_indices = sorted([
                a["statement_idx"]
                for a in assignments
                if a["keep"] == 1
            ])
            # Filter statements to only kept ones
            if len(stmt_dicts) > len(kept_indices):
                stmt_dicts = [stmt_dicts[i] for i in kept_indices]
    
    # Get output directory for this persona count
    data_dir = get_output_dir_for_n_personas(rep_dir, ablation, n_sample_personas)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=" * 60)
    logger.info(f"Topic: {topic_slug}")
    logger.info(f"Rep: {rep_idx}, Ablation: {ablation}, Personas: {n_sample_personas}")
    logger.info(f"Output: {data_dir}")
    logger.info(f"=" * 60)
    
    # Compute seed for this repetition
    seed = BASE_SEED + rep_idx
    
    for sample_idx in range(n_persona_samples):
        sample_dir = data_dir / f"sample{sample_idx}"
        
        if check_cache_exists(sample_dir, "results.json"):
            logger.info(f"  Sample {sample_idx}: cached, skipping")
            continue
        
        logger.info(f"  Sample {sample_idx}: Running with {n_sample_personas} personas...")
        
        # Sample personas
        sample_seed = seed * 100 + sample_idx
        sampled_persona_indices = sample_personas_for_voting(
            N_PERSONAS, n_sample_personas, sample_seed
        )
        
        # Extract preferences for sampled personas
        sampled_prefs = extract_sampled_preferences(
            full_preferences, sampled_persona_indices
        )
        
        # Extract persona descriptions for sampled voters
        sampled_personas = None
        if all_personas is not None:
            sampled_personas = [all_personas[i] for i in sampled_persona_indices]
        
        # Save sampled data
        save_sampled_persona_indices(sampled_persona_indices, sample_dir)
        save_sampled_preferences(sampled_prefs, sample_dir)
        
        # Run voting methods (with personas for chatgpt_with_personas)
        results = run_all_voting_methods(
            sampled_prefs, stmt_dicts, openai_client, personas=sampled_personas
        )
        
        # Save results
        save_voting_results(results, sample_dir)
        
        logger.info(f"    Winners: " + ", ".join(
            f"{m}={r.get('winner')}" for m, r in results.items()
        ))
    
    logger.info(f"Completed: {topic_slug} rep{rep_idx} {ablation} {n_sample_personas}-personas")


def run_all_multi_persona_experiments(
    topics: list,
    ablations: list,
    persona_counts: list,
    output_dir: Path,
    openai_client: OpenAI,
    test_mode: bool = False
) -> None:
    """
    Run voting experiments for multiple persona counts across all topics.
    
    Args:
        topics: List of topic slugs
        ablations: List of ablation types
        persona_counts: List of persona counts to test (e.g., [5, 10])
        output_dir: Output directory
        openai_client: OpenAI client
        test_mode: Whether to use reduced parameters
    """
    n_persona_samples = N_PERSONA_SAMPLES_TEST if test_mode else N_PERSONA_SAMPLES
    
    # Count total reps available
    data_dir = output_dir / "data"
    n_reps = 5  # Standard number of reps
    
    logger.info("=" * 80)
    logger.info(f"STARTING MULTI-PERSONA EXPERIMENT")
    logger.info(f"Topics: {len(topics)}")
    logger.info(f"Ablations: {ablations}")
    logger.info(f"Persona counts: {persona_counts}")
    logger.info(f"Persona samples per rep: {n_persona_samples}")
    logger.info("=" * 80)
    
    for topic_idx, topic_slug in enumerate(topics):
        logger.info(f"\n{'#' * 80}")
        logger.info(f"TOPIC {topic_idx + 1}/{len(topics)}: {topic_slug}")
        logger.info(f"{'#' * 80}")
        
        topic_dir = data_dir / topic_slug
        if not topic_dir.exists():
            logger.warning(f"Topic directory not found: {topic_dir}")
            continue
        
        # Find available reps
        rep_dirs = sorted(topic_dir.glob("rep*"))
        
        for rep_dir in rep_dirs:
            rep_idx = int(rep_dir.name.replace("rep", ""))
            
            for ablation in ablations:
                for n_personas in persona_counts:
                    try:
                        run_multi_persona_voting(
                            topic_slug,
                            rep_idx,
                            ablation,
                            n_personas,
                            output_dir,
                            openai_client,
                            n_persona_samples=n_persona_samples
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed: {topic_slug} rep{rep_idx} {ablation} "
                            f"{n_personas}-personas: {e}"
                        )
                        import traceback
                        logger.error(traceback.format_exc())
    
    logger.info("\nMULTI-PERSONA EXPERIMENT COMPLETE")


def main():
    parser = argparse.ArgumentParser(
        description="Run voting experiments with different persona counts"
    )
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
        "--persona-counts",
        nargs="+",
        type=int,
        default=[5, 10],
        help="Persona counts to test (default: 5 10)"
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
    logger.info(f"Persona counts: {args.persona_counts}")
    
    # Create OpenAI client with 60s read timeout
    openai_client = OpenAI(timeout=60.0)
    
    # Run experiments
    run_all_multi_persona_experiments(
        topics,
        ablations,
        args.persona_counts,
        args.output_dir,
        openai_client,
        test_mode=args.test
    )


if __name__ == "__main__":
    main()

