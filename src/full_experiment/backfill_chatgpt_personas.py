"""
Backfill chatgpt_with_personas for existing results that don't have it.

Usage:
    uv run python -m src.full_experiment.backfill_chatgpt_personas
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from .config import OUTPUT_DIR, ABLATIONS
from .data_loader import load_all_statements
from .voting_runner import (
    run_chatgpt_with_personas,
    compute_epsilon_for_winner,
)

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path) -> None:
    """Set up logging to file and console with timestamps."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"backfill_chatgpt_personas_{timestamp}.log"
    
    # Create handlers
    file_handler = logging.FileHandler(log_file, mode='a')
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Set format with timestamps
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
    
    logger.info(f"Logging to {log_file}")


def load_statements_and_personas_for_ablation(rep_dir: Path, ablation: str, topic_slug: str):
    """
    Load statements and personas for a specific ablation.
    
    For 'full' ablation: load filtered statements based on filter_assignments.json
    For 'no_filtering': load all bridging statements
    For 'no_bridging': load original Polis statements (not bridging statements)
    
    Returns:
        (stmt_dicts, all_personas) tuple
    """
    if ablation == "no_bridging":
        # no_bridging uses original Polis statements, not bridging statements
        # Load sampled indices from rep_dir
        indices_path = rep_dir / "sampled_indices.json"
        if not indices_path.exists():
            return None, None
        
        with open(indices_path) as f:
            sampled_indices = json.load(f)
        
        # Load original statements from Polis data
        all_entries = load_all_statements(topic_slug)
        
        # Get sampled entries
        all_personas = [all_entries[i]["persona"] for i in sampled_indices]
        stmt_dicts = [{"statement": all_entries[i]["statement"]} for i in sampled_indices]
        
        return stmt_dicts, all_personas
    else:
        bridging_path = rep_dir / "bridging_statements.json"
    
        if not bridging_path.exists():
            return None, None
        
        with open(bridging_path) as f:
            bridging_statements = json.load(f)
        
        all_personas = [s.get("persona", f"Voter {i}") for i, s in enumerate(bridging_statements)]
        stmt_dicts = [{"statement": s["statement"]} for s in bridging_statements]
        
        # For 'full' ablation, filter statements based on filter_assignments
        if ablation == "full":
            filter_path = rep_dir / "filter_assignments.json"
            if filter_path.exists():
                with open(filter_path) as f:
                    assignments = json.load(f)
                kept_indices = sorted([
                    a["statement_idx"]
                    for a in assignments
                    if a["keep"] == 1
                ])
                if len(stmt_dicts) > len(kept_indices):
                    stmt_dicts = [stmt_dicts[i] for i in kept_indices]
        
        return stmt_dicts, all_personas


def needs_backfill(results: dict) -> bool:
    """
    Check if a results dict needs backfill.
    
    Returns True if:
    - chatgpt_with_personas is missing, OR
    - chatgpt_with_personas exists but epsilon is null (buggy run)
    """
    if "chatgpt_with_personas" not in results:
        return True
    
    # Check if epsilon is null (from buggy run)
    chatgpt_result = results.get("chatgpt_with_personas", {})
    if chatgpt_result.get("epsilon") is None:
        return True
    
    return False


def process_sample_dir(
    sample_dir: Path,
    stmt_dicts: list,
    all_personas: list,
    openai_client: OpenAI,
    data_dir: Path
) -> tuple:
    """
    Process a single sample directory.
    
    Returns:
        (updated: bool, skipped: bool, error: bool)
    """
    results_path = sample_dir / "results.json"
    if not results_path.exists():
        return False, False, False
    
    with open(results_path) as f:
        results = json.load(f)
    
    # Check if needs backfill
    if not needs_backfill(results):
        return False, True, False  # skipped
    
    # Load sampled persona indices (try both naming conventions)
    indices_path = sample_dir / "sampled_persona_indices.json"
    if not indices_path.exists():
        indices_path = sample_dir / "persona_indices.json"
    if not indices_path.exists():
        # For 20-persona samples, all personas are used (indices 0-19)
        # Check if we have preferences to determine the number of voters
        prefs_path = sample_dir / "sampled_preferences.json"
        if not prefs_path.exists():
            prefs_path = sample_dir / "preferences.json"
        if prefs_path.exists():
            with open(prefs_path) as f:
                sampled_prefs = json.load(f)
            # Use indices based on number of voters in preferences
            sampled_indices = list(range(len(sampled_prefs)))
        else:
            logger.warning(f"  Missing indices and preferences: {sample_dir}")
            return False, False, True
    else:
        with open(indices_path) as f:
            sampled_indices = json.load(f)
    
    # Load sampled preferences for epsilon computation
    prefs_path = sample_dir / "sampled_preferences.json"
    if not prefs_path.exists():
        prefs_path = sample_dir / "preferences.json"
    if not prefs_path.exists():
        logger.warning(f"  Missing preferences: {sample_dir}")
        return False, False, True
    
    with open(prefs_path) as f:
        sampled_prefs = json.load(f)
    
    # Get personas for sampled voters
    try:
        sampled_personas = [all_personas[i] for i in sampled_indices]
    except IndexError:
        # If indices are out of range, use available personas
        sampled_personas = all_personas[:len(sampled_indices)]
    
    # Run chatgpt_with_personas
    try:
        rel_path = sample_dir.relative_to(data_dir)
        logger.info(f"  Running: {rel_path}")
        result = run_chatgpt_with_personas(stmt_dicts, sampled_personas, openai_client)
        
        # Compute epsilon
        winner = result.get("winner")
        if winner is not None:
            try:
                epsilon = compute_epsilon_for_winner(sampled_prefs, winner)
                result["epsilon"] = epsilon
            except Exception as e:
                logger.warning(f"    Epsilon computation failed: {e}")
                result["epsilon"] = None
        else:
            result["epsilon"] = None
        
        # Update results
        results["chatgpt_with_personas"] = result
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"    Winner={winner}, epsilon={result.get('epsilon')}")
        return True, False, False  # updated
        
    except Exception as e:
        logger.error(f"    Error: {e}")
        return False, False, True  # error


def backfill_chatgpt_with_personas(output_dir: Path = OUTPUT_DIR):
    """Backfill chatgpt_with_personas for all samples missing it or with null epsilon."""
    openai_client = OpenAI(timeout=60.0)
    data_dir = output_dir / "data"
    
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    logger.info("=" * 60)
    logger.info("Starting backfill of chatgpt_with_personas")
    logger.info("(includes 20-persona data and fixing null epsilon samples)")
    logger.info("=" * 60)
    
    for topic_dir in sorted(data_dir.iterdir()):
        if not topic_dir.is_dir() or topic_dir.name.startswith("pvc"):
            continue
        
        logger.info(f"\nProcessing topic: {topic_dir.name}")
        
        for rep_dir in sorted(topic_dir.glob("rep*")):
            # Process each ablation separately
            for ablation in ABLATIONS:
                # Load statements and personas for this ablation
                stmt_dicts, all_personas = load_statements_and_personas_for_ablation(
                    rep_dir, ablation, topic_dir.name
                )
                if stmt_dicts is None:
                    continue
                
                # Determine base directory for this ablation
                if ablation == "full":
                    ablation_base = rep_dir
                else:
                    ablation_base = rep_dir / f"ablation_{ablation}"
                
                if not ablation_base.exists():
                    continue
                
                # Process 20-persona samples (directly in ablation_base/sample*)
                for sample_dir in sorted(ablation_base.glob("sample*")):
                    if not sample_dir.is_dir():
                        continue
                    updated, skipped, error = process_sample_dir(
                        sample_dir, stmt_dicts, all_personas, openai_client, data_dir
                    )
                    if updated:
                        updated_count += 1
                    if skipped:
                        skipped_count += 1
                    if error:
                        error_count += 1
                
                # Process 5-persona and 10-persona subdirectories
                for n_personas_dir in ["5-personas", "10-personas"]:
                    personas_dir = ablation_base / n_personas_dir
                    
                    if not personas_dir.exists():
                        continue
                    
                    for sample_dir in sorted(personas_dir.glob("sample*")):
                        if not sample_dir.is_dir():
                            continue
                        updated, skipped, error = process_sample_dir(
                            sample_dir, stmt_dicts, all_personas, openai_client, data_dir
                        )
                        if updated:
                            updated_count += 1
                        if skipped:
                            skipped_count += 1
                        if error:
                            error_count += 1
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Backfill complete!")
    logger.info(f"  Updated: {updated_count}")
    logger.info(f"  Skipped (already had valid data): {skipped_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info("=" * 60)


def main():
    setup_logging(OUTPUT_DIR)
    logger.info(f"Start time: {datetime.now().isoformat()}")
    backfill_chatgpt_with_personas()
    logger.info(f"End time: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
