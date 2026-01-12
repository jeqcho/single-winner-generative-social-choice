"""
Re-run GPT voting methods (chatgpt, chatgpt_with_rankings, chatgpt_with_personas)
using gpt-5.2 with temperature=1 for all persona counts and ablations.

This script overwrites existing GPT voting results while preserving
non-GPT methods (schulze, borda, irv, plurality, veto_by_consumption).

Usage:
    uv run python -m src.full_experiment.rerun_gpt_voting
"""

import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

from .config import OUTPUT_DIR, ABLATIONS, ALL_TOPICS
from .data_loader import load_all_statements
from .voting_runner import compute_epsilon_for_winner

logger = logging.getLogger(__name__)

# Model configuration for this re-run
GPT_MODEL = "gpt-5.2"
GPT_TEMPERATURE = 1


def setup_logging(output_dir: Path) -> None:
    """Set up logging to file and console with timestamps."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"rerun_gpt_voting_{timestamp}.log"
    
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


# =============================================================================
# GPT Voting Methods (with explicit model parameter)
# =============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def run_chatgpt(
    statements: List[Dict],
    openai_client: OpenAI,
    model: str = GPT_MODEL,
    temperature: float = GPT_TEMPERATURE
) -> Dict:
    """Run ChatGPT baseline selection."""
    statements_text = "\n\n".join([
        f"Statement {i}: {stmt['statement']}"
        for i, stmt in enumerate(statements)
    ])
    
    n = len(statements)
    
    system_prompt = "You are a helpful assistant that selects consensus statements. Return ONLY valid JSON, no other text."
    
    user_prompt = f"""Here are {n} statements from a discussion:

{statements_text}

Which statement do you think would be the best choice as a consensus/bridging statement? 
Consider which one best represents a reasonable middle ground that could satisfy diverse perspectives.

Return your choice as a JSON object with this format:
{{"selected_statement_index": 0}}

Where the value is the index (0-{n-1}) of the statement you select.
Return only the JSON, no additional text."""

    try:
        start_time = time.time()
        response = openai_client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
        )
        duration = time.time() - start_time
        
        result = json.loads(response.output_text)
        winner = str(result.get("selected_statement_index"))
        
        return {"winner": winner}
    except Exception as e:
        logger.error(f"ChatGPT failed: {e}")
        return {"winner": None, "error": str(e)}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def run_chatgpt_with_rankings(
    statements: List[Dict],
    preferences: List[List[str]],
    openai_client: OpenAI,
    model: str = GPT_MODEL,
    temperature: float = GPT_TEMPERATURE
) -> Dict:
    """Run ChatGPT with preference rankings."""
    statements_text = "\n\n".join([
        f"Statement {i}: {stmt['statement']}"
        for i, stmt in enumerate(statements)
    ])
    
    n = len(statements)
    n_voters = len(preferences[0]) if preferences else 0
    
    # Format rankings (show first 10 voters)
    rankings_summary = []
    for voter in range(min(n_voters, 10)):
        ranking = [preferences[rank][voter] for rank in range(len(preferences))]
        rankings_summary.append(f"Voter {voter+1}: {' > '.join(ranking[:10])}...")
    
    if n_voters > 10:
        rankings_summary.append(f"... and {n_voters - 10} more voters")
    
    rankings_text = "\n".join(rankings_summary)
    
    system_prompt = "You are a helpful assistant that selects consensus statements. Return ONLY valid JSON, no other text."
    
    user_prompt = f"""Here are {n} statements from a discussion:

{statements_text}

Here are preference rankings from {n_voters} voters:

{rankings_text}

Based on both the statements and the preference rankings, which statement do you think would be the best choice as a consensus/bridging statement? 
Consider which one best represents a reasonable middle ground that could satisfy diverse perspectives, taking into account how the voters ranked them.

Return your choice as a JSON object with this format:
{{"selected_statement_index": 0}}

Where the value is the index (0-{n-1}) of the statement you select.
Return only the JSON, no additional text."""

    try:
        start_time = time.time()
        response = openai_client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
        )
        duration = time.time() - start_time
        
        result = json.loads(response.output_text)
        winner = str(result.get("selected_statement_index"))
        
        return {"winner": winner}
    except Exception as e:
        logger.error(f"ChatGPT with rankings failed: {e}")
        return {"winner": None, "error": str(e)}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def run_chatgpt_with_personas(
    statements: List[Dict],
    personas: List[str],
    openai_client: OpenAI,
    model: str = GPT_MODEL,
    temperature: float = GPT_TEMPERATURE
) -> Dict:
    """Run ChatGPT with persona descriptions of the voters."""
    statements_text = "\n\n".join([
        f"Statement {i}: {stmt['statement']}"
        for i, stmt in enumerate(statements)
    ])
    
    n = len(statements)
    n_voters = len(personas)
    
    # Format personas
    personas_text = "\n\n".join([
        f"Voter {i+1}: {persona}"
        for i, persona in enumerate(personas)
    ])
    
    system_prompt = "You are a helpful assistant that selects consensus statements. Return ONLY valid JSON, no other text."
    
    user_prompt = f"""Here are {n} statements from a discussion:

{statements_text}

Here are the {n_voters} voters who will be voting on these statements:

{personas_text}

Based on both the statements and the voter personas, which statement do you think would be the best choice as a consensus/bridging statement? 
Consider which one best represents a reasonable middle ground that could satisfy these diverse voters.

Return your choice as a JSON object with this format:
{{"selected_statement_index": 0}}

Where the value is the index (0-{n-1}) of the statement you select.
Return only the JSON, no additional text."""

    try:
        start_time = time.time()
        response = openai_client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
        )
        duration = time.time() - start_time
        
        result = json.loads(response.output_text)
        winner = str(result.get("selected_statement_index"))
        
        return {"winner": winner}
    except Exception as e:
        logger.error(f"ChatGPT with personas failed: {e}")
        return {"winner": None, "error": str(e)}


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_statements_and_personas_for_ablation(rep_dir: Path, ablation: str, topic_slug: str):
    """
    Load statements and personas for a specific ablation.
    
    Returns:
        (stmt_dicts, all_personas) tuple
    """
    if ablation == "no_bridging":
        # no_bridging uses original Polis statements
        indices_path = rep_dir / "sampled_indices.json"
        if not indices_path.exists():
            return None, None
        
        with open(indices_path) as f:
            sampled_indices = json.load(f)
        
        all_entries = load_all_statements(topic_slug)
        
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


def load_sample_data(sample_dir: Path) -> Tuple[Optional[List[List[str]]], Optional[List[int]]]:
    """
    Load preferences and persona indices for a sample.
    
    Returns:
        (preferences, persona_indices) tuple
    """
    # Load sampled preferences
    prefs_path = sample_dir / "sampled_preferences.json"
    if not prefs_path.exists():
        prefs_path = sample_dir / "preferences.json"
    if not prefs_path.exists():
        return None, None
    
    with open(prefs_path) as f:
        preferences = json.load(f)
    
    # Load sampled persona indices
    indices_path = sample_dir / "sampled_persona_indices.json"
    if not indices_path.exists():
        indices_path = sample_dir / "persona_indices.json"
    if not indices_path.exists():
        # For 20-persona samples, infer from preferences
        n_voters = len(preferences[0]) if preferences else 0
        persona_indices = list(range(n_voters))
    else:
        with open(indices_path) as f:
            persona_indices = json.load(f)
    
    return preferences, persona_indices


# =============================================================================
# Processing Functions
# =============================================================================

def process_sample_dir(
    sample_dir: Path,
    stmt_dicts: list,
    all_personas: list,
    openai_client: OpenAI,
    data_dir: Path
) -> Tuple[bool, bool]:
    """
    Process a single sample directory - re-run all GPT voting methods.
    
    Returns:
        (updated: bool, error: bool)
    """
    results_path = sample_dir / "results.json"
    if not results_path.exists():
        return False, False
    
    with open(results_path) as f:
        results = json.load(f)
    
    # Load sample data
    preferences, persona_indices = load_sample_data(sample_dir)
    if preferences is None:
        logger.warning(f"  Missing preferences: {sample_dir}")
        return False, True
    
    # Get personas for sampled voters
    try:
        sampled_personas = [all_personas[i] for i in persona_indices]
    except IndexError:
        sampled_personas = all_personas[:len(persona_indices)]
    
    rel_path = sample_dir.relative_to(data_dir)
    logger.info(f"  Processing: {rel_path}")
    
    try:
        # Run chatgpt
        logger.debug("    Running chatgpt...")
        chatgpt_result = run_chatgpt(stmt_dicts, openai_client)
        winner = chatgpt_result.get("winner")
        if winner is not None:
            try:
                epsilon = compute_epsilon_for_winner(preferences, winner)
                chatgpt_result["epsilon"] = epsilon
            except Exception as e:
                logger.warning(f"    Epsilon computation failed for chatgpt: {e}")
                chatgpt_result["epsilon"] = None
        else:
            chatgpt_result["epsilon"] = None
        results["chatgpt"] = chatgpt_result
        logger.info(f"    chatgpt: winner={winner}, epsilon={chatgpt_result.get('epsilon')}")
        
        # Run chatgpt_with_rankings
        logger.debug("    Running chatgpt_with_rankings...")
        rankings_result = run_chatgpt_with_rankings(stmt_dicts, preferences, openai_client)
        winner = rankings_result.get("winner")
        if winner is not None:
            try:
                epsilon = compute_epsilon_for_winner(preferences, winner)
                rankings_result["epsilon"] = epsilon
            except Exception as e:
                logger.warning(f"    Epsilon computation failed for chatgpt_with_rankings: {e}")
                rankings_result["epsilon"] = None
        else:
            rankings_result["epsilon"] = None
        results["chatgpt_with_rankings"] = rankings_result
        logger.info(f"    chatgpt_with_rankings: winner={winner}, epsilon={rankings_result.get('epsilon')}")
        
        # Run chatgpt_with_personas
        logger.debug("    Running chatgpt_with_personas...")
        personas_result = run_chatgpt_with_personas(stmt_dicts, sampled_personas, openai_client)
        winner = personas_result.get("winner")
        if winner is not None:
            try:
                epsilon = compute_epsilon_for_winner(preferences, winner)
                personas_result["epsilon"] = epsilon
            except Exception as e:
                logger.warning(f"    Epsilon computation failed for chatgpt_with_personas: {e}")
                personas_result["epsilon"] = None
        else:
            personas_result["epsilon"] = None
        results["chatgpt_with_personas"] = personas_result
        logger.info(f"    chatgpt_with_personas: winner={winner}, epsilon={personas_result.get('epsilon')}")
        
        # Save updated results
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return True, False  # updated
        
    except Exception as e:
        logger.error(f"    Error processing {rel_path}: {e}")
        return False, True  # error


def rerun_gpt_voting(output_dir: Path = OUTPUT_DIR):
    """Re-run GPT voting methods for all samples."""
    openai_client = OpenAI(timeout=60.0)
    data_dir = output_dir / "data"
    
    updated_count = 0
    error_count = 0
    total_samples = 0
    
    logger.info("=" * 60)
    logger.info(f"Re-running GPT voting methods with model={GPT_MODEL}, temperature={GPT_TEMPERATURE}")
    logger.info("=" * 60)
    
    for topic_dir in sorted(data_dir.iterdir()):
        if not topic_dir.is_dir() or topic_dir.name.startswith("pvc"):
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing topic: {topic_dir.name}")
        logger.info(f"{'='*60}")
        
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
                
                logger.info(f"\n  Ablation: {ablation} ({rep_dir.name})")
                
                # Process 20-persona samples (directly in ablation_base/sample*)
                for sample_dir in sorted(ablation_base.glob("sample*")):
                    if not sample_dir.is_dir():
                        continue
                    total_samples += 1
                    updated, error = process_sample_dir(
                        sample_dir, stmt_dicts, all_personas, openai_client, data_dir
                    )
                    if updated:
                        updated_count += 1
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
                        total_samples += 1
                        updated, error = process_sample_dir(
                            sample_dir, stmt_dicts, all_personas, openai_client, data_dir
                        )
                        if updated:
                            updated_count += 1
                        if error:
                            error_count += 1
    
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Re-run complete!")
    logger.info(f"  Total samples processed: {total_samples}")
    logger.info(f"  Updated: {updated_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info("=" * 60)


def main():
    setup_logging(OUTPUT_DIR)
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info(f"Using model: {GPT_MODEL}")
    logger.info(f"Using temperature: {GPT_TEMPERATURE}")
    rerun_gpt_voting()
    logger.info(f"End time: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
