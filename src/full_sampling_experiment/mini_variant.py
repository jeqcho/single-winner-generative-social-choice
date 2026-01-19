"""
Mini variant experiment module for immigration topic.

Each original statement author rewrites their statement as a "bridging statement"
after seeing all 100 statements. The experiment then runs with these transformed
statements instead of the originals.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

from .config import (
    MODEL_VOTING,
    TEMPERATURE,
    MAX_WORKERS,
    TOPIC_QUESTIONS,
    api_timer,
)

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def generate_persona_bridging_statement(
    persona: str,
    all_statements: List[Dict],
    topic: str,
    openai_client: OpenAI,
    model: str = MODEL_VOTING,
    temperature: float = TEMPERATURE
) -> str:
    """
    Given a persona and all 100 statements, generate a bridging statement.
    
    The persona writes from their perspective but aims to bridge viewpoints
    after seeing the diversity of opinions.
    
    Args:
        persona: The persona string description
        all_statements: All 100 statement dicts with 'statement' key
        topic: The topic/question being discussed
        openai_client: OpenAI client instance
        model: Model to use
        temperature: Temperature for sampling
    
    Returns:
        Generated bridging statement (2-4 sentences)
    """
    # Build numbered statement list
    statements_text = "\n".join(
        f"{i}: {stmt['statement']}"
        for i, stmt in enumerate(all_statements)
    )
    
    system_prompt = "You are writing a bridging statement from your persona's perspective. Return ONLY the statement text, no JSON."
    
    user_prompt = f"""You are a person with the following characteristics:
{persona}

Topic: "{topic}"

Here are 100 statements from people with diverse perspectives on this topic:

{statements_text}

Now, write a NEW bridging statement that:
- Reflects your background and values
- Acknowledges the diverse perspectives you've seen
- Aims to find common ground or bridge different viewpoints
- Is 2-4 sentences long

Write only the statement, no additional commentary."""

    start_time = time.time()
    response = openai_client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        reasoning={"effort": "minimal"},
    )
    api_timer.record(time.time() - start_time)
    
    return response.output_text.strip()


def transform_statements_to_bridging(
    statements: List[Dict],
    topic: str,
    openai_client: OpenAI,
    model: str = MODEL_VOTING,
    max_workers: int = MAX_WORKERS
) -> List[Dict]:
    """
    Transform 100 original statements into 100 bridging statements.
    
    Each original persona rewrites their statement after seeing all 100.
    
    Args:
        statements: List of statement dicts, each with 'persona' and 'statement'
        topic: The topic/question being discussed
        openai_client: OpenAI client instance
        model: Model to use
        max_workers: Maximum parallel workers
    
    Returns:
        List of dicts with same structure but updated 'statement' field
        Also includes 'original_statement' for reference
    """
    n = len(statements)
    
    logger.info(f"Transforming {n} statements into bridging statements...")
    
    def process_statement(args):
        """Process a single statement transformation."""
        idx, stmt = args
        persona = stmt['persona']
        
        bridging = generate_persona_bridging_statement(
            persona, statements, topic, openai_client, model
        )
        
        return idx, bridging
    
    # Initialize results
    bridging_statements = [None] * n
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_statement, (i, stmt)): i
            for i, stmt in enumerate(statements)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="Generating bridging statements", unit="stmt"):
            try:
                idx, bridging = future.result()
                bridging_statements[idx] = bridging
            except Exception as e:
                logger.error(f"Failed to generate bridging for statement {futures[future]}: {e}")
                # Keep original statement if transformation fails
                bridging_statements[futures[future]] = statements[futures[future]]['statement']
    
    # Build result list
    result = []
    for i, stmt in enumerate(statements):
        result.append({
            'persona': stmt['persona'],
            'statement': bridging_statements[i],
            'original_statement': stmt['statement']
        })
    
    logger.info(f"Transformed {n} statements into bridging statements")
    
    return result


def save_bridging_statements(
    statements: List[Dict],
    output_path: Path
) -> None:
    """Save bridging statements to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(statements, f, indent=2)
    
    logger.info(f"Saved bridging statements to {output_path}")


def load_bridging_statements(input_path: Path) -> List[Dict]:
    """Load bridging statements from JSON file."""
    with open(input_path, 'r') as f:
        statements = json.load(f)
    
    logger.info(f"Loaded {len(statements)} bridging statements from {input_path}")
    
    return statements


def run_mini_variant_rep(
    rep_idx: int,
    original_statements: List[Dict],
    voter_personas: List[str],
    topic_slug: str,
    output_dir: Path,
    openai_client: OpenAI
) -> Dict:
    """
    Run a single replication of the mini variant experiment.
    
    Args:
        rep_idx: Replication index
        original_statements: Original 100 statements (with 'persona' and 'statement')
        voter_personas: 100 voter personas (independent from statement authors)
        topic_slug: Topic slug
        output_dir: Output directory for this rep
        openai_client: OpenAI client
    
    Returns:
        Dict with experiment results
    """
    from .config import (
        MODEL_RANKING,
        K_VALUE,
        P_VALUE,
        N_SAMPLES_PER_REP,
        BASE_SEED,
    )
    from src.sampling_experiment.data_loader import (
        sample_kp,
        extract_subprofile,
        check_cache_exists,
    )
    from src.sampling_experiment.preference_builder import build_full_preferences
    from src.sampling_experiment.epsilon_calculator import (
        precompute_all_epsilons,
        lookup_epsilon,
        save_precomputed_epsilons,
        load_precomputed_epsilons,
    )
    from .likert_experiment import (
        collect_likert_scores,
        save_likert_scores,
        load_likert_scores,
    )
    
    rep_dir = output_dir / f"rep{rep_idx}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    
    topic = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    seed = BASE_SEED + rep_idx
    
    logger.info(f"Mini variant rep {rep_idx}: {rep_dir}")
    
    # Step 1: Transform statements to bridging statements
    bridging_path = rep_dir / "bridging_statements.json"
    
    if bridging_path.exists():
        logger.info("  Loading cached bridging statements...")
        bridging_statements = load_bridging_statements(bridging_path)
    else:
        logger.info("  Generating bridging statements...")
        bridging_statements = transform_statements_to_bridging(
            original_statements, topic, openai_client
        )
        save_bridging_statements(bridging_statements, bridging_path)
    
    # Step 2: Build preference profile with bridging statements
    prefs_path = rep_dir / "full_preferences.json"
    
    if prefs_path.exists():
        logger.info("  Loading cached preferences...")
        with open(prefs_path, 'r') as f:
            full_preferences = json.load(f)
    else:
        logger.info("  Building preference profile...")
        full_preferences = build_full_preferences(
            voter_personas, bridging_statements, topic_slug, openai_client,
            model=MODEL_RANKING
        )
        with open(prefs_path, 'w') as f:
            json.dump(full_preferences, f, indent=2)
    
    # Step 3: Precompute epsilons
    eps_path = rep_dir / "precomputed_epsilons.json"
    
    if eps_path.exists():
        logger.info("  Loading cached epsilons...")
        precomputed_epsilons = load_precomputed_epsilons(rep_dir)
    else:
        logger.info("  Precomputing epsilons...")
        precomputed_epsilons = precompute_all_epsilons(full_preferences)
        save_precomputed_epsilons(precomputed_epsilons, rep_dir)
    
    # Step 4: Collect Likert scores (only rep 0)
    likert_scores = None
    if rep_idx == 0:
        likert_path = rep_dir / "likert_scores.json"
        
        if likert_path.exists():
            logger.info("  Loading cached Likert scores...")
            likert_scores = load_likert_scores(likert_path)
        else:
            logger.info("  Collecting Likert scores...")
            likert_scores = collect_likert_scores(
                voter_personas, bridging_statements, topic, openai_client
            )
            save_likert_scores(likert_scores, likert_path)
    else:
        logger.info("  Skipping Likert scores (only collected for rep 0)")
    
    # Step 5: Run samples
    logger.info(f"  Running {N_SAMPLES_PER_REP} samples...")
    
    sample_results = []
    
    for sample_idx in range(N_SAMPLES_PER_REP):
        sample_dir = rep_dir / f"sample{sample_idx}"
        results_path = sample_dir / "results.json"
        
        if results_path.exists():
            logger.info(f"    Sample {sample_idx}: cached, skipping")
            with open(results_path, 'r') as f:
                sample_results.append(json.load(f))
            continue
        
        logger.info(f"    Sample {sample_idx}: running...")
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample K voters and P alternatives
        sample_seed = seed * 1000 + K_VALUE * 100 + P_VALUE + sample_idx * 10000
        voter_sample, alt_sample = sample_kp(
            len(voter_personas), len(bridging_statements), 
            K_VALUE, P_VALUE, sample_seed
        )
        
        # Extract subprofile
        sample_prefs, alt_mapping = extract_subprofile(
            full_preferences, voter_sample, alt_sample
        )
        
        # Get sample statements and personas
        sample_statements = [bridging_statements[i] for i in alt_sample]
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
        
        # Import and run voting methods
        from src.sampling_experiment.voting_methods import (
            run_schulze, run_borda, run_irv, run_plurality, run_veto_by_consumption,
            run_chatgpt, run_chatgpt_with_rankings, run_chatgpt_with_personas,
            run_chatgpt_star, run_chatgpt_star_with_rankings, run_chatgpt_star_with_personas,
            run_chatgpt_double_star, run_chatgpt_double_star_with_rankings,
            run_chatgpt_double_star_with_personas, insert_new_statement_into_rankings,
            run_chatgpt_triple_star,
        )
        from src.sampling_experiment.epsilon_calculator import compute_epsilon_for_new_statement
        from .config import MODEL_VOTING
        
        results = {}
        
        # Traditional methods
        for method_name, method_fn in [
            ("schulze", run_schulze),
            ("borda", run_borda),
            ("irv", run_irv),
            ("plurality", run_plurality),
            ("veto_by_consumption", run_veto_by_consumption),
        ]:
            result = method_fn(sample_prefs)
            if result["winner"] is not None:
                full_winner = str(alt_mapping[int(result["winner"])])
                result["winner_full"] = full_winner
                result["epsilon"] = lookup_epsilon(precomputed_epsilons, full_winner)
            results[method_name] = result
        
        # ChatGPT methods (select from P)
        for method_name, method_fn, args in [
            ("chatgpt", run_chatgpt, (sample_statements, openai_client, MODEL_VOTING)),
            ("chatgpt_rankings", run_chatgpt_with_rankings, 
             (sample_statements, sample_prefs, openai_client, MODEL_VOTING)),
            ("chatgpt_personas", run_chatgpt_with_personas,
             (sample_statements, sample_personas, openai_client, MODEL_VOTING)),
        ]:
            result = method_fn(*args)
            if result["winner"] is not None:
                full_winner = str(alt_mapping[int(result["winner"])])
                result["winner_full"] = full_winner
                result["epsilon"] = lookup_epsilon(precomputed_epsilons, full_winner)
            results[method_name] = result
        
        # ChatGPT* methods (select from all 100)
        for method_name, method_fn, args in [
            ("chatgpt_star", run_chatgpt_star,
             (bridging_statements, sample_statements, openai_client, MODEL_VOTING)),
            ("chatgpt_star_rankings", run_chatgpt_star_with_rankings,
             (bridging_statements, sample_statements, sample_prefs, openai_client, MODEL_VOTING)),
            ("chatgpt_star_personas", run_chatgpt_star_with_personas,
             (bridging_statements, sample_personas, openai_client, MODEL_VOTING)),
        ]:
            result = method_fn(*args)
            if result["winner"] is not None:
                result["epsilon"] = lookup_epsilon(precomputed_epsilons, result["winner"])
            results[method_name] = result
        
        # ChatGPT** methods (generate new)
        for method_name, method_fn, args in [
            ("chatgpt_double_star", run_chatgpt_double_star,
             (sample_statements, bridging_statements, sample_personas,
              full_preferences, voter_sample, topic, openai_client, MODEL_VOTING)),
            ("chatgpt_double_star_rankings", run_chatgpt_double_star_with_rankings,
             (sample_statements, sample_prefs, bridging_statements, sample_personas,
              full_preferences, voter_sample, topic, openai_client, MODEL_VOTING)),
            ("chatgpt_double_star_personas", run_chatgpt_double_star_with_personas,
             (sample_statements, sample_personas, bridging_statements,
              full_preferences, voter_sample, topic, openai_client, MODEL_VOTING)),
        ]:
            result = method_fn(*args)
            if result.get("new_statement"):
                selected_personas = [voter_personas[i] for i in voter_sample]
                updated_prefs = insert_new_statement_into_rankings(
                    result["new_statement"], bridging_statements, selected_personas,
                    voter_sample, full_preferences, topic, openai_client
                )
                epsilon = compute_epsilon_for_new_statement(updated_prefs, len(bridging_statements))
                result["epsilon"] = epsilon
            results[method_name] = result
        
        # ChatGPT*** (blind bridging)
        result = run_chatgpt_triple_star(
            topic, bridging_statements, voter_personas, full_preferences,
            openai_client, model=MODEL_VOTING
        )
        results["chatgpt_triple_star"] = result
        
        # Save results
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        sample_results.append(results)
    
    return {
        "rep_idx": rep_idx,
        "n_samples": len(sample_results),
        "likert_shape": list(likert_scores.shape) if hasattr(likert_scores, 'shape') else None,
    }
