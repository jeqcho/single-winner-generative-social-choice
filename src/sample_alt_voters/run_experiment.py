"""
Main entry point for Phase 2 of the sample-alt-voters experiment.

Runs preference ranking, voting method evaluation, and epsilon computation
across the factorial design:
- 4 Alternative Distributions: Alt1-4
- 2 Voter Distributions: Uniform (10 reps), Clustered (2 reps)
- 2 Topics: abortion, electoral
- 5 Mini-reps per rep (20 voters × 20 statements)

Usage:
    # Run all conditions for uniform voter distribution
    uv run python -m src.sample_alt_voters.run_experiment --voter-dist uniform --all-topics --all-alts
    
    # Run specific condition
    uv run python -m src.sample_alt_voters.run_experiment --voter-dist uniform --topic abortion --alt-dist persona_no_context --rep 0
    
    # Run clustered voter distribution
    uv run python -m src.sample_alt_voters.run_experiment --voter-dist clustered --all-topics --all-alts
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from .config import (
    PERSONAS_PATH,
    SAMPLED_STATEMENTS_DIR,
    SAMPLED_CONTEXT_DIR,
    PHASE2_DATA_DIR,
    TOPICS,
    TOPIC_QUESTIONS,
    TOPIC_SHORT_NAMES,
    ALT_DISTRIBUTIONS,
    N_ALTERNATIVES,
    N_VOTERS,
    K_SAMPLE,
    P_SAMPLE,
    N_SAMPLES_PER_REP,
    N_REPS_UNIFORM,
    N_REPS_CLUSTERED,
    IDEOLOGY_CLUSTERS,
    REASONING_EFFORT,
    BASE_SEED,
)
from .voter_samplers import sample_uniform, sample_from_cluster
from .preference_builder_iterative import (
    build_full_preferences_iterative,
    save_preferences,
    load_preferences,
    subsample_preferences,
)
from src.sampling_experiment.epsilon_calculator import (
    precompute_all_epsilons,
    lookup_epsilon,
    save_precomputed_epsilons,
    load_precomputed_epsilons,
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
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


# =============================================================================
# Data Loading
# =============================================================================

def load_personas() -> List[str]:
    """Load all adult personas."""
    with open(PERSONAS_PATH) as f:
        return json.load(f)


def load_statements_for_rep(
    topic_slug: str,
    alt_dist: str,
    rep_id: int
) -> List[Dict]:
    """
    Load statements for a specific (topic, alt_dist, rep_id) combination.
    
    For pre-generated distributions (Alt1, Alt4), samples 100 from the pool.
    For per-rep distributions (Alt2, Alt3), loads the pre-generated per-rep file.
    
    Returns:
        List of statement dicts with 'statement' key and 'id' key
    """
    topic_short = TOPIC_SHORT_NAMES.get(topic_slug, topic_slug)
    
    if alt_dist in ["persona_no_context", "no_persona_no_context"]:
        # Pre-generated pool - sample 100 using context indices
        pool_path = SAMPLED_STATEMENTS_DIR / alt_dist / f"{topic_short}.json"
        with open(pool_path) as f:
            pool_data = json.load(f)
        
        # Load context to get which 100 to use
        context_path = SAMPLED_CONTEXT_DIR / topic_short / f"rep{rep_id}.json"
        with open(context_path) as f:
            context_data = json.load(f)
        
        if alt_dist == "persona_no_context":
            # Use the context persona IDs
            context_ids = context_data["context_persona_ids"]
            statements = []
            for pid in context_ids:
                if pid in pool_data["statements"]:
                    statements.append({
                        "id": pid,
                        "statement": pool_data["statements"][pid]
                    })
        else:
            # Alt4: sample 100 from the list
            # The no_persona_no_context format has statements as a list
            pool_statements = pool_data.get("statements", [])
            if isinstance(pool_statements, dict):
                # Some formats have dict
                all_stmts = list(pool_statements.values())
            else:
                all_stmts = pool_statements
            
            rng = random.Random(BASE_SEED + rep_id)
            sampled = rng.sample(all_stmts, min(N_ALTERNATIVES, len(all_stmts)))
            statements = [{"id": str(i), "statement": s} for i, s in enumerate(sampled)]
    
    else:
        # Per-rep generated (Alt2, Alt3)
        rep_path = SAMPLED_STATEMENTS_DIR / alt_dist / topic_short / f"rep{rep_id}.json"
        with open(rep_path) as f:
            rep_data = json.load(f)
        
        stmts = rep_data.get("statements", [])
        if isinstance(stmts, dict):
            statements = [{"id": k, "statement": v} for k, v in stmts.items()]
        else:
            statements = [{"id": str(i), "statement": s} for i, s in enumerate(stmts)]
    
    return statements[:N_ALTERNATIVES]


# =============================================================================
# Voting Methods
# =============================================================================

TRADITIONAL_METHODS = {
    "schulze": run_schulze,
    "borda": run_borda,
    "irv": run_irv,
    "plurality": run_plurality,
    "veto_by_consumption": run_veto_by_consumption,
}


def run_traditional_voting_methods(
    preferences: List[List[str]]
) -> Dict[str, Dict]:
    """Run all traditional voting methods on a preference profile."""
    results = {}
    for name, method in TRADITIONAL_METHODS.items():
        try:
            result = method(preferences)
            results[name] = result
        except Exception as e:
            logger.error(f"Error running {name}: {e}")
            results[name] = {"winner": None, "error": str(e)}
    return results


def run_chatgpt_voting_methods(
    statements: List[Dict],
    preferences: List[List[str]],
    voter_personas: List[str],
    openai_client: OpenAI
) -> Dict[str, Dict]:
    """Run ChatGPT-based voting methods."""
    results = {}
    
    # Base ChatGPT
    try:
        results["chatgpt"] = run_chatgpt(statements, openai_client)
    except Exception as e:
        logger.error(f"Error running chatgpt: {e}")
        results["chatgpt"] = {"winner": None, "error": str(e)}
    
    # ChatGPT with rankings
    try:
        results["chatgpt_rankings"] = run_chatgpt_with_rankings(
            statements, preferences, openai_client
        )
    except Exception as e:
        logger.error(f"Error running chatgpt_rankings: {e}")
        results["chatgpt_rankings"] = {"winner": None, "error": str(e)}
    
    # ChatGPT with personas
    try:
        results["chatgpt_personas"] = run_chatgpt_with_personas(
            statements, voter_personas, openai_client
        )
    except Exception as e:
        logger.error(f"Error running chatgpt_personas: {e}")
        results["chatgpt_personas"] = {"winner": None, "error": str(e)}
    
    return results


# =============================================================================
# Mini-Rep Evaluation
# =============================================================================

def run_mini_rep(
    full_preferences: List[List[str]],
    full_epsilons: Dict[str, float],
    statements: List[Dict],
    voter_personas: List[str],
    mini_rep_id: int,
    openai_client: OpenAI,
    run_chatgpt_methods: bool = True
) -> Dict:
    """
    Run voting evaluation on a mini-rep (20×20 subsample).
    
    Args:
        full_preferences: Full 100×100 preference matrix
        full_epsilons: Precomputed epsilons for all 100 alternatives
        statements: All 100 statements
        voter_personas: All 100 voter persona strings
        mini_rep_id: Index of this mini-rep (0-4)
        openai_client: OpenAI client
        run_chatgpt_methods: Whether to run ChatGPT-based methods
        
    Returns:
        Dict with results for all voting methods
    """
    # Subsample to 20×20
    seed = BASE_SEED + mini_rep_id * 100
    sample_prefs, voter_indices, alt_indices = subsample_preferences(
        full_preferences,
        k_voters=K_SAMPLE,
        p_alts=P_SAMPLE,
        seed=seed
    )
    
    # Get sampled statements and personas
    sample_statements = [statements[i] for i in alt_indices]
    sample_personas = [voter_personas[i] for i in voter_indices]
    
    # Run traditional voting methods
    results = run_traditional_voting_methods(sample_prefs)
    
    # Run ChatGPT methods if requested
    if run_chatgpt_methods:
        chatgpt_results = run_chatgpt_voting_methods(
            sample_statements, sample_prefs, sample_personas, openai_client
        )
        results.update(chatgpt_results)
    
    # Look up epsilon for each winner
    # Map back from sample index to full index
    alt_mapping = {str(i): str(alt_indices[i]) for i in range(len(alt_indices))}
    
    for method_name, result in results.items():
        winner = result.get("winner")
        if winner is not None and winner in alt_mapping:
            full_winner = alt_mapping[winner]
            epsilon = lookup_epsilon(full_epsilons, full_winner)
            result["epsilon"] = epsilon
            result["full_winner_idx"] = full_winner
    
    return {
        "mini_rep_id": mini_rep_id,
        "voter_indices": voter_indices,
        "alt_indices": alt_indices,
        "results": results
    }


# =============================================================================
# Main Experiment Runner
# =============================================================================

def run_single_condition(
    topic_slug: str,
    alt_dist: str,
    voter_dist: str,
    rep_id: int,
    personas: List[str],
    openai_client: OpenAI,
    skip_if_exists: bool = True,
    run_chatgpt_methods: bool = True
) -> Optional[Dict]:
    """
    Run experiment for a single (topic, alt_dist, voter_dist, rep_id) condition.
    
    Returns:
        Dict with all results, or None if skipped
    """
    topic_short = TOPIC_SHORT_NAMES.get(topic_slug, topic_slug)
    
    # Determine output directory
    if voter_dist == "uniform":
        output_dir = PHASE2_DATA_DIR / topic_short / "uniform" / alt_dist / f"rep{rep_id}"
    else:
        cluster_name = IDEOLOGY_CLUSTERS[rep_id] if rep_id < len(IDEOLOGY_CLUSTERS) else voter_dist
        output_dir = PHASE2_DATA_DIR / topic_short / "clustered" / alt_dist / f"rep{rep_id}_{cluster_name}"
    
    # Check if already exists
    if skip_if_exists and (output_dir / "preferences.json").exists():
        logger.info(f"Skipping {output_dir} (already exists)")
        return None
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {topic_short}/{voter_dist}/{alt_dist}/rep{rep_id}")
    logger.info(f"{'='*60}")
    
    # Load statements
    logger.info("Loading statements...")
    statements = load_statements_for_rep(topic_slug, alt_dist, rep_id)
    logger.info(f"Loaded {len(statements)} statements")
    
    # Sample voters
    logger.info(f"Sampling {N_VOTERS} voters ({voter_dist})...")
    if voter_dist == "uniform":
        voter_indices, voter_personas = sample_uniform(
            personas, n_voters=N_VOTERS, seed=BASE_SEED + rep_id
        )
    else:
        cluster_name = IDEOLOGY_CLUSTERS[rep_id] if rep_id < len(IDEOLOGY_CLUSTERS) else voter_dist
        voter_indices, voter_personas = sample_from_cluster(
            personas, cluster_name, n_voters=N_VOTERS, seed=BASE_SEED + rep_id
        )
    logger.info(f"Sampled voters: {voter_indices[:5]}...")
    
    # Save voter sample
    with open(output_dir / "voters.json", 'w') as f:
        json.dump({
            "voter_dist": voter_dist,
            "voter_indices": voter_indices,
            "n_voters": len(voter_indices)
        }, f, indent=2)
    
    # Build preference matrix
    logger.info("Building preference matrix (A*-low)...")
    start_time = time.time()
    topic_question = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    
    preferences, pref_stats = build_full_preferences_iterative(
        voter_personas=voter_personas,
        statements=statements,
        topic=topic_question,
        openai_client=openai_client,
        reasoning_effort=REASONING_EFFORT,
        max_workers=50,
        show_progress=True
    )
    
    pref_time = time.time() - start_time
    logger.info(f"Built preferences in {pref_time:.1f}s")
    
    # Save preferences
    save_preferences(preferences, pref_stats, output_dir)
    
    # Precompute epsilons
    logger.info("Precomputing epsilons for all alternatives...")
    epsilons = precompute_all_epsilons(preferences)
    save_precomputed_epsilons(epsilons, output_dir)
    
    # Run mini-reps
    logger.info(f"Running {N_SAMPLES_PER_REP} mini-reps...")
    mini_rep_results = []
    
    for mini_rep_id in range(N_SAMPLES_PER_REP):
        logger.info(f"  Mini-rep {mini_rep_id}...")
        result = run_mini_rep(
            full_preferences=preferences,
            full_epsilons=epsilons,
            statements=statements,
            voter_personas=voter_personas,
            mini_rep_id=mini_rep_id,
            openai_client=openai_client,
            run_chatgpt_methods=run_chatgpt_methods
        )
        mini_rep_results.append(result)
        
        # Save mini-rep result
        mini_rep_dir = output_dir / f"mini_rep{mini_rep_id}"
        mini_rep_dir.mkdir(exist_ok=True)
        with open(mini_rep_dir / "results.json", 'w') as f:
            json.dump(result, f, indent=2)
    
    # Compile summary
    summary = {
        "topic": topic_slug,
        "topic_short": topic_short,
        "alt_dist": alt_dist,
        "voter_dist": voter_dist,
        "rep_id": rep_id,
        "n_statements": len(statements),
        "n_voters": len(voter_personas),
        "preference_stats": pref_stats,
        "preference_build_time": pref_time,
        "n_mini_reps": len(mini_rep_results),
        "timestamp": datetime.now().isoformat()
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Completed {topic_short}/{voter_dist}/{alt_dist}/rep{rep_id}")
    
    return summary


def run_all_conditions(
    voter_dist: str,
    topics: Optional[List[str]] = None,
    alt_dists: Optional[List[str]] = None,
    reps: Optional[List[int]] = None,
    openai_client: OpenAI = None,
    skip_if_exists: bool = True,
    run_chatgpt_methods: bool = True
):
    """
    Run all conditions for a voter distribution.
    
    Args:
        voter_dist: "uniform" or "clustered"
        topics: List of topic slugs (default: all)
        alt_dists: List of alt distributions (default: all)
        reps: List of rep IDs (default: all for the voter_dist)
        openai_client: OpenAI client
        skip_if_exists: Skip conditions that already have results
        run_chatgpt_methods: Whether to run ChatGPT-based methods
    """
    if topics is None:
        topics = TOPICS
    if alt_dists is None:
        alt_dists = ALT_DISTRIBUTIONS
    if reps is None:
        if voter_dist == "uniform":
            reps = list(range(N_REPS_UNIFORM))
        else:
            reps = list(range(N_REPS_CLUSTERED))
    
    # Load personas once
    logger.info("Loading personas...")
    personas = load_personas()
    logger.info(f"Loaded {len(personas)} personas")
    
    # Count conditions
    n_conditions = len(topics) * len(alt_dists) * len(reps)
    logger.info(f"Running {n_conditions} conditions for voter_dist={voter_dist}")
    
    completed = 0
    skipped = 0
    
    for topic in topics:
        for alt_dist in alt_dists:
            for rep_id in reps:
                result = run_single_condition(
                    topic_slug=topic,
                    alt_dist=alt_dist,
                    voter_dist=voter_dist,
                    rep_id=rep_id,
                    personas=personas,
                    openai_client=openai_client,
                    skip_if_exists=skip_if_exists,
                    run_chatgpt_methods=run_chatgpt_methods
                )
                
                if result is None:
                    skipped += 1
                else:
                    completed += 1
                
                logger.info(f"Progress: {completed + skipped}/{n_conditions} "
                          f"(completed={completed}, skipped={skipped})")
    
    logger.info(f"\nDone! Completed={completed}, Skipped={skipped}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 2 of the sample-alt-voters experiment"
    )
    
    parser.add_argument(
        "--voter-dist",
        choices=["uniform", "clustered"],
        required=True,
        help="Voter distribution to use"
    )
    parser.add_argument(
        "--topic",
        choices=["abortion", "electoral"],
        help="Specific topic to run (default: all)"
    )
    parser.add_argument(
        "--all-topics",
        action="store_true",
        help="Run all topics"
    )
    parser.add_argument(
        "--alt-dist",
        choices=ALT_DISTRIBUTIONS,
        help="Specific alternative distribution (default: all)"
    )
    parser.add_argument(
        "--all-alts",
        action="store_true",
        help="Run all alternative distributions"
    )
    parser.add_argument(
        "--rep",
        type=int,
        help="Specific replication ID (default: all)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip conditions that already have results"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if results exist"
    )
    parser.add_argument(
        "--no-chatgpt",
        action="store_true",
        help="Skip ChatGPT-based voting methods"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Determine topics
    topic_slug_map = {
        "abortion": "what-should-guide-laws-concerning-abortion",
        "electoral": "what-reforms-if-any-should-replace-or-modify-the-e"
    }
    
    if args.all_topics:
        topics = TOPICS
    elif args.topic:
        topics = [topic_slug_map[args.topic]]
    else:
        parser.error("Must specify --topic or --all-topics")
    
    # Determine alt_dists
    if args.all_alts:
        alt_dists = ALT_DISTRIBUTIONS
    elif args.alt_dist:
        alt_dists = [args.alt_dist]
    else:
        parser.error("Must specify --alt-dist or --all-alts")
    
    # Determine reps
    if args.rep is not None:
        reps = [args.rep]
    else:
        reps = None  # All reps for the voter_dist
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set in environment")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    # Run
    skip_existing = not args.force
    run_chatgpt_methods = not args.no_chatgpt
    
    run_all_conditions(
        voter_dist=args.voter_dist,
        topics=topics,
        alt_dists=alt_dists,
        reps=reps,
        openai_client=client,
        skip_if_exists=skip_existing,
        run_chatgpt_methods=run_chatgpt_methods
    )


if __name__ == "__main__":
    main()
