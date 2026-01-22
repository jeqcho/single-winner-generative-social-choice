"""
CLI script to generate Alt2 and Alt3 statements for all reps.

These are per-rep generators that depend on sampling 100 context statements first.

Usage:
    uv run python -m src.sample_alt_voters.generate_per_rep_statements --all
    uv run python -m src.sample_alt_voters.generate_per_rep_statements --topic abortion --rep 0
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from openai import OpenAI

from .config import (
    PERSONAS_PATH,
    SAMPLED_STATEMENTS_DIR,
    SAMPLED_CONTEXT_DIR,
    TOPICS,
    TOPIC_SHORT_NAMES,
    N_ALTERNATIVES,
    N_REPS,
    BASE_SEED,
)
from .alternative_generators import persona_context, no_persona_context

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def load_personas() -> dict:
    """Load adult personas from file."""
    logger.info(f"Loading personas from {PERSONAS_PATH}")
    with open(PERSONAS_PATH, 'r') as f:
        personas_list = json.load(f)
    
    # Convert list to dict with string indices as keys
    personas = {str(i): p for i, p in enumerate(personas_list)}
    logger.info(f"Loaded {len(personas)} personas")
    return personas


def load_alt1_statements(topic_slug: str) -> dict:
    """Load pre-generated Alt1 statements."""
    short_name = TOPIC_SHORT_NAMES.get(topic_slug, topic_slug)
    path = SAMPLED_STATEMENTS_DIR / "persona_no_context" / f"{short_name}.json"
    
    logger.info(f"Loading Alt1 statements from {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    
    statements = data.get("statements", {})
    logger.info(f"Loaded {len(statements)} Alt1 statements")
    return statements


def sample_context(
    alt1_statements: dict,
    n: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """
    Sample N context statements from Alt1 pool.
    
    Returns:
        Tuple of (list of persona_ids, list of statement texts)
    """
    random.seed(seed)
    persona_ids = list(alt1_statements.keys())
    sampled_ids = random.sample(persona_ids, min(n, len(persona_ids)))
    sampled_statements = [alt1_statements[pid] for pid in sampled_ids]
    return sampled_ids, sampled_statements


def save_context_sample(
    persona_ids: list[str],
    topic_slug: str,
    rep_id: int,
) -> Path:
    """Save the sampled context indices."""
    short_name = TOPIC_SHORT_NAMES.get(topic_slug, topic_slug)
    output_dir = SAMPLED_CONTEXT_DIR / short_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rep{rep_id}.json"
    
    data = {
        "topic": topic_slug,
        "rep_id": rep_id,
        "context_persona_ids": persona_ids,
        "count": len(persona_ids),
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved context sample to {output_path}")
    return output_path


def generate_rep(
    topic_slug: str,
    rep_id: int,
    personas: dict,
    alt1_statements: dict,
    client: OpenAI,
) -> None:
    """Generate Alt2 and Alt3 statements for a single rep."""
    short_name = TOPIC_SHORT_NAMES.get(topic_slug, topic_slug)
    
    # Sample 100 context statements
    seed = BASE_SEED + rep_id * 1000 + hash(topic_slug) % 1000
    context_persona_ids, context_statements = sample_context(
        alt1_statements, N_ALTERNATIVES, seed
    )
    
    # Save context sample
    save_context_sample(context_persona_ids, topic_slug, rep_id)
    
    # Get the personas who wrote the context statements
    context_personas = {pid: personas[pid] for pid in context_persona_ids}
    
    logger.info(f"=== Rep {rep_id}: Generating Alt2 ({len(context_personas)} personas) ===")
    
    # Generate Alt2 statements
    alt2_output_path = SAMPLED_STATEMENTS_DIR / "persona_context" / short_name / f"rep{rep_id}.json"
    alt2_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    persona_context.generate_for_rep(
        personas=context_personas,
        context_statements=context_statements,
        topic_slug=topic_slug,
        rep_id=rep_id,
        client=client,
        output_path=alt2_output_path,
    )
    
    logger.info(f"=== Rep {rep_id}: Generating Alt3 (20 batches × 5 = 100 statements) ===")
    
    # Generate Alt3 statements
    alt3_output_path = SAMPLED_STATEMENTS_DIR / "no_persona_context" / short_name / f"rep{rep_id}.json"
    alt3_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    no_persona_context.generate_for_rep(
        context_statements=context_statements,
        topic_slug=topic_slug,
        rep_id=rep_id,
        client=client,
        n_statements=N_ALTERNATIVES,
        output_path=alt3_output_path,
    )
    
    logger.info(f"=== Rep {rep_id} complete ===")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Alt2 and Alt3 statements for all reps"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate for all topics and all reps"
    )
    parser.add_argument(
        "--topic",
        type=str,
        choices=["abortion", "electoral"],
        help="Topic to generate for (use short name)"
    )
    parser.add_argument(
        "--rep",
        type=int,
        help="Specific rep to generate (0-9)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and args.topic is None:
        parser.error("Must specify --all or --topic")
    
    # Map short names to full slugs
    topic_map = {v: k for k, v in TOPIC_SHORT_NAMES.items()}
    
    # Load data
    personas = load_personas()
    
    # Initialize OpenAI client
    client = OpenAI()
    
    if args.all:
        # Generate for all topics and all reps
        logger.info("Generating Alt2 and Alt3 for all topics and all reps")
        for topic_slug in TOPICS:
            alt1_statements = load_alt1_statements(topic_slug)
            for rep_id in range(N_REPS):
                generate_rep(topic_slug, rep_id, personas, alt1_statements, client)
    else:
        # Get full topic slug
        topic_slug = topic_map.get(args.topic, args.topic)
        alt1_statements = load_alt1_statements(topic_slug)
        
        if args.rep is not None:
            # Single rep
            generate_rep(topic_slug, args.rep, personas, alt1_statements, client)
        else:
            # All reps for this topic
            for rep_id in range(N_REPS):
                generate_rep(topic_slug, rep_id, personas, alt1_statements, client)
    
    logger.info("=== All generation complete ===")


if __name__ == "__main__":
    main()
