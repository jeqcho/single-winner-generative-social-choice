"""
CLI script to pre-generate Alt1 and Alt4 statements.

Usage:
    # Pre-generate Alt1 (all 815 personas × 2 topics)
    uv run python -m src.sample_alt_voters.generate_statements --alt1 --topic abortion
    uv run python -m src.sample_alt_voters.generate_statements --alt1 --topic electoral

    # Pre-generate Alt4 (815 statements × 2 topics)
    uv run python -m src.sample_alt_voters.generate_statements --alt4 --topic abortion --n 815
    uv run python -m src.sample_alt_voters.generate_statements --alt4 --topic electoral --n 815

    # Generate both Alt1 and Alt4 for all topics
    uv run python -m src.sample_alt_voters.generate_statements --all
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from openai import OpenAI

from .config import (
    PERSONAS_PATH,
    SAMPLED_STATEMENTS_DIR,
    TOPICS,
    TOPIC_SHORT_NAMES,
    N_GLOBAL_ALT1,
    N_GLOBAL_ALT4,
)
from .alternative_generators import persona_no_context, no_persona_no_context

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


def get_output_path(alt_type: str, topic_slug: str) -> Path:
    """Get the output path for generated statements."""
    short_name = TOPIC_SHORT_NAMES.get(topic_slug, topic_slug)
    return SAMPLED_STATEMENTS_DIR / alt_type / f"{short_name}.json"


def generate_alt1(topic_slug: str, client: OpenAI) -> None:
    """Generate Alt1 statements for all personas on a topic."""
    logger.info(f"=== Generating Alt1 for topic: {topic_slug} ===")
    
    personas = load_personas()
    output_path = get_output_path("persona_no_context", topic_slug)
    
    logger.info(f"Output path: {output_path}")
    logger.info(f"Generating for {len(personas)} personas")
    
    results = persona_no_context.generate_all_statements(
        personas=personas,
        topic_slug=topic_slug,
        client=client,
        output_path=output_path,
    )
    
    logger.info(f"Generated {len(results)} Alt1 statements")
    logger.info(f"Saved to {output_path}")


def generate_alt4(topic_slug: str, n: int, client: OpenAI) -> None:
    """Generate Alt4 statements for a topic."""
    logger.info(f"=== Generating Alt4 for topic: {topic_slug} ===")
    
    output_path = get_output_path("no_persona_no_context", topic_slug)
    
    logger.info(f"Output path: {output_path}")
    logger.info(f"Generating {n} statements ({(n + 4) // 5} API calls)")
    
    results = no_persona_no_context.generate_n_statements(
        n=n,
        topic_slug=topic_slug,
        client=client,
        output_path=output_path,
    )
    
    logger.info(f"Generated {len(results)} Alt4 statements")
    logger.info(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate Alt1 and Alt4 statements for sample-alt-voters experiment"
    )
    
    parser.add_argument(
        "--alt1",
        action="store_true",
        help="Generate Alt1 statements (persona, no context)"
    )
    parser.add_argument(
        "--alt4",
        action="store_true",
        help="Generate Alt4 statements (no persona, no context, verbalized)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate both Alt1 and Alt4 for all topics"
    )
    parser.add_argument(
        "--topic",
        type=str,
        choices=["abortion", "electoral"],
        help="Topic to generate for (use short name)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N_GLOBAL_ALT4,
        help=f"Number of Alt4 statements to generate (default: {N_GLOBAL_ALT4})"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.alt1 and not args.alt4 and not args.all:
        parser.error("Must specify --alt1, --alt4, or --all")
    
    if (args.alt1 or args.alt4) and not args.topic and not args.all:
        parser.error("Must specify --topic when using --alt1 or --alt4")
    
    # Map short names to full slugs
    topic_map = {v: k for k, v in TOPIC_SHORT_NAMES.items()}
    
    # Initialize OpenAI client
    client = OpenAI()
    
    if args.all:
        # Generate everything
        logger.info("Generating all Alt1 and Alt4 statements for all topics")
        for topic_slug in TOPICS:
            generate_alt1(topic_slug, client)
            generate_alt4(topic_slug, args.n, client)
    else:
        # Get full topic slug
        topic_slug = topic_map.get(args.topic, args.topic)
        
        if args.alt1:
            generate_alt1(topic_slug, client)
        
        if args.alt4:
            generate_alt4(topic_slug, args.n, client)
    
    logger.info("=== Generation complete ===")


if __name__ == "__main__":
    main()
