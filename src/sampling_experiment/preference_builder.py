"""
Build preference profiles using single-call sorting.
"""

import json
import logging
from typing import List, Dict
from pathlib import Path
from openai import OpenAI

from .config import (
    MODEL,
    TEMPERATURE,
    MAX_WORKERS,
    TOPIC_QUESTIONS,
)
from .single_call_ranking import get_preference_matrix_single_call

logger = logging.getLogger(__name__)


def build_full_preferences(
    voter_personas: List[str],
    alt_statements: List[Dict],
    topic_slug: str,
    openai_client: OpenAI,
    max_workers: int = MAX_WORKERS,
    model: str = None,
    temperature: float = None
) -> List[List[str]]:
    """
    Build full preference matrix (100 voters x 100 alternatives).
    
    Uses single-call sorting where each voter ranks all alternatives
    in a single API call.
    
    Args:
        voter_personas: List of voter persona strings
        alt_statements: List of statement dicts
        topic_slug: Topic slug
        openai_client: OpenAI client
        max_workers: Max parallel workers
        model: Model to use (defaults to config.MODEL)
        temperature: Temperature for sampling (defaults to config.TEMPERATURE)
    
    Returns:
        Preference matrix where preferences[rank][voter] is the alternative
        index (as string) at that rank for that voter
    """
    # Use defaults from config if not specified
    if model is None:
        model = MODEL
    if temperature is None:
        temperature = TEMPERATURE
    
    topic = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    n_voters = len(voter_personas)
    n_alts = len(alt_statements)
    
    logger.info(f"Building full preferences: {n_voters} voters x {n_alts} alternatives")
    logger.info(f"Topic: {topic}")
    logger.info(f"Using single-call ranking with model={model}")
    
    # Build preference matrix using single-call ranking
    preferences = get_preference_matrix_single_call(
        personas=voter_personas,
        statements=alt_statements,
        topic=topic,
        openai_client=openai_client,
        max_workers=max_workers,
        model=model,
        temperature=temperature
    )
    
    logger.info(f"Built preference matrix: {len(preferences)} x {len(preferences[0])}")
    
    return preferences


def save_full_preferences(preferences: List[List[str]], output_dir: Path) -> None:
    """Save full preference matrix to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "full_preferences.json", 'w') as f:
        json.dump(preferences, f, indent=2)
    
    logger.info(f"Saved full preferences to {output_dir}")


def load_full_preferences(output_dir: Path) -> List[List[str]]:
    """Load full preference matrix from JSON."""
    with open(output_dir / "full_preferences.json", 'r') as f:
        return json.load(f)
