"""
Build preference profiles and Likert ratings using LLM-based ranking.

Uses the hybrid insertion sort from large_scale for efficient ranking.
"""

import json
import logging
from typing import List, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

import time

from .config import (
    MODEL,
    TEMPERATURE,
    MAX_WORKERS,
    TOPIC_QUESTIONS,
    api_timer,
)

# Import the hybrid insertion sort from large_scale
from src.large_scale.insertion_ranking import get_preference_matrix_hybrid

logger = logging.getLogger(__name__)


# =============================================================================
# Preference Ranking (using hybrid insertion sort)
# =============================================================================

def build_full_preferences(
    personas: List[str],
    statements: List[Dict],
    topic_slug: str,
    openai_client: OpenAI,
    max_workers: int = MAX_WORKERS
) -> List[List[str]]:
    """
    Build full preference matrix (100 personas x 100 statements).
    
    Uses the hybrid insertion sort algorithm which:
    - For sorted list < threshold (70): Uses single LLM call
    - For sorted list >= threshold: Uses binary search with pairwise comparisons
    
    Args:
        personas: List of persona strings
        statements: List of statement dicts
        topic_slug: Topic slug
        openai_client: OpenAI client
        max_workers: Max parallel workers
    
    Returns:
        Preference matrix where preferences[rank][voter] is the statement index
        at that rank for that voter (as string)
    """
    topic = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    n_personas = len(personas)
    n_statements = len(statements)
    
    logger.info(f"Building preferences: {n_personas} personas x {n_statements} statements")
    logger.info(f"Using hybrid insertion sort (model={MODEL}, temperature={TEMPERATURE})")
    
    # Use the hybrid insertion sort from large_scale
    preferences = get_preference_matrix_hybrid(
        personas=personas,
        statements=statements,
        topic=topic,
        openai_client=openai_client,
        threshold=70,  # Default threshold for switching to binary search
        max_workers=max_workers,
        model_name=MODEL,
        temperature=TEMPERATURE
    )
    
    logger.info(f"Built preference matrix: {len(preferences)} x {len(preferences[0])}")
    
    return preferences


# =============================================================================
# Likert Ratings
# =============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def _get_single_likert_rating(
    persona: str,
    statement: Dict,
    topic: str,
    openai_client: OpenAI
) -> int:
    """
    Get a single Likert rating from a persona for a statement.
    
    Returns:
        Rating from 1 to 5
    """
    system_prompt = "You are rating statements based on the given persona. Return ONLY valid JSON, no other text."
    
    user_prompt = f"""You are a person with the following characteristics:
{persona}

Given the topic: "{topic}"

Please rate how much you agree with the following statement on a scale of 1-5:

Statement: {statement['statement']}

Rating scale:
1 = Strongly disagree
2 = Disagree  
3 = Neutral / Neither agree nor disagree
4 = Agree
5 = Strongly agree

Consider:
- How well this statement aligns with your values and perspective
- Whether you would support or endorse this position
- How much this statement represents your views on the topic

Return your rating as a JSON object with this format:
{{"rating": 3}}

Where the rating is an integer from 1 to 5.
Return only the JSON, no additional text."""

    start_time = time.time()
    response = openai_client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=TEMPERATURE,
        reasoning={"effort": "minimal"}
    )
    api_timer.record(time.time() - start_time)
    
    result = json.loads(response.output_text)
    rating = result.get("rating", 3)
    
    # Ensure valid range
    rating = max(1, min(5, int(rating)))
    
    return rating


def build_full_likert(
    personas: List[str],
    statements: List[Dict],
    topic_slug: str,
    openai_client: OpenAI,
    max_workers: int = MAX_WORKERS
) -> List[List[int]]:
    """
    Build full Likert rating matrix (100 personas x 100 statements).
    
    Args:
        personas: List of persona strings
        statements: List of statement dicts
        topic_slug: Topic slug
        openai_client: OpenAI client
        max_workers: Max parallel workers
    
    Returns:
        Rating matrix where ratings[persona_idx][statement_idx] is the Likert rating
    """
    topic = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    n_personas = len(personas)
    n_statements = len(statements)
    total_ratings = n_personas * n_statements
    
    logger.info(f"Building Likert ratings: {n_personas} personas x {n_statements} statements = {total_ratings} ratings")
    
    # Initialize ratings matrix
    ratings = [[None for _ in range(n_statements)] for _ in range(n_personas)]
    
    # Create all tasks
    tasks = [
        (p_idx, s_idx, personas[p_idx], statements[s_idx])
        for p_idx in range(n_personas)
        for s_idx in range(n_statements)
    ]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_coords = {}
        for p_idx, s_idx, persona, statement in tasks:
            future = executor.submit(
                _get_single_likert_rating,
                persona, statement, topic, openai_client
            )
            future_to_coords[future] = (p_idx, s_idx)
        
        with tqdm(total=total_ratings, desc="Getting Likert ratings", unit="rating") as pbar:
            for future in as_completed(future_to_coords):
                p_idx, s_idx = future_to_coords[future]
                try:
                    rating = future.result()
                    ratings[p_idx][s_idx] = rating
                except Exception as e:
                    logger.error(f"Failed rating for persona {p_idx}, statement {s_idx}: {e}")
                    ratings[p_idx][s_idx] = 3  # Default to neutral
                pbar.update(1)
    
    logger.info(f"Built Likert rating matrix: {n_personas} x {n_statements}")
    
    return ratings


# =============================================================================
# Save/Load Functions
# =============================================================================

def save_preferences(preferences: List[List[str]], output_dir: Path) -> None:
    """Save preference matrix to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "full_preferences.json", 'w') as f:
        json.dump(preferences, f, indent=2)
    logger.info(f"Saved preferences to {output_dir / 'full_preferences.json'}")


def load_preferences(output_dir: Path) -> List[List[str]]:
    """Load preference matrix from JSON."""
    with open(output_dir / "full_preferences.json", 'r') as f:
        return json.load(f)


def save_likert(ratings: List[List[int]], output_dir: Path) -> None:
    """Save Likert ratings to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "full_likert.json", 'w') as f:
        json.dump(ratings, f, indent=2)
    logger.info(f"Saved Likert ratings to {output_dir / 'full_likert.json'}")


def load_likert(output_dir: Path) -> List[List[int]]:
    """Load Likert ratings from JSON."""
    with open(output_dir / "full_likert.json", 'r') as f:
        return json.load(f)

