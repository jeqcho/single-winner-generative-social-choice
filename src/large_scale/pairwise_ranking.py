"""
Implement pairwise comparison-based ranking using Python's built-in sort.
"""

import json
import time
from functools import cmp_to_key
from typing import List, Dict, Callable
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import api_timer for timing tracking
from src.full_experiment.config import api_timer


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def pairwise_compare(
    persona: str,
    statement_a: Dict,
    statement_b: Dict,
    topic: str,
    openai_client: OpenAI,
    model_name: str = "gpt-5-nano",
    temperature: float = 1.0
) -> int:
    """
    Ask persona to compare two statements and return preference.
    
    Args:
        persona: Persona string description
        statement_a: First statement dict with 'statement' key
        statement_b: Second statement dict with 'statement' key
        topic: The topic/question being discussed
        openai_client: OpenAI client instance
        model_name: Name of the model to use (default: gpt-5-nano)
        temperature: Temperature for sampling (default: 1.0)
    
    Returns:
        -1 if persona prefers A, 1 if persona prefers B, 0 if equal
    """
    prompt = f"""You are a person with the following characteristics:
{persona}

Given the topic: "{topic}"

Please compare these two statements and indicate which one you prefer:

Statement A: {statement_a['statement']}

Statement B: {statement_b['statement']}

Consider which statement:
- Better aligns with your values and perspective
- You would more likely support or agree with
- Represents a better position on this topic

Return your choice as a JSON object with this format:
{{"preference": "A"}}  // if you prefer Statement A
{{"preference": "B"}}  // if you prefer Statement B
{{"preference": "equal"}}  // if you have no preference

Return only the JSON, no additional text."""

    start_time = time.time()
    response = openai_client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": "You are evaluating statements based on the given persona. Return ONLY valid JSON, no other text."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        reasoning={"effort": "minimal"}
    )
    api_timer.record(time.time() - start_time)
    
    result = json.loads(response.output_text)
    preference = result.get("preference", "equal").lower()
    
    if preference == "a":
        return -1
    elif preference == "b":
        return 1
    else:
        return 0


def merge_sort_with_comparisons(
    items: List[Dict],
    compare_func: Callable[[Dict, Dict], int]
) -> List[Dict]:
    """
    Sort items using merge sort with a custom comparison function.
    
    Args:
        items: List of items to sort
        compare_func: Function that takes two items and returns -1, 0, or 1
    
    Returns:
        Sorted list of items (most preferred first)
    """
    if len(items) <= 1:
        return items.copy()
    
    # Split in half
    mid = len(items) // 2
    left = merge_sort_with_comparisons(items[:mid], compare_func)
    right = merge_sort_with_comparisons(items[mid:], compare_func)
    
    # Merge
    merged = []
    i, j = 0, 0
    
    while i < len(left) and j < len(right):
        # compare_func returns -1 if left[i] is preferred over right[j]
        comparison = compare_func(left[i], right[j])
        
        if comparison <= 0:  # left[i] is preferred or equal
            merged.append(left[i])
            i += 1
        else:  # right[j] is preferred
            merged.append(right[j])
            j += 1
    
    # Append remaining items
    merged.extend(left[i:])
    merged.extend(right[j:])
    
    return merged


def rank_statements_pairwise(
    persona: str,
    statements: List[Dict],
    topic: str,
    openai_client: OpenAI
) -> List[int]:
    """
    Rank statements using pairwise comparisons with Python's built-in sort.
    
    Args:
        persona: Persona string description
        statements: List of statement dicts
        topic: The topic/question
        openai_client: OpenAI client instance
    
    Returns:
        List of statement indices in order from most to least preferred
    """
    # Add indices to statements for tracking
    indexed_statements = [
        {"index": i, "statement": stmt["statement"]}
        for i, stmt in enumerate(statements)
    ]
    
    # Track comparison count
    comparison_count = [0]
    
    def compare_with_persona(a: Dict, b: Dict) -> int:
        """Wrapper to compare two statements using the persona."""
        comparison_count[0] += 1
        return pairwise_compare(persona, a, b, topic, openai_client)
    
    # Sort using Python's built-in sort with cmp_to_key
    sorted_statements = sorted(indexed_statements, key=cmp_to_key(compare_with_persona))
    
    # Extract indices
    ranking = [stmt["index"] for stmt in sorted_statements]
    
    logger.info(f"  Completed ranking with {comparison_count[0]} pairwise comparisons (expected ~{len(statements)*np.log2(len(statements)):.0f})")
    
    return ranking


# Import numpy for logging
import numpy as np


def get_preference_matrix_pairwise(
    personas: List[str],
    statements: List[Dict],
    topic: str,
    openai_client: OpenAI
) -> List[List[str]]:
    """
    Get preference matrix using pairwise comparisons for all personas.
    
    Args:
        personas: List of persona string descriptions
        statements: List of statement dicts
        topic: The topic/question
        openai_client: OpenAI client instance
    
    Returns:
        Preference matrix where preferences[rank][voter] is the alternative at rank 'rank' for voter 'voter'
    """
    n_statements = len(statements)
    n_personas = len(personas)
    
    expected_comparisons_per_persona = int(n_statements * np.log2(n_statements)) if n_statements > 0 else 0
    total_expected = expected_comparisons_per_persona * n_personas
    
    logger.info(f"Getting preference rankings from {n_personas} personas for {n_statements} statements")
    logger.info(f"Expected comparisons: ~{expected_comparisons_per_persona} per persona, ~{total_expected} total")
    
    # Get ranking from each persona (parallelized)
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def process_persona(persona_idx_pair):
        """Process a single persona and return (index, ranking)."""
        idx, persona = persona_idx_pair
        logger.info(f"Processing persona {idx+1}/{n_personas}")
        ranking = rank_statements_pairwise(persona, statements, topic, openai_client)
        return idx, ranking
    
    rankings = [None] * n_personas
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {
            executor.submit(process_persona, (i, persona)): i 
            for i, persona in enumerate(personas)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Ranking personas", unit="persona"):
            idx, ranking = future.result()
            rankings[idx] = ranking
    
    # Convert to preference matrix format: preferences[rank][voter]
    preferences = []
    for rank in range(n_statements):
        rank_row = []
        for voter in range(n_personas):
            # The statement index at this rank for this voter
            statement_idx = rankings[voter][rank]
            rank_row.append(str(statement_idx))
        preferences.append(rank_row)
    
    return preferences

