"""
Hybrid insertion-based ranking using LLM calls.

This module implements a hybrid sorting approach that optimizes API costs:
- For small lists (n < threshold): Single LLM call to find insertion position
- For large lists (n >= threshold): Binary search with pairwise comparisons,
  then switch to single-call when range drops below threshold

This is more economical than pure pairwise comparison sorting for n < 70.
"""

import json
import logging
import time
from typing import List, Dict, Callable
from functools import cmp_to_key
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import numpy as np

# Import pairwise_compare from existing module for binary search comparisons
from src.large_scale.pairwise_ranking import pairwise_compare
# Import api_timer for timing tracking
from src.full_experiment.config import api_timer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default threshold for switching between binary search and single-call
DEFAULT_THRESHOLD = 70


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def find_insertion_position_single_call(
    persona: str,
    sorted_statements: List[Dict],
    new_statement: Dict,
    topic: str,
    openai_client: OpenAI,
    model_name: str = "gpt-5-nano",
    temperature: float = 1.0
) -> int:
    """
    Single LLM call to find where to insert a new statement into a sorted list.
    
    Args:
        persona: Persona string description
        sorted_statements: List of statement dicts already sorted by preference (most preferred first)
        new_statement: Statement dict to insert
        topic: The topic/question being discussed
        openai_client: OpenAI client instance
        model_name: Name of the model to use (default: gpt-5-nano)
        temperature: Temperature for sampling (default: 1.0)
    
    Returns:
        Index where the new statement should be inserted (0 = most preferred position)
    """
    n = len(sorted_statements)
    
    # Handle empty list case
    if n == 0:
        return 0
    
    # Build the numbered list of statements
    statements_list = "\n".join(
        f"{i}. {stmt['statement']}"
        for i, stmt in enumerate(sorted_statements)
    )
    
    prompt = f"""You are a person with the following characteristics:
{persona}

Given the topic: "{topic}"

Below is a list of statements ranked from MOST preferred (position 0) to LEAST preferred:

{statements_list}

NEW STATEMENT to insert: {new_statement['statement']}

Determine where this new statement should be inserted to maintain your preference order.
- Return 0 if it should become the MOST preferred (before position 0)
- Return {n} if it should become the LEAST preferred (after position {n-1})
- Return any position 1 to {n-1} to insert between existing statements

Return your answer as JSON: {{"position": <number>}}
Return only JSON, no other text."""

    start_time = time.time()
    response = openai_client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": "You are evaluating statements based on the given persona. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        reasoning={"effort": "minimal"}
    )
    api_timer.record(time.time() - start_time)
    
    result = json.loads(response.output_text)
    position = result.get("position", n // 2)
    
    # Clamp position to valid range
    position = max(0, min(n, int(position)))
    
    return position


def find_insertion_position_hybrid(
    persona: str,
    sorted_statements: List[Dict],
    new_statement: Dict,
    topic: str,
    openai_client: OpenAI,
    threshold: int = DEFAULT_THRESHOLD,
    model_name: str = "gpt-5-nano",
    temperature: float = 1.0
) -> int:
    """
    Find insertion position using hybrid approach:
    - Binary search with pairwise comparisons while range >= threshold
    - Single LLM call when range < threshold
    
    Args:
        persona: Persona string description
        sorted_statements: List of statement dicts already sorted by preference
        new_statement: Statement dict to insert
        topic: The topic/question being discussed
        openai_client: OpenAI client instance
        threshold: Size threshold for switching to single-call (default 70)
        model_name: Name of the model to use (default: gpt-5-nano)
        temperature: Temperature for sampling (default: 1.0)
    
    Returns:
        Index where the new statement should be inserted
    """
    n = len(sorted_statements)
    
    # If list is small enough, use single call directly
    if n < threshold:
        return find_insertion_position_single_call(
            persona, sorted_statements, new_statement, topic, openai_client,
            model_name=model_name, temperature=temperature
        )
    
    # Binary search until range is small enough
    left, right = 0, n
    
    while (right - left) >= threshold:
        mid = (left + right) // 2
        
        # Compare new_statement vs statement at mid
        # pairwise_compare returns -1 if first is preferred, 1 if second is preferred
        comparison = pairwise_compare(
            persona, new_statement, sorted_statements[mid], topic, openai_client,
            model_name=model_name, temperature=temperature
        )
        
        if comparison <= 0:
            # new_statement is preferred over or equal to mid, go left
            right = mid
        else:
            # mid is preferred over new_statement, go right
            left = mid + 1
    
    # Range is now < threshold, use single call on the sublist
    sublist = sorted_statements[left:right]
    
    if len(sublist) == 0:
        return left
    
    relative_pos = find_insertion_position_single_call(
        persona, sublist, new_statement, topic, openai_client,
        model_name=model_name, temperature=temperature
    )
    
    return left + relative_pos


def insertion_sort_hybrid(
    items: List[Dict],
    persona: str,
    topic: str,
    openai_client: OpenAI,
    threshold: int = DEFAULT_THRESHOLD,
    model_name: str = "gpt-5-nano",
    temperature: float = 1.0
) -> List[Dict]:
    """
    Sort items using hybrid insertion sort.
    
    Builds a sorted list by inserting items one at a time using the hybrid
    position-finding approach.
    
    Args:
        items: List of items to sort
        persona: Persona string description
        topic: The topic/question being discussed
        openai_client: OpenAI client instance
        threshold: Size threshold for switching to single-call
        model_name: Name of the model to use (default: gpt-5-nano)
        temperature: Temperature for sampling (default: 1.0)
    
    Returns:
        Sorted list of items (most preferred first)
    """
    if len(items) <= 1:
        return items.copy()
    
    # Start with first item
    sorted_list = [items[0]]
    
    # Insert remaining items one by one
    for i, item in enumerate(items[1:], start=1):
        position = find_insertion_position_hybrid(
            persona, sorted_list, item, topic, openai_client, threshold,
            model_name=model_name, temperature=temperature
        )
        sorted_list.insert(position, item)
        
        if (i + 1) % 10 == 0:
            logger.debug(f"    Inserted {i + 1}/{len(items)} items")
    
    return sorted_list


def rank_statements_hybrid(
    persona: str,
    statements: List[Dict],
    topic: str,
    openai_client: OpenAI,
    threshold: int = DEFAULT_THRESHOLD,
    model_name: str = "gpt-5-nano",
    temperature: float = 1.0
) -> List[int]:
    """
    Rank statements using hybrid insertion sort.
    
    This is more efficient than pure pairwise comparison sorting:
    - For n < threshold: Uses single LLM calls (n-1 calls total)
    - For n >= threshold: Uses binary search + single-call finish
    
    Args:
        persona: Persona string description
        statements: List of statement dicts
        topic: The topic/question
        openai_client: OpenAI client instance
        threshold: Size threshold for single-call approach (default 70)
        model_name: Name of the model to use (default: gpt-5-nano)
        temperature: Temperature for sampling (default: 1.0)
    
    Returns:
        List of statement indices in order from most to least preferred
    """
    # Add indices to statements for tracking
    indexed_statements = [
        {"index": i, "statement": stmt["statement"]}
        for i, stmt in enumerate(statements)
    ]
    
    n = len(statements)
    
    # Estimate API calls
    if n < threshold:
        expected_calls = n - 1  # Single call per insertion
    else:
        # Binary search calls + single calls
        binary_calls_per_insert = max(0, np.log2(n / threshold))
        expected_calls = int((n - 1) * (binary_calls_per_insert + 1))
    
    logger.info(f"  Ranking {n} statements with hybrid sort (threshold={threshold})")
    logger.info(f"  Expected API calls: ~{expected_calls}")
    
    # Sort using hybrid insertion sort
    sorted_statements = insertion_sort_hybrid(
        indexed_statements, persona, topic, openai_client, threshold,
        model_name=model_name, temperature=temperature
    )
    
    # Extract indices
    ranking = [stmt["index"] for stmt in sorted_statements]
    
    logger.info(f"  Completed hybrid ranking for {n} statements")
    
    return ranking


def get_preference_matrix_hybrid(
    personas: List[str],
    statements: List[Dict],
    topic: str,
    openai_client: OpenAI,
    threshold: int = DEFAULT_THRESHOLD,
    max_workers: int = 20,
    model_name: str = "gpt-5-nano",
    temperature: float = 1.0
) -> List[List[str]]:
    """
    Get preference matrix using hybrid insertion sort for all personas.
    
    Args:
        personas: List of persona string descriptions
        statements: List of statement dicts
        topic: The topic/question
        openai_client: OpenAI client instance
        threshold: Size threshold for single-call approach
        max_workers: Maximum parallel workers for persona processing
        model_name: Name of the model to use (default: gpt-5-nano)
        temperature: Temperature for sampling (default: 1.0)
    
    Returns:
        Preference matrix where preferences[rank][voter] is the alternative 
        at rank 'rank' for voter 'voter'
    """
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    n_statements = len(statements)
    n_personas = len(personas)
    
    logger.info(f"Getting preference rankings from {n_personas} personas for {n_statements} statements")
    logger.info(f"Using hybrid insertion sort with threshold={threshold}")
    logger.info(f"Model: {model_name}, Temperature: {temperature}")
    
    def process_persona(persona_idx_pair):
        """Process a single persona and return (index, ranking)."""
        idx, persona = persona_idx_pair
        logger.info(f"Processing persona {idx+1}/{n_personas}")
        ranking = rank_statements_hybrid(
            persona, statements, topic, openai_client, threshold,
            model_name=model_name, temperature=temperature
        )
        return idx, ranking
    
    rankings = [None] * n_personas
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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

