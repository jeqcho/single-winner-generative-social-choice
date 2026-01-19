"""
Single-call ranking using GPT-5.2.

Instead of pairwise comparisons or insertion sort, we ask the model to
rank all statements in a single API call and return sorted IDs.
"""

import json
import logging
import time
from typing import List, Dict
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import MODEL, TEMPERATURE, api_timer

logger = logging.getLogger(__name__)


# Sentinel value for invalid/failed rankings
INVALID_RANKING_VALUE = -999
MAX_RANKING_RETRIES = 10


def _make_single_ranking_api_call(
    persona: str,
    statements: List[Dict],
    topic: str,
    openai_client: OpenAI,
    model: str,
    temperature: float
) -> List[int]:
    """
    Make a single API call for ranking. Returns the ranking or raises an exception.
    This is a helper function that does NOT handle validation - just the API call.
    """
    n = len(statements)
    
    # Build numbered statement list
    statements_text = "\n".join(
        f"{i}: {stmt['statement']}"
        for i, stmt in enumerate(statements)
    )
    
    system_prompt = "You are ranking statements based on the given persona. Return ONLY valid JSON, no other text."
    
    user_prompt = f"""You are a person with the following characteristics:
{persona}

Given the topic: "{topic}"

Below are {n} statements. Rank them from MOST preferred (rank 1) to LEAST preferred (rank {n}) based on your persona.

{statements_text}

Return a JSON object mapping each rank position to the statement index you assign to that rank:
{{"1": <most_preferred_idx>, "2": <second_preferred_idx>, ..., "{n}": <least_preferred_idx>}}

IMPORTANT: 
- Keys must be "1" through "{n}" (rank positions, where 1 = most preferred)
- Values must be unique integers from 0 to {n-1} (statement indices)
- Each statement index must appear exactly once
Return only the JSON, no additional text."""

    start_time = time.time()
    response = openai_client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        reasoning={"effort": "low"},
    )
    api_timer.record(time.time() - start_time)
    
    result = json.loads(response.output_text)
    # Convert {"1": idx, "2": idx, ...} to [idx_at_rank1, idx_at_rank2, ...]
    ranking = [result[str(i)] for i in range(1, n + 1)]
    
    return ranking


def _validate_ranking(ranking: List[int], n: int) -> bool:
    """Validate that a ranking is a valid permutation of 0 to n-1."""
    if not isinstance(ranking, list):
        return False
    if len(ranking) != n:
        return False
    if set(ranking) != set(range(n)):
        return False
    return True


def rank_all_statements_single_call(
    persona: str,
    statements: List[Dict],
    topic: str,
    openai_client: OpenAI,
    model: str = MODEL,
    temperature: float = TEMPERATURE
) -> List[int]:
    """
    Rank all statements in a single API call with retry logic.
    
    Args:
        persona: Persona string description
        statements: List of statement dicts with 'statement' key
        topic: The topic/question being discussed
        openai_client: OpenAI client instance
        model: Model to use (default: GPT-5.2)
        temperature: Temperature for sampling
    
    Returns:
        List of statement indices in order from most to least preferred.
        Returns [INVALID_RANKING_VALUE] * n if all retries fail (hard fail).
    """
    n = len(statements)
    
    for attempt in range(1, MAX_RANKING_RETRIES + 1):
        try:
            ranking = _make_single_ranking_api_call(
                persona, statements, topic, openai_client, model, temperature
            )
            
            # Validate ranking
            if _validate_ranking(ranking, n):
                if attempt > 1:
                    logger.info(f"Valid ranking obtained on attempt {attempt}")
                return ranking
            else:
                # Log the invalid ranking details
                actual_len = len(ranking) if isinstance(ranking, list) else "N/A"
                has_duplicates = len(ranking) != len(set(ranking)) if isinstance(ranking, list) else "N/A"
                logger.warning(
                    f"Invalid ranking on attempt {attempt}/{MAX_RANKING_RETRIES}: "
                    f"length={actual_len} (expected {n}), has_duplicates={has_duplicates}, "
                    f"ranking={ranking[:20]}..." if len(str(ranking)) > 100 else f"ranking={ranking}"
                )
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error on attempt {attempt}/{MAX_RANKING_RETRIES}: {e}")
        except Exception as e:
            logger.warning(f"API error on attempt {attempt}/{MAX_RANKING_RETRIES}: {type(e).__name__}: {e}")
    
    # All retries exhausted - hard fail with invalid marker
    logger.error(
        f"HARD FAIL: All {MAX_RANKING_RETRIES} attempts failed to produce a valid ranking. "
        f"Returning invalid ranking marker (all {INVALID_RANKING_VALUE}). "
        f"Persona preview: {persona[:100]}..."
    )
    return [INVALID_RANKING_VALUE] * n


def get_preference_matrix_single_call(
    personas: List[str],
    statements: List[Dict],
    topic: str,
    openai_client: OpenAI,
    max_workers: int = 50,
    model: str = MODEL,
    temperature: float = TEMPERATURE
) -> List[List[str]]:
    """
    Build preference matrix using single-call ranking for all personas.
    
    Args:
        personas: List of persona string descriptions
        statements: List of statement dicts
        topic: The topic/question
        openai_client: OpenAI client instance
        max_workers: Maximum parallel workers
        model: Model to use
        temperature: Temperature for sampling
    
    Returns:
        Preference matrix where preferences[rank][voter] is the alternative
        index (as string) at rank 'rank' for voter 'voter'.
        Invalid rankings will contain INVALID_RANKING_VALUE (-999) for all positions.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    
    n_statements = len(statements)
    n_personas = len(personas)
    
    logger.info(f"Building preference matrix: {n_personas} personas x {n_statements} statements")
    logger.info(f"Using single-call ranking with model={model}")
    logger.info(f"Max retries per persona: {MAX_RANKING_RETRIES}")
    
    def process_persona(args):
        """Process a single persona and return (index, ranking)."""
        idx, persona = args
        ranking = rank_all_statements_single_call(
            persona, statements, topic, openai_client, model, temperature
        )
        return idx, ranking
    
    rankings = [None] * n_personas
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_persona, (i, persona)): i
            for i, persona in enumerate(personas)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="Ranking personas", unit="persona"):
            idx, ranking = future.result()
            rankings[idx] = ranking
    
    # Count invalid rankings (those that failed all retries)
    invalid_count = sum(
        1 for r in rankings 
        if r and r[0] == INVALID_RANKING_VALUE
    )
    
    if invalid_count > 0:
        invalid_voter_ids = [i for i, r in enumerate(rankings) if r and r[0] == INVALID_RANKING_VALUE]
        logger.error(
            f"PREFERENCE MATRIX CONTAINS {invalid_count}/{n_personas} INVALID RANKINGS "
            f"(voters: {invalid_voter_ids}). These voters have all {INVALID_RANKING_VALUE} values."
        )
    else:
        logger.info(f"All {n_personas} rankings are valid.")
    
    # Convert to preference matrix format: preferences[rank][voter]
    preferences = []
    for rank in range(n_statements):
        rank_row = []
        for voter in range(n_personas):
            # The statement index at this rank for this voter
            statement_idx = rankings[voter][rank]
            rank_row.append(str(statement_idx))
        preferences.append(rank_row)
    
    logger.info(f"Built preference matrix: {len(preferences)} x {len(preferences[0])}")
    
    return preferences


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def insert_statement_into_ranking(
    persona: str,
    current_ranking: List[int],
    statements: List[Dict],
    new_statement: str,
    topic: str,
    openai_client: OpenAI,
    model: str = MODEL,
    temperature: float = TEMPERATURE
) -> List[int]:
    """
    Insert a new statement into an existing ranking.
    
    Used for ChatGPT** where we need to re-query voters to insert
    the newly generated statement.
    
    Args:
        persona: Persona string description
        current_ranking: Current ranking (list of statement indices, most to least preferred)
        statements: Original list of statement dicts
        new_statement: The new statement text to insert
        topic: The topic/question
        openai_client: OpenAI client instance
        model: Model to use
        temperature: Temperature for sampling
    
    Returns:
        Updated ranking with new statement index (len(statements)) inserted
    """
    n = len(current_ranking)
    new_idx = len(statements)  # New statement gets the next available index
    
    # Build the ranked statements for context
    ranked_text = "\n".join(
        f"Rank {i+1} (ID {idx}): {statements[idx]['statement']}"
        for i, idx in enumerate(current_ranking)
    )
    
    system_prompt = "You are inserting a new statement into your preference ranking. Return ONLY valid JSON."
    
    user_prompt = f"""You are a person with the following characteristics:
{persona}

Given the topic: "{topic}"

You previously ranked these statements from most to least preferred:

{ranked_text}

NEW STATEMENT (ID {new_idx}): {new_statement}

Where should this new statement be inserted in your ranking?
- Return 0 to make it your MOST preferred (before rank 1)
- Return {n} to make it your LEAST preferred (after rank {n})
- Return any position 1-{n-1} to insert it between existing ranks

Return JSON: {{"insert_position": <number>}}"""

    start_time = time.time()
    response = openai_client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        reasoning={"effort": "low"},
    )
    api_timer.record(time.time() - start_time)
    
    result = json.loads(response.output_text)
    position = result.get("insert_position", n // 2)
    
    # Clamp position to valid range
    position = max(0, min(n, int(position)))
    
    # Insert new statement at position
    new_ranking = current_ranking.copy()
    new_ranking.insert(position, new_idx)
    
    return new_ranking
