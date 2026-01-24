"""
Iterative Top-K/Bottom-K ranking (Approach A*).

Variant of Approach A where bottom-K is requested with "least preferred first"
instead of "least preferred last". The hypothesis is that it's cognitively
easier to output the worst statement first rather than "reserving space" for it.

Key differences from Approach A:
- Bottom-K prompt: "least preferred first" (most disliked first)
- Assembly logic: append instead of prepend for bottom rankings
"""

import json
import logging
import random
import time
from typing import Any

from openai import OpenAI

from .config import (
    MODEL,
    TEMPERATURE,
    K_TOP_BOTTOM,
    N_ROUNDS,
    MAX_RETRIES,
    HASH_SEED,
    SYSTEM_PROMPT_TEMPLATE,
    RANKING_TASK,
    api_timer,
)
from .hash_identifiers import id_to_hash, hash_to_id, build_hash_lookup
from .degeneracy_detector import (
    is_partial_degenerate,
    is_degenerate,
    validate_top_bottom_k,
    validate_final_ranking,
)

logger = logging.getLogger(__name__)


def build_system_prompt(persona: str) -> str:
    """Build the system prompt with persona injection."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        persona=persona,
        task_description=RANKING_TASK
    )


def build_top_bottom_prompt(
    topic: str,
    statements: list[tuple[str, str]],
    k: int = K_TOP_BOTTOM
) -> str:
    """
    Build user prompt for top-K/bottom-K selection.
    
    A* variant: asks for bottom-K with "least preferred FIRST" (most disliked first).
    
    Args:
        topic: The topic question
        statements: List of (hash, text) tuples in presentation order
        k: Number to select for top and bottom
    
    Returns:
        The user prompt string.
    """
    n = len(statements)
    
    # Build statements list
    stmt_lines = "\n".join(f"{h}: \"{text}\"" for h, text in statements)
    
    # KEY CHANGE: "least preferred first" instead of "least preferred last"
    prompt = f"""Topic: "{topic}"

Here are {n} statements (identified by 4-letter codes):
{stmt_lines}

From these {n} statements, identify:
1. Your TOP {k} most preferred (in order, most preferred first)
2. Your BOTTOM {k} least preferred (in order, least preferred first)

IMPORTANT: Do NOT simply list codes in the order they appear above.
Your preferences should reflect your persona's values and background.

Return JSON: {{"top_{k}": ["code1", "code2", ...], "bottom_{k}": ["code1", "code2", ...]}}"""
    
    return prompt


def build_final_ranking_prompt(
    topic: str,
    statements: list[tuple[str, str]]
) -> str:
    """
    Build user prompt for final round (rank all remaining).
    
    Args:
        topic: The topic question
        statements: List of (hash, text) tuples in presentation order
    
    Returns:
        The user prompt string.
    """
    n = len(statements)
    
    # Build statements list
    stmt_lines = "\n".join(f"{h}: \"{text}\"" for h, text in statements)
    
    prompt = f"""Topic: "{topic}"

Here are {n} statements (identified by 4-letter codes):
{stmt_lines}

Rank ALL of these statements from most to least preferred.

IMPORTANT: Do NOT simply list codes in the order they appear above.
Your preferences should reflect your persona's values and background.

Return JSON: {{"ranking": ["most_preferred", "second", ..., "least_preferred"]}}"""
    
    return prompt


def call_api_for_top_bottom(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str,
    k: int = K_TOP_BOTTOM
) -> tuple[list[str], list[str]]:
    """
    Make API call to get top-K and bottom-K selections.
    
    Args:
        client: OpenAI client
        system_prompt: System prompt with persona
        user_prompt: User prompt with statements
        reasoning_effort: "minimal", "low", or "medium"
        k: Expected count for each list
    
    Returns:
        Tuple of (top_k, bottom_k) lists of hash strings.
    
    Raises:
        Exception on API or parsing error.
    """
    start_time = time.time()
    
    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=TEMPERATURE,
        reasoning={"effort": reasoning_effort},
    )
    
    api_timer.record(time.time() - start_time)
    
    # Parse response
    result = json.loads(response.output_text)
    
    top_k = result.get(f"top_{k}", result.get("top_10", []))
    bottom_k = result.get(f"bottom_{k}", result.get("bottom_10", []))
    
    return top_k, bottom_k


def call_api_for_final_ranking(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str
) -> list[str]:
    """
    Make API call to get final ranking of remaining statements.
    
    Args:
        client: OpenAI client
        system_prompt: System prompt with persona
        user_prompt: User prompt with statements
        reasoning_effort: "minimal", "low", or "medium"
    
    Returns:
        List of hash strings in preference order (most to least preferred).
    
    Raises:
        Exception on API or parsing error.
    """
    start_time = time.time()
    
    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=TEMPERATURE,
        reasoning={"effort": reasoning_effort},
    )
    
    api_timer.record(time.time() - start_time)
    
    # Parse response
    result = json.loads(response.output_text)
    ranking = result.get("ranking", [])
    
    return ranking


def get_top_bottom_with_retry(
    client: OpenAI,
    persona: str,
    topic: str,
    statements: list[tuple[str, str]],
    presentation_order: list[str],
    reasoning_effort: str,
    k: int = K_TOP_BOTTOM,
    max_retries: int = MAX_RETRIES
) -> tuple[list[str], list[str], int, bool]:
    """
    Get top-K/bottom-K with validation and retry logic.
    
    Args:
        client: OpenAI client
        persona: Persona string
        topic: Topic question
        statements: List of (hash, text) tuples
        presentation_order: Order of hashes as presented
        reasoning_effort: Reasoning effort level
        k: Number to select for top/bottom
        max_retries: Maximum retry attempts
    
    Returns:
        Tuple of (top_k, bottom_k, retry_count, is_valid).
    """
    system_prompt = build_system_prompt(persona)
    user_prompt = build_top_bottom_prompt(topic, statements, k)
    valid_hashes = set(presentation_order)
    
    top_k, bottom_k = None, None
    
    for attempt in range(max_retries + 1):
        try:
            top_k, bottom_k = call_api_for_top_bottom(
                client, system_prompt, user_prompt, reasoning_effort, k
            )
            
            # Validate structural correctness
            is_valid, error_msg = validate_top_bottom_k(top_k, bottom_k, valid_hashes, k)
            if not is_valid:
                logger.warning(f"Validation failed on attempt {attempt + 1}: {error_msg}")
                continue
            
            # Check for degeneracy
            if is_partial_degenerate(top_k, bottom_k, presentation_order):
                logger.warning(f"Degenerate output on attempt {attempt + 1}")
                continue
            
            # Success!
            return top_k, bottom_k, attempt, True
            
        except Exception as e:
            logger.warning(f"Exception on attempt {attempt + 1}: {e}")
    
    # All retries exhausted - return last result (may be invalid)
    logger.error(f"All {max_retries + 1} attempts failed for top-bottom selection")
    return top_k or [], bottom_k or [], max_retries, False


def get_final_ranking_with_retry(
    client: OpenAI,
    persona: str,
    topic: str,
    statements: list[tuple[str, str]],
    presentation_order: list[str],
    reasoning_effort: str,
    max_retries: int = MAX_RETRIES
) -> tuple[list[str], int, bool]:
    """
    Get final ranking with validation and retry logic.
    
    Args:
        client: OpenAI client
        persona: Persona string
        topic: Topic question
        statements: List of (hash, text) tuples
        presentation_order: Order of hashes as presented
        reasoning_effort: Reasoning effort level
        max_retries: Maximum retry attempts
    
    Returns:
        Tuple of (ranking, retry_count, is_valid).
    """
    system_prompt = build_system_prompt(persona)
    user_prompt = build_final_ranking_prompt(topic, statements)
    valid_hashes = set(presentation_order)
    
    ranking = None
    
    for attempt in range(max_retries + 1):
        try:
            ranking = call_api_for_final_ranking(
                client, system_prompt, user_prompt, reasoning_effort
            )
            
            # Validate structural correctness
            is_valid, error_msg = validate_final_ranking(ranking, valid_hashes)
            if not is_valid:
                logger.warning(f"Validation failed on attempt {attempt + 1}: {error_msg}")
                continue
            
            # Check for degeneracy
            if is_degenerate(ranking, presentation_order):
                logger.warning(f"Degenerate output on attempt {attempt + 1}")
                continue
            
            # Success!
            return ranking, attempt, True
            
        except Exception as e:
            logger.warning(f"Exception on attempt {attempt + 1}: {e}")
    
    # All retries exhausted
    logger.error(f"All {max_retries + 1} attempts failed for final ranking")
    return ranking or [], max_retries, False


def iterative_rank(
    client: OpenAI,
    persona: str,
    statements: list[dict],
    topic: str,
    reasoning_effort: str,
    voter_seed: int,
    hash_seed: int = HASH_SEED
) -> dict:
    """
    Build full ranking through 5 rounds of top-K/bottom-K selection (A* variant).
    
    Each round shuffles remaining statements to break presentation order bias.
    Degeneracy is checked against THAT ROUND's presentation order.
    
    A* difference: bottom_k is now "least preferred first", so we append
    instead of prepend when assembling the final ranking.
    
    Args:
        client: OpenAI client
        persona: Persona string
        statements: List of statement dicts with 'statement' key
        topic: Topic question
        reasoning_effort: "minimal", "low", or "medium"
        voter_seed: Seed for per-voter randomization
        hash_seed: Seed for hash generation
    
    Returns:
        Dictionary with:
        - 'ranking': Full ranking (list of statement IDs, most to least preferred)
        - 'round_details': Per-round metadata
        - 'total_retries': Total retries across all rounds
        - 'all_valid': True if all rounds succeeded
    """
    n = len(statements)
    remaining_ids = list(range(n))
    
    top_rankings = []      # Accumulate top selections (in order)
    bottom_rankings = []   # Accumulate bottom selections (A*: append, not prepend)
    
    round_details = []
    total_retries = 0
    all_valid = True
    
    # Build hash lookup
    hash_lookup = build_hash_lookup(n, hash_seed)
    
    for round_num in range(1, N_ROUNDS + 1):
        # Shuffle remaining statements for THIS round
        round_seed = voter_seed * 10 + round_num
        rng = random.Random(round_seed)
        shuffled_ids = remaining_ids.copy()
        rng.shuffle(shuffled_ids)
        
        # Build presentation for this round
        presentation_order = [id_to_hash(sid, hash_seed) for sid in shuffled_ids]
        round_statements = [
            (id_to_hash(sid, hash_seed), statements[sid]['statement']) 
            for sid in shuffled_ids
        ]
        
        round_info = {
            'round': round_num,
            'n_statements': len(round_statements),
            'presentation_order': presentation_order,
        }
        
        if round_num < N_ROUNDS:
            # Rounds 1-4: Get top 10 and bottom 10
            top_k_hashes, bottom_k_hashes, retries, is_valid = get_top_bottom_with_retry(
                client=client,
                persona=persona,
                topic=topic,
                statements=round_statements,
                presentation_order=presentation_order,
                reasoning_effort=reasoning_effort,
            )
            
            round_info['type'] = 'top_bottom'
            round_info['retries'] = retries
            round_info['is_valid'] = is_valid
            round_info['top_k'] = top_k_hashes
            round_info['bottom_k'] = bottom_k_hashes
            
            total_retries += retries
            if not is_valid:
                all_valid = False
            
            # Convert hashes back to IDs
            top_k_ids = [hash_lookup[h] for h in top_k_hashes if h in hash_lookup]
            bottom_k_ids = [hash_lookup[h] for h in bottom_k_hashes if h in hash_lookup]
            
            # Accumulate rankings
            top_rankings.extend(top_k_ids)
            # KEY CHANGE: Append instead of prepend (A* uses "least preferred first")
            bottom_rankings.extend(bottom_k_ids)
            
            # Remove from remaining
            placed = set(top_k_ids + bottom_k_ids)
            remaining_ids = [sid for sid in remaining_ids if sid not in placed]
            
        else:
            # Round 5: Rank all 20 remaining
            final_hashes, retries, is_valid = get_final_ranking_with_retry(
                client=client,
                persona=persona,
                topic=topic,
                statements=round_statements,
                presentation_order=presentation_order,
                reasoning_effort=reasoning_effort,
            )
            
            round_info['type'] = 'final_ranking'
            round_info['retries'] = retries
            round_info['is_valid'] = is_valid
            round_info['ranking'] = final_hashes
            
            total_retries += retries
            if not is_valid:
                all_valid = False
            
            # Convert hashes back to IDs
            middle_ranking_ids = [hash_lookup[h] for h in final_hashes if h in hash_lookup]
        
        round_details.append(round_info)
    
    # Assemble final ranking: top_rankings + middle_ranking + bottom_rankings
    final_ranking = top_rankings + middle_ranking_ids + bottom_rankings
    
    return {
        'ranking': final_ranking,
        'round_details': round_details,
        'total_retries': total_retries,
        'all_valid': all_valid,
    }


def rank_voter(
    client: OpenAI,
    voter_idx: int,
    persona: str,
    statements: list[dict],
    topic: str,
    reasoning_effort: str,
    hash_seed: int = HASH_SEED
) -> dict:
    """
    Rank statements for a single voter (A* variant).
    
    Wrapper around iterative_rank with voter index for seeding.
    
    Args:
        client: OpenAI client
        voter_idx: Index of the voter (used for seeding)
        persona: Persona string
        statements: List of statement dicts
        topic: Topic question
        reasoning_effort: Reasoning effort level
        hash_seed: Seed for hash generation
    
    Returns:
        Result dict from iterative_rank with voter_idx added.
    """
    result = iterative_rank(
        client=client,
        persona=persona,
        statements=statements,
        topic=topic,
        reasoning_effort=reasoning_effort,
        voter_seed=voter_idx,
        hash_seed=hash_seed,
    )
    result['voter_idx'] = voter_idx
    return result
