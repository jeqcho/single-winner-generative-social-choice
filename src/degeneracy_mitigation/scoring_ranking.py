"""
Scoring-based ranking (Approach B).

Each voter scores all statements from -100 to +100, then scores are
converted to a ranking. Duplicate scores are resolved through follow-up
rounds (max 3 total rounds).

Key features:
- Simple -100 to +100 scale (decimals allowed)
- No anchor statements needed
- Unique scores required for clean ranking
- Iterative dedup for any duplicate scores
"""

import json
import logging
import random
import time
from collections import Counter
from typing import Any

from openai import OpenAI

from .config import (
    MODEL,
    TEMPERATURE,
    MAX_DEDUP_ROUNDS,
    HASH_SEED,
    SYSTEM_PROMPT_TEMPLATE,
    SCORING_TASK,
    api_timer,
)
from .hash_identifiers import id_to_hash, build_hash_lookup
from .degeneracy_detector import validate_scores

logger = logging.getLogger(__name__)


def build_system_prompt(persona: str) -> str:
    """Build the system prompt with persona injection."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        persona=persona,
        task_description=SCORING_TASK
    )


def build_scoring_prompt(
    topic: str,
    statements: list[tuple[str, str]]
) -> str:
    """
    Build user prompt for scoring all statements.
    
    Args:
        topic: The topic question
        statements: List of (hash, text) tuples
    
    Returns:
        The user prompt string.
    """
    n = len(statements)
    
    # Build statements list
    stmt_lines = "\n".join(f"{h}: \"{text}\"" for h, text in statements)
    
    prompt = f"""Topic: "{topic}"

Here are {n} statements (identified by 4-letter codes):
{stmt_lines}

Score each statement based on how much you agree with it:
- +100 = strongly agree/support
- 0 = neutral
- -100 = strongly disagree/oppose

IMPORTANT: 
- Do NOT output duplicate scores. Each statement must have a unique score.
- Decimal points are allowed if necessary (e.g., 75.5).

Return JSON: {{"hash1": score1, "hash2": score2, ...}}"""
    
    return prompt


def build_dedup_prompt(
    statements: list[tuple[str, str]]
) -> str:
    """
    Build user prompt for re-scoring statements with duplicate scores.
    
    Args:
        statements: List of (hash, text) tuples that had duplicate scores
    
    Returns:
        The user prompt string.
    """
    # Build statements list
    stmt_lines = "\n".join(f"{h}: \"{text}\"" for h, text in statements)
    
    prompt = f"""The following statements previously received the same score. 
Please re-score them with UNIQUE scores to differentiate your preference.

{stmt_lines}

Use the same scale (-100 to +100, decimals allowed).
Each statement must have a unique score.

Return JSON: {{"hash1": score1, "hash2": score2, ...}}"""
    
    return prompt


def call_api_for_scores(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str
) -> dict[str, float]:
    """
    Make API call to get scores for statements.
    
    Args:
        client: OpenAI client
        system_prompt: System prompt with persona
        user_prompt: User prompt with statements
        reasoning_effort: "minimal", "low", or "medium"
    
    Returns:
        Dictionary mapping hash to score.
    
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
    
    # Convert all values to float
    scores = {k: float(v) for k, v in result.items()}
    
    return scores


def find_duplicate_scores(scores: dict[str, float]) -> list[str]:
    """
    Find all hashes that share a score with another hash.
    
    Args:
        scores: Dictionary mapping hash to score
    
    Returns:
        List of hashes that have duplicate score values.
    """
    score_counts = Counter(scores.values())
    duplicate_values = {v for v, count in score_counts.items() if count > 1}
    return [h for h, v in scores.items() if v in duplicate_values]


def score_with_dedup(
    client: OpenAI,
    persona: str,
    statements: list[dict],
    topic: str,
    reasoning_effort: str,
    voter_seed: int,
    hash_seed: int = HASH_SEED
) -> dict:
    """
    Score statements, then resolve any duplicates with follow-up rounds.
    
    Max 3 rounds total (1 initial + 2 dedup rounds).
    
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
        - 'scores': Final scores dict (hash -> score)
        - 'ranking': List of statement IDs (most to least preferred)
        - 'dedup_rounds': Number of dedup rounds needed
        - 'has_unresolved_duplicates': True if duplicates remain
        - 'round_details': Per-round metadata
    """
    n = len(statements)
    
    # Build hash lookup and statement mapping
    hash_lookup = build_hash_lookup(n, hash_seed)
    
    # Shuffle presentation order for this voter
    rng = random.Random(voter_seed)
    shuffled_ids = list(range(n))
    rng.shuffle(shuffled_ids)
    
    # Build statements with hashes
    stmt_with_hashes = [
        (id_to_hash(sid, hash_seed), statements[sid]['statement'])
        for sid in shuffled_ids
    ]
    valid_hashes = set(h for h, _ in stmt_with_hashes)
    
    system_prompt = build_system_prompt(persona)
    
    round_details = []
    dedup_rounds = 0
    
    # Round 1: Score all statements
    user_prompt = build_scoring_prompt(topic, stmt_with_hashes)
    
    try:
        scores = call_api_for_scores(client, system_prompt, user_prompt, reasoning_effort)
        
        # Validate scores
        is_valid, error_msg = validate_scores(scores, valid_hashes)
        if not is_valid:
            logger.warning(f"Initial scoring validation failed: {error_msg}")
            # Try to continue anyway if we have most scores
        
        round_details.append({
            'round': 1,
            'type': 'initial',
            'n_scores': len(scores),
            'duplicates_found': len(find_duplicate_scores(scores)),
        })
        
    except Exception as e:
        logger.error(f"Initial scoring failed: {e}")
        # Return empty/invalid result
        return {
            'scores': {},
            'ranking': [],
            'dedup_rounds': 0,
            'has_unresolved_duplicates': True,
            'round_details': [{'round': 1, 'error': str(e)}],
        }
    
    # Dedup rounds (max 2 more)
    while dedup_rounds < MAX_DEDUP_ROUNDS - 1:  # -1 because initial round counts
        duplicates = find_duplicate_scores(scores)
        if not duplicates:
            break  # No duplicates, we're done
        
        dedup_rounds += 1
        logger.info(f"Dedup round {dedup_rounds}: {len(duplicates)} hashes with duplicate scores")
        
        # Build dedup statements
        dup_statements = [(h, text) for h, text in stmt_with_hashes if h in duplicates]
        
        # Make dedup API call
        dedup_prompt = build_dedup_prompt(dup_statements)
        
        try:
            new_scores = call_api_for_scores(client, system_prompt, dedup_prompt, reasoning_effort)
            
            # Update scores
            for h, score in new_scores.items():
                if h in scores:
                    scores[h] = score
            
            round_details.append({
                'round': dedup_rounds + 1,
                'type': 'dedup',
                'n_rescored': len(new_scores),
                'duplicates_remaining': len(find_duplicate_scores(scores)),
            })
            
        except Exception as e:
            logger.warning(f"Dedup round {dedup_rounds} failed: {e}")
            round_details.append({
                'round': dedup_rounds + 1,
                'type': 'dedup',
                'error': str(e),
            })
            break
    
    # Check for unresolved duplicates
    final_duplicates = find_duplicate_scores(scores)
    has_unresolved = len(final_duplicates) > 0
    
    if has_unresolved:
        logger.warning(f"Unresolved duplicates after {dedup_rounds} dedup rounds: {final_duplicates}")
    
    # Convert scores to ranking
    ranking_hashes = scores_to_ranking(scores)
    ranking_ids = [hash_lookup[h] for h in ranking_hashes if h in hash_lookup]
    
    return {
        'scores': scores,
        'ranking': ranking_ids,
        'dedup_rounds': dedup_rounds,
        'has_unresolved_duplicates': has_unresolved,
        'round_details': round_details,
    }


def scores_to_ranking(scores: dict[str, float]) -> list[str]:
    """
    Convert scores to ranking (highest score = rank 1).
    
    Args:
        scores: Dictionary mapping hash to score
    
    Returns:
        List of hashes sorted by score (highest first).
    """
    return sorted(scores.keys(), key=lambda h: scores[h], reverse=True)


def score_voter(
    client: OpenAI,
    voter_idx: int,
    persona: str,
    statements: list[dict],
    topic: str,
    reasoning_effort: str,
    hash_seed: int = HASH_SEED
) -> dict:
    """
    Score statements for a single voter.
    
    Wrapper around score_with_dedup with voter index for seeding.
    
    Args:
        client: OpenAI client
        voter_idx: Index of the voter (used for seeding)
        persona: Persona string
        statements: List of statement dicts
        topic: Topic question
        reasoning_effort: Reasoning effort level
        hash_seed: Seed for hash generation
    
    Returns:
        Result dict from score_with_dedup with voter_idx added.
    """
    result = score_with_dedup(
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
