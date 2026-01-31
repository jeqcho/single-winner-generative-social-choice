"""
Batched iterative insertion for GPT** and GPT*** methods.

Instead of inserting new statements one at a time, this module batches all
new statements from a rep together with the original 100 statements and
runs a single iterative ranking per voter.

Key benefits:
- 16x cost reduction (1 ranking per voter vs 16)
- Positions are extracted relative to original statements only
- Consistent with iterative ranking methodology (stable, accurate)
"""

import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from src.degeneracy_mitigation.iterative_ranking import rank_voter
from src.degeneracy_mitigation.config import HASH_SEED
from src.experiment_utils.config import (
    RANKING_REASONING,
    N_ALT_POOL,
    MAX_WORKERS,
)
from src.experiment_utils.epsilon_calculator import compute_critical_epsilon_custom

logger = logging.getLogger(__name__)


def extract_position_among_originals(
    ranking: List[int],
    new_statement_idx: int,
    n_originals: int = 100
) -> int:
    """
    Find position of new statement counting only original statements.
    
    IMPORTANT: We discount other new statements in the ranking.
    If all 16 new statements are ranked positions 0-15, they all get position 0
    (because 0 originals are ranked before any of them).
    
    Args:
        ranking: Full ranking (list of statement indices, e.g., 0-115)
        new_statement_idx: Index of the new statement to find (e.g., 100-115)
        n_originals: Number of original statements (default 100)
    
    Returns:
        Position among originals only (0 = most preferred, n_originals = least preferred)
    
    Raises:
        ValueError: If new_statement_idx is not found in the ranking
    """
    position_among_originals = 0
    for idx in ranking:
        if idx == new_statement_idx:
            return position_among_originals
        if idx < n_originals:  # Only count original statements
            position_among_originals += 1
    raise ValueError(f"Statement {new_statement_idx} not found in ranking")


def rank_with_new_statements(
    persona: str,
    original_statements: List[Dict],
    new_statements: List[Dict],
    topic: str,
    openai_client: OpenAI,
    voter_idx: int,
    hash_seed: int = HASH_SEED,
    voter_dist: str = None,
    alt_dist: str = None,
    rep: int = None,
) -> Dict[str, int]:
    """
    Rank all statements together and return positions of new statements
    RELATIVE TO ORIGINAL STATEMENTS ONLY.
    
    Args:
        persona: Voter persona string
        original_statements: 100 original statements (list of {"statement": str, ...})
        new_statements: List of new statements with metadata:
            [{"statement": str, "method": str, "mini_rep": int or None}, ...]
        topic: Topic question string
        openai_client: OpenAI client instance
        voter_idx: Voter index (used for seeding)
        hash_seed: Seed for hash generation
        voter_dist: Voter distribution (for metadata)
        alt_dist: Alternative distribution (for metadata)
        rep: Replication number (for metadata)
    
    Returns:
        Dict mapping method keys to positions among originals (0 to n_originals)
        e.g., {"gpt_triple_star": 42, "gpt_double_star_base_mr0": 15, ...}
    """
    n_originals = len(original_statements)
    
    # Combine: original (indices 0 to n_originals-1) + new (indices n_originals+)
    all_statements = original_statements + [
        {"statement": s["statement"]} for s in new_statements
    ]
    
    # Run iterative ranking on all statements (e.g., 116)
    result = rank_voter(
        client=openai_client,
        voter_idx=voter_idx,
        persona=persona,
        statements=all_statements,
        topic=topic,
        reasoning_effort=RANKING_REASONING,
        hash_seed=hash_seed,
        voter_dist=voter_dist,
        alt_dist=alt_dist,
        rep=rep,
    )
    
    # Check if ranking succeeded
    if not result.get('all_valid', False) or not result.get('ranking'):
        logger.error(f"Ranking failed for voter {voter_idx}")
        # Return None positions to indicate failure
        return {_make_method_key(s): None for s in new_statements}
    
    # Extract positions for new statements, counting only originals
    ranking = result['ranking']
    positions = {}
    
    for i, new_stmt in enumerate(new_statements):
        new_idx = n_originals + i
        try:
            position = extract_position_among_originals(ranking, new_idx, n_originals)
        except ValueError as e:
            logger.error(f"Position extraction failed: {e}")
            position = None
        
        key = _make_method_key(new_stmt)
        positions[key] = position
    
    return positions


def _make_method_key(new_stmt: Dict) -> str:
    """Create a unique key for a new statement based on method and mini_rep."""
    method = new_stmt.get("method", "unknown")
    mini_rep = new_stmt.get("mini_rep")
    if mini_rep is not None:
        return f"{method}_mr{mini_rep}"
    return method


def run_batched_ranking_for_rep(
    original_statements: List[Dict],
    new_statements: List[Dict],
    voter_personas: List[str],
    topic: str,
    openai_client: OpenAI,
    hash_seed: int = HASH_SEED,
    voter_dist: str = None,
    alt_dist: str = None,
    rep: int = None,
    max_workers: int = MAX_WORKERS,
) -> Dict[str, List[Optional[int]]]:
    """
    Run batched iterative ranking for all voters in a rep.
    
    Args:
        original_statements: 100 original statements
        new_statements: List of new statements with metadata
        voter_personas: List of 100 voter persona strings
        topic: Topic question string
        openai_client: OpenAI client instance
        hash_seed: Seed for hash generation
        voter_dist: Voter distribution (for metadata)
        alt_dist: Alternative distribution (for metadata)
        rep: Replication number (for metadata)
        max_workers: Maximum parallel workers
    
    Returns:
        Dict mapping method keys to list of 100 positions (one per voter)
        e.g., {"gpt_triple_star": [42, 15, 3, ...], ...}
    """
    n_voters = len(voter_personas)
    
    # Initialize result structure
    method_keys = [_make_method_key(s) for s in new_statements]
    all_positions: Dict[str, List[Optional[int]]] = {k: [None] * n_voters for k in method_keys}
    
    logger.info(f"Running batched ranking for {n_voters} voters with {len(new_statements)} new statements")
    
    def process_voter(voter_idx: int) -> Dict[str, int]:
        return rank_with_new_statements(
            persona=voter_personas[voter_idx],
            original_statements=original_statements,
            new_statements=new_statements,
            topic=topic,
            openai_client=openai_client,
            voter_idx=voter_idx,
            hash_seed=hash_seed,
            voter_dist=voter_dist,
            alt_dist=alt_dist,
            rep=rep,
        )
    
    # Run in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_voter, i): i for i in range(n_voters)}
        
        for future in tqdm(as_completed(futures), total=n_voters,
                          desc="Batched ranking", unit="voter"):
            voter_idx = futures[future]
            try:
                positions = future.result()
                for key, pos in positions.items():
                    all_positions[key][voter_idx] = pos
            except Exception as e:
                logger.error(f"Voter {voter_idx} failed: {e}")
    
    return all_positions


def compute_epsilon_from_positions(
    positions: List[int],
    original_statements: List[Dict],
    n_voters: int = 100,
) -> Optional[float]:
    """
    Compute epsilon for a new statement given its positions in each voter's ranking.
    
    This converts positions to a preference matrix format and uses the custom
    epsilon computation that treats the new statement as an additional alternative.
    
    Args:
        positions: List of positions (0-100) for each voter
        original_statements: The 100 original statements
        n_voters: Number of voters (should match len(positions))
    
    Returns:
        Critical epsilon value, or None if computation fails
    """
    n_originals = len(original_statements)
    
    # Filter out None positions
    valid_positions = [(i, p) for i, p in enumerate(positions) if p is not None]
    if len(valid_positions) < n_voters:
        logger.warning(f"Only {len(valid_positions)}/{n_voters} valid positions")
        if len(valid_positions) == 0:
            return None
    
    # Build a preference matrix with n_originals + 1 alternatives
    # The new statement is at index n_originals
    n_total = n_originals + 1
    new_stmt_idx = n_originals
    
    # Create preference matrix: [rank][voter] = alternative_index (as string)
    preferences: List[List[str]] = [[""] * n_voters for _ in range(n_total)]
    
    for voter_idx, position in valid_positions:
        # Position is where the new statement falls among originals
        # We need to build a full ranking: originals 0 to n_originals-1, plus new_stmt_idx
        
        # For simplicity, we assume original statements are ranked 0, 1, 2, ..., n_originals-1
        # and we insert the new statement at the given position
        
        # Build the ranking for this voter
        ranking = list(range(n_originals))  # Original indices
        ranking.insert(position, new_stmt_idx)  # Insert new statement
        
        # Write to preferences matrix
        for rank, alt_idx in enumerate(ranking):
            preferences[rank][voter_idx] = str(alt_idx)
    
    # Handle voters with missing data (fill with new statement at bottom)
    missing_voters = [i for i in range(n_voters) if positions[i] is None]
    for voter_idx in missing_voters:
        ranking = list(range(n_originals)) + [new_stmt_idx]
        for rank, alt_idx in enumerate(ranking):
            preferences[rank][voter_idx] = str(alt_idx)
    
    # Compute epsilon for the new statement
    alternatives = [str(i) for i in range(n_total)]
    winner = str(new_stmt_idx)
    
    try:
        epsilon = compute_critical_epsilon_custom(
            preferences, alternatives, winner, m_override=n_originals
        )
        return epsilon
    except Exception as e:
        logger.error(f"Epsilon computation failed: {e}")
        return None
