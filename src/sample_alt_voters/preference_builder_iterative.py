"""
Preference builder using iterative A*-low ranking.

Wraps the degeneracy-mitigated iterative ranking (A*-low) to build
full preference matrices (100 voters × 100 alternatives).

Key features:
- Uses 5 rounds of top-K/bottom-K selection per voter
- Hash identifiers to prevent index/rank conflation
- Per-round shuffling to break presentation order bias
- Bottom-K requested as "least preferred first" (A* variant)
- Parallel execution with configurable workers
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from openai import OpenAI
from tqdm import tqdm

from src.degeneracy_mitigation.iterative_ranking_star import rank_voter
from src.degeneracy_mitigation.config import HASH_SEED

logger = logging.getLogger(__name__)


def build_full_preferences_iterative(
    voter_personas: List[str],
    statements: List[Dict],
    topic: str,
    openai_client: OpenAI,
    reasoning_effort: str = "low",
    max_workers: int = 50,
    hash_seed: int = HASH_SEED,
    show_progress: bool = True
) -> Tuple[List[List[str]], Dict]:
    """
    Build full preference matrix using iterative A*-low ranking.
    
    Each voter ranks all statements through 5 rounds of top-K/bottom-K selection.
    This approach achieves 97% valid rankings with near-zero presentation order bias.
    
    Args:
        voter_personas: List of voter persona description strings
        statements: List of statement dicts with 'statement' key
        topic: Topic question string
        openai_client: OpenAI client instance
        reasoning_effort: Reasoning effort level ("low" recommended for A*-low)
        max_workers: Maximum parallel workers for API calls
        hash_seed: Seed for hash identifier generation
        show_progress: Whether to show progress bar
        
    Returns:
        Tuple of (preferences, stats):
        - preferences: Preference matrix where preferences[rank][voter] is the 
          alternative index (as string) at that rank for that voter
        - stats: Dict with statistics (valid_count, total_retries, etc.)
    """
    n_voters = len(voter_personas)
    n_alts = len(statements)
    
    logger.info(f"Building preference matrix: {n_voters} voters × {n_alts} alternatives")
    logger.info(f"Using A*-low iterative ranking with reasoning_effort={reasoning_effort}")
    logger.info(f"5 API rounds per voter = {n_voters * 5} total API calls")
    
    def process_voter(args):
        """Process a single voter and return (index, result)."""
        idx, persona = args
        result = rank_voter(
            client=openai_client,
            voter_idx=idx,
            persona=persona,
            statements=statements,
            topic=topic,
            reasoning_effort=reasoning_effort,
            hash_seed=hash_seed
        )
        return idx, result
    
    # Process all voters in parallel
    results = [None] * n_voters
    total_retries = 0
    valid_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_voter, (i, persona)): i
            for i, persona in enumerate(voter_personas)
        }
        
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), 
                          desc="Building preferences", unit="voter")
        
        for future in iterator:
            idx, result = future.result()
            results[idx] = result
            total_retries += result.get('total_retries', 0)
            if result.get('all_valid', False):
                valid_count += 1
    
    # Convert to preference matrix format [rank][voter]
    # Each result['ranking'] is a list of statement indices in preference order
    # IMPORTANT: We must ensure no duplicates in rankings for epsilon calculation
    preferences = []
    invalid_voter_indices = []
    
    # First pass: build initial matrix
    for rank in range(n_alts):
        rank_row = []
        for voter in range(n_voters):
            if results[voter] and 'ranking' in results[voter]:
                ranking = results[voter]['ranking']
                if rank < len(ranking):
                    rank_row.append(str(ranking[rank]))
                else:
                    rank_row.append("-1")
            else:
                rank_row.append("-1")
        preferences.append(rank_row)
    
    # Second pass: identify invalid voters (duplicates, -1s, wrong length)
    for voter in range(n_voters):
        voter_ranking = [preferences[rank][voter] for rank in range(n_alts)]
        has_invalid = "-1" in voter_ranking
        has_duplicates = len(voter_ranking) != len(set(voter_ranking))
        if has_invalid or has_duplicates:
            invalid_voter_indices.append(voter)
    
    # Log invalid voters but do NOT replace with random data
    if invalid_voter_indices:
        logger.warning(f"Found {len(invalid_voter_indices)} voters with invalid/duplicate rankings: {invalid_voter_indices}")
        valid_count = max(0, n_voters - len(invalid_voter_indices))
    
    # Compile stats
    stats = {
        "n_voters": n_voters,
        "n_alternatives": n_alts,
        "valid_count": valid_count,
        "invalid_count": n_voters - valid_count,
        "valid_rate": valid_count / n_voters if n_voters > 0 else 0,
        "total_retries": total_retries,
        "reasoning_effort": reasoning_effort,
    }
    
    logger.info(f"Built preference matrix: {len(preferences)} × {len(preferences[0])}")
    logger.info(f"Valid rankings: {valid_count}/{n_voters} ({100*stats['valid_rate']:.1f}%)")
    logger.info(f"Total retries: {total_retries}")
    
    return preferences, stats


def save_preferences(
    preferences: List[List[str]], 
    stats: Dict,
    output_dir: Path
) -> None:
    """
    Save preferences and stats to JSON files.
    
    Args:
        preferences: Preference matrix [rank][voter]
        stats: Statistics dict
        output_dir: Directory to save to
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "preferences.json", 'w') as f:
        json.dump(preferences, f)
    
    with open(output_dir / "preference_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved preferences to {output_dir}")


def validate_preferences(
    preferences: List[List[str]]
) -> Tuple[List[int], Dict[str, any]]:
    """
    Validate preference matrix and identify invalid voters.
    
    Checks for:
    - Duplicate alternatives in voter rankings
    - Invalid values ("-1")
    - Wrong number of alternatives
    
    Does NOT modify the preferences - just reports issues.
    
    Args:
        preferences: Preference matrix [rank][voter]
        
    Returns:
        Tuple of (invalid_voter_indices, validation_info)
    """
    n_alts = len(preferences)
    n_voters = len(preferences[0]) if preferences else 0
    
    invalid_voters = []
    voters_with_duplicates = []
    voters_with_invalid_values = []
    
    for voter in range(n_voters):
        voter_ranking = [preferences[rank][voter] for rank in range(n_alts)]
        has_invalid = "-1" in voter_ranking
        has_duplicates = len(voter_ranking) != len(set(voter_ranking))
        
        if has_invalid:
            voters_with_invalid_values.append(voter)
        if has_duplicates:
            voters_with_duplicates.append(voter)
        if has_invalid or has_duplicates:
            invalid_voters.append(voter)
    
    validation_info = {
        "n_voters": n_voters,
        "n_alternatives": n_alts,
        "n_invalid": len(invalid_voters),
        "n_valid": n_voters - len(invalid_voters),
        "voters_with_duplicates": voters_with_duplicates,
        "voters_with_invalid_values": voters_with_invalid_values,
    }
    
    if invalid_voters:
        logger.warning(f"Found {len(invalid_voters)} invalid voters: {invalid_voters}")
    
    return invalid_voters, validation_info


def load_preferences(
    output_dir: Path,
    validate: bool = True
) -> Tuple[List[List[str]], Dict]:
    """
    Load preferences and stats from JSON files.
    
    Args:
        output_dir: Directory to load from
        validate: If True, validate and report invalid rankings
        
    Returns:
        Tuple of (preferences, stats)
    """
    with open(output_dir / "preferences.json") as f:
        preferences = json.load(f)
    
    stats_path = output_dir / "preference_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    else:
        stats = {}
    
    # Validate if requested
    if validate:
        invalid_voters, validation_info = validate_preferences(preferences)
        stats["invalid_voters"] = invalid_voters
        stats["validation_info"] = validation_info
    
    return preferences, stats


def subsample_preferences(
    preferences: List[List[str]],
    k_voters: int = 20,
    p_alts: int = 20,
    voter_indices: Optional[List[int]] = None,
    alt_indices: Optional[List[int]] = None,
    seed: int = None
) -> Tuple[List[List[str]], List[int], List[int]]:
    """
    Subsample a preference matrix to k voters × p alternatives.
    
    Used for mini-rep evaluation (20×20 from 100×100).
    
    Args:
        preferences: Full preference matrix [rank][voter]
        k_voters: Number of voters to sample
        p_alts: Number of alternatives to sample
        voter_indices: Specific voter indices to use (overrides k_voters)
        alt_indices: Specific alternative indices to use (overrides p_alts)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (subsampled_preferences, voter_indices, alt_indices)
    """
    import random
    
    n_ranks = len(preferences)
    n_voters = len(preferences[0]) if preferences else 0
    
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Sample voter indices
    if voter_indices is None:
        voter_indices = rng.sample(range(n_voters), min(k_voters, n_voters))
    
    # Sample alternative indices
    if alt_indices is None:
        alt_indices = rng.sample(range(n_ranks), min(p_alts, n_ranks))
    alt_set = set(str(a) for a in alt_indices)
    
    # Build mapping from old alt index to new alt index
    alt_mapping = {str(old): str(new) for new, old in enumerate(alt_indices)}
    
    # Extract subsampled preferences
    # For each sampled voter, filter to only include sampled alternatives
    # and remap indices
    subsampled = []
    for voter_idx in voter_indices:
        # Get this voter's full ranking
        voter_ranking = [preferences[rank][voter_idx] for rank in range(n_ranks)]
        
        # Filter to only sampled alternatives and remap
        filtered_ranking = []
        for alt in voter_ranking:
            if alt in alt_set:
                filtered_ranking.append(alt_mapping[alt])
        
        subsampled.append(filtered_ranking)
    
    # Convert to [rank][voter] format
    result = []
    for rank in range(len(alt_indices)):
        rank_row = []
        for voter_ranking in subsampled:
            if rank < len(voter_ranking):
                rank_row.append(voter_ranking[rank])
            else:
                rank_row.append("-1")
        result.append(rank_row)
    
    return result, voter_indices, alt_indices
