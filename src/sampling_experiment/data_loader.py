"""
Data loading and sampling functions for the sampling experiment.
"""

import json
import logging
import random
from typing import List, Dict, Tuple
from pathlib import Path

from .config import (
    STATEMENTS_DIR,
    N_VOTER_POOL,
    N_ALT_POOL,
    BASE_SEED,
)

logger = logging.getLogger(__name__)


def load_all_entries(topic_slug: str) -> List[Dict]:
    """
    Load all persona+statement entries for a topic.
    
    Args:
        topic_slug: The topic slug
    
    Returns:
        List of dicts with 'persona' and 'statement' keys
    """
    filepath = STATEMENTS_DIR / f"{topic_slug}.json"
    
    with open(filepath, 'r') as f:
        entries = json.load(f)
    
    logger.info(f"Loaded {len(entries)} entries from {filepath}")
    
    return entries


def sample_pools(
    all_entries: List[Dict],
    n_voters: int = N_VOTER_POOL,
    n_alts: int = N_ALT_POOL,
    seed: int = BASE_SEED
) -> Tuple[List[int], List[int], List[str], List[Dict]]:
    """
    Sample voter pool and alternative pool from all entries.
    
    The voter pool and alternative pool are sampled independently,
    meaning they may have some overlap.
    
    Args:
        all_entries: All persona+statement entries
        n_voters: Number of personas for voter pool
        n_alts: Number of alternatives for alternative pool
        seed: Random seed
    
    Returns:
        Tuple of (voter_indices, alt_indices, voter_personas, alt_statements)
        - voter_indices: Indices into all_entries for voter pool
        - alt_indices: Indices into all_entries for alternative pool
        - voter_personas: List of persona strings for voters
        - alt_statements: List of statement dicts for alternatives
    """
    random.seed(seed)
    
    n_total = len(all_entries)
    
    # Sample voter pool (we use personas from these entries)
    voter_indices = sorted(random.sample(range(n_total), n_voters))
    
    # Sample alternative pool independently (we use statements from these entries)
    # Reset seed to get different sample
    random.seed(seed + 1000)
    alt_indices = sorted(random.sample(range(n_total), n_alts))
    
    # Extract personas and statements
    voter_personas = [all_entries[i]["persona"] for i in voter_indices]
    alt_statements = [{"statement": all_entries[i]["statement"], "original_idx": i} 
                      for i in alt_indices]
    
    logger.info(f"Sampled {n_voters} voters and {n_alts} alternatives")
    
    return voter_indices, alt_indices, voter_personas, alt_statements


def sample_kp(
    n_voters: int,
    n_alts: int,
    k: int,
    p: int,
    seed: int
) -> Tuple[List[int], List[int]]:
    """
    Sample K voters and P alternatives from the pools.
    
    Args:
        n_voters: Total number of voters in pool
        n_alts: Total number of alternatives in pool
        k: Number of voters to sample
        p: Number of alternatives to sample
        seed: Random seed
    
    Returns:
        Tuple of (voter_sample_indices, alt_sample_indices)
        These are indices into the respective pools (0 to n-1)
    """
    random.seed(seed)
    
    voter_sample = sorted(random.sample(range(n_voters), k))
    
    random.seed(seed + 500)
    alt_sample = sorted(random.sample(range(n_alts), p))
    
    return voter_sample, alt_sample


def extract_subprofile(
    full_preferences: List[List[str]],
    voter_indices: List[int],
    alt_indices: List[int]
) -> Tuple[List[List[str]], Dict[int, int]]:
    """
    Extract a K x P subprofile from the full preference matrix.
    
    Args:
        full_preferences: Full preference matrix [rank][voter]
        voter_indices: Indices of voters to include
        alt_indices: Indices of alternatives to include
    
    Returns:
        Tuple of (subprofile, alt_mapping)
        - subprofile: K x P preference matrix with remapped alternative indices
        - alt_mapping: Maps subprofile alt index -> full profile alt index
    """
    # Create mapping from full alt index to subprofile alt index
    alt_set = set(alt_indices)
    full_to_sub = {full_idx: sub_idx for sub_idx, full_idx in enumerate(alt_indices)}
    sub_to_full = {sub_idx: full_idx for sub_idx, full_idx in enumerate(alt_indices)}
    
    # For each sampled voter, extract their ranking over sampled alternatives
    k = len(voter_indices)
    p = len(alt_indices)
    
    # Build subprofile
    # For each voter, we need to preserve their relative ordering of the P alternatives
    subprofile_rankings = []
    
    for voter_idx in voter_indices:
        # Get this voter's full ranking
        full_ranking = [int(full_preferences[rank][voter_idx]) 
                       for rank in range(len(full_preferences))]
        
        # Filter to only include sampled alternatives, preserving order
        sub_ranking = [full_to_sub[alt] for alt in full_ranking if alt in alt_set]
        
        subprofile_rankings.append(sub_ranking)
    
    # Convert to preferences[rank][voter] format
    subprofile = []
    for rank in range(p):
        rank_row = []
        for voter in range(k):
            alt_idx = subprofile_rankings[voter][rank]
            rank_row.append(str(alt_idx))
        subprofile.append(rank_row)
    
    return subprofile, sub_to_full


def save_pool_data(
    voter_indices: List[int],
    alt_indices: List[int],
    voter_personas: List[str],
    alt_statements: List[Dict],
    output_dir: Path
) -> None:
    """Save pool sampling data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = {
        "voter_indices": voter_indices,
        "alt_indices": alt_indices,
        "voter_personas": voter_personas,
        "alt_statements": alt_statements,
    }
    
    with open(output_dir / "pool_data.json", 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved pool data to {output_dir}")


def load_pool_data(output_dir: Path) -> Tuple[List[int], List[int], List[str], List[Dict]]:
    """Load pool sampling data."""
    with open(output_dir / "pool_data.json", 'r') as f:
        data = json.load(f)
    
    return (
        data["voter_indices"],
        data["alt_indices"],
        data["voter_personas"],
        data["alt_statements"],
    )


def save_preferences(preferences: List[List[str]], output_dir: Path) -> None:
    """Save preference matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "full_preferences.json", 'w') as f:
        json.dump(preferences, f, indent=2)
    
    logger.info(f"Saved preferences to {output_dir}")


def load_preferences(output_dir: Path) -> List[List[str]]:
    """Load preference matrix."""
    with open(output_dir / "full_preferences.json", 'r') as f:
        return json.load(f)


def save_epsilons(epsilons: Dict[str, float], output_dir: Path) -> None:
    """Save precomputed epsilons."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "precomputed_epsilons.json", 'w') as f:
        json.dump(epsilons, f, indent=2)
    
    logger.info(f"Saved epsilons to {output_dir}")


def load_epsilons(output_dir: Path) -> Dict[str, float]:
    """Load precomputed epsilons."""
    with open(output_dir / "precomputed_epsilons.json", 'r') as f:
        return json.load(f)


def check_cache_exists(output_dir: Path, filename: str) -> bool:
    """Check if a cache file exists."""
    return (output_dir / filename).exists()
