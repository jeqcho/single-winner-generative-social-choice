"""
Epsilon precomputation and lookup for the sampling experiment.

Precomputes epsilon for all 100 alternatives against the full 100x100 profile,
then provides fast lookup when voting methods return winners.

Includes a custom epsilon computation that allows overriding m for new statements.
"""

import json
import logging
from typing import List, Dict, Optional, Sequence
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from pvc_toolbox import compute_critical_epsilon

# Import FlowNetwork for custom epsilon computation
try:
    from pvc_toolbox._flow import FlowNetwork
    _HAS_FLOW = True
except ImportError:
    _HAS_FLOW = False

from .config import N_ALT_POOL

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Epsilon Computation with m Override
# =============================================================================

def _compute_critical_epsilon_with_m(
    alternative: str,
    profile: List[List[str]],
    candidates: List[str],
    n: int,
    m_actual: int,
    m_for_veto: int,
) -> float:
    """
    Compute critical epsilon with a custom m value for veto power.
    
    This allows computing epsilon for a new statement (index 100) while
    using m=100 for the veto power calculation so the new statement
    doesn't get extra veto power.
    
    Args:
        alternative: The alternative to compute epsilon for
        profile: List of voter rankings (each is a list of alternatives)
        candidates: List of all candidate names (including new statement)
        n: Number of voters
        m_actual: Actual number of alternatives in the profile
        m_for_veto: m value to use for veto power calculation
    
    Returns:
        Critical epsilon value
    """
    if not _HAS_FLOW:
        raise ImportError("pvc_toolbox._flow not available for custom epsilon computation")
    
    # Build flow network
    # Node indexing:
    #   0              : source S
    #   1..n           : voter nodes
    #   n+1..n+(m_actual-1) : candidate nodes (excluding the target alternative)
    #   last           : sink T
    sink_index = 1 + n + (m_actual - 1)
    network = FlowNetwork(sink_index + 1)
    
    S = 0
    T = sink_index
    
    # Precompute positions for each voter
    pos: List[Dict[str, int]] = []
    for voter_ranking in profile:
        pos.append({c: i for i, c in enumerate(voter_ranking)})
    
    # Add S -> voter edges with capacity r = m_for_veto (not m_actual)
    # This is the key change - we use m_for_veto for capacity
    for vi in range(n):
        network.add_edge(S, 1 + vi, m_for_veto)
    
    # Map candidates (except alternative) to node IDs
    cand_to_node: Dict[str, int] = {}
    node_cursor = 1 + n
    for d in candidates:
        if d == alternative:
            continue
        cand_to_node[d] = node_cursor
        network.add_edge(node_cursor, T, n)
        node_cursor += 1
    
    # For each voter, connect to candidates ranked WORSE than alternative
    INF = n * m_actual
    for vi in range(n):
        v_node = 1 + vi
        if alternative not in pos[vi]:
            # Alternative not in this voter's ranking - skip
            continue
        rank_alt = pos[vi][alternative]
        worse_tail = profile[vi][rank_alt + 1:]
        for d in worse_tail:
            if d in cand_to_node:
                d_node = cand_to_node[d]
                network.add_edge(v_node, d_node, INF)
    
    # Compute max flow
    F = network.max_flow(S, T)
    
    # Compute critical epsilon using m_for_veto
    # Note: total_vertices = source_capacity + sink_capacity
    #   source_capacity = n * m_for_veto (we use m_for_veto for veto power)
    #   sink_capacity = (m_actual - 1) * n (actual number of candidate nodes)
    total_vertices = m_for_veto * n + (m_actual - 1) * n
    S_a = total_vertices - F
    epsilon_star = (S_a / (m_for_veto * n)) - 1.0
    
    return epsilon_star


def compute_critical_epsilon_custom(
    preferences: Sequence[Sequence[str]],
    alternatives: Sequence[str],
    alternative: str,
    m_override: Optional[int] = None,
) -> float:
    """
    Compute critical epsilon with optional m override.
    
    This is like pvc_toolbox.compute_critical_epsilon but allows
    specifying a different m value for the veto power calculation.
    
    Args:
        preferences: Preference matrix [rank][voter]
        alternatives: List of all alternative names
        alternative: The alternative to compute epsilon for
        m_override: If provided, use this m for veto power calculation
                   instead of len(alternatives)
    
    Returns:
        Critical epsilon value
    """
    if alternative not in alternatives:
        raise ValueError(f"Alternative '{alternative}' is not in the alternatives list")
    
    # Convert preferences to profile format (list of voter rankings)
    n_ranks = len(preferences)
    n_voters = len(preferences[0]) if preferences else 0
    
    profile: List[List[str]] = []
    for voter in range(n_voters):
        ranking = [preferences[rank][voter] for rank in range(n_ranks)]
        profile.append(ranking)
    
    candidates = list(alternatives)
    n = n_voters
    m_actual = len(candidates)
    m_for_veto = m_override if m_override is not None else m_actual
    
    # Trivial case
    if m_actual == 1:
        return -1.0
    
    return _compute_critical_epsilon_with_m(
        alternative, profile, candidates, n, m_actual, m_for_veto
    )


def compute_epsilon_for_alternative(
    preferences: List[List[str]],
    alt_index: int,
    n_alternatives: int = None
) -> Optional[float]:
    """
    Compute the critical epsilon for a single alternative.
    
    Args:
        preferences: Preference matrix [rank][voter]
        alt_index: Index of the alternative to compute epsilon for
        n_alternatives: Number of alternatives (for generating alternatives list)
    
    Returns:
        Critical epsilon value, or None if computation fails
    """
    if n_alternatives is None:
        n_alternatives = len(preferences)
    
    alternatives = [str(i) for i in range(n_alternatives)]
    winner = str(alt_index)
    
    try:
        epsilon = compute_critical_epsilon(preferences, alternatives, winner)
        return epsilon
    except Exception as e:
        logger.error(f"Epsilon computation failed for alt {alt_index}: {e}")
        return None


def precompute_all_epsilons(
    preferences: List[List[str]],
    max_workers: int = 10
) -> Dict[str, float]:
    """
    Precompute epsilon for all alternatives in the preference profile.
    
    Args:
        preferences: Full preference matrix [rank][voter] (100x100)
        max_workers: Maximum parallel workers
    
    Returns:
        Dict mapping alternative index (as string) to epsilon value
    """
    n_alternatives = len(preferences)
    
    logger.info(f"Precomputing epsilon for all {n_alternatives} alternatives...")
    
    epsilons = {}
    
    # Compute epsilon for each alternative
    # Note: pvc_toolbox is CPU-bound, so parallelization helps
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(compute_epsilon_for_alternative, preferences, i, n_alternatives): i
            for i in range(n_alternatives)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="Computing epsilons", unit="alt"):
            alt_idx = futures[future]
            try:
                epsilon = future.result()
                epsilons[str(alt_idx)] = epsilon
            except Exception as e:
                logger.error(f"Failed to compute epsilon for alt {alt_idx}: {e}")
                epsilons[str(alt_idx)] = None
    
    # Log statistics
    valid_epsilons = [e for e in epsilons.values() if e is not None]
    if valid_epsilons:
        mean_eps = sum(valid_epsilons) / len(valid_epsilons)
        min_eps = min(valid_epsilons)
        max_eps = max(valid_epsilons)
        logger.info(f"Epsilon stats: mean={mean_eps:.4f}, min={min_eps:.4f}, max={max_eps:.4f}")
    
    return epsilons


def lookup_epsilon(
    epsilons: Dict[str, float],
    winner: str
) -> Optional[float]:
    """
    Look up precomputed epsilon for a winner.
    
    Args:
        epsilons: Precomputed epsilon dict
        winner: Winner index (as string)
    
    Returns:
        Epsilon value or None
    """
    return epsilons.get(winner)


def compute_epsilon_for_new_statement(
    preferences: List[List[str]],
    new_statement_index: int,
    m_for_epsilon: int = N_ALT_POOL
) -> Optional[float]:
    """
    Compute epsilon for a newly generated statement (ChatGPT**).
    
    The new statement is inserted into the preference profile, but we keep
    m=100 for epsilon calculation so the new alt doesn't have veto power.
    
    Args:
        preferences: Updated preference matrix with new statement inserted
        new_statement_index: Index of the new statement in the updated profile
        m_for_epsilon: Number of alternatives to use for veto power (keep at 100)
    
    Returns:
        Critical epsilon value
    """
    # The preferences matrix now has 101 alternatives (or more)
    n_total = len(preferences)
    
    # Include ALL alternatives in the list (including the new statement)
    alternatives = [str(i) for i in range(n_total)]
    
    # The winner is the new statement
    winner = str(new_statement_index)
    
    try:
        # Use custom epsilon computation with m_override
        # This uses all alternatives but calculates veto power with m=100
        epsilon = compute_critical_epsilon_custom(
            preferences, alternatives, winner, m_override=m_for_epsilon
        )
        return epsilon
    except Exception as e:
        logger.error(f"Epsilon computation failed for new statement: {e}")
        return None


def save_precomputed_epsilons(epsilons: Dict[str, float], output_dir: Path) -> None:
    """Save precomputed epsilons to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "precomputed_epsilons.json", 'w') as f:
        json.dump(epsilons, f, indent=2)
    
    logger.info(f"Saved precomputed epsilons to {output_dir}")


def load_precomputed_epsilons(output_dir: Path) -> Dict[str, float]:
    """Load precomputed epsilons from JSON."""
    with open(output_dir / "precomputed_epsilons.json", 'r') as f:
        return json.load(f)


def get_mean_epsilon(epsilons: Dict[str, float]) -> float:
    """
    Calculate mean epsilon across all alternatives.
    
    Args:
        epsilons: Dict of alternative index -> epsilon
    
    Returns:
        Mean epsilon value
    """
    valid = [e for e in epsilons.values() if e is not None]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)
