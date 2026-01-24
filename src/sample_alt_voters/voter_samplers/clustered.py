"""
Ideology-based voter sampling.

Samples voters from specific ideology clusters:
- progressive_liberal: ~431 personas
- conservative_traditional: ~255 personas
"""

import random
from typing import List, Tuple, Dict, Optional

from ..ideology_classifier import load_cluster_assignments


# Cache for cluster assignments
_cluster_cache: Optional[Dict[str, List[int]]] = None


def get_clusters() -> Dict[str, List[int]]:
    """
    Get cached cluster assignments or load from file.
    
    Returns:
        Dict mapping cluster names to lists of persona indices
    """
    global _cluster_cache
    
    if _cluster_cache is None:
        _cluster_cache = load_cluster_assignments()
    
    return _cluster_cache


def sample_from_cluster(
    personas: List[str],
    cluster_name: str,
    n_voters: int = 100,
    seed: int = None
) -> Tuple[List[int], List[str]]:
    """
    Sample n_voters from a specific ideology cluster.
    
    Args:
        personas: List of all persona description strings (815 total)
        cluster_name: One of "progressive_liberal" or "conservative_traditional"
        n_voters: Number of voters to sample (default: 100)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (persona_indices, persona_texts):
        - persona_indices: List of indices into the original personas list
        - persona_texts: List of sampled persona description strings
        
    Raises:
        ValueError: If cluster_name is not valid
    """
    valid_clusters = ["progressive_liberal", "conservative_traditional"]
    if cluster_name not in valid_clusters:
        raise ValueError(
            f"Invalid cluster_name: {cluster_name}. "
            f"Must be one of: {valid_clusters}"
        )
    
    clusters = get_clusters()
    cluster_indices = clusters[cluster_name]
    
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Sample from this cluster
    # If cluster is smaller than n_voters, sample with replacement or just take all
    if len(cluster_indices) >= n_voters:
        sampled_indices = rng.sample(cluster_indices, n_voters)
    else:
        # Cluster too small - sample with replacement
        sampled_indices = rng.choices(cluster_indices, k=n_voters)
    
    # Get corresponding persona texts
    sampled_personas = [personas[idx] for idx in sampled_indices]
    
    return sampled_indices, sampled_personas


def get_cluster_size(cluster_name: str) -> int:
    """
    Get the size of a cluster.
    
    Args:
        cluster_name: One of "progressive_liberal" or "conservative_traditional"
        
    Returns:
        Number of personas in the cluster
    """
    clusters = get_clusters()
    return len(clusters.get(cluster_name, []))
