"""
Uniform voter sampling from all adult personas.

Samples n_voters uniformly at random from all 815 adult personas.
"""

import random
from typing import List, Tuple


def sample_uniform(
    personas: List[str],
    n_voters: int = 100,
    seed: int = None
) -> Tuple[List[int], List[str]]:
    """
    Sample n_voters uniformly from all personas.
    
    Args:
        personas: List of all persona description strings (815 total)
        n_voters: Number of voters to sample (default: 100)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (persona_indices, persona_texts):
        - persona_indices: List of indices into the original personas list
        - persona_texts: List of sampled persona description strings
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Sample indices without replacement
    all_indices = list(range(len(personas)))
    sampled_indices = rng.sample(all_indices, min(n_voters, len(personas)))
    
    # Get corresponding persona texts
    sampled_personas = [personas[idx] for idx in sampled_indices]
    
    return sampled_indices, sampled_personas
