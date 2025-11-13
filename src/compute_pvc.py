"""
Compute Proportional Veto Core (PVC) using veto-by-consumption algorithm.
Translated from reference.ts
"""

from typing import List


def compute_pvc(preferences: List[List[str]], alternatives: List[str]) -> List[str]:
    """
    Compute the Proportional Veto Core (PVC) for a given preference profile by successive elimination.
    
    Args:
        preferences: Matrix where preferences[rank][voter] is the alternative at rank 'rank' for voter 'voter'
        alternatives: List of all alternative strings
    
    Returns:
        Array of alternatives in the PVC
    """
    m = len(alternatives)
    n = len(preferences[0]) if preferences else 0  # number of voters (columns)
    
    if m == 0 or n == 0:
        return []
    
    # Map alternatives to indices for number representation
    alt_to_index = {alt: idx for idx, alt in enumerate(alternatives)}
    
    # Convert preference matrix to profile format (each voter's complete ordering)
    # Note that `profile` is [voter][alternative] while `preferences` is the transpose
    profile: List[List[int]] = []
    for voter in range(n):
        voter_prefs: List[int] = []
        for rank in range(m):
            alt = preferences[rank][voter]
            index = alt_to_index.get(alt, -1)
            if index == -1:
                raise ValueError(f"Alternative '{alt}' not found in alternatives list")
            voter_prefs.append(index)
        profile.append(voter_prefs)
    
    # Veto by consumption
    # Initialize each alternative tank
    tanks = [1.0] * m
    remaining_alts = set(range(m))
    eps = 1e-9
    
    # Run the clock
    while len(remaining_alts) > 1:
        # Count voters "eating" from each alternative (least preferred)
        num_voter_eating = [0.0] * m
        for voter in range(n):
            voter_profile = profile[voter]
            if len(voter_profile) > 0:
                least_preferred = voter_profile[-1]
                num_voter_eating[least_preferred] += 1
        
        # Find t_delta (minimum time until next elimination)
        t_delta = 1.0
        for alt in range(m):
            if tanks[alt] == 0 or num_voter_eating[alt] == 0:
                continue
            t_delta = min(t_delta, tanks[alt] / num_voter_eating[alt])
        
        # Let t_delta pass
        eliminated_now = []
        for alt in range(m):
            if tanks[alt] == 0:
                continue
            tanks[alt] -= t_delta * num_voter_eating[alt]
            if tanks[alt] < eps:
                tanks[alt] = 0
                remaining_alts.discard(alt)
                eliminated_now.append(alt)
        
        # Remove eliminated alternatives from voter rankings
        for voter in range(n):
            while len(profile[voter]) > 0 and tanks[profile[voter][-1]] == 0:
                profile[voter].pop()
        
        if len(remaining_alts) == 0:
            # Ties - return all that were eliminated in this round
            return [alternatives[idx] for idx in eliminated_now]
    
    # Return remaining alternatives
    return [alternatives[idx] for idx in remaining_alts]

