"""
Get preference rankings from personas using OpenAI API.
"""

from typing import List, Dict
from openai import OpenAI


def get_preference_rankings(summaries: List[Dict], personas: List[Dict], openai_client: OpenAI = None) -> List[List[str]]:
    """
    Get preference rankings from 10 personas on the summaries.
    
    Args:
        summaries: List of summary dicts (from generate_summaries)
        personas: List of 10 personas (different from statement generators)
        openai_client: OpenAI client instance
    
    Returns:
        Preference matrix where preferences[rank][voter] is the alternative at rank 'rank' for voter 'voter'
        Each alternative is identified by its index (0-9) as a string
    """
    if openai_client is None:
        raise ValueError("OpenAI client is required")
    
    if len(personas) != 10:
        raise ValueError(f"Expected 10 personas, got {len(personas)}")
    
    if len(summaries) != 10:
        raise ValueError(f"Expected 10 summaries, got {len(summaries)}")
    
    # Format summaries with indices
    summaries_text = "\n\n".join([
        f"Summary {i}: {summary['summary']}"
        for i, summary in enumerate(summaries)
    ])
    
    rankings = []
    
    for persona in personas:
        persona_desc = f"""Name: {persona.get('name', 'Unknown')}
Background: {persona.get('background', 'Not specified')}
Perspective: {persona.get('perspective', 'Not specified')}
Values: {persona.get('values', 'Not specified')}
Communication Style: {persona.get('communication_style', 'Not specified')}"""
        
        prompt = f"""You are {persona.get('name', 'a person')} with the following characteristics:

{persona_desc}

Here are 10 summaries of a discussion:

{summaries_text}

Rank these summaries from most preferred to least preferred (1 = most preferred, 10 = least preferred).
Consider which summaries best align with your values, perspective, and what you think would be good consensus statements.

Return your ranking as a JSON object with this exact format:
{{"ranking": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}}

Where the array contains the summary indices (0-9) in order from most preferred (first) to least preferred (last).
Return only the JSON, no additional text."""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are {persona.get('name', 'a person')} with the characteristics described. Return ONLY valid JSON, no other text."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.5
            )
        except Exception:
            # Fallback if response_format not supported
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are {persona.get('name', 'a person')} with the characteristics described. Return ONLY valid JSON, no other text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
        
        import json
        result = json.loads(response.choices[0].message.content)
        ranking = result.get("ranking", [])
        
        # Validate ranking
        if len(ranking) != 10:
            raise ValueError(f"Invalid ranking length: {len(ranking)}, expected 10")
        if set(ranking) != set(range(10)):
            raise ValueError(f"Invalid ranking: must contain all indices 0-9, got {ranking}")
        
        # Convert to strings for PVC format
        ranking_str = [str(idx) for idx in ranking]
        rankings.append(ranking_str)
    
    # Convert to matrix format: preferences[rank][voter]
    # Each ranking is a list of summary indices from most to least preferred
    # We need to transpose: for each rank position, which summary index is at that rank for each voter
    m = len(summaries)  # number of alternatives
    n = len(personas)   # number of voters
    
    preferences = []
    for rank in range(m):
        rank_row = []
        for voter in range(n):
            # The summary index at this rank for this voter
            summary_idx = rankings[voter][rank]
            rank_row.append(summary_idx)
        preferences.append(rank_row)
    
    return preferences

