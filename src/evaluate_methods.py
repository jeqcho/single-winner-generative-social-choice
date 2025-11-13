"""
Evaluate alternative voting methods using VoteKit and OpenAI.
"""

from typing import List, Dict
from openai import OpenAI
from votekit import RankProfile, RankBallot
from votekit.elections import Plurality, Borda, IRV


def evaluate_methods(
    preference_matrix: List[List[str]],
    summaries: List[Dict],
    openai_client: OpenAI = None
) -> Dict:
    """
    Evaluate alternative voting methods on the preference rankings.
    
    Args:
        preference_matrix: Matrix where preferences[rank][voter] is alternative index at rank 'rank' for voter 'voter'
        summaries: List of summary dicts (for ChatGPT selection)
        openai_client: OpenAI client instance
    
    Returns:
        Dict with winner for each method and whether winner is in PVC
    """
    if openai_client is None:
        raise ValueError("OpenAI client is required")
    
    # Convert preference matrix to VoteKit RankProfile format
    # VoteKit expects RankBallot objects
    # Our matrix format: preferences[rank][voter] = alternative index
    # Need to convert to: ballots[voter] = [alt0, alt1, ..., alt9] (most to least preferred)
    
    m = len(preference_matrix)  # number of alternatives (ranks)
    n = len(preference_matrix[0]) if preference_matrix else 0  # number of voters
    
    # Create candidate names (use indices as strings)
    candidates = [str(i) for i in range(m)]
    
    # Convert to VoteKit format: each voter's ranking as RankBallot
    ballots = []
    for voter in range(n):
        ranking = []
        for rank in range(m):
            alt_idx = preference_matrix[rank][voter]
            ranking.append(alt_idx)
        ballots.append(RankBallot(ranking=ranking))
    
    # Create RankProfile
    profile = RankProfile(ballots=ballots, candidates=candidates)
    
    results = {}
    
    # Plurality
    try:
        plurality_election = Plurality(profile, m=1, tiebreak="random")
        elected = plurality_election.get_elected()
        winner = list(elected[0])[0] if elected and elected[0] else None
        results["plurality"] = {
            "winner": winner,
            "in_pvc": None  # Will be set after PVC computation
        }
    except Exception as e:
        results["plurality"] = {"winner": None, "error": str(e), "in_pvc": None}
    
    # Borda Count
    try:
        borda_election = Borda(profile, m=1, tiebreak="random")
        elected = borda_election.get_elected()
        winner = list(elected[0])[0] if elected and elected[0] else None
        results["borda"] = {
            "winner": winner,
            "in_pvc": None
        }
    except Exception as e:
        results["borda"] = {"winner": None, "error": str(e), "in_pvc": None}
    
    # IRV (Instant Runoff Voting)
    try:
        irv_election = IRV(profile, tiebreak="random")
        elected = irv_election.get_elected()
        winner = list(elected[0])[0] if elected and elected[0] else None
        results["irv"] = {
            "winner": winner,
            "in_pvc": None
        }
    except Exception as e:
        results["irv"] = {"winner": None, "error": str(e), "in_pvc": None}
    
    # ChatGPT selection
    summaries_text = "\n\n".join([
        f"Summary {i}: {summary['summary']}"
        for i, summary in enumerate(summaries)
    ])
    
    chatgpt_prompt = f"""Here are 10 summaries of a discussion:

{summaries_text}

Which summary do you think would be the best choice as a consensus/bridging statement? 
Consider which one best represents a reasonable middle ground that could satisfy diverse perspectives.

Return your choice as a JSON object with this format:
{{"selected_summary_index": 0}}

Where the value is the index (0-9) of the summary you select.
Return only the JSON, no additional text."""

    try:
        try:
            chatgpt_response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that selects consensus statements. Return ONLY valid JSON, no other text."},
                    {"role": "user", "content": chatgpt_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.5
            )
        except Exception:
            # Fallback if response_format not supported
            chatgpt_response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that selects consensus statements. Return ONLY valid JSON, no other text."},
                    {"role": "user", "content": chatgpt_prompt}
                ],
                temperature=0.5
            )
        
        import json
        chatgpt_result = json.loads(chatgpt_response.choices[0].message.content)
        selected_idx = chatgpt_result.get("selected_summary_index")
        
        results["chatgpt"] = {
            "winner": str(selected_idx) if selected_idx is not None else None,
            "in_pvc": None
        }
    except Exception as e:
        results["chatgpt"] = {"winner": None, "error": str(e), "in_pvc": None}
    
    return results

