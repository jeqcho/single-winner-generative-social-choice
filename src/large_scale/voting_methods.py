"""
Evaluate alternative voting methods using VoteKit and OpenAI (with ChatGPT variants and successive veto).
"""

import json
from typing import List, Dict
from openai import OpenAI
from votekit import RankProfile, RankBallot
from votekit.elections import Plurality, Borda, IRV, RankedPairs, Schulze


def evaluate_all_methods(
    preference_matrix: List[List[str]],
    statements: List[Dict],
    discriminative_personas: List[str],
    openai_client: OpenAI,
    pvc: List[str] = None
) -> Dict:
    """
    Evaluate all voting methods on the preference rankings.
    
    Args:
        preference_matrix: Matrix where preferences[rank][voter] is alternative index at rank 'rank' for voter 'voter'
        statements: List of statement dicts
        discriminative_personas: List of discriminative persona strings (for ChatGPT+Profiles)
        openai_client: OpenAI client instance
        pvc: List of statement indices in the PVC (precomputed)
    
    Returns:
        Dict with winner for each method
    """
    m = len(preference_matrix)  # number of alternatives (ranks)
    n = len(preference_matrix[0]) if preference_matrix else 0  # number of voters
    
    # Create candidate names with prefix to avoid VoteKit parsing multi-digit numbers as multiple candidates
    # Using format "c0", "c1", ... "c19" instead of "0", "1", ... "19"
    candidates = [f"c{i}" for i in range(m)]
    
    # Convert to VoteKit format: each voter's ranking as RankBallot
    # Each rank position should contain a frozenset of candidates at that rank
    # Since we have no ties, each frozenset contains exactly one candidate
    ballots = []
    for voter in range(n):
        ranking = []
        for rank in range(m):
            alt_idx = int(preference_matrix[rank][voter])  # Convert string to int
            ranking.append(frozenset([f"c{alt_idx}"]))  # Convert to "c{idx}" format
        ballots.append(RankBallot(ranking=tuple(ranking)))
    
    # Create RankProfile
    profile = RankProfile(ballots=ballots, candidates=candidates)
    
    results = {}
    
    # Helper function to extract index from "c{idx}" format
    def extract_winner_idx(winner):
        """Convert winner from 'c{idx}' format back to '{idx}' string."""
        if winner and isinstance(winner, str) and winner.startswith('c'):
            return winner[1:]  # Remove 'c' prefix
        return None
    
    # Plurality
    try:
        plurality_election = Plurality(profile, m=1, tiebreak="random")
        elected = plurality_election.get_elected()
        winner = list(elected[0])[0] if elected and elected[0] else None
        results["plurality"] = {
            "winner": extract_winner_idx(winner),
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
            "winner": extract_winner_idx(winner),
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
            "winner": extract_winner_idx(winner),
            "in_pvc": None
        }
    except Exception as e:
        results["irv"] = {"winner": None, "error": str(e), "in_pvc": None}
    
    # Schulze method (Condorcet method)
    try:
        schulze_election = Schulze(profile, tiebreak="random")
        elected = schulze_election.get_elected()
        winner = list(elected[0])[0] if elected and elected[0] else None
        results["schulze"] = {
            "winner": extract_winner_idx(winner),
            "in_pvc": None
        }
    except Exception as e:
        results["schulze"] = {"winner": None, "error": str(e), "in_pvc": None}
    
    # RankedPairs method (Condorcet method, similar to Schulze)
    try:
        rankedpairs_election = RankedPairs(profile, tiebreak="random")
        elected = rankedpairs_election.get_elected()
        winner = list(elected[0])[0] if elected and elected[0] else None
        results["rankedpairs"] = {
            "winner": extract_winner_idx(winner),
            "in_pvc": None
        }
    except Exception as e:
        results["rankedpairs"] = {"winner": None, "error": str(e), "in_pvc": None}
    
    # Successive Veto (always picks from PVC)
    try:
        from src.compute_pvc import compute_pvc
        alternatives = [str(i) for i in range(m)]
        successive_veto_winners = compute_pvc(preference_matrix, alternatives)
        # Take first winner if multiple in PVC
        successive_veto_winner = successive_veto_winners[0] if successive_veto_winners else None
        results["successive_veto"] = {
            "winner": successive_veto_winner,
            "in_pvc": None  # Will be set after
        }
    except Exception as e:
        results["successive_veto"] = {"winner": None, "error": str(e), "in_pvc": None}
    
    # ChatGPT baseline (just sees statements)
    results["chatgpt"] = chatgpt_select_baseline(statements, openai_client)
    
    # ChatGPT + Rankings
    results["chatgpt_rankings"] = chatgpt_select_with_rankings(
        statements, preference_matrix, openai_client
    )
    
    # ChatGPT + Profiles
    results["chatgpt_profiles"] = chatgpt_select_with_profiles(
        statements, discriminative_personas, openai_client
    )
    
    # ChatGPT + Rankings + Profiles
    results["chatgpt_rankings_profiles"] = chatgpt_select_with_rankings_and_profiles(
        statements, preference_matrix, discriminative_personas, openai_client
    )
    
    # Update in_pvc status for all methods if PVC was provided
    if pvc is not None:
        pvc_set = set(pvc)
        for method, result in results.items():
            if result.get("winner") is not None and "error" not in result:
                results[method]["in_pvc"] = result["winner"] in pvc_set
    
    return results


def chatgpt_select_baseline(
    statements: List[Dict],
    openai_client: OpenAI
) -> Dict:
    """
    ChatGPT baseline selection (just sees statements).
    
    Args:
        statements: List of statement dicts
        openai_client: OpenAI client instance
    
    Returns:
        Dict with winner and in_pvc flag
    """
    statements_text = "\n\n".join([
        f"Statement {i}: {statement['statement']}"
        for i, statement in enumerate(statements)
    ])
    
    prompt = f"""Here are {len(statements)} statements from a discussion:

{statements_text}

Which statement do you think would be the best choice as a consensus/bridging statement? 
Consider which one best represents a reasonable middle ground that could satisfy diverse perspectives.

Return your choice as a JSON object with this format:
{{"selected_statement_index": 0}}

Where the value is the index (0-{len(statements)-1}) of the statement you select.
Return only the JSON, no additional text."""

    try:
        return _execute_chatgpt_selection(prompt, openai_client)
    except Exception as e:
        return {"winner": None, "error": str(e), "in_pvc": None}


def chatgpt_select_with_rankings(
    statements: List[Dict],
    preference_matrix: List[List[str]],
    openai_client: OpenAI
) -> Dict:
    """
    ChatGPT selection with preference rankings.
    
    Args:
        statements: List of statement dicts
        preference_matrix: Preference matrix from discriminative personas
        openai_client: OpenAI client instance
    
    Returns:
        Dict with winner and in_pvc flag
    """
    statements_text = "\n\n".join([
        f"Statement {i}: {statement['statement']}"
        for i, statement in enumerate(statements)
    ])
    
    # Format rankings for display
    n_voters = len(preference_matrix[0]) if preference_matrix else 0
    rankings_summary = []
    for voter in range(min(n_voters, 10)):  # Show first 10 voters to avoid token limit
        ranking = [preference_matrix[rank][voter] for rank in range(len(preference_matrix))]
        rankings_summary.append(f"Voter {voter+1}: {' > '.join(ranking[:10])}... (most to least preferred)")
    
    if n_voters > 10:
        rankings_summary.append(f"... and {n_voters - 10} more voters")
    
    rankings_text = "\n".join(rankings_summary)
    
    prompt = f"""Here are {len(statements)} statements from a discussion:

{statements_text}

Here are preference rankings from {n_voters} voters:

{rankings_text}

Based on both the statements and the preference rankings, which statement do you think would be the best choice as a consensus/bridging statement? 
Consider which one best represents a reasonable middle ground that could satisfy diverse perspectives, taking into account how the voters ranked them.

Return your choice as a JSON object with this format:
{{"selected_statement_index": 0}}

Where the value is the index (0-{len(statements)-1}) of the statement you select.
Return only the JSON, no additional text."""

    try:
        return _execute_chatgpt_selection(prompt, openai_client)
    except Exception as e:
        return {"winner": None, "error": str(e), "in_pvc": None}


def chatgpt_select_with_profiles(
    statements: List[Dict],
    discriminative_personas: List[str],
    openai_client: OpenAI
) -> Dict:
    """
    ChatGPT selection with discriminative persona profiles.
    
    Args:
        statements: List of statement dicts
        discriminative_personas: List of discriminative persona strings
        openai_client: OpenAI client instance
    
    Returns:
        Dict with winner and in_pvc flag
    """
    statements_text = "\n\n".join([
        f"Statement {i}: {statement['statement']}"
        for i, statement in enumerate(statements)
    ])
    
    # Show first 10 personas to avoid token limit
    personas_text = "\n\n".join([
        f"Voter {i+1}: {persona[:200]}..." if len(persona) > 200 else f"Voter {i+1}: {persona}"
        for i, persona in enumerate(discriminative_personas[:10])
    ])
    
    if len(discriminative_personas) > 10:
        personas_text += f"\n\n... and {len(discriminative_personas) - 10} more voters with diverse characteristics"
    
    prompt = f"""Here are {len(statements)} statements from a discussion:

{statements_text}

Here are the characteristics of the {len(discriminative_personas)} voters who will be evaluating these statements:

{personas_text}

Based on both the statements and the voter profiles, which statement do you think would be the best choice as a consensus/bridging statement? 
Consider which one would best satisfy this diverse group of voters.

Return your choice as a JSON object with this format:
{{"selected_statement_index": 0}}

Where the value is the index (0-{len(statements)-1}) of the statement you select.
Return only the JSON, no additional text."""

    try:
        return _execute_chatgpt_selection(prompt, openai_client)
    except Exception as e:
        return {"winner": None, "error": str(e), "in_pvc": None}


def chatgpt_select_with_rankings_and_profiles(
    statements: List[Dict],
    preference_matrix: List[List[str]],
    discriminative_personas: List[str],
    openai_client: OpenAI
) -> Dict:
    """
    ChatGPT selection with both preference rankings and discriminative persona profiles.
    
    Args:
        statements: List of statement dicts
        preference_matrix: Preference matrix from discriminative personas
        discriminative_personas: List of discriminative persona strings
        openai_client: OpenAI client instance
    
    Returns:
        Dict with winner and in_pvc flag
    """
    statements_text = "\n\n".join([
        f"Statement {i}: {statement['statement']}"
        for i, statement in enumerate(statements)
    ])
    
    # Format personas
    personas_text = "\n\n".join([
        f"Voter {i+1}: {persona[:200]}..." if len(persona) > 200 else f"Voter {i+1}: {persona}"
        for i, persona in enumerate(discriminative_personas[:10])
    ])
    
    if len(discriminative_personas) > 10:
        personas_text += f"\n\n... and {len(discriminative_personas) - 10} more voters with diverse characteristics"
    
    # Format rankings
    n_voters = len(preference_matrix[0]) if preference_matrix else 0
    rankings_summary = []
    for voter in range(min(n_voters, 10)):
        ranking = [preference_matrix[rank][voter] for rank in range(len(preference_matrix))]
        rankings_summary.append(f"Voter {voter+1}: {' > '.join(ranking[:10])}... (most to least preferred)")
    
    if n_voters > 10:
        rankings_summary.append(f"... and {n_voters - 10} more voters")
    
    rankings_text = "\n".join(rankings_summary)
    
    prompt = f"""Here are {len(statements)} statements from a discussion:

{statements_text}

Here are the characteristics of the {len(discriminative_personas)} voters:

{personas_text}

Here are their preference rankings:

{rankings_text}

Based on the statements, voter profiles, and their preferences, which statement would be the best choice as a consensus/bridging statement? 
Consider which one would best satisfy this diverse group given what we know about them and their preferences.

Return your choice as a JSON object with this format:
{{"selected_statement_index": 0}}

Where the value is the index (0-{len(statements)-1}) of the statement you select.
Return only the JSON, no additional text."""

    try:
        return _execute_chatgpt_selection(prompt, openai_client)
    except Exception as e:
        return {"winner": None, "error": str(e), "in_pvc": None}


def _execute_chatgpt_selection(prompt: str, openai_client: OpenAI) -> Dict:
    """
    Execute ChatGPT selection with given prompt.
    
    Args:
        prompt: The prompt to send to ChatGPT
        openai_client: OpenAI client instance
    
    Returns:
        Dict with winner and in_pvc flag
    """
    response = openai_client.responses.create(
        model="gpt-5-nano",
        input=[
            {"role": "system", "content": "You are a helpful assistant that selects consensus statements. Return ONLY valid JSON, no other text."},
            {"role": "user", "content": prompt}
        ]
    )
    
    result = json.loads(response.output_text)
    selected_idx = result.get("selected_statement_index")
    
    return {
        "winner": str(selected_idx) if selected_idx is not None else None,
        "in_pvc": None
    }


def evaluate_six_methods(
    preference_matrix: List[List[str]],
    statements: List[Dict],
    openai_client: OpenAI,
    pvc: List = None
) -> Dict:
    """
    Evaluate the 6 voting methods: Plurality, Borda, IRV, ChatGPT, Schulze, Veto by Consumption.
    
    Args:
        preference_matrix: Matrix where preferences[rank][voter] is alternative index at rank 'rank' for voter 'voter'
        statements: List of statement dicts
        openai_client: OpenAI client instance
        pvc: List of statement indices in the PVC (precomputed)
    
    Returns:
        Dict with winner for each method
    """
    m = len(preference_matrix)  # number of alternatives (ranks)
    n = len(preference_matrix[0]) if preference_matrix else 0  # number of voters
    
    # Create candidate names with prefix
    candidates = [f"c{i}" for i in range(m)]
    
    # Convert to VoteKit format
    ballots = []
    for voter in range(n):
        ranking = []
        for rank in range(m):
            alt_idx = int(preference_matrix[rank][voter])
            ranking.append(frozenset([f"c{alt_idx}"]))
        ballots.append(RankBallot(ranking=tuple(ranking)))
    
    profile = RankProfile(ballots=ballots, candidates=candidates)
    
    results = {}
    
    def extract_winner_idx(winner):
        """Convert winner from 'c{idx}' format back to '{idx}' string."""
        if winner and isinstance(winner, str) and winner.startswith('c'):
            return winner[1:]
        return None
    
    # 1. Plurality
    try:
        plurality_election = Plurality(profile, m=1, tiebreak="random")
        elected = plurality_election.get_elected()
        winner = list(elected[0])[0] if elected and elected[0] else None
        results["plurality"] = {"winner": extract_winner_idx(winner), "in_pvc": None}
    except Exception as e:
        results["plurality"] = {"winner": None, "error": str(e), "in_pvc": None}
    
    # 2. Borda
    try:
        borda_election = Borda(profile, m=1, tiebreak="random")
        elected = borda_election.get_elected()
        winner = list(elected[0])[0] if elected and elected[0] else None
        results["borda"] = {"winner": extract_winner_idx(winner), "in_pvc": None}
    except Exception as e:
        results["borda"] = {"winner": None, "error": str(e), "in_pvc": None}
    
    # 3. IRV
    try:
        irv_election = IRV(profile, tiebreak="random")
        elected = irv_election.get_elected()
        winner = list(elected[0])[0] if elected and elected[0] else None
        results["irv"] = {"winner": extract_winner_idx(winner), "in_pvc": None}
    except Exception as e:
        results["irv"] = {"winner": None, "error": str(e), "in_pvc": None}
    
    # 4. ChatGPT
    results["chatgpt"] = chatgpt_select_baseline(statements, openai_client)
    
    # 5. Schulze
    try:
        schulze_election = Schulze(profile, tiebreak="random")
        elected = schulze_election.get_elected()
        winner = list(elected[0])[0] if elected and elected[0] else None
        results["schulze"] = {"winner": extract_winner_idx(winner), "in_pvc": None}
    except Exception as e:
        results["schulze"] = {"winner": None, "error": str(e), "in_pvc": None}
    
    # 6. Veto by Consumption (Successive Veto)
    try:
        from src.compute_pvc import compute_pvc
        alternatives = [str(i) for i in range(m)]
        veto_winners = compute_pvc(preference_matrix, alternatives)
        veto_winner = veto_winners[0] if veto_winners else None
        results["veto_by_consumption"] = {"winner": veto_winner, "in_pvc": None}
    except Exception as e:
        results["veto_by_consumption"] = {"winner": None, "error": str(e), "in_pvc": None}
    
    # Update in_pvc status if PVC was provided
    if pvc is not None:
        pvc_set = set(str(p) for p in pvc)
        for method, result in results.items():
            if result.get("winner") is not None and "error" not in result:
                results[method]["in_pvc"] = str(result["winner"]) in pvc_set
    
    return results

