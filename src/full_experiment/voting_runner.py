"""
Run voting methods and compute epsilon-PVC.
"""

import json
import random
import logging
from typing import List, Dict, Tuple
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
from votekit import RankProfile, RankBallot
from votekit.elections import Plurality, Borda, IRV, RankedPairs
from pvc_toolbox import compute_critical_epsilon

import time

from .config import (
    MODEL,
    TEMPERATURE,
    N_SAMPLE_PERSONAS,
    TOPIC_QUESTIONS,
    VOTING_METHODS,
    api_timer,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Sample Personas
# =============================================================================

def sample_personas_for_voting(
    n_total: int,
    n_sample: int = N_SAMPLE_PERSONAS,
    seed: int = 42
) -> List[int]:
    """
    Sample persona indices for voting.
    
    Args:
        n_total: Total number of personas
        n_sample: Number to sample
        seed: Random seed
    
    Returns:
        List of sampled persona indices
    """
    random.seed(seed)
    indices = sorted(random.sample(range(n_total), n_sample))
    logger.info(f"Sampled {n_sample} personas with seed {seed}")
    return indices


def extract_sampled_preferences(
    preferences: List[List[str]],
    persona_indices: List[int]
) -> List[List[str]]:
    """
    Extract preferences for sampled personas.
    
    Args:
        preferences: Full preference matrix [rank][voter]
        persona_indices: Indices of personas to extract
    
    Returns:
        Sampled preference matrix
    """
    n_statements = len(preferences)
    sampled = []
    for rank in range(n_statements):
        rank_row = [preferences[rank][idx] for idx in persona_indices]
        sampled.append(rank_row)
    return sampled


# =============================================================================
# VoteKit-based Methods
# =============================================================================

def _preferences_to_votekit(preferences: List[List[str]]) -> Tuple[RankProfile, List[str]]:
    """
    Convert preference matrix to VoteKit format.
    
    Args:
        preferences: Matrix where preferences[rank][voter] is statement index
    
    Returns:
        Tuple of (RankProfile, candidate_names)
    """
    n_statements = len(preferences)
    n_voters = len(preferences[0]) if preferences else 0
    
    # Create candidate names
    candidates = [f"c{i}" for i in range(n_statements)]
    
    # Create ballots
    ballots = []
    for voter in range(n_voters):
        ranking = []
        for rank in range(n_statements):
            stmt_idx = int(preferences[rank][voter])
            ranking.append(frozenset([f"c{stmt_idx}"]))
        ballots.append(RankBallot(ranking=tuple(ranking)))
    
    profile = RankProfile(ballots=ballots, candidates=candidates)
    return profile, candidates


def _extract_winner(elected, prefix: str = "c") -> str:
    """Extract winner index from VoteKit election result."""
    if elected and elected[0]:
        winner = list(elected[0])[0]
        if winner and isinstance(winner, str) and winner.startswith(prefix):
            return winner[1:]  # Remove prefix
    return None


def run_schulze(preferences: List[List[str]]) -> Dict:
    """Run Schulze/RankedPairs method (Condorcet)."""
    try:
        profile, _ = _preferences_to_votekit(preferences)
        # Use RankedPairs as a Condorcet method (Schulze not available in this votekit version)
        election = RankedPairs(profile, tiebreak="random")
        elected = election.get_elected()
        winner = _extract_winner(elected)
        return {"winner": winner}
    except Exception as e:
        logger.error(f"Schulze/RankedPairs failed: {e}")
        return {"winner": None, "error": str(e)}


def run_borda(preferences: List[List[str]]) -> Dict:
    """Run Borda count."""
    try:
        profile, _ = _preferences_to_votekit(preferences)
        election = Borda(profile, m=1, tiebreak="random")
        elected = election.get_elected()
        winner = _extract_winner(elected)
        return {"winner": winner}
    except Exception as e:
        logger.error(f"Borda failed: {e}")
        return {"winner": None, "error": str(e)}


def run_irv(preferences: List[List[str]]) -> Dict:
    """Run Instant Runoff Voting."""
    try:
        profile, _ = _preferences_to_votekit(preferences)
        election = IRV(profile, tiebreak="random")
        elected = election.get_elected()
        winner = _extract_winner(elected)
        return {"winner": winner}
    except Exception as e:
        logger.error(f"IRV failed: {e}")
        return {"winner": None, "error": str(e)}


def run_plurality(preferences: List[List[str]]) -> Dict:
    """Run Plurality voting."""
    try:
        profile, _ = _preferences_to_votekit(preferences)
        election = Plurality(profile, m=1, tiebreak="random")
        elected = election.get_elected()
        winner = _extract_winner(elected)
        return {"winner": winner}
    except Exception as e:
        logger.error(f"Plurality failed: {e}")
        return {"winner": None, "error": str(e)}


# =============================================================================
# Veto by Consumption (PVC)
# =============================================================================

def run_veto_by_consumption(preferences: List[List[str]]) -> Dict:
    """Run veto by consumption (PVC) method."""
    try:
        from src.compute_pvc import compute_pvc
        
        n_statements = len(preferences)
        alternatives = [str(i) for i in range(n_statements)]
        
        pvc_result = compute_pvc(preferences, alternatives)
        winner = pvc_result[0] if pvc_result else None
        
        return {"winner": winner}
    except Exception as e:
        logger.error(f"Veto by consumption failed: {e}")
        return {"winner": None, "error": str(e)}


# =============================================================================
# ChatGPT Methods
# =============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def run_chatgpt(
    statements: List[Dict],
    openai_client: OpenAI
) -> Dict:
    """Run ChatGPT baseline selection."""
    statements_text = "\n\n".join([
        f"Statement {i}: {stmt['statement']}"
        for i, stmt in enumerate(statements)
    ])
    
    n = len(statements)
    
    system_prompt = "You are a helpful assistant that selects consensus statements. Return ONLY valid JSON, no other text."
    
    user_prompt = f"""Here are {n} statements from a discussion:

{statements_text}

Which statement do you think would be the best choice as a consensus/bridging statement? 
Consider which one best represents a reasonable middle ground that could satisfy diverse perspectives.

Return your choice as a JSON object with this format:
{{"selected_statement_index": 0}}

Where the value is the index (0-{n-1}) of the statement you select.
Return only the JSON, no additional text."""

    try:
        start_time = time.time()
        response = openai_client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            reasoning={"effort": "minimal"}
        )
        api_timer.record(time.time() - start_time)
        
        result = json.loads(response.output_text)
        winner = str(result.get("selected_statement_index"))
        
        return {"winner": winner}
    except Exception as e:
        logger.error(f"ChatGPT failed: {e}")
        return {"winner": None, "error": str(e)}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def run_chatgpt_with_rankings(
    statements: List[Dict],
    preferences: List[List[str]],
    openai_client: OpenAI
) -> Dict:
    """Run ChatGPT with preference rankings."""
    statements_text = "\n\n".join([
        f"Statement {i}: {stmt['statement']}"
        for i, stmt in enumerate(statements)
    ])
    
    n = len(statements)
    n_voters = len(preferences[0]) if preferences else 0
    
    # Format rankings (show first 10 voters)
    rankings_summary = []
    for voter in range(min(n_voters, 10)):
        ranking = [preferences[rank][voter] for rank in range(len(preferences))]
        rankings_summary.append(f"Voter {voter+1}: {' > '.join(ranking[:10])}...")
    
    if n_voters > 10:
        rankings_summary.append(f"... and {n_voters - 10} more voters")
    
    rankings_text = "\n".join(rankings_summary)
    
    system_prompt = "You are a helpful assistant that selects consensus statements. Return ONLY valid JSON, no other text."
    
    user_prompt = f"""Here are {n} statements from a discussion:

{statements_text}

Here are preference rankings from {n_voters} voters:

{rankings_text}

Based on both the statements and the preference rankings, which statement do you think would be the best choice as a consensus/bridging statement? 
Consider which one best represents a reasonable middle ground that could satisfy diverse perspectives, taking into account how the voters ranked them.

Return your choice as a JSON object with this format:
{{"selected_statement_index": 0}}

Where the value is the index (0-{n-1}) of the statement you select.
Return only the JSON, no additional text."""

    try:
        start_time = time.time()
        response = openai_client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            reasoning={"effort": "minimal"}
        )
        api_timer.record(time.time() - start_time)
        
        result = json.loads(response.output_text)
        winner = str(result.get("selected_statement_index"))
        
        return {"winner": winner}
    except Exception as e:
        logger.error(f"ChatGPT with rankings failed: {e}")
        return {"winner": None, "error": str(e)}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def run_chatgpt_with_personas(
    statements: List[Dict],
    personas: List[str],
    openai_client: OpenAI
) -> Dict:
    """Run ChatGPT with persona descriptions of the voters."""
    statements_text = "\n\n".join([
        f"Statement {i}: {stmt['statement']}"
        for i, stmt in enumerate(statements)
    ])
    
    n = len(statements)
    n_voters = len(personas)
    
    # Format personas
    personas_text = "\n\n".join([
        f"Voter {i+1}: {persona}"
        for i, persona in enumerate(personas)
    ])
    
    system_prompt = "You are a helpful assistant that selects consensus statements. Return ONLY valid JSON, no other text."
    
    user_prompt = f"""Here are {n} statements from a discussion:

{statements_text}

Here are the {n_voters} voters who will be voting on these statements:

{personas_text}

Based on both the statements and the voter personas, which statement do you think would be the best choice as a consensus/bridging statement? 
Consider which one best represents a reasonable middle ground that could satisfy these diverse voters.

Return your choice as a JSON object with this format:
{{"selected_statement_index": 0}}

Where the value is the index (0-{n-1}) of the statement you select.
Return only the JSON, no additional text."""

    try:
        start_time = time.time()
        response = openai_client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            reasoning={"effort": "minimal"}
        )
        api_timer.record(time.time() - start_time)
        
        result = json.loads(response.output_text)
        winner = str(result.get("selected_statement_index"))
        
        return {"winner": winner}
    except Exception as e:
        logger.error(f"ChatGPT with personas failed: {e}")
        return {"winner": None, "error": str(e)}


# =============================================================================
# Epsilon Computation
# =============================================================================

def compute_epsilon_for_winner(
    preferences: List[List[str]],
    winner: str
) -> float:
    """
    Compute the critical epsilon for a winner.
    
    Args:
        preferences: Preference matrix [rank][voter]
        winner: Winner statement index (as string)
    
    Returns:
        Critical epsilon value
    """
    if winner is None:
        return None
    
    n_statements = len(preferences)
    
    # pvc_toolbox expects preferences[rank][voter] - which is exactly what we have
    # alternatives should match the statement indices in preferences
    alternatives = [str(i) for i in range(n_statements)]
    
    try:
        epsilon = compute_critical_epsilon(preferences, alternatives, winner)
        return epsilon
    except Exception as e:
        logger.error(f"Epsilon computation failed for winner {winner}: {e}")
        return None


# =============================================================================
# Run All Methods
# =============================================================================

def run_all_voting_methods(
    preferences: List[List[str]],
    statements: List[Dict],
    openai_client: OpenAI,
    personas: List[str] = None
) -> Dict[str, Dict]:
    """
    Run all voting methods and compute epsilon for each winner.
    
    Args:
        preferences: Preference matrix [rank][voter]
        statements: List of statement dicts
        openai_client: OpenAI client
        personas: Optional list of persona descriptions for sampled voters
    
    Returns:
        Dict mapping method name to result dict with winner and epsilon
    """
    results = {}
    
    # Schulze
    logger.info("Running Schulze...")
    results["schulze"] = run_schulze(preferences)
    
    # Borda
    logger.info("Running Borda...")
    results["borda"] = run_borda(preferences)
    
    # IRV
    logger.info("Running IRV...")
    results["irv"] = run_irv(preferences)
    
    # Plurality
    logger.info("Running Plurality...")
    results["plurality"] = run_plurality(preferences)
    
    # Veto by consumption
    logger.info("Running Veto by Consumption...")
    results["veto_by_consumption"] = run_veto_by_consumption(preferences)
    
    # ChatGPT
    logger.info("Running ChatGPT...")
    results["chatgpt"] = run_chatgpt(statements, openai_client)
    
    # ChatGPT with rankings
    logger.info("Running ChatGPT with rankings...")
    results["chatgpt_with_rankings"] = run_chatgpt_with_rankings(
        statements, preferences, openai_client
    )
    
    # ChatGPT with personas (only if personas provided)
    if personas is not None:
        logger.info("Running ChatGPT with personas...")
        results["chatgpt_with_personas"] = run_chatgpt_with_personas(
            statements, personas, openai_client
        )
    
    # Compute epsilon for each winner
    logger.info("Computing epsilon values...")
    for method, result in results.items():
        winner = result.get("winner")
        if winner is not None:
            epsilon = compute_epsilon_for_winner(preferences, winner)
            result["epsilon"] = epsilon
            logger.info(f"  {method}: winner={winner}, epsilon={epsilon}")
        else:
            result["epsilon"] = None
    
    return results


# =============================================================================
# Save/Load Functions
# =============================================================================

def save_voting_results(results: Dict, output_dir: Path) -> None:
    """Save voting results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved voting results to {output_dir}")


def load_voting_results(output_dir: Path) -> Dict:
    """Load voting results from JSON."""
    with open(output_dir / "results.json", 'r') as f:
        return json.load(f)


def save_sampled_persona_indices(indices: List[int], output_dir: Path) -> None:
    """Save sampled persona indices."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "persona_indices.json", 'w') as f:
        json.dump(indices, f, indent=2)


def save_sampled_preferences(preferences: List[List[str]], output_dir: Path) -> None:
    """Save sampled preferences."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "preferences.json", 'w') as f:
        json.dump(preferences, f, indent=2)

