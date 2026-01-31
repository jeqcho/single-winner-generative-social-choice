"""
Pairwise Borda insertion algorithm for statement ranking.

This is an experimental alternative to chunked insertion where we compare
the new statement against each existing statement individually (pairwise).

Key advantages:
1. Blind comparison - model doesn't know which statement is "new"
2. Simpler task - binary choice easier than position among 20-100
3. No anchoring - each comparison is independent
4. Fine-grained - 99 comparisons provides more signal than 5 chunks

The Borda score is the count of existing statements that the new one beats.
Final position = n_statements - borda_score (more wins = more preferred).
"""

import json
import logging
import random
import time
from typing import List, Dict, Tuple
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import RANKING_MODEL, RANKING_REASONING, TEMPERATURE, api_timer, build_api_metadata

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def compare_statements_pairwise(
    persona: str,
    statement_1: str,
    statement_2: str,
    topic: str,
    openai_client: OpenAI,
    randomize: bool = True,
    model: str = RANKING_MODEL,
    temperature: float = TEMPERATURE,
    voter_dist: str = None,
    alt_dist: str = None,
    method: str = None,
    rep: int = None,
    voter_idx: int = None,
) -> str:
    """
    Compare two statements and return which one the persona prefers.
    
    The statements are presented as "Statement A" and "Statement B" without
    any indication of which is the "new" statement being inserted.
    
    Args:
        persona: Persona string description
        statement_1: First statement text
        statement_2: Second statement text
        topic: The topic/question
        openai_client: OpenAI client instance
        randomize: If True, randomly assign which statement is A vs B
        model: Model to use
        temperature: Temperature for sampling
        voter_dist: Voter distribution (for metadata)
        alt_dist: Alternative distribution (for metadata)
        method: Voting method name (for metadata)
        rep: Replication number (for metadata)
        voter_idx: Voter index (for metadata)
        comparison_idx: Index of this comparison (for metadata)
    
    Returns:
        "1" if statement_1 is preferred, "2" if statement_2 is preferred
    """
    # Optionally randomize which statement is A vs B
    if randomize:
        if random.random() < 0.5:
            statement_a, statement_b = statement_1, statement_2
            a_is_1 = True
        else:
            statement_a, statement_b = statement_2, statement_1
            a_is_1 = False
    else:
        statement_a, statement_b = statement_1, statement_2
        a_is_1 = True
    
    system_prompt = f"""You are simulating a single, internally consistent person defined by the following persona:
{persona}

You must evaluate each statement solely through the lens of this persona's values, background, beliefs, and preferences.

Your task is to compare two statements and indicate which you prefer. Return valid JSON only.
Do not include explanations, commentary, or extra text."""
    
    user_prompt = f"""Topic: "{topic}"

Compare these two statements and indicate which one you prefer:

Statement A: "{statement_a}"

Statement B: "{statement_b}"

Which statement better reflects a position you would support on this topic?

Return JSON: {{"preferred": "A"}} or {{"preferred": "B"}}"""
    
    start_time = time.time()
    response = openai_client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        reasoning={"effort": RANKING_REASONING},
        metadata=build_api_metadata(
            phase="4_insertion",
            component="pairwise_comparison",
            topic=topic,
            voter_dist=voter_dist,
            alt_dist=alt_dist,
            method=method,
            rep=rep,
            voter_idx=voter_idx,
        ),
    )
    api_timer.record(time.time() - start_time)
    
    result = json.loads(response.output_text)
    preferred = result.get("preferred", "A").upper()
    
    # Convert back to "1" or "2" based on which statement was preferred
    if preferred == "A":
        return "1" if a_is_1 else "2"
    else:
        return "2" if a_is_1 else "1"


def insert_statement_pairwise_borda(
    persona: str,
    current_ranking: List[int],
    statements: List[Dict],
    new_statement: str,
    topic: str,
    openai_client: OpenAI,
    model: str = RANKING_MODEL,
    temperature: float = TEMPERATURE,
    voter_dist: str = None,
    alt_dist: str = None,
    method: str = None,
    rep: int = None,
    voter_idx: int = None,
) -> Tuple[List[int], float, int]:
    """
    Insert a new statement using pairwise Borda scoring.
    
    For each existing statement in the ranking, we compare it against the new
    statement. The Borda score is the count of existing statements that the
    new statement beats. Final position = n_statements - borda_score.
    
    Args:
        persona: Persona string description
        current_ranking: Current ranking (list of statement indices, most to least preferred)
        statements: List of statement dicts
        new_statement: The new statement text to insert
        topic: The topic/question
        openai_client: OpenAI client instance
        model: Model to use
        temperature: Temperature for sampling
        voter_dist: Voter distribution (for metadata)
        alt_dist: Alternative distribution (for metadata)
        method: Voting method name (for metadata)
        rep: Replication number (for metadata)
        voter_idx: Voter index (for metadata)
    
    Returns:
        Tuple of:
        - new_ranking: Updated ranking with new statement index inserted
        - borda_score: Normalized Borda score (0-1, where 1 = beats all)
        - final_position: Position in the final ranking (0 = most preferred)
    """
    n = len(current_ranking)
    new_idx = len(statements)  # New statement gets the next available index
    
    # Compare new statement against each existing statement
    wins = 0
    comparison_results = []
    
    for i, existing_idx in enumerate(current_ranking):
        existing_statement = statements[existing_idx]["statement"]
        
        result = compare_statements_pairwise(
            persona=persona,
            statement_1=new_statement,  # statement_1 is the new one
            statement_2=existing_statement,
            topic=topic,
            openai_client=openai_client,
            randomize=True,
            model=model,
            temperature=temperature,
            voter_dist=voter_dist,
            alt_dist=alt_dist,
            method=method,
            rep=rep,
            voter_idx=voter_idx,
        )
        
        # "1" means new statement won, "2" means existing statement won
        new_wins = (result == "1")
        if new_wins:
            wins += 1
        
        comparison_results.append({
            "rank": i,
            "existing_idx": existing_idx,
            "new_wins": new_wins,
        })
    
    # Borda score = number of wins (0 to n)
    borda_score = wins / n  # Normalized to 0-1
    
    # Final position: more wins = more preferred = lower position number
    # position = n - wins means if you beat all (wins=n), you're at position 0
    final_position = n - wins
    
    # Create new ranking with new statement inserted at final_position
    new_ranking = current_ranking.copy()
    new_ranking.insert(final_position, new_idx)
    
    return new_ranking, borda_score, final_position


def get_pairwise_detail(
    persona: str,
    current_ranking: List[int],
    statements: List[Dict],
    new_statement: str,
    topic: str,
    openai_client: OpenAI,
    model: str = RANKING_MODEL,
    temperature: float = TEMPERATURE,
    voter_dist: str = None,
    alt_dist: str = None,
    method: str = None,
    rep: int = None,
    voter_idx: int = None,
) -> Dict:
    """
    Get detailed pairwise comparison results for analysis.
    
    Returns dict with:
    - final_position: Position in the final ranking
    - borda_score: Normalized Borda score (0-1)
    - wins: Raw number of wins
    - total_comparisons: Total number of comparisons made
    - win_rate: wins / total_comparisons
    """
    n = len(current_ranking)
    new_idx = len(statements)
    
    wins = 0
    comparison_details = []
    
    for i, existing_idx in enumerate(current_ranking):
        existing_statement = statements[existing_idx]["statement"]
        
        result = compare_statements_pairwise(
            persona=persona,
            statement_1=new_statement,
            statement_2=existing_statement,
            topic=topic,
            openai_client=openai_client,
            randomize=True,
            model=model,
            temperature=temperature,
            voter_dist=voter_dist,
            alt_dist=alt_dist,
            method=method,
            rep=rep,
            voter_idx=voter_idx,
        )
        
        new_wins = (result == "1")
        if new_wins:
            wins += 1
        
        comparison_details.append({
            "rank": i,
            "existing_idx": existing_idx,
            "new_wins": new_wins,
        })
    
    borda_score = wins / n
    final_position = n - wins
    
    return {
        "final_position": final_position,
        "borda_score": borda_score,
        "wins": wins,
        "total_comparisons": n,
        "win_rate": wins / n,
        "comparison_details": comparison_details,
    }
