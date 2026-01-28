"""
Statement insertion into existing rankings using RANKING_MODEL.

Used for GPT** and GPT*** methods where we need to insert a newly
generated statement into all voter rankings to compute epsilon.
"""

import json
import logging
import time
from typing import List, Dict
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
def insert_statement_into_ranking(
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
    mini_rep: int = None,
    voter_idx: int = None,
) -> List[int]:
    """
    Insert a new statement into an existing ranking.
    
    Used for ChatGPT** where we need to re-query voters to insert
    the newly generated statement.
    
    Args:
        persona: Persona string description
        current_ranking: Current ranking (list of statement indices, most to least preferred)
        statements: Original list of statement dicts
        new_statement: The new statement text to insert
        topic: The topic/question
        openai_client: OpenAI client instance
        model: Model to use
        temperature: Temperature for sampling
        voter_dist: Voter distribution (for metadata)
        alt_dist: Alternative distribution (for metadata)
        method: Voting method name (for metadata)
        rep: Replication number (for metadata)
        mini_rep: Mini-rep index (for metadata)
        voter_idx: Voter index (for metadata)
    
    Returns:
        Updated ranking with new statement index (len(statements)) inserted
    """
    n = len(current_ranking)
    new_idx = len(statements)  # New statement gets the next available index
    
    # Build the ranked statements for context
    ranked_text = "\n".join(
        f"Rank {i+1} (ID {idx}): {statements[idx]['statement']}"
        for i, idx in enumerate(current_ranking)
    )
    
    system_prompt = "You are inserting a new statement into your preference ranking. Return ONLY valid JSON."
    
    user_prompt = f"""You are a person with the following characteristics:
{persona}

Given the topic: "{topic}"

You previously ranked these statements from most to least preferred:

{ranked_text}

NEW STATEMENT (ID {new_idx}): {new_statement}

Where should this new statement be inserted in your ranking?
- Return 0 to make it your MOST preferred (before rank 1)
- Return {n} to make it your LEAST preferred (after rank {n})
- Return any position 1-{n-1} to insert it between existing ranks

Return JSON: {{"insert_position": <number>}}"""

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
            component="statement_insertion",
            topic=topic,
            voter_dist=voter_dist,
            alt_dist=alt_dist,
            method=method,
            rep=rep,
            mini_rep=mini_rep,
            voter_idx=voter_idx,
        ),
    )
    api_timer.record(time.time() - start_time)
    
    result = json.loads(response.output_text)
    position = result.get("insert_position", n // 2)
    
    # Clamp position to valid range
    position = max(0, min(n, int(position)))
    
    # Insert new statement at position
    new_ranking = current_ranking.copy()
    new_ranking.insert(position, new_idx)
    
    return new_ranking
