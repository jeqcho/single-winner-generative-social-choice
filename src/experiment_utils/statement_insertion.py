"""
Statement insertion into existing rankings using RANKING_MODEL.

DEPRECATED: This single-call insertion method has known position bias.
Use batched iterative ranking instead:
  src/experiment_utils/batched_iterative_insertion.py

This module is kept for backward compatibility but should not be used
for new experiments.
"""

import json
import logging
import time
import warnings
from typing import List, Dict
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import RANKING_MODEL, RANKING_REASONING, TEMPERATURE, api_timer, build_api_metadata
from src.degeneracy_mitigation.hash_identifiers import id_to_hash
from src.degeneracy_mitigation.config import HASH_SEED

logger = logging.getLogger(__name__)

# =============================================================================
# DEPRECATION WARNING - This module is deprecated!
# =============================================================================
warnings.warn(
    "\n" + "=" * 70 + "\n"
    "DEPRECATION WARNING: statement_insertion.py is deprecated!\n"
    "This single-call insertion method has known position bias.\n"
    "Use batched iterative ranking instead:\n"
    "  src/experiment_utils/batched_iterative_insertion.py\n"
    + "=" * 70,
    DeprecationWarning,
    stacklevel=2
)


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
    # Runtime deprecation warning
    print("\n" + "!" * 70)
    print("WARNING: Using deprecated single-call insertion!")
    print("This method has position bias. Use batched iterative ranking instead:")
    print("  src/experiment_utils/batched_iterative_insertion.py")
    print("!" * 70 + "\n")
    
    n = len(current_ranking)
    new_idx = len(statements)  # New statement gets the next available index
    
    # Generate hash for new statement
    new_hash = id_to_hash(new_idx, HASH_SEED)
    
    # Build the ranked statements for context with hash codes
    ranked_text = "\n".join(
        f"Rank {i+1} ({id_to_hash(idx, HASH_SEED)}): \"{statements[idx]['statement']}\""
        for i, idx in enumerate(current_ranking)
    )
    
    system_prompt = f"""You are simulating a single, internally consistent person defined by the following persona:
{persona}

You must evaluate each statement solely through the lens of this persona's values, background, beliefs, and preferences.

Your task is to determine where a new statement fits in your preference ranking and return valid JSON only.
Do not include explanations, commentary, or extra text."""
    
    user_prompt = f"""Topic: "{topic}"

Here is your current ranking from most to least preferred (identified by 4-letter codes):
{ranked_text}

NEW STATEMENT ({new_hash}): "{new_statement}"

Where should this new statement be inserted in your ranking?
- Return 0 to place it MOST preferred (before rank 1)
- Return {n} to place it LEAST preferred (after rank {n})
- Return any position 1-{n-1} to insert it between existing ranks

IMPORTANT: Your decision should reflect your persona's values and background.

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
