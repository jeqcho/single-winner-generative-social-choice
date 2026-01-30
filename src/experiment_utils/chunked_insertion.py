"""
Chunked Borda insertion algorithm for statement ranking.

This is an experimental alternative to the standard insertion that addresses
the bias toward too-preferred positions. Instead of asking the model to insert
into all 100 statements at once, we:

1. Split the 100 alternatives into 5 chunks of 20 consecutive statements
2. Ask the model where to insert within each chunk
3. Calculate a Borda score from the 5 insertion positions
4. Use the Borda score to determine the final position

The key insight is that by not revealing which chunk (top/middle/bottom) the
model is seeing, we avoid biasing it toward middle insertions.
"""

import json
import logging
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
def insert_statement_into_chunk(
    persona: str,
    chunk_ranking: List[int],
    statements: List[Dict],
    new_statement: str,
    topic: str,
    openai_client: OpenAI,
    chunk_idx: int = 0,
    model: str = RANKING_MODEL,
    temperature: float = TEMPERATURE,
    voter_dist: str = None,
    alt_dist: str = None,
    method: str = None,
    rep: int = None,
    voter_idx: int = None,
) -> int:
    """
    Ask model where to insert new statement among chunk alternatives.
    
    Uses RANKING_MODEL with RANKING_REASONING effort (same as current insertion).
    
    Args:
        persona: Persona string description
        chunk_ranking: List of statement indices in this chunk (ordered best to worst)
        statements: Full list of statement dicts
        new_statement: The new statement text to insert
        topic: The topic/question
        openai_client: OpenAI client instance
        chunk_idx: Index of this chunk (0-4) for metadata
        model: Model to use
        temperature: Temperature for sampling
        voter_dist: Voter distribution (for metadata)
        alt_dist: Alternative distribution (for metadata)
        method: Voting method name (for metadata)
        rep: Replication number (for metadata)
        voter_idx: Voter index (for metadata)
    
    Returns:
        Insert position (0-20) within this chunk
    """
    chunk_size = len(chunk_ranking)
    
    # Build chunk statements as numbered list (no hashes)
    chunk_text = "\n".join(
        f'{i+1}. "{statements[idx]["statement"]}"'
        for i, idx in enumerate(chunk_ranking)
    )
    
    system_prompt = f"""You are simulating a single, internally consistent person defined by the following persona:
{persona}

You must evaluate each statement solely through the lens of this persona's values, background, beliefs, and preferences.

Your task is to determine where a new statement fits in your preference ranking and return valid JSON only.
Do not include explanations, commentary, or extra text."""
    
    n_total = len(statements)
    user_prompt = f"""Topic: "{topic}"

You have ranked {n_total} statements on this topic. To help you evaluate a new statement, 
we have divided your ranking into 5 chunks of consecutive statements.

Below is ONE of these chunks. The {chunk_size} statements are shown in order from best to worst 
within this chunk:

{chunk_text}

NEW STATEMENT: "{new_statement}"

Compare this new statement to the {chunk_size} statements above. Where should it be inserted?
- Return 0 if it is BETTER than all {chunk_size} shown
- Return {chunk_size} if it is WORSE than all {chunk_size} shown
- Return 1-{chunk_size - 1} to insert between existing positions

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
            component="chunked_insertion",
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
    position = result.get("insert_position", chunk_size // 2)
    
    # Clamp position to valid range
    position = max(0, min(chunk_size, int(position)))
    
    return position


def insert_statement_chunked_borda(
    persona: str,
    current_ranking: List[int],
    statements: List[Dict],
    new_statement: str,
    topic: str,
    openai_client: OpenAI,
    n_chunks: int = 5,
    model: str = RANKING_MODEL,
    temperature: float = TEMPERATURE,
    voter_dist: str = None,
    alt_dist: str = None,
    method: str = None,
    rep: int = None,
    voter_idx: int = None,
) -> Tuple[List[int], float, int]:
    """
    Insert a new statement using chunked Borda approach.
    
    Splits the ranking into chunks, asks for insertion position in each chunk,
    then uses Borda scoring to determine the final position.
    
    Args:
        persona: Persona string description
        current_ranking: Current ranking (list of statement indices, most to least preferred)
        statements: Original list of statement dicts
        new_statement: The new statement text to insert
        topic: The topic/question
        openai_client: OpenAI client instance
        n_chunks: Number of chunks to split the ranking into (default 5)
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
        - borda_score: Normalized Borda score (0-1, lower = more preferred)
        - final_position: Position where the new statement was inserted (0-indexed)
    """
    n = len(current_ranking)
    chunk_size = n // n_chunks
    new_idx = len(statements)  # New statement gets the next available index
    
    # Split ranking into chunks
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else n  # Last chunk gets remainder
        chunks.append(current_ranking[start:end])
    
    # Get insertion position for each chunk
    chunk_positions = []
    for chunk_idx, chunk in enumerate(chunks):
        position = insert_statement_into_chunk(
            persona=persona,
            chunk_ranking=chunk,
            statements=statements,
            new_statement=new_statement,
            topic=topic,
            openai_client=openai_client,
            chunk_idx=chunk_idx,
            model=model,
            temperature=temperature,
            voter_dist=voter_dist,
            alt_dist=alt_dist,
            method=method,
            rep=rep,
            voter_idx=voter_idx,
        )
        chunk_positions.append(position)
    
    # Calculate Borda score
    # For each chunk, if position is p, the new statement "beats" (chunk_size - p) alternatives
    # Sum these to get total Borda points
    # Lower score = more preferred
    borda_points = 0
    for chunk_idx, (chunk, position) in enumerate(zip(chunks, chunk_positions)):
        chunk_len = len(chunk)
        # Number of alternatives in this chunk that the new statement beats
        beats = chunk_len - position
        borda_points += beats
    
    # Normalize to 0-1 (0 = beats all, 1 = beats none)
    borda_score = 1.0 - (borda_points / n)
    
    # Convert Borda score to position
    # borda_score of 0 means position 0 (most preferred)
    # borda_score of 1 means position n (least preferred)
    final_position = int(round(borda_score * n))
    final_position = max(0, min(n, final_position))
    
    # Insert new statement at the calculated position
    new_ranking = current_ranking.copy()
    new_ranking.insert(final_position, new_idx)
    
    return new_ranking, borda_score, final_position


def get_chunk_positions_detail(
    persona: str,
    current_ranking: List[int],
    statements: List[Dict],
    new_statement: str,
    topic: str,
    openai_client: OpenAI,
    n_chunks: int = 5,
    model: str = RANKING_MODEL,
    temperature: float = TEMPERATURE,
    voter_dist: str = None,
    alt_dist: str = None,
    method: str = None,
    rep: int = None,
    voter_idx: int = None,
) -> Dict:
    """
    Get detailed chunk-by-chunk insertion results for analysis.
    
    Similar to insert_statement_chunked_borda but returns detailed info
    for each chunk for analysis purposes.
    
    Returns:
        Dict with:
        - chunk_positions: List of positions returned for each chunk
        - chunk_sizes: List of sizes for each chunk
        - borda_points: Total Borda points
        - borda_score: Normalized Borda score (0-1)
        - final_position: Calculated final position
    """
    n = len(current_ranking)
    chunk_size = n // n_chunks
    
    # Split ranking into chunks
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < n_chunks - 1 else n
        chunks.append(current_ranking[start:end])
    
    # Get insertion position for each chunk
    chunk_positions = []
    for chunk_idx, chunk in enumerate(chunks):
        position = insert_statement_into_chunk(
            persona=persona,
            chunk_ranking=chunk,
            statements=statements,
            new_statement=new_statement,
            topic=topic,
            openai_client=openai_client,
            chunk_idx=chunk_idx,
            model=model,
            temperature=temperature,
            voter_dist=voter_dist,
            alt_dist=alt_dist,
            method=method,
            rep=rep,
            voter_idx=voter_idx,
        )
        chunk_positions.append(position)
    
    # Calculate Borda score
    borda_points = 0
    chunk_sizes = []
    for chunk_idx, (chunk, position) in enumerate(zip(chunks, chunk_positions)):
        chunk_len = len(chunk)
        chunk_sizes.append(chunk_len)
        beats = chunk_len - position
        borda_points += beats
    
    borda_score = 1.0 - (borda_points / n)
    final_position = int(round(borda_score * n))
    final_position = max(0, min(n, final_position))
    
    return {
        "chunk_positions": chunk_positions,
        "chunk_sizes": chunk_sizes,
        "borda_points": borda_points,
        "borda_score": borda_score,
        "final_position": final_position,
    }
