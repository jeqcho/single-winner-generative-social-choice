"""
Alt3: No persona, WITH context statement generation using verbalized sampling.

This generator creates bridging statements without a persona but after reading
100 existing statements, using verbalized sampling to get diverse outputs.

Key features:
- Generated per-rep (depends on which 100 statements are sampled as context)
- Shared across voter distributions within the same rep
- Uses verbalized sampling (20 API calls * 5 = 100 statements per rep)
- No persona conditioning - just synthesizes based on context
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

from ..config import (
    MODEL,
    TEMPERATURE,
    MAX_WORKERS,
    TOPIC_QUESTIONS,
    api_timer,
)
from ..verbalized_sampling import (
    parse_verbalized_response,
    get_verbalized_system_prompt,
    format_statements_for_context,
)

logger = logging.getLogger(__name__)


def _build_user_prompt(topic: str, statements_list: str) -> str:
    """Build the user prompt for Alt3 generation (no persona, with context)."""
    return f"""Topic: "{topic}"

Here are 100 statements from people with diverse perspectives on this topic:

{statements_list}

Write a NEW bridging statement on this topic. Your statement should:
- Synthesize key themes across the different viewpoints
- Aim to find common ground or bridge different viewpoints
- Be 2-4 sentences long
- Be self-contained (do not reference "the statements above" or "other people")"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def generate_batch(
    context_statements: List[str],
    topic_slug: str,
    client: OpenAI,
    batch_id: int = 0,
) -> List[str]:
    """
    Generate a batch of 5 Alt3 statements using verbalized sampling.
    
    Args:
        context_statements: List of 100 statements to show as context
        topic_slug: Topic slug (used to look up full question)
        client: OpenAI client instance
        batch_id: Batch identifier for logging
        
    Returns:
        List of 5 statement strings
    """
    topic_question = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    system_prompt = get_verbalized_system_prompt()
    statements_list = format_statements_for_context(context_statements)
    user_prompt = _build_user_prompt(topic_question, statements_list)
    
    start_time = time.time()
    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=TEMPERATURE,
    )
    api_timer.record(time.time() - start_time)
    
    raw_text = response.output_text
    statements = parse_verbalized_response(raw_text)
    
    if len(statements) < 5:
        logger.warning(f"Batch {batch_id}: Expected 5 statements, got {len(statements)}")
    
    return statements


def generate_for_rep(
    context_statements: List[str],
    topic_slug: str,
    rep_id: int,
    client: OpenAI,
    n_statements: int = 100,
    max_workers: int = MAX_WORKERS,
    output_path: Optional[Path] = None,
) -> List[str]:
    """
    Generate Alt3 statements for a rep using verbalized sampling.
    
    Each API call returns 5 statements, so we make n_statements/5 calls.
    
    Args:
        context_statements: List of 100 statements to show as context
        topic_slug: Topic slug for the question
        rep_id: Replication identifier
        client: OpenAI client instance
        n_statements: Number of statements to generate (default 100)
        max_workers: Maximum parallel API calls
        output_path: Optional path to save results
        
    Returns:
        List of generated statement strings
    """
    topic_question = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    n_batches = (n_statements + 4) // 5  # 20 batches for 100 statements
    
    logger.info(f"Generating {n_statements} Alt3 statements for rep {rep_id} ({n_batches} API calls)")
    logger.info(f"Topic: {topic_question}")
    logger.info(f"Context: {len(context_statements)} statements")
    
    all_statements = []
    errors = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(
                generate_batch,
                context_statements,
                topic_slug,
                client,
                i,
            ): i
            for i in range(n_batches)
        }
        
        with tqdm(total=n_batches, desc=f"Alt3 rep{rep_id}", unit="batch") as pbar:
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_statements = future.result()
                    all_statements.extend(batch_statements)
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Failed to generate batch {batch_id}: {e}")
                    errors.append((batch_id, str(e)))
                    pbar.update(1)
    
    # Trim to exact number needed
    all_statements = all_statements[:n_statements]
    
    # Save results
    if output_path:
        _save_results(all_statements, topic_slug, rep_id, output_path)
    
    if errors:
        logger.warning(f"Failed to generate {len(errors)} batches")
    
    logger.info(f"Successfully generated {len(all_statements)} Alt3 statements for rep {rep_id}")
    
    return all_statements


def _save_results(statements: List[str], topic_slug: str, rep_id: int, output_path: Path) -> None:
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "topic": topic_slug,
        "rep_id": rep_id,
        "alt_type": "no_persona_context",
        "count": len(statements),
        "statements": statements,  # List of statement strings
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.debug(f"Saved {len(statements)} statements to {output_path}")


def load_statements(path: Path) -> List[str]:
    """
    Load Alt3 statements from file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        List of statement strings
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    statements = data.get("statements", [])
    logger.info(f"Loaded {len(statements)} Alt3 statements from {path}")
    
    return statements
