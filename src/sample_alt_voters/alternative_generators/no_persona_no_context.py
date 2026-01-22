"""
Alt4: No persona, no context statement generation using verbalized sampling.

This generator creates bridging statements without any persona or context,
using verbalized sampling to get diverse statements (5 per API call).

Key features:
- Pre-generated (no persona, no context = can generate all upfront)
- Uses verbalized sampling (5 responses per call, all 5 are used)
- 815 total statements = 163 API calls (163 * 5 = 815)
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
)

logger = logging.getLogger(__name__)


def _build_user_prompt(topic: str) -> str:
    """Build the user prompt for Alt4 generation (no persona, no context)."""
    return f"""Topic: "{topic}"

Write a bridging statement on this topic. Your statement should:
- Aim to find common ground or bridge different viewpoints
- Be 2-4 sentences long"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def generate_batch(
    topic_slug: str,
    client: OpenAI,
    batch_id: int = 0,
) -> List[str]:
    """
    Generate a batch of 5 Alt4 statements using verbalized sampling.
    
    Args:
        topic_slug: Topic slug (used to look up full question)
        client: OpenAI client instance
        batch_id: Batch identifier for logging
        
    Returns:
        List of 5 statement strings
    """
    topic_question = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    system_prompt = get_verbalized_system_prompt()
    user_prompt = _build_user_prompt(topic_question)
    
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


def generate_n_statements(
    n: int,
    topic_slug: str,
    client: OpenAI,
    max_workers: int = MAX_WORKERS,
    output_path: Optional[Path] = None,
) -> List[str]:
    """
    Generate N Alt4 statements using verbalized sampling.
    
    Each API call returns 5 statements, so we make ceil(n/5) calls.
    
    Args:
        n: Number of statements to generate
        topic_slug: Topic slug for the question
        client: OpenAI client instance
        max_workers: Maximum parallel API calls
        output_path: Optional path to save results incrementally
        
    Returns:
        List of generated statement strings
    """
    topic_question = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    n_batches = (n + 4) // 5  # Ceiling division
    
    logger.info(f"Generating {n} Alt4 statements ({n_batches} API calls)")
    logger.info(f"Topic: {topic_question}")
    
    all_statements = []
    errors = []
    
    # If output path exists, load existing results to resume
    if output_path and output_path.exists():
        with open(output_path, 'r') as f:
            existing = json.load(f)
        all_statements = existing.get("statements", [])
        logger.info(f"Resuming from {len(all_statements)} existing statements")
    
    # Calculate remaining batches needed
    current_batches = (len(all_statements) + 4) // 5
    remaining_batches = max(0, n_batches - current_batches)
    
    if remaining_batches == 0:
        logger.info("Already have enough statements")
        return all_statements[:n]
    
    logger.info(f"Generating {remaining_batches} more batches")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(
                generate_batch,
                topic_slug,
                client,
                current_batches + i,
            ): current_batches + i
            for i in range(remaining_batches)
        }
        
        with tqdm(total=remaining_batches, desc="Alt4 generation", unit="batch") as pbar:
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_statements = future.result()
                    all_statements.extend(batch_statements)
                    pbar.update(1)
                    
                    # Save incrementally every 10 batches (50 statements)
                    if output_path and len(all_statements) % 50 == 0:
                        _save_results(all_statements, topic_slug, output_path)
                        
                except Exception as e:
                    logger.error(f"Failed to generate batch {batch_id}: {e}")
                    errors.append((batch_id, str(e)))
                    pbar.update(1)
    
    # Final save
    if output_path:
        _save_results(all_statements, topic_slug, output_path)
    
    if errors:
        logger.warning(f"Failed to generate {len(errors)} batches")
    
    logger.info(f"Successfully generated {len(all_statements)} Alt4 statements")
    
    # Return exactly n statements (or as many as we have)
    return all_statements[:n]


def _save_results(statements: List[str], topic_slug: str, output_path: Path) -> None:
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "topic": topic_slug,
        "alt_type": "no_persona_no_context",
        "count": len(statements),
        "statements": statements,  # List of statement strings
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.debug(f"Saved {len(statements)} statements to {output_path}")


def load_statements(path: Path) -> List[str]:
    """
    Load pre-generated Alt4 statements from file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        List of statement strings
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    statements = data.get("statements", [])
    logger.info(f"Loaded {len(statements)} Alt4 statements from {path}")
    
    return statements
