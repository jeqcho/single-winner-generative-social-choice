"""
Generate bridging statements from personas synthesizing all statements.
"""

import json
import logging
from typing import List, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

import time

from .config import (
    BRIDGING_MODEL,
    MAX_WORKERS,
    TOPIC_QUESTIONS,
    api_timer,
)

logger = logging.getLogger(__name__)


def _format_statements_list(statements: List[Dict]) -> str:
    """Format statements into a numbered list for the prompt."""
    lines = []
    for i, stmt in enumerate(statements):
        lines.append(f"{i + 1}. {stmt['statement']}")
    return "\n".join(lines)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def _generate_single_bridging_statement(
    persona: str,
    persona_idx: int,
    statements_list: str,
    topic: str,
    openai_client: OpenAI
) -> Dict:
    """
    Generate a bridging statement from a single persona.
    
    Args:
        persona: Persona string description
        persona_idx: Index of the persona
        statements_list: Formatted list of all statements
        topic: The topic/question
        openai_client: OpenAI client instance
    
    Returns:
        Dict with persona_idx, persona, and bridging statement
    """
    system_prompt = (
        "You are a person with the characteristics described. "
        "Write only the bridging statement, no additional commentary."
    )
    
    user_prompt = f"""You are a person with the following characteristics:
{persona}

You have read the following 100 statements from a discussion on the topic: "{topic}"

{statements_list}

Write a bridging statement that could serve as a consensus or compromise position. Your statement should:
- Reflect YOUR perspective and values based on your characteristics
- Synthesize key themes you find important across the discussion  
- Be self-contained (do not reference other people or their statements)
- Be concise (2-4 sentences)

Write only the bridging statement:"""

    start_time = time.time()
    response = openai_client.responses.create(
        model=BRIDGING_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )
    api_timer.record(time.time() - start_time)
    
    bridging_statement = response.output_text.strip()
    
    return {
        "persona_idx": persona_idx,
        "persona": persona,
        "statement": bridging_statement
    }


def generate_bridging_statements(
    personas: List[str],
    statements: List[Dict],
    topic_slug: str,
    openai_client: OpenAI,
    max_workers: int = MAX_WORKERS
) -> List[Dict]:
    """
    Generate bridging statements from all personas.
    
    Args:
        personas: List of persona strings
        statements: List of statement dicts
        topic_slug: Topic slug for looking up the full question
        openai_client: OpenAI client instance
        max_workers: Maximum parallel workers
    
    Returns:
        List of bridging statement dicts, each with:
        - persona_idx: index of the persona
        - persona: persona string
        - statement: generated bridging statement
    """
    topic = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    statements_list = _format_statements_list(statements)
    
    logger.info(f"Generating bridging statements for {len(personas)} personas")
    logger.info(f"Topic: {topic}")
    
    bridging_statements = [None] * len(personas)
    errors = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _generate_single_bridging_statement,
                persona,
                idx,
                statements_list,
                topic,
                openai_client
            ): idx
            for idx, persona in enumerate(personas)
        }
        
        with tqdm(total=len(personas), desc="Generating bridging statements", unit="stmt") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    bridging_statements[idx] = result
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Failed to generate bridging statement for persona {idx}: {e}")
                    errors.append((idx, str(e)))
                    pbar.update(1)
    
    # Filter out None values (failed generations)
    bridging_statements = [s for s in bridging_statements if s is not None]
    
    if errors:
        logger.warning(f"Failed to generate {len(errors)} bridging statements")
    
    logger.info(f"Successfully generated {len(bridging_statements)}/{len(personas)} bridging statements")
    
    return bridging_statements


def save_bridging_statements(
    bridging_statements: List[Dict],
    output_dir: Path
) -> None:
    """
    Save bridging statements to JSON file.
    
    Args:
        bridging_statements: List of bridging statement dicts
        output_dir: Directory to save to
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / "bridging_statements.json"
    
    with open(filepath, 'w') as f:
        json.dump(bridging_statements, f, indent=2)
    
    logger.info(f"Saved {len(bridging_statements)} bridging statements to {filepath}")


def load_bridging_statements(output_dir: Path) -> List[Dict]:
    """
    Load bridging statements from JSON file.
    
    Args:
        output_dir: Directory containing the file
    
    Returns:
        List of bridging statement dicts
    """
    filepath = output_dir / "bridging_statements.json"
    
    with open(filepath, 'r') as f:
        bridging_statements = json.load(f)
    
    logger.info(f"Loaded {len(bridging_statements)} bridging statements from {filepath}")
    
    return bridging_statements

