"""
Alt1: Persona-conditioned statement generation WITHOUT seeing other statements.

This generator creates bridging statements where each persona writes based on
their characteristics alone, without any context from existing statements.

Key features:
- Pre-generated for all 815 personas (one API call per persona)
- Prompts instruct model to avoid self-referential phrases like "As a progressive"
- Statements aim to find common ground on the topic
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

logger = logging.getLogger(__name__)


# System prompt for Alt1 - simple perspective statement
SYSTEM_PROMPT = """You are writing a statement that reflects your perspective on a topic."""


def _build_user_prompt(persona: str, topic: str) -> str:
    """Build the user prompt for Alt1 generation."""
    return f"""You are a person with the following characteristics:
{persona}

Topic: "{topic}"

Write a bridging statement expressing your views on this topic. Your statement should:
- Reflect your background, values, and life experiences
- Aim to find common ground or bridge different viewpoints
- Be 2-4 sentences long
- NOT write in first-person
- NOT explicitly reference your identity or demographics (avoid "As a [X]...")

Write only the statement:"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def generate_single_statement(
    persona: str,
    persona_id: str,
    topic_slug: str,
    client: OpenAI
) -> Dict:
    """
    Generate a single Alt1 statement for one persona.
    
    Args:
        persona: Persona description string
        persona_id: Unique identifier for this persona
        topic_slug: Topic slug (used to look up full question)
        client: OpenAI client instance
        
    Returns:
        Dict with persona_id, persona, statement, and topic
    """
    topic_question = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    user_prompt = _build_user_prompt(persona, topic_question)
    
    start_time = time.time()
    response = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=TEMPERATURE,
    )
    api_timer.record(time.time() - start_time)
    
    statement = response.output_text.strip()
    
    return {
        "persona_id": persona_id,
        "persona": persona,
        "statement": statement,
        "topic": topic_slug,
    }


def generate_all_statements(
    personas: Dict[str, str],
    topic_slug: str,
    client: OpenAI,
    max_workers: int = MAX_WORKERS,
    output_path: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Generate Alt1 statements for all personas in parallel.
    
    Args:
        personas: Dict mapping persona_id to persona description
        topic_slug: Topic slug for the question
        client: OpenAI client instance
        max_workers: Maximum parallel API calls
        output_path: Optional path to save results incrementally
        
    Returns:
        Dict mapping persona_id to generated statement
    """
    topic_question = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    logger.info(f"Generating Alt1 statements for {len(personas)} personas")
    logger.info(f"Topic: {topic_question}")
    
    results = {}
    errors = []
    
    # If output path exists, load existing results to resume
    if output_path and output_path.exists():
        with open(output_path, 'r') as f:
            existing = json.load(f)
        results = existing.get("statements", {})
        logger.info(f"Resuming from {len(results)} existing statements")
    
    # Filter to only personas we haven't processed yet
    remaining_personas = {
        pid: persona for pid, persona in personas.items()
        if pid not in results
    }
    
    if not remaining_personas:
        logger.info("All personas already processed")
        return results
    
    logger.info(f"Generating statements for {len(remaining_personas)} remaining personas")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pid = {
            executor.submit(
                generate_single_statement,
                persona,
                pid,
                topic_slug,
                client
            ): pid
            for pid, persona in remaining_personas.items()
        }
        
        with tqdm(total=len(remaining_personas), desc="Alt1 generation", unit="stmt") as pbar:
            for future in as_completed(future_to_pid):
                pid = future_to_pid[future]
                try:
                    result = future.result()
                    results[pid] = result["statement"]
                    pbar.update(1)
                    
                    # Save incrementally every 50 statements
                    if output_path and len(results) % 50 == 0:
                        _save_results(results, topic_slug, output_path)
                        
                except Exception as e:
                    logger.error(f"Failed to generate statement for persona {pid}: {e}")
                    errors.append((pid, str(e)))
                    pbar.update(1)
    
    # Final save
    if output_path:
        _save_results(results, topic_slug, output_path)
    
    if errors:
        logger.warning(f"Failed to generate {len(errors)} statements")
    
    logger.info(f"Successfully generated {len(results)}/{len(personas)} Alt1 statements")
    
    return results


def _save_results(results: Dict[str, str], topic_slug: str, output_path: Path) -> None:
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "topic": topic_slug,
        "alt_type": "persona_no_context",
        "count": len(results),
        "statements": results,  # {persona_id: statement_text}
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.debug(f"Saved {len(results)} statements to {output_path}")


def load_statements(path: Path) -> Dict[str, str]:
    """
    Load pre-generated Alt1 statements from file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Dict mapping persona_id to statement text
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    statements = data.get("statements", {})
    logger.info(f"Loaded {len(statements)} Alt1 statements from {path}")
    
    return statements
