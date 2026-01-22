"""
Alt2: Persona-conditioned statement generation WITH context (Ben's bridging setup).

This generator creates bridging statements where each persona reads 100 existing
statements first, then writes a new bridging statement synthesizing what they read.

Key features:
- The 100 personas who wrote the context Alt1 statements each generate 1 new statement
- Generated per-rep (depends on which 100 statements are sampled as context)
- Shared across voter distributions within the same rep
- Prompts instruct model to synthesize themes while avoiding self-referential phrases
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
from ..verbalized_sampling import format_statements_for_context

logger = logging.getLogger(__name__)


# System prompt for Alt2 - simple perspective statement
SYSTEM_PROMPT = """You are writing a statement that reflects your perspective on a topic."""


def _build_user_prompt(persona: str, topic: str, statements_list: str) -> str:
    """Build the user prompt for Alt2 generation (persona + context)."""
    return f"""You are a person with the following characteristics:
{persona}

Topic: "{topic}"

Here are 100 statements from people with diverse perspectives on this topic:

{statements_list}

Write a NEW bridging statement expressing your views on this topic. Your statement should:
- Reflect your background, values, and life experiences
- Synthesize key themes you observed across the discussion
- Aim to find common ground or bridge different viewpoints
- Be 2-4 sentences long
- NOT write in first-person
- NOT explicitly reference your identity or demographics (avoid "As a [X]...")
- Be self-contained (do not reference "the statements above" or "other people")

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
    context_statements: List[str],
    topic_slug: str,
    client: OpenAI
) -> Dict:
    """
    Generate a single Alt2 statement for one persona after reading context.
    
    Args:
        persona: Persona description string
        persona_id: Unique identifier for this persona
        context_statements: List of 100 statements to show as context
        topic_slug: Topic slug (used to look up full question)
        client: OpenAI client instance
        
    Returns:
        Dict with persona_id, persona, statement, and topic
    """
    topic_question = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    statements_list = format_statements_for_context(context_statements)
    user_prompt = _build_user_prompt(persona, topic_question, statements_list)
    
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


def generate_for_rep(
    personas: Dict[str, str],
    context_statements: List[str],
    topic_slug: str,
    rep_id: int,
    client: OpenAI,
    max_workers: int = MAX_WORKERS,
    output_path: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Generate Alt2 statements for all personas in a rep.
    
    The personas should be the same 100 who wrote the context statements.
    
    Args:
        personas: Dict mapping persona_id to persona description (should be 100)
        context_statements: List of 100 statements to show as context
        topic_slug: Topic slug for the question
        rep_id: Replication identifier
        client: OpenAI client instance
        max_workers: Maximum parallel API calls
        output_path: Optional path to save results
        
    Returns:
        Dict mapping persona_id to generated statement
    """
    topic_question = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    logger.info(f"Generating Alt2 statements for rep {rep_id}: {len(personas)} personas")
    logger.info(f"Topic: {topic_question}")
    logger.info(f"Context: {len(context_statements)} statements")
    
    results = {}
    errors = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pid = {
            executor.submit(
                generate_single_statement,
                persona,
                pid,
                context_statements,
                topic_slug,
                client
            ): pid
            for pid, persona in personas.items()
        }
        
        with tqdm(total=len(personas), desc=f"Alt2 rep{rep_id}", unit="stmt") as pbar:
            for future in as_completed(future_to_pid):
                pid = future_to_pid[future]
                try:
                    result = future.result()
                    results[pid] = result["statement"]
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Failed to generate statement for persona {pid}: {e}")
                    errors.append((pid, str(e)))
                    pbar.update(1)
    
    # Save results
    if output_path:
        _save_results(results, topic_slug, rep_id, output_path)
    
    if errors:
        logger.warning(f"Failed to generate {len(errors)} statements")
    
    logger.info(f"Successfully generated {len(results)}/{len(personas)} Alt2 statements for rep {rep_id}")
    
    return results


def _save_results(results: Dict[str, str], topic_slug: str, rep_id: int, output_path: Path) -> None:
    """Save results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "topic": topic_slug,
        "rep_id": rep_id,
        "alt_type": "persona_context",
        "count": len(results),
        "statements": results,  # {persona_id: statement_text}
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.debug(f"Saved {len(results)} statements to {output_path}")


def load_statements(path: Path) -> Dict[str, str]:
    """
    Load Alt2 statements from file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Dict mapping persona_id to statement text
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    statements = data.get("statements", {})
    logger.info(f"Loaded {len(statements)} Alt2 statements from {path}")
    
    return statements
