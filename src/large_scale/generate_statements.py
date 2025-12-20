"""
Generate statements from personas using OpenAI API (scaled version with parallelization).
"""

import json
import os
from typing import List, Dict
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def generate_statement_from_persona(
    topic: str,
    persona: str,
    openai_client: OpenAI,
    persona_idx: int = None
) -> Dict:
    """
    Generate a statement from a persona on a given topic (with retry logic).
    
    Args:
        topic: The topic/question to generate statement about
        persona: Persona string description
        openai_client: OpenAI client instance
        persona_idx: Index of persona for logging
    
    Returns:
        Dict with persona and statement
    """
    try:
        prompt = f"""You are a person with the following characteristics:
{persona}

Given the topic: "{topic}"

Write a statement expressing your views on this topic. The statement should:
- Reflect your background, perspective, and values
- Be clear and substantive (2-4 sentences)
- Represent a genuine viewpoint someone with your characteristics might hold

Write only the statement, no additional commentary."""

        response = openai_client.responses.create(
            model="gpt-5.1",
            input=[
                {"role": "system", "content": f"You are a person with the characteristics described."},
                {"role": "user", "content": prompt}
            ],
            timeout=60.0
        )
        
        statement_text = response.output_text.strip()
        
        return {
            "persona": persona,
            "statement": statement_text
        }
    except Exception as e:
        logger.error(f"Error generating statement for persona {persona_idx}: {e}")
        raise


def generate_all_statements(
    topic: str,
    personas: List[str],
    openai_client: OpenAI,
    max_workers: int = 20
) -> List[Dict]:
    """
    Generate statements from all personas on a given topic (parallelized).
    
    Args:
        topic: The topic/question to generate statements about
        personas: List of persona string descriptions
        openai_client: OpenAI client instance
        max_workers: Maximum number of parallel workers (default: 20)
    
    Returns:
        List of statement dicts, each with:
        - persona: persona string
        - statement: generated statement string
    """
    logger.info(f"Generating statements for topic: {topic}")
    logger.info(f"Total personas: {len(personas)}, Max workers: {max_workers}")
    
    statements = [None] * len(personas)  # Pre-allocate to maintain order
    errors = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(
                generate_statement_from_persona,
                topic,
                personas[i],
                openai_client,
                i
            ): i
            for i in range(len(personas))
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(personas), desc="Generating statements", unit="stmt") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    statements[idx] = result
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Failed to generate statement for persona {idx} after retries: {e}")
                    errors.append((idx, str(e)))
                    pbar.update(1)
    
    # Filter out None values (failed generations)
    statements = [s for s in statements if s is not None]
    
    if errors:
        logger.warning(f"Failed to generate {len(errors)} statements: {errors}")
    
    logger.info(f"Successfully generated {len(statements)}/{len(personas)} statements")
    
    return statements


def save_statements(
    statements: List[Dict],
    topic_slug: str,
    output_dir: str = "data/large_scale/statements"
) -> None:
    """
    Save statements to JSON file.
    
    Args:
        statements: List of statement dicts
        topic_slug: Slugified topic name for filename
        output_dir: Directory to save statements
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{topic_slug}.json")
    
    with open(filepath, 'w') as f:
        json.dump(statements, f, indent=2)
    
    logger.info(f"Statements saved to {filepath}")


def load_statements(
    topic_slug: str,
    input_dir: str = "data/large_scale/statements"
) -> List[Dict]:
    """
    Load statements from JSON file.
    
    Args:
        topic_slug: Slugified topic name for filename
        input_dir: Directory containing statements
    
    Returns:
        List of statement dicts
    """
    filepath = os.path.join(input_dir, f"{topic_slug}.json")
    
    with open(filepath, 'r') as f:
        statements = json.load(f)
    
    logger.info(f"Loaded {len(statements)} statements from {filepath}")
    
    return statements
