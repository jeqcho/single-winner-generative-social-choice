"""
Get Likert scale ratings from evaluative personas (with parallelization).
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
def get_likert_rating(
    persona: str,
    statement: Dict,
    topic: str,
    openai_client: OpenAI,
    persona_idx: int = None,
    stmt_idx: int = None
) -> int:
    """
    Get Likert scale rating (1-5) from persona for a statement (with retry logic).
    
    Args:
        persona: Persona string description
        statement: Statement dict with 'statement' key
        topic: The topic/question
        openai_client: OpenAI client instance
        persona_idx: Index of persona for logging
        stmt_idx: Index of statement for logging
    
    Returns:
        Likert rating from 1 (strongly disagree) to 5 (strongly agree)
    """
    try:
        prompt = f"""You are a person with the following characteristics:
{persona}

Given the topic: "{topic}"

Please rate how much you agree with the following statement on a scale of 1-5:

Statement: {statement['statement']}

Rating scale:
1 = Strongly disagree
2 = Disagree
3 = Neutral / Neither agree nor disagree
4 = Agree
5 = Strongly agree

Consider:
- How well this statement aligns with your values and perspective
- Whether you would support or endorse this position
- How much this statement represents your views on the topic

Return your rating as a JSON object with this format:
{{"rating": 3}}

Where the rating is an integer from 1 to 5.
Return only the JSON, no additional text."""

        response = openai_client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": "You are rating statements based on the given persona. Return ONLY valid JSON, no other text."},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = json.loads(response.output_text)
        rating = result.get("rating", 3)
        
        # Ensure rating is in valid range
        rating = max(1, min(5, int(rating)))
        
        return rating
    except Exception as e:
        logger.error(f"Error getting rating from persona {persona_idx} for statement {stmt_idx}: {e}")
        raise


def get_all_ratings(
    personas: List[str],
    statements: List[Dict],
    topic: str,
    openai_client: OpenAI,
    max_workers: int = 20
) -> List[Dict]:
    """
    Get Likert ratings from all evaluative personas for all statements (parallelized).
    
    Args:
        personas: List of evaluative persona strings
        statements: List of statement dicts
        topic: The topic/question
        openai_client: OpenAI client instance
        max_workers: Maximum number of parallel workers (default: 20)
    
    Returns:
        List of rating dicts, each with:
        - persona: persona string
        - ratings: list of ratings (one per statement)
    """
    n_statements = len(statements)
    n_personas = len(personas)
    total_tasks = n_personas * n_statements
    
    logger.info(f"Getting Likert ratings: {n_personas} personas × {n_statements} statements = {total_tasks} ratings")
    logger.info(f"Max workers: {max_workers}")
    
    # Initialize results structure
    all_ratings = [{
        "persona": persona,
        "ratings": [None] * n_statements
    } for persona in personas]
    
    errors = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all rating tasks
        future_to_coords = {}
        for p_idx, persona in enumerate(personas):
            for s_idx, statement in enumerate(statements):
                future = executor.submit(
                    get_likert_rating,
                    persona,
                    statement,
                    topic,
                    openai_client,
                    p_idx,
                    s_idx
                )
                future_to_coords[future] = (p_idx, s_idx)
        
        # Process completed tasks with progress bar
        with tqdm(total=total_tasks, desc="Getting evaluative ratings", unit="rating") as pbar:
            for future in as_completed(future_to_coords):
                p_idx, s_idx = future_to_coords[future]
                try:
                    rating = future.result()
                    all_ratings[p_idx]["ratings"][s_idx] = rating
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Failed to get rating from persona {p_idx} for statement {s_idx}: {e}")
                    errors.append((p_idx, s_idx, str(e)))
                    # Use default rating of 3 (neutral) for failed ratings
                    all_ratings[p_idx]["ratings"][s_idx] = 3
                    pbar.update(1)
    
    if errors:
        logger.warning(f"Failed to get {len(errors)} ratings (using default value of 3)")
    
    logger.info(f"Completed evaluative ratings: {total_tasks - len(errors)}/{total_tasks} successful")
    
    return all_ratings


def save_evaluations(
    evaluations: List[Dict],
    topic_slug: str,
    output_dir: str = "data/large_scale/evaluations"
) -> None:
    """
    Save evaluations to JSON file.
    
    Args:
        evaluations: List of evaluation dicts
        topic_slug: Slugified topic name for filename
        output_dir: Directory to save evaluations
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{topic_slug}.json")
    
    with open(filepath, 'w') as f:
        json.dump(evaluations, f, indent=2)
    
    logger.info(f"Evaluations saved to {filepath}")


def load_evaluations(
    topic_slug: str,
    input_dir: str = "data/large_scale/evaluations"
) -> List[Dict]:
    """
    Load evaluations from JSON file.
    
    Args:
        topic_slug: Slugified topic name for filename
        input_dir: Directory containing evaluations
    
    Returns:
        List of evaluation dicts
    """
    filepath = os.path.join(input_dir, f"{topic_slug}.json")
    
    with open(filepath, 'r') as f:
        evaluations = json.load(f)
    
    n_personas = len(evaluations)
    n_statements = len(evaluations[0]["ratings"]) if evaluations else 0
    logger.info(f"Loaded evaluations from {filepath}: {n_personas} personas, {n_statements} statements")
    
    return evaluations
