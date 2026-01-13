"""
Generate 1000x1000 Likert score matrices using GPT-5.2.

Each of 1000 personas rates all 1000 statements on a 1-5 scale.
Statements are batched (default 100 per API call) for efficiency.
"""

import argparse
import csv
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# All 13 topics
ALL_TOPICS = [
    "how-should-we-increase-the-general-publics-trust-i",
    "what-are-the-best-policies-to-prevent-littering-in",
    "what-are-your-thoughts-on-the-way-university-campu",
    "what-balance-should-be-struck-between-environmenta",
    "what-balance-should-exist-between-gun-safety-laws-",
    "what-limits-if-any-should-exist-on-free-speech-reg",
    "what-principles-should-guide-immigration-policy-an",
    "what-reforms-if-any-should-replace-or-modify-the-e",
    "what-responsibilities-should-tech-companies-have-w",
    "what-role-should-artificial-intelligence-play-in-s",
    "what-role-should-the-government-play-in-ensuring-u",
    "what-should-guide-laws-concerning-abortion",
    "what-strategies-should-guide-policing-to-address-b",
]

# Topic slug to full question mapping
TOPIC_QUESTIONS = {
    "how-should-we-increase-the-general-publics-trust-i": 
        "How should we increase the general public's trust in institutions?",
    "what-are-the-best-policies-to-prevent-littering-in":
        "What are the best policies to prevent littering in public spaces?",
    "what-are-your-thoughts-on-the-way-university-campu":
        "What are your thoughts on the way university campuses handle free speech?",
    "what-balance-should-be-struck-between-environmenta":
        "What balance should be struck between environmental protection and economic growth?",
    "what-balance-should-exist-between-gun-safety-laws-":
        "What balance should exist between gun safety laws and Second Amendment rights?",
    "what-limits-if-any-should-exist-on-free-speech-reg":
        "What limits, if any, should exist on free speech regarding hate speech?",
    "what-principles-should-guide-immigration-policy-an":
        "What principles should guide immigration policy and the path to citizenship?",
    "what-reforms-if-any-should-replace-or-modify-the-e":
        "What reforms, if any, should replace or modify the electoral college?",
    "what-responsibilities-should-tech-companies-have-w":
        "What responsibilities should tech companies have with user data and privacy?",
    "what-role-should-artificial-intelligence-play-in-s":
        "What role should artificial intelligence play in society?",
    "what-role-should-the-government-play-in-ensuring-u":
        "What role should the government play in ensuring universal healthcare?",
    "what-should-guide-laws-concerning-abortion":
        "What should guide laws concerning abortion?",
    "what-strategies-should-guide-policing-to-address-b":
        "What strategies should guide policing to address both safety and civil rights?",
}

MODEL = "gpt-5.2"
PROJECT_ROOT = Path(__file__).parent.parent.parent
STATEMENTS_DIR = PROJECT_ROOT / "data" / "large_scale" / "prod" / "statements"
OUTPUT_DIR = PROJECT_ROOT / "data" / "large_scale" / "likert_1000x1000"


def load_statements_and_personas(topic_slug: str) -> Tuple[List[str], List[str]]:
    """
    Load personas and statements from the topic's statements file.
    
    Args:
        topic_slug: The topic slug (e.g., 'what-should-guide-laws-concerning-abortion')
    
    Returns:
        Tuple of (personas list, statements list)
    """
    filepath = STATEMENTS_DIR / f"{topic_slug}.json"
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    personas = [item['persona'] for item in data]
    statements = [item['statement'] for item in data]
    
    logger.info(f"Loaded {len(personas)} personas and {len(statements)} statements from {filepath}")
    
    return personas, statements


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def get_batched_likert_ratings(
    persona: str,
    statements: List[str],
    topic: str,
    openai_client: OpenAI,
    persona_idx: int = None,
    batch_idx: int = None
) -> List[int]:
    """
    Get Likert scale ratings (1-5) from a persona for a batch of statements.
    
    Args:
        persona: Persona string description
        statements: List of statement strings to rate
        topic: The topic/question
        openai_client: OpenAI client instance
        persona_idx: Index of persona for logging
        batch_idx: Index of batch for logging
    
    Returns:
        List of Likert ratings (1-5) for each statement
    """
    # Build numbered statements list
    statements_text = "\n".join(
        f"{i+1}. {stmt}" for i, stmt in enumerate(statements)
    )
    
    prompt = f"""You are a person with the following characteristics:
{persona}

Given the topic: "{topic}"

Rate how much you agree with each of the following statements on a scale of 1-5:
1 = Strongly disagree
2 = Disagree
3 = Neutral / Neither agree nor disagree
4 = Agree
5 = Strongly agree

Consider for each statement:
- How well it aligns with your values and perspective
- Whether you would support or endorse this position
- How much it represents your views on the topic

Statements:
{statements_text}

Return ONLY a JSON object with a "ratings" array containing {len(statements)} integers (1-5), one for each statement in order.
Example format: {{"ratings": [4, 3, 5, 2, ...]}}"""

    try:
        response = openai_client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": "You are rating statements based on the given persona. Return ONLY valid JSON, no other text."},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = json.loads(response.output_text)
        ratings = result.get("ratings", [])
        
        # Validate ratings
        if len(ratings) != len(statements):
            logger.warning(
                f"Persona {persona_idx} batch {batch_idx}: Expected {len(statements)} ratings, got {len(ratings)}. Padding/truncating."
            )
            # Pad with 3s if too short, truncate if too long
            if len(ratings) < len(statements):
                ratings.extend([3] * (len(statements) - len(ratings)))
            else:
                ratings = ratings[:len(statements)]
        
        # Ensure ratings are in valid range
        ratings = [max(1, min(5, int(r))) for r in ratings]
        
        return ratings
        
    except Exception as e:
        logger.error(f"Error getting ratings from persona {persona_idx} batch {batch_idx}: {e}")
        raise


def process_persona(
    persona_idx: int,
    persona: str,
    statements: List[str],
    topic: str,
    openai_client: OpenAI,
    batch_size: int
) -> Tuple[int, List[int]]:
    """
    Process all statements for a single persona in batches.
    
    Args:
        persona_idx: Index of the persona
        persona: Persona string
        statements: All statements to rate
        topic: The topic question
        openai_client: OpenAI client
        batch_size: Number of statements per API call
    
    Returns:
        Tuple of (persona_idx, all_ratings)
    """
    all_ratings = []
    n_batches = (len(statements) + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(statements))
        batch_statements = statements[start:end]
        
        try:
            ratings = get_batched_likert_ratings(
                persona=persona,
                statements=batch_statements,
                topic=topic,
                openai_client=openai_client,
                persona_idx=persona_idx,
                batch_idx=batch_idx
            )
            all_ratings.extend(ratings)
        except Exception as e:
            logger.error(f"Failed persona {persona_idx} batch {batch_idx}: {e}. Using default ratings.")
            all_ratings.extend([3] * len(batch_statements))
    
    return persona_idx, all_ratings


def save_checkpoint(
    matrix: List[List[int]],
    completed_personas: int,
    topic_slug: str,
    output_dir: Path
) -> None:
    """Save a checkpoint of the current progress."""
    checkpoint_path = output_dir / f"{topic_slug}_checkpoint.json"
    
    checkpoint_data = {
        "completed_personas": completed_personas,
        "matrix": matrix,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f)
    
    logger.info(f"Checkpoint saved: {completed_personas} personas completed")


def load_checkpoint(topic_slug: str, output_dir: Path) -> Optional[Dict]:
    """Load a checkpoint if it exists."""
    checkpoint_path = output_dir / f"{topic_slug}_checkpoint.json"
    
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None


def save_results(
    matrix: List[List[int]],
    personas: List[str],
    statements: List[str],
    topic_slug: str,
    output_dir: Path
) -> None:
    """
    Save results in both JSON and CSV formats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON (machine-readable)
    json_path = output_dir / f"{topic_slug}_likert_matrix.json"
    json_data = {
        "metadata": {
            "topic": topic_slug,
            "topic_question": TOPIC_QUESTIONS.get(topic_slug, ""),
            "model": MODEL,
            "n_personas": len(personas),
            "n_statements": len(statements),
            "timestamp": datetime.now().isoformat()
        },
        "matrix": matrix
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    logger.info(f"JSON saved to {json_path}")
    
    # Save CSV (human-readable)
    csv_path = output_dir / f"{topic_slug}_likert_matrix.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header row: persona_idx, S0, S1, ..., S999
        header = ["persona_idx"] + [f"S{i}" for i in range(len(statements))]
        writer.writerow(header)
        
        # Data rows
        for idx, ratings in enumerate(matrix):
            writer.writerow([idx] + ratings)
    
    logger.info(f"CSV saved to {csv_path}")
    
    # Remove checkpoint file after successful save
    checkpoint_path = output_dir / f"{topic_slug}_checkpoint.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint file removed after successful completion")


def run_topic(
    topic_slug: str,
    max_workers: int = 50,
    batch_size: int = 100,
    checkpoint_interval: int = 100,
    n_personas: Optional[int] = None,
    n_statements: Optional[int] = None,
    resume: bool = True
) -> None:
    """
    Run Likert scoring for a single topic.
    
    Args:
        topic_slug: Topic to process
        max_workers: Maximum parallel API calls
        batch_size: Statements per API call
        checkpoint_interval: Save checkpoint every N personas
        n_personas: Limit number of personas (for testing)
        n_statements: Limit number of statements (for testing)
        resume: Whether to resume from checkpoint
    """
    logger.info(f"Starting Likert scoring for topic: {topic_slug}")
    
    # Load data
    personas, statements = load_statements_and_personas(topic_slug)
    
    # Apply limits if specified
    if n_personas:
        personas = personas[:n_personas]
    if n_statements:
        statements = statements[:n_statements]
    
    topic_question = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    
    n_total_personas = len(personas)
    n_total_statements = len(statements)
    n_batches_per_persona = (n_total_statements + batch_size - 1) // batch_size
    total_api_calls = n_total_personas * n_batches_per_persona
    
    logger.info(f"Personas: {n_total_personas}, Statements: {n_total_statements}")
    logger.info(f"Batch size: {batch_size}, Batches per persona: {n_batches_per_persona}")
    logger.info(f"Total API calls: {total_api_calls}")
    
    # Check for checkpoint
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(topic_slug, OUTPUT_DIR) if resume else None
    
    if checkpoint:
        matrix = checkpoint["matrix"]
        start_persona = checkpoint["completed_personas"]
        logger.info(f"Resuming from checkpoint: {start_persona} personas already completed")
    else:
        matrix = []
        start_persona = 0
    
    # Initialize OpenAI client
    openai_client = OpenAI()
    
    # Process personas in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks for remaining personas
        futures = {}
        for idx in range(start_persona, n_total_personas):
            future = executor.submit(
                process_persona,
                idx,
                personas[idx],
                statements,
                topic_question,
                openai_client,
                batch_size
            )
            futures[future] = idx
        
        # Track progress
        completed_since_checkpoint = 0
        
        with tqdm(total=n_total_personas - start_persona, desc="Processing personas", unit="persona") as pbar:
            for future in as_completed(futures):
                try:
                    persona_idx, ratings = future.result()
                    
                    # Ensure matrix is large enough
                    while len(matrix) <= persona_idx:
                        matrix.append(None)
                    matrix[persona_idx] = ratings
                    
                    pbar.update(1)
                    completed_since_checkpoint += 1
                    
                    # Save checkpoint periodically
                    if completed_since_checkpoint >= checkpoint_interval:
                        completed_count = sum(1 for m in matrix if m is not None)
                        save_checkpoint(matrix, completed_count, topic_slug, OUTPUT_DIR)
                        completed_since_checkpoint = 0
                        
                except Exception as e:
                    persona_idx = futures[future]
                    logger.error(f"Failed to process persona {persona_idx}: {e}")
                    # Use default ratings
                    while len(matrix) <= persona_idx:
                        matrix.append(None)
                    matrix[persona_idx] = [3] * n_total_statements
                    pbar.update(1)
    
    # Final save
    save_results(matrix, personas, statements, topic_slug, OUTPUT_DIR)
    
    logger.info(f"Completed topic: {topic_slug}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 1000x1000 Likert score matrices using GPT-5.2"
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="Single topic slug to process"
    )
    parser.add_argument(
        "--all-topics",
        action="store_true",
        help="Process all 13 topics"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=50,
        help="Maximum parallel API calls (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Statements per API call (default: 100)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N personas (default: 100)"
    )
    parser.add_argument(
        "--n-personas",
        type=int,
        help="Limit number of personas (for testing)"
    )
    parser.add_argument(
        "--n-statements",
        type=int,
        help="Limit number of statements (for testing)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any existing checkpoint"
    )
    
    args = parser.parse_args()
    
    if not args.topic and not args.all_topics:
        parser.error("Must specify --topic or --all-topics")
    
    topics_to_process = ALL_TOPICS if args.all_topics else [args.topic]
    
    for topic in topics_to_process:
        if topic not in ALL_TOPICS:
            logger.error(f"Unknown topic: {topic}")
            logger.info(f"Available topics: {ALL_TOPICS}")
            continue
        
        run_topic(
            topic_slug=topic,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval,
            n_personas=args.n_personas,
            n_statements=args.n_statements,
            resume=not args.no_resume
        )


if __name__ == "__main__":
    main()
