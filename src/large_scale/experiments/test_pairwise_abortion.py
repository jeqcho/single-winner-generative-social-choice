"""
Test script for pairwise comparisons - uses small subset for quick verification.

This test:
- Uses only 2 personas (instead of 5)
- Uses only first 10 statements (instead of 900)
- Should produce 10 × 9 = 90 comparisons per persona = 180 total
- Verifies CSV format, API calls, logging
- Expected runtime: ~1-2 minutes
"""

import json
import csv
import os
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from itertools import islice
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI()

# File locks for concurrent CSV writing
file_locks = {}

# Setup logging
def setup_logging():
    """Setup timestamped logging directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "test_pairwise_abortion.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_dir

def load_statements(limit: int = 10) -> List[str]:
    """Load first N statements for abortion topic."""
    statements_file = Path("data/large_scale/prod/statements/what-should-guide-laws-concerning-abortion.json")
    with open(statements_file, 'r') as f:
        data = json.load(f)
    
    # Extract only the "statement" field, not the "persona" field
    statements = [item["statement"] for item in data[:limit]]
    logging.info(f"Loaded {len(statements)} statements (limited to {limit}) from {statements_file}")
    return statements

def load_and_sample_personas(seed: int = 42, num_personas: int = 2) -> List[Dict]:
    """Load discriminative personas and sample a subset."""
    personas_file = Path("data/personas/prod/discriminative.json")
    with open(personas_file, 'r') as f:
        personas = json.load(f)
    
    random.seed(seed)
    sampled = random.sample(list(enumerate(personas)), num_personas)
    logging.info(f"Sampled {num_personas} personas with seed={seed}")
    logging.info(f"Persona indices: {[idx for idx, _ in sampled]}")
    return sampled

def generate_all_pairs(n: int) -> List[Tuple[int, int]]:
    """Generate all ordered pairs (i, j) where i != j for n items."""
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    logging.info(f"Generated {len(pairs)} ordered pairs for {n} statements")
    return pairs

def batch_pairs(pairs: List[Tuple[int, int]], batch_size: int = 10):
    """Yield batches of pairs."""
    it = iter(pairs)
    while batch := list(islice(it, batch_size)):
        yield batch

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
def compare_batch_api(persona_desc: str, statements: List[str], batch: List[Tuple[int, int]]) -> List[Dict]:
    """
    Compare a batch of statement pairs via OpenAI API.
    
    Returns list of preferences: [{"pair": 1, "preference": "A"}, ...]
    """
    # Build prompt
    comparisons_text = []
    for idx, (i, j) in enumerate(batch, 1):
        comparisons_text.append(f"{idx}. A: {statements[i]} | B: {statements[j]}")
    
    comparisons_str = "\n".join(comparisons_text)
    
    prompt = f"""You are: {persona_desc}

Topic: What should guide laws concerning abortion?

Compare these {len(batch)} pairs. For each, you MUST choose either A or B (no ties allowed):

{comparisons_str}

Return a JSON array with your preferred statement for each pair.
Format: [{{"pair": 1, "preference": "A"}}, {{"pair": 2, "preference": "B"}}, ...]"""

    try:
        response = client.responses.create(
            model="gpt-5-nano",
            input=[{"role": "user", "content": prompt}]
        )
        
        # Find the actual content
        content = None
        if hasattr(response, 'output') and response.output:
            # Look for a message output item (type='message', not 'reasoning')
            for output_item in response.output:
                item_type = getattr(output_item, 'type', None)
                if item_type == 'message':
                    # content is a list of ResponseOutputText objects
                    if hasattr(output_item, 'content') and output_item.content:
                        if isinstance(output_item.content, list) and len(output_item.content) > 0:
                            # Extract text from first content item
                            content = output_item.content[0].text
                        else:
                            content = output_item.content
                        break
        
        if not content and hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
        
        if not content:
            logging.error(f"Could not find content in response")
            raise ValueError("Could not extract content from response")
        
        result = json.loads(content)
        
        # Validate result
        if not isinstance(result, list) or len(result) != len(batch):
            logging.warning(f"Invalid response length: expected {len(batch)}, got {len(result)}")
            raise ValueError("Invalid response format")
        
        return result
        
    except Exception as e:
        logging.error(f"API call failed: {e}")
        raise

def write_results_to_csv(persona_idx: int, results: List[Tuple[int, int, int]], test: bool = True):
    """
    Write comparison results to CSV file for a persona.
    
    Args:
        persona_idx: Index of persona
        results: List of (statement_i, statement_j, preferred_id)
        test: If True, prefix filename with 'test_'
    """
    output_dir = Path("data/large_scale/prod")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = "test_" if test else ""
    output_file = output_dir / f"{prefix}pairwise_abortion_persona_{persona_idx}.csv"
    
    # Get or create lock for this file
    if persona_idx not in file_locks:
        file_locks[persona_idx] = Lock()
    
    with file_locks[persona_idx]:
        # Check if file exists to decide on writing header
        file_exists = output_file.exists()
        
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['statement_id_i', 'statement_id_j', 'preferred_statement_id'])
            writer.writerows(results)

def process_batch(persona_idx: int, persona_desc: str, statements: List[str], batch: List[Tuple[int, int]]) -> int:
    """
    Process a single batch of comparisons for a persona.
    
    Returns number of comparisons processed.
    """
    try:
        # Call API
        api_results = compare_batch_api(persona_desc, statements, batch)
        
        # Parse results and map A/B to statement IDs
        csv_rows = []
        for result, (i, j) in zip(api_results, batch):
            preference = result.get('preference', '').upper()
            if preference == 'A':
                preferred_id = i
            elif preference == 'B':
                preferred_id = j
            else:
                # Force a choice if invalid
                logging.warning(f"Invalid preference '{preference}' for pair ({i},{j}), defaulting to A")
                preferred_id = i
            
            csv_rows.append((i, j, preferred_id))
        
        # Write immediately to CSV
        write_results_to_csv(persona_idx, csv_rows, test=True)
        
        return len(batch)
        
    except Exception as e:
        logging.error(f"Failed to process batch for persona {persona_idx}: {e}")
        return 0

def test_pairwise_comparisons(num_personas: int = 2, num_statements: int = 10, seed: int = 42, max_workers: int = 10):
    """
    Test function to compute pairwise comparisons on a small subset.
    
    Args:
        num_personas: Number of personas to sample (default: 2)
        num_statements: Number of statements to use (default: 10)
        seed: Random seed for persona sampling
        max_workers: Number of parallel workers (default: 10)
    """
    log_dir = setup_logging()
    logging.info("=" * 80)
    logging.info("STARTING TEST: O(n^2) PAIRWISE COMPARISON")
    logging.info("=" * 80)
    
    # Load data
    statements = load_statements(limit=num_statements)
    sampled_personas = load_and_sample_personas(seed=seed, num_personas=num_personas)
    
    # Generate all pairs
    all_pairs = generate_all_pairs(len(statements))
    total_comparisons = len(all_pairs) * num_personas
    
    logging.info(f"Total comparisons to compute: {total_comparisons:,}")
    logging.info(f"Batches of 10: {(total_comparisons + 9) // 10:,}")
    logging.info(f"Parallel workers: {max_workers}")
    
    # Create work queue: (persona_idx, persona_desc, batch)
    work_items = []
    for persona_idx, persona_desc in sampled_personas:
        # Batch the pairs
        for batch in batch_pairs(all_pairs, batch_size=10):
            work_items.append((persona_idx, persona_desc, statements, batch))
    
    total_batches = len(work_items)
    logging.info(f"Total batches to process: {total_batches:,}")
    
    # Process batches in parallel
    comparisons_completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all work
        futures = {
            executor.submit(process_batch, persona_idx, persona_desc, statements, batch): (persona_idx, batch)
            for persona_idx, persona_desc, statements, batch in work_items
        }
        
        # Progress bar
        with tqdm(total=total_batches, desc="Processing batches") as pbar:
            for future in as_completed(futures):
                persona_idx, batch = futures[future]
                try:
                    count = future.result()
                    comparisons_completed += count
                    pbar.update(1)
                except Exception as e:
                    logging.error(f"Batch failed for persona {persona_idx}: {e}")
                    pbar.update(1)
    
    logging.info("=" * 80)
    logging.info(f"TEST COMPLETED: {comparisons_completed:,} comparisons processed")
    logging.info(f"Logs saved to: {log_dir}")
    logging.info("=" * 80)
    
    # Verify output files
    logging.info("\nVerifying output files:")
    for persona_idx, _ in sampled_personas:
        output_file = Path(f"data/large_scale/prod/test_pairwise_abortion_persona_{persona_idx}.csv")
        if output_file.exists():
            with open(output_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                logging.info(f"Persona {persona_idx}: {len(rows)} rows in CSV")
                
                # Show first few rows
                logging.info(f"Sample rows for persona {persona_idx}:")
                for row in rows[:3]:
                    logging.info(f"  {row}")
        else:
            logging.warning(f"Output file not found for persona {persona_idx}")

if __name__ == "__main__":
    test_pairwise_comparisons(num_personas=2, num_statements=10, seed=42, max_workers=10)

