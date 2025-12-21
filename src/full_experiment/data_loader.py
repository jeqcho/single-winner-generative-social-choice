"""
Data loading and sampling utilities for the full experiment.

Each entry in the statements file contains BOTH persona and statement bundled together.
We sample entries, and extract both from the same sampled indices.
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple

from .config import (
    STATEMENTS_DIR,
    N_STATEMENTS,
    BASE_SEED,
)

logger = logging.getLogger(__name__)


def load_all_statements(topic_slug: str) -> List[Dict]:
    """
    Load all statements for a given topic.
    
    Args:
        topic_slug: The topic slug (filename without .json)
    
    Returns:
        List of statement dicts with 'persona' and 'statement' keys
    """
    filepath = STATEMENTS_DIR / f"{topic_slug}.json"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Statements file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        statements = json.load(f)
    
    logger.info(f"Loaded {len(statements)} statements from {filepath}")
    return statements


def sample_entries(
    all_entries: List[Dict],
    n_entries: int = N_STATEMENTS,
    seed: int = BASE_SEED
) -> Tuple[List[int], List[Dict], List[str]]:
    """
    Sample entries from the full list. Each entry has both persona and statement.
    
    Args:
        all_entries: Full list of entry dicts (each has 'persona' and 'statement')
        n_entries: Number of entries to sample
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (sampled_indices, sampled_statements, sampled_personas)
        - sampled_indices: indices into original file
        - sampled_statements: list of statement dicts (with 'statement' key)
        - sampled_personas: list of persona strings
    """
    random.seed(seed)
    
    if len(all_entries) < n_entries:
        raise ValueError(
            f"Not enough entries. Need {n_entries}, have {len(all_entries)}"
        )
    
    # Sample indices
    all_indices = list(range(len(all_entries)))
    sampled_indices = sorted(random.sample(all_indices, n_entries))
    
    # Extract both statements and personas from sampled entries
    sampled_entries = [all_entries[i] for i in sampled_indices]
    sampled_statements = [{"statement": e["statement"]} for e in sampled_entries]
    sampled_personas = [e["persona"] for e in sampled_entries]
    
    logger.info(f"Sampled {n_entries} entries with seed {seed}")
    
    return sampled_indices, sampled_statements, sampled_personas


def save_sampled_data(
    output_dir: Path,
    sampled_indices: List[int],
) -> None:
    """
    Save sampled indices to JSON file.
    
    Only the indices are saved - statements and personas can be derived
    from the original file using these indices.
    
    Args:
        output_dir: Directory to save files
        sampled_indices: Indices of sampled entries
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save indices only (statements and personas are derived from these)
    with open(output_dir / "sampled_indices.json", 'w') as f:
        json.dump(sampled_indices, f, indent=2)
    
    logger.info(f"Saved sampled indices to {output_dir}")


def load_sampled_data(
    output_dir: Path,
    all_entries: List[Dict]
) -> Tuple[List[int], List[Dict], List[str]]:
    """
    Load previously sampled data from JSON files.
    
    Args:
        output_dir: Directory containing saved files
        all_entries: Full list of entries to index into
    
    Returns:
        Tuple of (sampled_indices, sampled_statements, sampled_personas)
    """
    with open(output_dir / "sampled_indices.json", 'r') as f:
        sampled_indices = json.load(f)
    
    # Derive statements and personas from indices
    sampled_entries = [all_entries[i] for i in sampled_indices]
    sampled_statements = [{"statement": e["statement"]} for e in sampled_entries]
    sampled_personas = [e["persona"] for e in sampled_entries]
    
    logger.info(f"Loaded sampled data from {output_dir}")
    
    return sampled_indices, sampled_statements, sampled_personas


def check_cache_exists(output_dir: Path, filename: str) -> bool:
    """
    Check if a cached file exists.
    
    Args:
        output_dir: Directory to check
        filename: Name of the file to check
    
    Returns:
        True if file exists, False otherwise
    """
    return (output_dir / filename).exists()


def load_json_cache(output_dir: Path, filename: str):
    """
    Load a cached JSON file.
    
    Args:
        output_dir: Directory containing the file
        filename: Name of the file
    
    Returns:
        Parsed JSON content
    """
    with open(output_dir / filename, 'r') as f:
        return json.load(f)


def save_json_cache(output_dir: Path, filename: str, data) -> None:
    """
    Save data to a JSON cache file.
    
    Args:
        output_dir: Directory to save to
        filename: Name of the file
        data: Data to save (must be JSON serializable)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / filename, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {filename} to {output_dir}")

