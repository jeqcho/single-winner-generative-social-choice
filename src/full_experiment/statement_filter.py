"""
Filter similar statements using LLM-based clustering.
"""

import json
import logging
from typing import List, Dict, Tuple
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

import time

from .config import (
    FILTERING_MODEL,
    TOPIC_QUESTIONS,
    api_timer,
)

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def cluster_statements(
    statements: List[Dict],
    topic_slug: str,
    openai_client: OpenAI
) -> List[Dict]:
    """
    Cluster similar statements using LLM.
    
    Args:
        statements: List of statement dicts with 'statement' key
        topic_slug: Topic slug
        openai_client: OpenAI client
    
    Returns:
        List of assignment dicts with:
        - statement_idx: index of the statement
        - cluster_id: which cluster it belongs to
        - keep: 1 if this is the representative, 0 otherwise
    """
    topic = TOPIC_QUESTIONS.get(topic_slug, topic_slug)
    
    # Format statements
    numbered_statements = "\n".join(
        f"{i}. {stmt['statement']}"
        for i, stmt in enumerate(statements)
    )
    
    system_prompt = "You are an expert at identifying semantically similar statements. Return ONLY valid JSON."
    
    user_prompt = f"""Below are {len(statements)} statements on the topic: "{topic}"

{numbered_statements}

Your task: Identify clusters of statements that express very similar ideas or positions. 
Two statements are "very similar" if they:
- Make essentially the same argument or point
- Differ only in minor wording or phrasing
- Would be considered near-duplicates in a survey

For each statement, assign:
1. A cluster_id (integer starting from 0)
2. A keep flag (1 if this is the best representative of its cluster, 0 otherwise)

Rules:
- Statements with unique positions should be their own cluster (singleton)
- Within each cluster, exactly ONE statement should have keep=1
- The statement with keep=1 should be the clearest/most well-written in that cluster

Return a JSON array with {len(statements)} objects. For example:
[
  {{"statement_idx": 0, "cluster_id": 0, "keep": 1}},
  {{"statement_idx": 1, "cluster_id": 0, "keep": 0}},
  {{"statement_idx": 2, "cluster_id": 1, "keep": 1}},
  ...
]

Return only the JSON array, no other text."""

    logger.info(f"Clustering {len(statements)} statements...")
    
    start_time = time.time()
    response = openai_client.responses.create(
        model=FILTERING_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )
    api_timer.record(time.time() - start_time)
    
    assignments = json.loads(response.output_text)
    
    # Validate assignments
    _validate_assignments(assignments, len(statements))
    
    # Log clustering statistics
    n_clusters = len(set(a["cluster_id"] for a in assignments))
    n_kept = sum(1 for a in assignments if a["keep"] == 1)
    
    logger.info(f"Clustered into {n_clusters} clusters, keeping {n_kept} statements")
    
    return assignments


def _validate_assignments(assignments: List[Dict], n_statements: int) -> None:
    """
    Validate the clustering assignments.
    
    Raises:
        ValueError: If assignments are invalid
    """
    if len(assignments) != n_statements:
        raise ValueError(
            f"Expected {n_statements} assignments, got {len(assignments)}"
        )
    
    # Check all statement indices are present
    indices = set(a["statement_idx"] for a in assignments)
    expected_indices = set(range(n_statements))
    if indices != expected_indices:
        raise ValueError(f"Missing or extra statement indices")
    
    # Check each cluster has exactly one keep=1
    cluster_keeps = {}
    for a in assignments:
        cluster_id = a["cluster_id"]
        if cluster_id not in cluster_keeps:
            cluster_keeps[cluster_id] = 0
        cluster_keeps[cluster_id] += a["keep"]
    
    for cluster_id, keep_count in cluster_keeps.items():
        if keep_count != 1:
            raise ValueError(
                f"Cluster {cluster_id} has {keep_count} keep=1, expected 1"
            )


def apply_filter_to_preferences(
    preferences: List[List[str]],
    assignments: List[Dict]
) -> Tuple[List[List[str]], List[int]]:
    """
    Apply the clustering filter to a preference matrix.
    
    Args:
        preferences: Original preference matrix [rank][voter]
        assignments: Clustering assignments
    
    Returns:
        Tuple of (filtered_preferences, kept_indices)
        - filtered_preferences: Preferences only over kept statements
        - kept_indices: Original indices of kept statements
    """
    # Get indices of statements to keep
    kept_indices = sorted([
        a["statement_idx"]
        for a in assignments
        if a["keep"] == 1
    ])
    kept_set = set(kept_indices)
    
    # Create mapping from old index to new index
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)}
    
    n_voters = len(preferences[0]) if preferences else 0
    
    # Build filtered preferences
    # For each voter, filter their ranking to only include kept statements
    filtered_rankings = []
    for voter in range(n_voters):
        # Get this voter's original ranking (as list of statement indices)
        original_ranking = [int(preferences[rank][voter]) for rank in range(len(preferences))]
        
        # Filter to only kept statements, maintaining relative order
        filtered_ranking = [
            old_to_new[idx]
            for idx in original_ranking
            if idx in kept_set
        ]
        filtered_rankings.append(filtered_ranking)
    
    # Convert back to preferences[rank][voter] format
    n_kept = len(kept_indices)
    filtered_preferences = []
    for rank in range(n_kept):
        rank_row = []
        for voter in range(n_voters):
            rank_row.append(str(filtered_rankings[voter][rank]))
        filtered_preferences.append(rank_row)
    
    logger.info(f"Filtered preferences: {len(preferences)} -> {n_kept} statements")
    
    return filtered_preferences, kept_indices


def apply_filter_to_likert(
    ratings: List[List[int]],
    assignments: List[Dict]
) -> Tuple[List[List[int]], List[int]]:
    """
    Apply the clustering filter to a Likert rating matrix.
    
    Args:
        ratings: Original rating matrix [persona][statement]
        assignments: Clustering assignments
    
    Returns:
        Tuple of (filtered_ratings, kept_indices)
    """
    # Get indices of statements to keep
    kept_indices = sorted([
        a["statement_idx"]
        for a in assignments
        if a["keep"] == 1
    ])
    
    # Filter ratings
    filtered_ratings = [
        [ratings[p_idx][s_idx] for s_idx in kept_indices]
        for p_idx in range(len(ratings))
    ]
    
    logger.info(f"Filtered Likert: {len(ratings[0])} -> {len(kept_indices)} statements")
    
    return filtered_ratings, kept_indices


def create_no_filter_assignments(n_statements: int) -> List[Dict]:
    """
    Create assignments where every statement is its own cluster (no filtering).
    
    Args:
        n_statements: Number of statements
    
    Returns:
        Assignments where each statement is kept
    """
    return [
        {"statement_idx": i, "cluster_id": i, "keep": 1}
        for i in range(n_statements)
    ]


# =============================================================================
# Save/Load Functions
# =============================================================================

def save_filter_assignments(assignments: List[Dict], output_dir: Path) -> None:
    """Save filter assignments to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "filter_assignments.json", 'w') as f:
        json.dump(assignments, f, indent=2)
    logger.info(f"Saved filter assignments to {output_dir}")


def load_filter_assignments(output_dir: Path) -> List[Dict]:
    """Load filter assignments from JSON."""
    with open(output_dir / "filter_assignments.json", 'r') as f:
        return json.load(f)


def save_filtered_preferences(preferences: List[List[str]], output_dir: Path) -> None:
    """Save filtered preferences to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "filtered_preferences.json", 'w') as f:
        json.dump(preferences, f, indent=2)
    logger.info(f"Saved filtered preferences to {output_dir}")


def load_filtered_preferences(output_dir: Path) -> List[List[str]]:
    """Load filtered preferences from JSON."""
    with open(output_dir / "filtered_preferences.json", 'r') as f:
        return json.load(f)


def save_filtered_likert(ratings: List[List[int]], output_dir: Path) -> None:
    """Save filtered Likert ratings to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "filtered_likert.json", 'w') as f:
        json.dump(ratings, f, indent=2)
    logger.info(f"Saved filtered Likert ratings to {output_dir}")


def load_filtered_likert(output_dir: Path) -> List[List[int]]:
    """Load filtered Likert ratings from JSON."""
    with open(output_dir / "filtered_likert.json", 'r') as f:
        return json.load(f)

