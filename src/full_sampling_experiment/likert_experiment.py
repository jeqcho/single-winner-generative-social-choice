"""
Likert experiment module for collecting agreement scores and plotting histograms.

Each voter rates all statements on a 1-10 Likert scale (agreement).
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

from .config import (
    MODEL_LIKERT,
    TEMPERATURE,
    MAX_WORKERS,
    TOPIC_DISPLAY_NAMES,
    api_timer,
)

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def collect_likert_scores_for_voter(
    persona: str,
    statements: List[Dict],
    topic: str,
    openai_client: OpenAI,
    model: str = MODEL_LIKERT,
    temperature: float = TEMPERATURE
) -> Dict[str, int]:
    """
    Collect Likert scores from a single voter for all statements.
    
    Args:
        persona: Voter persona string
        statements: List of statement dicts with 'statement' key
        topic: The topic/question being discussed
        openai_client: OpenAI client instance
        model: Model to use
        temperature: Temperature for sampling
    
    Returns:
        Dict mapping statement ID (as string) to Likert score (1-10)
    """
    n = len(statements)
    
    # Build numbered statement list
    statements_text = "\n".join(
        f"{i}: {stmt['statement']}"
        for i, stmt in enumerate(statements)
    )
    
    system_prompt = "You are rating statements based on the given persona. Return ONLY valid JSON."
    
    user_prompt = f"""You are a person with the following characteristics:
{persona}

Given the topic: "{topic}"

Rate each statement on a scale of 1-10 based on how much you agree with it:
- 1 = Strongly disagree
- 5 = Neutral
- 10 = Strongly agree

Statements:
{statements_text}

Return your ratings as a JSON object with statement IDs as keys:
{{"0": <score>, "1": <score>, ..., "{n-1}": <score>}}

IMPORTANT: Each score must be an integer from 1 to 10. Include all {n} statements."""

    start_time = time.time()
    response = openai_client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        reasoning={"effort": "minimal"},
    )
    api_timer.record(time.time() - start_time)
    
    result = json.loads(response.output_text)
    
    # Validate and convert to proper format
    scores = {}
    for i in range(n):
        key = str(i)
        if key in result:
            score = int(result[key])
            # Clamp to valid range
            score = max(1, min(10, score))
            scores[key] = score
        else:
            # Default to neutral if missing
            logger.warning(f"Missing score for statement {i}, defaulting to 5")
            scores[key] = 5
    
    return scores


def collect_likert_scores(
    personas: List[str],
    statements: List[Dict],
    topic: str,
    openai_client: OpenAI,
    model: str = MODEL_LIKERT,
    max_workers: int = MAX_WORKERS
) -> np.ndarray:
    """
    Collect Likert scores from all voters for all statements.
    
    Args:
        personas: List of voter persona strings
        statements: List of statement dicts
        topic: The topic/question being discussed
        openai_client: OpenAI client instance
        model: Model to use
        max_workers: Maximum parallel workers
    
    Returns:
        numpy array of shape (n_voters, n_statements) with Likert scores 1-10
    """
    n_voters = len(personas)
    n_statements = len(statements)
    
    logger.info(f"Collecting Likert scores: {n_voters} voters x {n_statements} statements")
    
    def process_voter(args):
        """Process a single voter and return (index, scores)."""
        idx, persona = args
        scores = collect_likert_scores_for_voter(
            persona, statements, topic, openai_client, model
        )
        return idx, scores
    
    # Initialize results array
    scores_matrix = np.zeros((n_voters, n_statements), dtype=np.int32)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_voter, (i, persona)): i
            for i, persona in enumerate(personas)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures),
                          desc="Collecting Likert scores", unit="voter"):
            try:
                idx, scores = future.result()
                for stmt_idx in range(n_statements):
                    scores_matrix[idx, stmt_idx] = scores.get(str(stmt_idx), 5)
            except Exception as e:
                logger.error(f"Failed to collect scores for voter {futures[future]}: {e}")
                # Default to neutral scores
                scores_matrix[futures[future], :] = 5
    
    logger.info(f"Collected Likert scores: shape={scores_matrix.shape}")
    logger.info(f"Score distribution: min={scores_matrix.min()}, max={scores_matrix.max()}, mean={scores_matrix.mean():.2f}")
    
    return scores_matrix


def save_likert_scores(
    scores: np.ndarray,
    output_path: Path
) -> None:
    """Save Likert scores to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to list for JSON serialization
    scores_list = scores.tolist()
    
    with open(output_path, 'w') as f:
        json.dump({
            "scores": scores_list,
            "shape": list(scores.shape),
            "n_voters": scores.shape[0],
            "n_statements": scores.shape[1]
        }, f, indent=2)
    
    logger.info(f"Saved Likert scores to {output_path}")


def load_likert_scores(input_path: Path) -> np.ndarray:
    """Load Likert scores from JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    scores = np.array(data["scores"], dtype=np.int32)
    logger.info(f"Loaded Likert scores from {input_path}: shape={scores.shape}")
    
    return scores


def plot_likert_histograms(
    all_topics_scores: Dict[str, np.ndarray],
    output_path: Path,
    figsize: tuple = (12, 20)
) -> None:
    """
    Plot histograms of Likert scores for all topics in a single figure.
    
    Args:
        all_topics_scores: Dict mapping topic_slug to scores array
                          Each array can be (n_voters, n_statements) or flattened
        output_path: Path to save the figure
        figsize: Figure size (width, height)
    """
    n_topics = len(all_topics_scores)
    
    if n_topics == 0:
        logger.warning("No topics to plot")
        return
    
    # Create figure with one row per topic
    fig, axes = plt.subplots(n_topics, 1, figsize=figsize)
    
    # Handle single topic case
    if n_topics == 1:
        axes = [axes]
    
    # Sort topics by display name for consistent ordering
    sorted_topics = sorted(all_topics_scores.keys(), 
                          key=lambda x: TOPIC_DISPLAY_NAMES.get(x, x))
    
    for idx, topic_slug in enumerate(sorted_topics):
        ax = axes[idx]
        scores = all_topics_scores[topic_slug]
        
        # Flatten if needed
        if scores.ndim > 1:
            scores = scores.flatten()
        
        # Get display name
        display_name = TOPIC_DISPLAY_NAMES.get(topic_slug, topic_slug)
        
        # Plot histogram
        bins = np.arange(0.5, 11.5, 1)  # Bins centered on 1-10
        counts, _, patches = ax.hist(scores, bins=bins, density=True, 
                                     alpha=0.7, color='steelblue', edgecolor='black')
        
        # Add mean line
        mean_score = np.mean(scores)
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_score:.2f}')
        
        # Formatting
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel('Likert Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{display_name} (n={len(scores):,})')
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Likert histograms to {output_path}")


def plot_likert_comparison(
    main_scores: Dict[str, np.ndarray],
    variant_scores: Dict[str, np.ndarray],
    output_path: Path,
    main_label: str = "Original",
    variant_label: str = "Bridging",
    figsize: tuple = (14, 6)
) -> None:
    """
    Plot comparison of Likert score distributions between main and variant experiments.
    
    Args:
        main_scores: Dict mapping topic_slug to scores array for main experiment
        variant_scores: Dict mapping topic_slug to scores array for variant
        output_path: Path to save the figure
        main_label: Label for main experiment
        variant_label: Label for variant experiment
        figsize: Figure size
    """
    # Get common topics
    common_topics = set(main_scores.keys()) & set(variant_scores.keys())
    
    if not common_topics:
        logger.warning("No common topics between main and variant")
        return
    
    n_topics = len(common_topics)
    
    fig, axes = plt.subplots(1, n_topics, figsize=figsize)
    
    if n_topics == 1:
        axes = [axes]
    
    sorted_topics = sorted(common_topics, 
                          key=lambda x: TOPIC_DISPLAY_NAMES.get(x, x))
    
    for idx, topic_slug in enumerate(sorted_topics):
        ax = axes[idx]
        
        main = main_scores[topic_slug].flatten()
        variant = variant_scores[topic_slug].flatten()
        
        display_name = TOPIC_DISPLAY_NAMES.get(topic_slug, topic_slug)
        
        # Plot overlapping histograms
        bins = np.arange(0.5, 11.5, 1)
        
        ax.hist(main, bins=bins, density=True, alpha=0.5, 
                color='blue', edgecolor='blue', label=main_label)
        ax.hist(variant, bins=bins, density=True, alpha=0.5, 
                color='orange', edgecolor='orange', label=variant_label)
        
        # Add mean lines
        ax.axvline(np.mean(main), color='blue', linestyle='--', linewidth=2)
        ax.axvline(np.mean(variant), color='orange', linestyle='--', linewidth=2)
        
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel('Likert Score')
        ax.set_ylabel('Density')
        ax.set_title(display_name)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Likert comparison to {output_path}")
