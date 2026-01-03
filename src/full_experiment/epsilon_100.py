"""
Compute epsilon values with respect to 100 personas using winners from N personas.

This module provides functions to compute and collect epsilon-100 values,
which measure how well a winner (selected using N sampled personas) performs
when evaluated against the full 100 persona preference profile.

Supports multiple persona counts (5, 10, 20) with corresponding directory structures:
- 20 personas: rep{i}/sample{j}/ (standard)
- 10 personas: rep{i}/10-personas/sample{j}/
- 5 personas: rep{i}/5-personas/sample{j}/
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

from pvc_toolbox import compute_critical_epsilon

from .config import VOTING_METHODS, OUTPUT_DIR

logger = logging.getLogger(__name__)

# Standard persona counts to support
PERSONA_COUNTS = [5, 10, 20]


def get_full_preferences_path(rep_dir: Path, ablation: str) -> Path:
    """
    Get the path to the 100-persona preferences file for a given ablation.
    
    Args:
        rep_dir: Path to the rep directory (e.g., data/topic/rep0)
        ablation: Ablation type ('full', 'no_filtering', 'no_bridging')
    
    Returns:
        Path to the preferences JSON file
    """
    if ablation == "full":
        # Full ablation uses filtered preferences
        return rep_dir / "filtered_preferences.json"
    elif ablation == "no_filtering":
        # No filtering uses full preferences (no filtering applied)
        return rep_dir / "full_preferences.json"
    elif ablation == "no_bridging":
        # No bridging uses its own full preferences
        return rep_dir / "ablation_no_bridging" / "full_preferences.json"
    else:
        raise ValueError(f"Unknown ablation: {ablation}")


def get_sample_results_dir(rep_dir: Path, ablation: str) -> Path:
    """
    Get the directory containing sample results for a given ablation.
    
    Args:
        rep_dir: Path to the rep directory (e.g., data/topic/rep0)
        ablation: Ablation type ('full', 'no_filtering', 'no_bridging')
    
    Returns:
        Path to the directory containing sample subdirectories
    """
    if ablation == "full":
        return rep_dir
    elif ablation == "no_filtering":
        return rep_dir / "ablation_no_filtering"
    elif ablation == "no_bridging":
        return rep_dir / "ablation_no_bridging"
    else:
        raise ValueError(f"Unknown ablation: {ablation}")


def get_sample_results_dir_for_n_personas(
    rep_dir: Path,
    ablation: str,
    n_personas: int
) -> Path:
    """
    Get the directory containing sample results for a given ablation and persona count.
    
    Args:
        rep_dir: Path to the rep directory (e.g., data/topic/rep0)
        ablation: Ablation type ('full', 'no_filtering', 'no_bridging')
        n_personas: Number of personas (5, 10, or 20)
    
    Returns:
        Path to the directory containing sample subdirectories
    """
    base_dir = get_sample_results_dir(rep_dir, ablation)
    
    if n_personas == 20:
        # Standard 20 personas go directly in base_dir
        return base_dir
    else:
        # Other counts go in subdirectories
        return base_dir / f"{n_personas}-personas"


def compute_epsilon_100_for_winner(
    full_preferences: List[List[str]],
    winner: str
) -> Optional[float]:
    """
    Compute epsilon for a winner using all 100 personas.
    
    Args:
        full_preferences: Preference matrix [rank][voter] with all 100 personas
        winner: Winner statement index (as string)
    
    Returns:
        Critical epsilon value, or None if computation fails
    """
    if winner is None:
        return None
    
    n_statements = len(full_preferences)
    alternatives = [str(i) for i in range(n_statements)]
    
    try:
        epsilon = compute_critical_epsilon(full_preferences, alternatives, winner)
        return epsilon
    except Exception as e:
        logger.error(f"Epsilon-100 computation failed for winner {winner}: {e}")
        return None


def collect_epsilon_100_for_topic(
    topic_slug: str,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full"
) -> Dict[str, List[float]]:
    """
    Collect epsilon-100 values for a topic across all repetitions and samples.
    
    Args:
        topic_slug: Topic slug
        output_dir: Output directory
        ablation: Ablation type
    
    Returns:
        Dict mapping method name to list of epsilon-100 values
    """
    results = {method: [] for method in VOTING_METHODS}
    
    topic_dir = output_dir / "data" / topic_slug
    
    if not topic_dir.exists():
        logger.warning(f"Topic directory not found: {topic_dir}")
        return results
    
    # Iterate through all rep directories
    for rep_dir in sorted(topic_dir.glob("rep*")):
        # Load 100-persona preferences
        prefs_path = get_full_preferences_path(rep_dir, ablation)
        if not prefs_path.exists():
            logger.warning(f"Preferences file not found: {prefs_path}")
            continue
        
        with open(prefs_path, 'r') as f:
            full_preferences = json.load(f)
        
        # Get sample results directory
        sample_base = get_sample_results_dir(rep_dir, ablation)
        
        # Iterate through all sample directories
        for sample_dir in sorted(sample_base.glob("sample*")):
            results_file = sample_dir / "results.json"
            if not results_file.exists():
                continue
            
            with open(results_file, 'r') as f:
                sample_results = json.load(f)
            
            # Compute epsilon-100 for each method's winner
            for method in VOTING_METHODS:
                if method in sample_results:
                    winner = sample_results[method].get("winner")
                    if winner is not None:
                        epsilon_100 = compute_epsilon_100_for_winner(
                            full_preferences, winner
                        )
                        if epsilon_100 is not None:
                            results[method].append(epsilon_100)
    
    return results


def collect_epsilon_100_clustered_for_topic(
    topic_slug: str,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full"
) -> Dict[str, List[List[float]]]:
    """
    Collect epsilon-100 values clustered by repetition for a topic.
    
    This is useful for computing cluster-aware confidence intervals.
    
    Args:
        topic_slug: Topic slug
        output_dir: Output directory
        ablation: Ablation type
    
    Returns:
        Dict mapping method name to list of lists (outer: reps, inner: samples)
    """
    results = {method: [] for method in VOTING_METHODS}
    
    topic_dir = output_dir / "data" / topic_slug
    
    if not topic_dir.exists():
        logger.warning(f"Topic directory not found: {topic_dir}")
        return results
    
    # Iterate through all rep directories
    for rep_dir in sorted(topic_dir.glob("rep*")):
        # Load 100-persona preferences
        prefs_path = get_full_preferences_path(rep_dir, ablation)
        if not prefs_path.exists():
            continue
        
        with open(prefs_path, 'r') as f:
            full_preferences = json.load(f)
        
        # Get sample results directory
        sample_base = get_sample_results_dir(rep_dir, ablation)
        
        # Collect all samples for this rep
        rep_results = {method: [] for method in VOTING_METHODS}
        
        for sample_dir in sorted(sample_base.glob("sample*")):
            results_file = sample_dir / "results.json"
            if not results_file.exists():
                continue
            
            with open(results_file, 'r') as f:
                sample_results = json.load(f)
            
            for method in VOTING_METHODS:
                if method in sample_results:
                    winner = sample_results[method].get("winner")
                    if winner is not None:
                        epsilon_100 = compute_epsilon_100_for_winner(
                            full_preferences, winner
                        )
                        if epsilon_100 is not None:
                            rep_results[method].append(epsilon_100)
        
        # Add this rep's results to the clustered results
        for method in VOTING_METHODS:
            if rep_results[method]:
                results[method].append(rep_results[method])
    
    return results


def collect_all_epsilon_100(
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full",
    topics: Optional[List[str]] = None
) -> Dict[str, List[float]]:
    """
    Collect all epsilon-100 values across all topics.
    
    Args:
        output_dir: Output directory
        ablation: Ablation type
        topics: List of topics to include (None = all)
    
    Returns:
        Dict mapping method name to list of epsilon-100 values
    """
    all_results = {method: [] for method in VOTING_METHODS}
    
    data_dir = output_dir / "data"
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return all_results
    
    # Get all topic directories
    if topics is None:
        topic_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    else:
        topic_dirs = [data_dir / t for t in topics if (data_dir / t).exists()]
    
    for topic_dir in topic_dirs:
        topic_results = collect_epsilon_100_for_topic(
            topic_dir.name, output_dir, ablation
        )
        for method in VOTING_METHODS:
            all_results[method].extend(topic_results[method])
    
    return all_results


def collect_all_epsilon_100_clustered(
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full",
    topics: Optional[List[str]] = None
) -> Dict[str, List[List[float]]]:
    """
    Collect all epsilon-100 values clustered by repetition across all topics.
    
    Args:
        output_dir: Output directory
        ablation: Ablation type
        topics: List of topics to include (None = all)
    
    Returns:
        Dict mapping method name to list of lists (outer: reps, inner: samples)
    """
    all_results = {method: [] for method in VOTING_METHODS}
    
    data_dir = output_dir / "data"
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return all_results
    
    # Get all topic directories
    if topics is None:
        topic_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    else:
        topic_dirs = [data_dir / t for t in topics if (data_dir / t).exists()]
    
    for topic_dir in topic_dirs:
        topic_results = collect_epsilon_100_clustered_for_topic(
            topic_dir.name, output_dir, ablation
        )
        for method in VOTING_METHODS:
            all_results[method].extend(topic_results[method])
    
    return all_results


# =============================================================================
# Multi-Persona Collection Functions
# =============================================================================


def collect_epsilon_100_for_n_personas_topic(
    topic_slug: str,
    n_personas: int,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full"
) -> Dict[str, List[float]]:
    """
    Collect epsilon-100 values for a specific persona count for a topic.
    
    Args:
        topic_slug: Topic slug
        n_personas: Number of personas (5, 10, or 20)
        output_dir: Output directory
        ablation: Ablation type
    
    Returns:
        Dict mapping method name to list of epsilon-100 values
    """
    results = {method: [] for method in VOTING_METHODS}
    
    topic_dir = output_dir / "data" / topic_slug
    
    if not topic_dir.exists():
        logger.warning(f"Topic directory not found: {topic_dir}")
        return results
    
    # Iterate through all rep directories
    for rep_dir in sorted(topic_dir.glob("rep*")):
        # Load 100-persona preferences
        prefs_path = get_full_preferences_path(rep_dir, ablation)
        if not prefs_path.exists():
            logger.warning(f"Preferences file not found: {prefs_path}")
            continue
        
        with open(prefs_path, 'r') as f:
            full_preferences = json.load(f)
        
        # Get sample results directory for this persona count
        sample_base = get_sample_results_dir_for_n_personas(rep_dir, ablation, n_personas)
        
        if not sample_base.exists():
            continue
        
        # Iterate through all sample directories
        for sample_dir in sorted(sample_base.glob("sample*")):
            results_file = sample_dir / "results.json"
            if not results_file.exists():
                continue
            
            with open(results_file, 'r') as f:
                sample_results = json.load(f)
            
            # Compute epsilon-100 for each method's winner
            for method in VOTING_METHODS:
                if method in sample_results:
                    winner = sample_results[method].get("winner")
                    if winner is not None:
                        epsilon_100 = compute_epsilon_100_for_winner(
                            full_preferences, winner
                        )
                        if epsilon_100 is not None:
                            results[method].append(epsilon_100)
    
    return results


def collect_epsilon_100_for_n_personas_clustered_topic(
    topic_slug: str,
    n_personas: int,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full"
) -> Dict[str, List[List[float]]]:
    """
    Collect epsilon-100 values clustered by repetition for a specific persona count.
    
    Args:
        topic_slug: Topic slug
        n_personas: Number of personas (5, 10, or 20)
        output_dir: Output directory
        ablation: Ablation type
    
    Returns:
        Dict mapping method name to list of lists (outer: reps, inner: samples)
    """
    results = {method: [] for method in VOTING_METHODS}
    
    topic_dir = output_dir / "data" / topic_slug
    
    if not topic_dir.exists():
        logger.warning(f"Topic directory not found: {topic_dir}")
        return results
    
    # Iterate through all rep directories
    for rep_dir in sorted(topic_dir.glob("rep*")):
        # Load 100-persona preferences
        prefs_path = get_full_preferences_path(rep_dir, ablation)
        if not prefs_path.exists():
            continue
        
        with open(prefs_path, 'r') as f:
            full_preferences = json.load(f)
        
        # Get sample results directory for this persona count
        sample_base = get_sample_results_dir_for_n_personas(rep_dir, ablation, n_personas)
        
        if not sample_base.exists():
            continue
        
        # Collect all samples for this rep
        rep_results = {method: [] for method in VOTING_METHODS}
        
        for sample_dir in sorted(sample_base.glob("sample*")):
            results_file = sample_dir / "results.json"
            if not results_file.exists():
                continue
            
            with open(results_file, 'r') as f:
                sample_results = json.load(f)
            
            for method in VOTING_METHODS:
                if method in sample_results:
                    winner = sample_results[method].get("winner")
                    if winner is not None:
                        epsilon_100 = compute_epsilon_100_for_winner(
                            full_preferences, winner
                        )
                        if epsilon_100 is not None:
                            rep_results[method].append(epsilon_100)
        
        # Add this rep's results to the clustered results
        for method in VOTING_METHODS:
            if rep_results[method]:
                results[method].append(rep_results[method])
    
    return results


def collect_all_epsilon_100_for_n_personas(
    n_personas: int,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full",
    topics: Optional[List[str]] = None
) -> Dict[str, List[float]]:
    """
    Collect all epsilon-100 values for a specific persona count across all topics.
    
    Args:
        n_personas: Number of personas (5, 10, or 20)
        output_dir: Output directory
        ablation: Ablation type
        topics: List of topics to include (None = all)
    
    Returns:
        Dict mapping method name to list of epsilon-100 values
    """
    all_results = {method: [] for method in VOTING_METHODS}
    
    data_dir = output_dir / "data"
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return all_results
    
    # Get all topic directories
    if topics is None:
        topic_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    else:
        topic_dirs = [data_dir / t for t in topics if (data_dir / t).exists()]
    
    for topic_dir in topic_dirs:
        topic_results = collect_epsilon_100_for_n_personas_topic(
            topic_dir.name, n_personas, output_dir, ablation
        )
        for method in VOTING_METHODS:
            all_results[method].extend(topic_results[method])
    
    return all_results


def collect_all_epsilon_100_for_n_personas_clustered(
    n_personas: int,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full",
    topics: Optional[List[str]] = None
) -> Dict[str, List[List[float]]]:
    """
    Collect all epsilon-100 values clustered by repetition for a specific persona count.
    
    Args:
        n_personas: Number of personas (5, 10, or 20)
        output_dir: Output directory
        ablation: Ablation type
        topics: List of topics to include (None = all)
    
    Returns:
        Dict mapping method name to list of lists (outer: reps, inner: samples)
    """
    all_results = {method: [] for method in VOTING_METHODS}
    
    data_dir = output_dir / "data"
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return all_results
    
    # Get all topic directories
    if topics is None:
        topic_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    else:
        topic_dirs = [data_dir / t for t in topics if (data_dir / t).exists()]
    
    for topic_dir in topic_dirs:
        topic_results = collect_epsilon_100_for_n_personas_clustered_topic(
            topic_dir.name, n_personas, output_dir, ablation
        )
        for method in VOTING_METHODS:
            all_results[method].extend(topic_results[method])
    
    return all_results

