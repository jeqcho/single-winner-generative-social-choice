"""
Get preference rankings from discriminative personas using pairwise comparisons.
"""

import json
import os
from typing import List, Dict
from openai import OpenAI
from src.large_scale.pairwise_ranking import get_preference_matrix_pairwise


def get_discriminative_rankings(
    personas: List[str],
    statements: List[Dict],
    topic: str,
    openai_client: OpenAI
) -> List[List[str]]:
    """
    Get preference rankings from discriminative personas.
    
    Args:
        personas: List of discriminative persona strings
        statements: List of statement dicts
        topic: The topic/question
        openai_client: OpenAI client instance
    
    Returns:
        Preference matrix where preferences[rank][voter] is the alternative at rank 'rank' for voter 'voter'
    """
    return get_preference_matrix_pairwise(personas, statements, topic, openai_client)


def save_preferences(
    preferences: List[List[str]],
    topic_slug: str,
    output_dir: str = "data/large_scale/preferences"
) -> None:
    """
    Save preference matrix to JSON file.
    
    Args:
        preferences: Preference matrix
        topic_slug: Slugified topic name for filename
        output_dir: Directory to save preferences
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{topic_slug}.json")
    
    with open(filepath, 'w') as f:
        json.dump(preferences, f, indent=2)
    
    print(f"Preferences saved to {filepath}")


def load_preferences(
    topic_slug: str,
    input_dir: str = "data/large_scale/preferences"
) -> List[List[str]]:
    """
    Load preference matrix from JSON file.
    
    Args:
        topic_slug: Slugified topic name for filename
        input_dir: Directory containing preferences
    
    Returns:
        Preference matrix
    """
    filepath = os.path.join(input_dir, f"{topic_slug}.json")
    
    with open(filepath, 'r') as f:
        preferences = json.load(f)
    
    n_alternatives = len(preferences)
    n_voters = len(preferences[0]) if preferences else 0
    print(f"Loaded preferences from {filepath}: {n_alternatives} alternatives, {n_voters} voters")
    
    return preferences


