"""
Load and split personas from HuggingFace SynthLabsAI/PERSONA dataset.
"""

import json
import os
import random
from typing import Dict, List, Tuple
from pathlib import Path


def load_personas_from_huggingface(cache_dir: str = "data/personas") -> List[str]:
    """
    Load unique personas from HuggingFace SynthLabsAI/PERSONA dataset.
    
    Args:
        cache_dir: Directory to cache the dataset
    
    Returns:
        List of unique persona strings
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    print("Loading SynthLabsAI/PERSONA dataset from HuggingFace...")
    dataset = load_dataset("SynthLabsAI/PERSONA", cache_dir=cache_dir)
    
    # Extract unique personas from the 'persona' column
    # The dataset might have different splits, so we'll check all of them
    all_personas = set()
    
    for split_name in dataset.keys():
        split = dataset[split_name]
        if 'persona' in split.column_names:
            personas = split['persona']
            all_personas.update(personas)
            print(f"  Found {len(personas)} personas in '{split_name}' split")
    
    unique_personas = list(all_personas)
    print(f"Total unique personas: {len(unique_personas)}")
    
    return unique_personas


def split_personas(
    personas: List[str],
    n_generative: int,
    n_discriminative: int,
    n_evaluative: int,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Randomly split personas into three groups.
    
    Args:
        personas: List of unique persona strings
        n_generative: Number of generative personas
        n_discriminative: Number of discriminative personas
        n_evaluative: Number of evaluative personas
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (generative, discriminative, evaluative) persona lists
    """
    total_needed = n_generative + n_discriminative + n_evaluative
    
    if len(personas) < total_needed:
        raise ValueError(
            f"Not enough unique personas. Need {total_needed}, but only have {len(personas)}"
        )
    
    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    shuffled = personas.copy()
    random.shuffle(shuffled)
    
    # Split into three groups
    generative = shuffled[:n_generative]
    discriminative = shuffled[n_generative:n_generative + n_discriminative]
    evaluative = shuffled[n_generative + n_discriminative:n_generative + n_discriminative + n_evaluative]
    
    print(f"\nSplit personas:")
    print(f"  Generative: {len(generative)}")
    print(f"  Discriminative: {len(discriminative)}")
    print(f"  Evaluative: {len(evaluative)}")
    
    return generative, discriminative, evaluative


def save_persona_splits(
    generative: List[str],
    discriminative: List[str],
    evaluative: List[str],
    output_dir: str = "data/personas",
    test_mode: bool = False
) -> None:
    """
    Save persona splits to JSON files.
    
    Args:
        generative: List of generative persona strings
        discriminative: List of discriminative persona strings
        evaluative: List of evaluative persona strings
        output_dir: Base directory to save persona splits (will add /test or /prod)
        test_mode: If True, save to test/ subdirectory, otherwise prod/
    """
    # Determine subdirectory based on mode
    subdir = "test" if test_mode else "prod"
    full_output_dir = os.path.join(output_dir, subdir)
    os.makedirs(full_output_dir, exist_ok=True)
    
    splits = {
        "generative.json": generative,
        "discriminative.json": discriminative,
        "evaluative.json": evaluative
    }
    
    for filename, personas in splits.items():
        filepath = os.path.join(full_output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(personas, f, indent=2)
        print(f"Saved {len(personas)} personas to {filepath}")
    
    print(f"\nPersona splits saved to {full_output_dir}")


def load_persona_splits(
    input_dir: str = "data/personas",
    test_mode: bool = False
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load persona splits from JSON files.
    
    Args:
        input_dir: Base directory containing persona split files
        test_mode: If True, load from test/ subdirectory, otherwise prod/
    
    Returns:
        Tuple of (generative, discriminative, evaluative) persona lists
    """
    # Determine subdirectory based on mode
    subdir = "test" if test_mode else "prod"
    full_input_dir = os.path.join(input_dir, subdir)
    
    generative_path = os.path.join(full_input_dir, "generative.json")
    discriminative_path = os.path.join(full_input_dir, "discriminative.json")
    evaluative_path = os.path.join(full_input_dir, "evaluative.json")
    
    with open(generative_path, 'r') as f:
        generative = json.load(f)
    
    with open(discriminative_path, 'r') as f:
        discriminative = json.load(f)
    
    with open(evaluative_path, 'r') as f:
        evaluative = json.load(f)
    
    print(f"Loaded persona splits:")
    print(f"  Generative: {len(generative)}")
    print(f"  Discriminative: {len(discriminative)}")
    print(f"  Evaluative: {len(evaluative)}")
    
    return generative, discriminative, evaluative


def main():
    """Main entry point for persona loading and splitting."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and split personas from HuggingFace")
    parser.add_argument(
        "--n-generative",
        type=int,
        default=20,
        help="Number of generative personas (default: 20 for testing, 900 for production)"
    )
    parser.add_argument(
        "--n-discriminative",
        type=int,
        default=5,
        help="Number of discriminative personas (default: 5 for testing, 50 for production)"
    )
    parser.add_argument(
        "--n-evaluative",
        type=int,
        default=5,
        help="Number of evaluative personas (default: 5 for testing, 50 for production)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/personas",
        help="Output directory for persona splits (default: data/personas)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Load personas from HuggingFace
    personas = load_personas_from_huggingface()
    
    # Split personas
    generative, discriminative, evaluative = split_personas(
        personas,
        args.n_generative,
        args.n_discriminative,
        args.n_evaluative,
        args.seed
    )
    
    # Save splits
    save_persona_splits(generative, discriminative, evaluative, args.output_dir)
    
    print(f"\nPersona splits saved to {args.output_dir}")


if __name__ == "__main__":
    main()


