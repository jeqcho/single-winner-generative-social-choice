"""
Ideology classifier for adult personas.

Classifies personas into two ideological clusters based on the 'ideology' field:
- Progressive/Liberal: Progressive, Liberal, Social Justice, Feminist, etc.
- Conservative/Traditional: Conservative, Traditional, Libertarian, etc.

Personas not matching either cluster are classified as "other".
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from .config import PERSONAS_PATH, DATA_DIR

logger = logging.getLogger(__name__)

# =============================================================================
# Ideology Keywords
# =============================================================================

# Keywords that indicate Progressive/Liberal ideology (case-insensitive matching)
PROGRESSIVE_LIBERAL_KEYWORDS = [
    "progressive",
    "liberal",
    "social liberal",
    "social justice",
    "feminist",
    "feminism",
    "egalitarian",
    "environmentalism",
    "environmentalist",
    "environmental conservation",
    "environmental protection",
    "social equality",
    "workers' rights",
    "social welfare",
    "humanism",
    "humanistic",
    "humanitarian",
]

# Keywords that indicate Conservative/Traditional ideology (case-insensitive matching)
CONSERVATIVE_TRADITIONAL_KEYWORDS = [
    "conservative",
    "traditional",
    "traditionalist",
    "libertarian",
    "libertarianism",
    "fiscal conservatism",
    "social conservatism",
    "christian values",
    "family values",
    "small government",
]

# Exact matches that should be classified as Progressive/Liberal
PROGRESSIVE_LIBERAL_EXACT = {
    "Liberal",
    "Progressive", 
    "Social Liberal",
    "Social Justice",
    "Feminist",
    "Feminism",
    "Egalitarian",
    "Environmentalism",
    "Environmentalist",
    "Humanism",
    "Humanistic",
}

# Exact matches that should be classified as Conservative/Traditional
CONSERVATIVE_TRADITIONAL_EXACT = {
    "Conservative",
    "Traditional",
    "Traditionalist",
    "Libertarian",
    "Libertarianism",
    "Social Conservatism",
}


def extract_ideology(persona_text: str) -> str:
    """
    Extract the ideology value from a persona text.
    
    Args:
        persona_text: Full persona description string
        
    Returns:
        The ideology value, or empty string if not found
    """
    for line in persona_text.split('\n'):
        if line.startswith('ideology:'):
            return line.split(':', 1)[1].strip()
    return ""


def classify_ideology(ideology: str) -> str:
    """
    Classify an ideology string into one of three categories.
    
    Args:
        ideology: The ideology string from the persona
        
    Returns:
        One of: "progressive_liberal", "conservative_traditional", "other"
    """
    if not ideology:
        return "other"
    
    ideology_lower = ideology.lower()
    
    # Check exact matches first
    if ideology in PROGRESSIVE_LIBERAL_EXACT:
        return "progressive_liberal"
    if ideology in CONSERVATIVE_TRADITIONAL_EXACT:
        return "conservative_traditional"
    
    # Check keyword matches (case-insensitive)
    for keyword in PROGRESSIVE_LIBERAL_KEYWORDS:
        if keyword in ideology_lower:
            return "progressive_liberal"
    
    for keyword in CONSERVATIVE_TRADITIONAL_KEYWORDS:
        if keyword in ideology_lower:
            return "conservative_traditional"
    
    return "other"


def classify_persona(persona_text: str) -> str:
    """
    Classify a persona into an ideology cluster.
    
    Args:
        persona_text: Full persona description string
        
    Returns:
        One of: "progressive_liberal", "conservative_traditional", "other"
    """
    ideology = extract_ideology(persona_text)
    return classify_ideology(ideology)


def get_ideology_clusters(personas: List[str]) -> Dict[str, List[int]]:
    """
    Classify all personas and return cluster assignments.
    
    Args:
        personas: List of persona description strings
        
    Returns:
        Dict mapping cluster names to lists of persona indices:
        {
            "progressive_liberal": [idx1, idx2, ...],
            "conservative_traditional": [idx1, idx2, ...],
            "other": [idx1, idx2, ...]
        }
    """
    clusters = {
        "progressive_liberal": [],
        "conservative_traditional": [],
        "other": []
    }
    
    for idx, persona in enumerate(personas):
        cluster = classify_persona(persona)
        clusters[cluster].append(idx)
    
    return clusters


def load_personas() -> List[str]:
    """Load personas from the configured path."""
    with open(PERSONAS_PATH) as f:
        return json.load(f)


def get_cluster_stats(clusters: Dict[str, List[int]]) -> Dict[str, int]:
    """Get counts for each cluster."""
    return {name: len(indices) for name, indices in clusters.items()}


def save_cluster_assignments(
    clusters: Dict[str, List[int]], 
    output_path: Path = None
) -> None:
    """
    Save cluster assignments to a JSON file.
    
    Args:
        clusters: Dict mapping cluster names to persona indices
        output_path: Path to save to (defaults to data/sample-alt-voters/ideology_clusters.json)
    """
    if output_path is None:
        output_path = DATA_DIR / "sample-alt-voters" / "ideology_clusters.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(clusters, f, indent=2)
    
    logger.info(f"Saved ideology clusters to {output_path}")


def load_cluster_assignments(
    input_path: Path = None
) -> Dict[str, List[int]]:
    """
    Load cluster assignments from a JSON file.
    
    Args:
        input_path: Path to load from (defaults to data/sample-alt-voters/ideology_clusters.json)
        
    Returns:
        Dict mapping cluster names to persona indices
    """
    if input_path is None:
        input_path = DATA_DIR / "sample-alt-voters" / "ideology_clusters.json"
    
    with open(input_path) as f:
        return json.load(f)


def main():
    """CLI entry point for generating and saving ideology clusters."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description="Classify personas into ideology clusters"
    )
    parser.add_argument(
        "--save", 
        action="store_true",
        help="Save cluster assignments to JSON file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Print detailed classification results"
    )
    args = parser.parse_args()
    
    # Load personas
    logger.info(f"Loading personas from {PERSONAS_PATH}")
    personas = load_personas()
    logger.info(f"Loaded {len(personas)} personas")
    
    # Classify
    clusters = get_ideology_clusters(personas)
    stats = get_cluster_stats(clusters)
    
    # Print summary
    print("\nIdeology Cluster Summary:")
    print("-" * 40)
    for cluster, count in sorted(stats.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(personas)
        print(f"  {cluster}: {count} ({pct:.1f}%)")
    print(f"  Total: {len(personas)}")
    
    # Verbose output - show ideology distribution within each cluster
    if args.verbose:
        from collections import Counter
        
        print("\n\nDetailed breakdown by ideology value:")
        print("=" * 60)
        
        for cluster_name in ["progressive_liberal", "conservative_traditional", "other"]:
            print(f"\n{cluster_name.upper()}:")
            print("-" * 40)
            
            ideologies = []
            for idx in clusters[cluster_name]:
                ideology = extract_ideology(personas[idx])
                ideologies.append(ideology)
            
            for ideology, count in Counter(ideologies).most_common():
                print(f"  {ideology}: {count}")
    
    # Save if requested
    if args.save:
        save_cluster_assignments(clusters)
        print(f"\nSaved cluster assignments to data/sample-alt-voters/ideology_clusters.json")


if __name__ == "__main__":
    main()
