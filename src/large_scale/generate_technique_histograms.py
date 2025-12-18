"""
Generate histogram visualization showing evaluative ratings for each technique's winner across all topics.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

# Global cache for topic mappings
_TOPIC_MAPPINGS: Optional[Dict[str, str]] = None


def load_topic_mappings() -> Dict[str, str]:
    """Load topic mappings from topic_mappings.json."""
    global _TOPIC_MAPPINGS
    if _TOPIC_MAPPINGS is not None:
        return _TOPIC_MAPPINGS
    
    mapping_file = Path("data/topic_mappings.json")
    if not mapping_file.exists():
        print("Warning: data/topic_mappings.json not found, using fallback shortening")
        _TOPIC_MAPPINGS = {}
        return _TOPIC_MAPPINGS
    
    with open(mapping_file, 'r') as f:
        _TOPIC_MAPPINGS = json.load(f)
    
    return _TOPIC_MAPPINGS


def load_results(results_dir: str = "data/large_scale/results") -> List[Dict]:
    """Load all result JSON files."""
    results: List[Dict] = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory {results_dir} does not exist")
        return results
    
    for json_file in sorted(results_path.glob("*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    return results


def shorten_topic(topic: str, max_length: int = 50) -> str:
    """Shorten topic using topic_mappings.json."""
    mappings = load_topic_mappings()
    
    # Try to find exact match in mappings
    if topic in mappings:
        return mappings[topic]
    
    # Fallback to old behavior if not found
    if len(topic) <= max_length:
        return topic
    
    # Remove question mark and common prefixes
    topic_clean = topic.replace("How should we ", "").replace("What are ", "").replace("What should ", "").replace("What ", "")
    topic_clean = topic_clean.replace("?", "").strip()
    
    if len(topic_clean) <= max_length:
        return topic_clean
    
    # Try to cut at a word boundary
    shortened = topic_clean[:max_length].rsplit(' ', 1)[0]
    if len(shortened) < max_length * 0.7:
        return topic_clean[:max_length-3] + "..."
    return shortened + "..."


def extract_ratings_for_winner(result: Dict, method: str) -> List[int]:
    """Extract evaluative ratings for a method's winner."""
    method_results = result.get("method_results", {})
    method_result = method_results.get(method, {})
    winner = method_result.get("winner")
    
    if winner is None:
        return []
    
    try:
        winner_idx = int(winner)
    except (ValueError, TypeError):
        return []
    
    # Extract ratings from evaluations
    evaluations = result.get("evaluations", [])
    ratings = []
    
    for eval_item in evaluations:
        persona_ratings = eval_item.get("ratings", [])
        if winner_idx < len(persona_ratings):
            ratings.append(persona_ratings[winner_idx])
    
    return ratings


def generate_technique_histogram(
    results: List[Dict],
    method: str,
    method_label: str,
    output_file: str
) -> None:
    """Generate histogram figure for one technique across all topics."""
    
    n_topics = len(results)
    
    if n_topics == 0:
        print(f"No results to plot for {method_label}")
        return
    
    # Create a figure with table layout: one row per topic, histogram in each row
    fig = plt.figure(figsize=(14, max(8, n_topics * 1.2)))
    
    # Collect all ratings for summary statistics
    all_ratings = []
    
    # Create a grid: left column for labels, right column for histograms
    gs = fig.add_gridspec(n_topics, 2, width_ratios=[0.4, 0.6], hspace=0.3, wspace=0.2)
    
    for idx, result in enumerate(results):
        topic = result.get("topic", "Unknown")
        ratings = extract_ratings_for_winner(result, method)
        all_ratings.extend(ratings)
        
        # Topic label axis (left)
        ax_label = fig.add_subplot(gs[idx, 0])
        ax_label.axis('off')
        topic_short = shorten_topic(topic, max_length=60)
        ax_label.text(1.0, 0.5, topic_short, 
                     horizontalalignment='right', verticalalignment='center',
                     transform=ax_label.transAxes, fontsize=10, wrap=True)
        
        # Histogram axis (right)
        ax_hist = fig.add_subplot(gs[idx, 1])
        
        if ratings:
            # Create histogram for Likert scale 1-5
            bins = np.arange(0.5, 6.5, 1)  # Bins from 0.5 to 6.5 for integer ratings 1-5
            counts, _, _ = ax_hist.hist(ratings, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
            
            ax_hist.set_xlim(0.5, 5.5)
            ax_hist.set_xticks(range(1, 6))
            ax_hist.set_xlabel('Rating (1-5)', fontsize=9)
            ax_hist.set_ylabel('Frequency', fontsize=9)
            
            # Add mean line
            mean_rating = np.mean(ratings)
            ax_hist.axvline(mean_rating, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            
            # Add mean text
            max_count = max(counts) if len(counts) > 0 else 1
            ax_hist.text(mean_rating, max_count * 0.9, f'μ={mean_rating:.1f}', 
                        fontsize=8, color='red', ha='center', va='bottom')
            
            # Add count
            ax_hist.text(0.98, 0.95, f'n={len(ratings)}', 
                        transform=ax_hist.transAxes, fontsize=8,
                        horizontalalignment='right', verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Add grid for better readability
            ax_hist.grid(True, alpha=0.3, axis='y')
        else:
            ax_hist.text(0.5, 0.5, 'No winner/ratings', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax_hist.transAxes, fontsize=12)
            ax_hist.set_xlim(0.5, 5.5)
            ax_hist.set_ylim(0, 1)
            ax_hist.set_xticks(range(1, 6))
    
    # Add overall title
    fig.suptitle(f'Evaluative Ratings Distribution for {method_label} Winners', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Histogram for {method_label} saved to {output_file}")
    
    # Print summary statistics
    if all_ratings:
        print(f"  Total ratings: {len(all_ratings)}")
        print(f"  Overall mean: {np.mean(all_ratings):.2f}")
        print(f"  Overall median: {np.median(all_ratings):.2f}")
        print(f"  Overall std: {np.std(all_ratings):.2f}")


def generate_all_histograms(
    results: List[Dict],
    output_dir: str = "."
) -> None:
    """Generate histograms for all techniques."""
    
    methods = {
        "plurality": "Plurality",
        "borda": "Borda",
        "irv": "IRV",
        "rankedpairs": "RankedPairs",
        "chatgpt": "ChatGPT",
        "chatgpt_rankings": "ChatGPT+Rankings",
        "chatgpt_profiles": "ChatGPT+Profiles",
        "chatgpt_rankings_profiles": "ChatGPT+Rankings+Profiles"
    }
    
    for method, label in methods.items():
        output_file = os.path.join(output_dir, f"histogram_{method}.png")
        print(f"\nGenerating histogram for {label}...")
        generate_technique_histogram(results, method, label, output_file)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate histogram visualizations of evaluative ratings by technique")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/large_scale/results",
        help="Directory containing result JSON files (default: data/large_scale/results)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for histogram PNG files (default: current directory)"
    )
    
    args = parser.parse_args()
    
    results = load_results(args.results_dir)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    generate_all_histograms(results, args.output_dir)


if __name__ == "__main__":
    main()

