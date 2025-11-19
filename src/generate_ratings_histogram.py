"""
Generate histogram visualization showing consensus ratings for each topic.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: str = "data/results") -> List[Dict]:
    """Load all result JSON files."""
    results = []
    results_path = Path(results_dir)
    
    for json_file in sorted(results_path.glob("*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    return results


def shorten_topic(topic: str, max_length: int = 50) -> str:
    """Shorten topic for display."""
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


def extract_ratings(result: Dict) -> List[int]:
    """Extract ratings from bridging evaluation."""
    bridging_eval = result.get("bridging_evaluation", [])
    ratings = []
    
    for eval_item in bridging_eval:
        evaluation = eval_item.get("evaluation", {})
        rating = evaluation.get("rating")
        if rating is not None:
            ratings.append(int(rating))
    
    return ratings


def generate_histogram_figure(results: List[Dict], output_file: str = "ratings_histogram.png") -> None:
    """Generate table-style figure with histograms for each topic."""
    
    n_topics = len(results)
    
    # Create a figure with table layout: one row per topic, histogram in each row
    # Each row will have: topic label (left) and histogram (right)
    fig = plt.figure(figsize=(14, max(8, n_topics * 1.2)))
    
    # Collect all ratings for summary statistics
    all_ratings = []
    
    # Create a grid: left column for labels, right column for histograms
    gs = fig.add_gridspec(n_topics, 2, width_ratios=[0.4, 0.6], hspace=0.3, wspace=0.2)
    
    for idx, result in enumerate(results):
        topic = result.get("topic", "Unknown")
        ratings = extract_ratings(result)
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
            # Create histogram
            bins = np.arange(0.5, 11.5, 1)  # Bins from 0.5 to 10.5 for integer ratings 1-10
            counts, _, _ = ax_hist.hist(ratings, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
            
            ax_hist.set_xlim(0.5, 10.5)
            ax_hist.set_xticks(range(1, 11))
            ax_hist.set_xlabel('Rating', fontsize=9)
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
            ax_hist.text(0.5, 0.5, 'No ratings available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax_hist.transAxes, fontsize=12)
            ax_hist.set_xlim(0, 10)
            ax_hist.set_ylim(0, 1)
    
    # Add overall title
    fig.suptitle('Consensus Ratings Distribution by Topic', fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Histogram figure saved to {output_file}")
    
    # Print summary statistics
    if all_ratings:
        print(f"\nSummary Statistics:")
        print(f"  Total ratings: {len(all_ratings)}")
        print(f"  Overall mean: {np.mean(all_ratings):.2f}")
        print(f"  Overall median: {np.median(all_ratings):.2f}")
        print(f"  Overall std: {np.std(all_ratings):.2f}")
        print(f"\nPer-topic statistics:")
        for result in results:
            topic = result.get("topic", "Unknown")
            ratings = extract_ratings(result)
            if ratings:
                print(f"  {shorten_topic(topic, 40)}: mean={np.mean(ratings):.2f}, n={len(ratings)}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate histogram visualization of consensus ratings")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/results",
        help="Directory containing result JSON files (default: data/results)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ratings_histogram.png",
        help="Output PNG file (default: ratings_histogram.png)"
    )
    
    args = parser.parse_args()
    
    results = load_results(args.results_dir)
    
    if not results:
        print(f"No results found in {args.results_dir}")
        return
    
    generate_histogram_figure(results, args.output)


if __name__ == "__main__":
    main()

