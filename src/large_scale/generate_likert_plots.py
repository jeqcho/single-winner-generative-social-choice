"""
Generate Likert scale plots with confidence intervals.

Creates:
1. Aggregate plot across all 13 topics with 95% confidence intervals
2. Individual plots for each of the 13 topics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from scipy import stats


OUTPUT_BASE = Path("data/large_scale/gen-200-disc-50-eval-50-nano-low")
RESULTS_DIR = OUTPUT_BASE / "results"
PLOTS_DIR = OUTPUT_BASE / "plots"

VOTING_METHODS = ["plurality", "borda", "irv", "chatgpt", "schulze", "veto_by_consumption"]
METHOD_LABELS = {
    "plurality": "Plurality",
    "borda": "Borda",
    "irv": "IRV",
    "chatgpt": "ChatGPT",
    "schulze": "Schulze",
    "veto_by_consumption": "Veto by\nConsumption"
}

# Colors for methods
COLORS = {
    "plurality": "#e74c3c",
    "borda": "#3498db",
    "irv": "#2ecc71",
    "chatgpt": "#9b59b6",
    "schulze": "#f39c12",
    "veto_by_consumption": "#1abc9c"
}


def load_all_results() -> List[Dict]:
    """Load all topic result files."""
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.stem == "all_topics_summary":
            continue
        with open(f, 'r') as fp:
            results.append(json.load(fp))
    return results


def compute_mean_and_ci(ratings: List[float], confidence: float = 0.95) -> tuple:
    """Compute mean and confidence interval."""
    n = len(ratings)
    if n == 0:
        return 3.0, 0.0, 0.0
    
    mean = np.mean(ratings)
    
    if n < 2:
        return mean, 0.0, 0.0
    
    se = stats.sem(ratings)
    ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return mean, mean - ci, mean + ci


def generate_aggregate_plot(all_results: List[Dict]):
    """Generate aggregate Likert plot with confidence intervals across all topics."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect all ratings by method across all topics
    method_all_ratings = {method: [] for method in VOTING_METHODS}
    method_topic_means = {method: [] for method in VOTING_METHODS}
    
    for result in all_results:
        likert = result.get("likert_ratings", {})
        for method in VOTING_METHODS:
            if method in likert and likert[method]:
                ratings = [r for r in likert[method] if r is not None]
                method_all_ratings[method].extend(ratings)
                if ratings:
                    method_topic_means[method].append(np.mean(ratings))
    
    # Compute statistics
    means = []
    ci_lower = []
    ci_upper = []
    labels = []
    colors = []
    
    for method in VOTING_METHODS:
        ratings = method_all_ratings[method]
        if ratings:
            mean, low, high = compute_mean_and_ci(ratings)
        else:
            mean, low, high = 3.0, 3.0, 3.0
        
        means.append(mean)
        ci_lower.append(mean - low)
        ci_upper.append(high - mean)
        labels.append(METHOD_LABELS.get(method, method))
        colors.append(COLORS.get(method, "#333333"))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(VOTING_METHODS))
    bars = ax.bar(x, means, color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)
    
    # Add error bars
    ax.errorbar(x, means, yerr=[ci_lower, ci_upper], fmt='none', color='black', 
                capsize=5, capthick=2, linewidth=2)
    
    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Likert Scale Rating (1-5)', fontsize=14)
    ax.set_xlabel('Voting Method', fontsize=14)
    ax.set_title('Evaluative Likert Ratings Across All Topics\n(95% Confidence Intervals)', fontsize=16, fontweight='bold')
    ax.set_ylim(1, 5)
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Neutral')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{mean:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "likert_aggregate.png", dpi=300, bbox_inches='tight')
    plt.savefig(PLOTS_DIR / "likert_aggregate.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Saved aggregate plot to {PLOTS_DIR / 'likert_aggregate.png'}")


def generate_topic_plot(result: Dict, topic_idx: int):
    """Generate Likert plot for a single topic."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    topic = result.get("topic", "Unknown")
    topic_slug = result.get("topic_slug", f"topic_{topic_idx}")
    likert = result.get("likert_ratings", {})
    
    means = []
    ci_lower = []
    ci_upper = []
    labels = []
    colors = []
    
    for method in VOTING_METHODS:
        ratings = likert.get(method, [])
        if ratings:
            ratings = [r for r in ratings if r is not None]
            mean, low, high = compute_mean_and_ci(ratings)
        else:
            mean, low, high = 3.0, 3.0, 3.0
        
        means.append(mean)
        ci_lower.append(mean - low)
        ci_upper.append(high - mean)
        labels.append(METHOD_LABELS.get(method, method))
        colors.append(COLORS.get(method, "#333333"))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(VOTING_METHODS))
    bars = ax.bar(x, means, color=colors, edgecolor='black', linewidth=1, alpha=0.85)
    
    ax.errorbar(x, means, yerr=[ci_lower, ci_upper], fmt='none', color='black', 
                capsize=4, capthick=1.5, linewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Likert Scale (1-5)', fontsize=12)
    ax.set_xlabel('Voting Method', fontsize=12)
    
    # Truncate title if too long
    title = topic if len(topic) < 60 else topic[:57] + "..."
    ax.set_title(f'Evaluative Ratings: {title}', fontsize=12, fontweight='bold')
    
    ax.set_ylim(1, 5)
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08, 
                f'{mean:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"likert_{topic_slug}.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved topic plot: likert_{topic_slug}.png")


def generate_all_plots():
    """Generate all Likert plots."""
    print("Loading results...")
    all_results = load_all_results()
    print(f"Found {len(all_results)} topic results")
    
    if not all_results:
        print("No results found!")
        return
    
    print("\nGenerating aggregate plot...")
    generate_aggregate_plot(all_results)
    
    print("\nGenerating individual topic plots...")
    for i, result in enumerate(all_results):
        generate_topic_plot(result, i)
    
    print(f"\nAll plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    generate_all_plots()

