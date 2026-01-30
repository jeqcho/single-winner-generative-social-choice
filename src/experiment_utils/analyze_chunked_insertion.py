"""
Analyze and visualize chunked Borda insertion test results.

This script:
1. Loads test results from both topics
2. Generates comparison analysis
3. Creates visualization plots

Usage:
    uv run python -m src.experiment_utils.analyze_chunked_insertion
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from pvc_toolbox import compute_critical_epsilon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "chunked_insertion_test"
FIGURES_DIR = OUTPUT_DIR / "figures"


def load_results(topic: str, rep_id: int = 0) -> Dict:
    """Load test results for a topic."""
    results_path = OUTPUT_DIR / topic / f"rep{rep_id}" / "test_results.json"
    summary_path = OUTPUT_DIR / topic / f"rep{rep_id}" / "summary.json"
    
    with open(results_path) as f:
        results = json.load(f)
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    return {"results": results, "summary": summary}


def create_comparison_report(topics: List[str]) -> Dict:
    """Generate comparison report across topics."""
    report = {
        "topics": {},
        "overall": {}
    }
    
    all_position_errors = []
    all_absolute_errors = []
    
    for topic in topics:
        data = load_results(topic)
        summary = data["summary"]
        results = data["results"]
        
        # Extract successful results
        successful = [r for r in results if r.get("success", False)]
        position_errors = [r["position_error"] for r in successful]
        absolute_errors = [r["absolute_error"] for r in successful]
        
        all_position_errors.extend(position_errors)
        all_absolute_errors.extend(absolute_errors)
        
        report["topics"][topic] = {
            "n_tests": summary["total_tests"],
            "successful_tests": summary["successful_tests"],
            "position_error_mean": summary["position_error_mean"],
            "position_error_std": summary["position_error_std"],
            "position_error_median": summary["position_error_median"],
            "absolute_error_mean": summary["absolute_error_mean"],
            "absolute_error_std": summary["absolute_error_std"],
            "absolute_error_median": summary["absolute_error_median"],
            "per_alternative": summary.get("per_alternative", {}),
        }
    
    # Overall statistics
    report["overall"] = {
        "total_tests": len(all_position_errors),
        "position_error_mean": np.mean(all_position_errors),
        "position_error_std": np.std(all_position_errors),
        "position_error_median": np.median(all_position_errors),
        "absolute_error_mean": np.mean(all_absolute_errors),
        "absolute_error_std": np.std(all_absolute_errors),
        "absolute_error_median": np.median(all_absolute_errors),
    }
    
    return report


def plot_position_error_distribution(topics: List[str]) -> None:
    """Plot histogram of position errors."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'abortion': '#2E86AB', 'environment': '#A23B72'}
    
    for topic in topics:
        data = load_results(topic)
        results = data["results"]
        successful = [r for r in results if r.get("success", False)]
        position_errors = [r["position_error"] for r in successful]
        
        ax.hist(position_errors, bins=30, alpha=0.6, label=topic.capitalize(),
                color=colors.get(topic, 'gray'), edgecolor='white')
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect (error=0)')
    
    ax.set_xlabel('Position Error (predicted - original)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Position Error Distribution\n(Negative = predicted too preferred)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add stats annotation
    for i, topic in enumerate(topics):
        data = load_results(topic)
        summary = data["summary"]
        text = f'{topic.capitalize()}:\nmean={summary["position_error_mean"]:.1f}\nstd={summary["position_error_std"]:.1f}'
        ax.text(0.02 + i*0.15, 0.98, text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "position_error_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved position_error_distribution.png")


def plot_predicted_vs_original_scatter(topics: List[str]) -> None:
    """Plot predicted vs original position scatter."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'abortion': '#2E86AB', 'environment': '#A23B72'}
    markers = {'abortion': 'o', 'environment': 's'}
    
    for topic in topics:
        data = load_results(topic)
        results = data["results"]
        successful = [r for r in results if r.get("success", False)]
        
        original = [r["original_position"] for r in successful]
        predicted = [r["predicted_position"] for r in successful]
        
        ax.scatter(original, predicted, alpha=0.3, label=topic.capitalize(),
                   color=colors.get(topic, 'gray'), marker=markers.get(topic, 'o'), s=20)
    
    # Perfect prediction line
    ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel('Original Position (1=most preferred)', fontsize=12)
    ax.set_ylabel('Predicted Position', fontsize=12)
    ax.set_title('Predicted vs Original Position\n(Points below line = predicted too preferred)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "predicted_vs_original_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved predicted_vs_original_scatter.png")


def plot_predicted_vs_original_scatter_binned(topics: List[str], bin_size: int = 5) -> None:
    """Plot predicted vs original position scatter, binned by rank."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'abortion': '#2E86AB', 'environment': '#A23B72'}
    bins = np.arange(0, 101, bin_size)
    
    for topic in topics:
        data = load_results(topic)
        results = data["results"]
        successful = [r for r in results if r.get("success", False)]
        
        original = np.array([r["original_position"] for r in successful])
        predicted = np.array([r["predicted_position"] for r in successful])
        
        # Bin the data
        bin_means = []
        bin_centers = []
        
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i+1]
            mask = (original >= low) & (original < high)
            if mask.sum() > 0:
                bin_centers.append((low + high) / 2)
                bin_means.append(np.mean(predicted[mask]))
        
        # Plot without error bars
        ax.plot(bin_centers, bin_means, 'o-', markersize=8,
                label=f'{topic.capitalize()} (n={len(successful)})',
                color=colors.get(topic, 'gray'), alpha=0.8, linewidth=2)
    
    # Perfect prediction line
    ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel(f'Original Position (binned by {bin_size} ranks)', fontsize=12)
    ax.set_ylabel('Mean Predicted Position', fontsize=12)
    ax.set_title('Binned Predicted vs Original Position\n(Points below line = predicted too preferred)', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "predicted_vs_original_scatter_binned.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved predicted_vs_original_scatter_binned.png")


def plot_epsilon_comparison(topics: List[str]) -> None:
    """Plot epsilon at original vs relationship with position error."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {'abortion': '#2E86AB', 'environment': '#A23B72'}
    
    # Plot 1: Position error vs original epsilon
    ax1 = axes[0]
    for topic in topics:
        data = load_results(topic)
        summary = data["summary"]
        per_alt = summary.get("per_alternative", {})
        
        epsilons = []
        errors = []
        for alt_id, alt_data in per_alt.items():
            if alt_data.get("original_epsilon") is not None:
                epsilons.append(alt_data["original_epsilon"])
                errors.append(alt_data["mean_position_error"])
        
        ax1.scatter(epsilons, errors, s=100, alpha=0.7, label=topic.capitalize(),
                    color=colors.get(topic, 'gray'))
    
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel('Original Epsilon', fontsize=12)
    ax1.set_ylabel('Mean Position Error', fontsize=12)
    ax1.set_title('Position Error vs Original Epsilon\n(per alternative)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Bar chart of mean error by topic
    ax2 = axes[1]
    topic_names = [t.capitalize() for t in topics]
    means = [load_results(t)["summary"]["position_error_mean"] for t in topics]
    stds = [load_results(t)["summary"]["position_error_std"] for t in topics]
    
    x = np.arange(len(topics))
    bars = ax2.bar(x, means, yerr=stds, capsize=5, 
                   color=[colors.get(t, 'gray') for t in topics], alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(topic_names)
    ax2.set_ylabel('Mean Position Error', fontsize=12)
    ax2.set_title('Mean Position Error by Topic\n(with std dev)', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{mean:.1f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "epsilon_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved epsilon_comparison.png")


def load_preferences(topic: str, rep_id: int = 0) -> List[List[str]]:
    """Load the original preference profile."""
    pref_path = PROJECT_ROOT / "outputs" / "sample_alt_voters" / "data" / topic / "uniform" / "persona_no_context" / f"rep{rep_id}" / "preferences.json"
    with open(pref_path) as f:
        return json.load(f)


def compute_epsilon_at_predicted_positions(topic: str, rep_id: int = 0) -> List[Dict]:
    """
    Compute epsilon for each test alternative using predicted positions.
    
    For each alternative, we create a hypothetical profile where each voter
    has the alternative at their predicted position, then compute epsilon.
    """
    data = load_results(topic, rep_id)
    results = data["results"]
    preferences = load_preferences(topic, rep_id)
    
    # Group results by test_alt_idx
    by_alt = defaultdict(list)
    for r in results:
        if r.get("success"):
            by_alt[r["test_alt_idx"]].append(r)
    
    n_voters = len(preferences[0])
    n_alts = len(preferences)
    
    epsilon_results = []
    
    for alt_idx, alt_results in by_alt.items():
        # Get original epsilon (same for all voters)
        original_epsilon = alt_results[0]["original_epsilon"]
        
        # Create a mapping of voter_idx -> predicted_position
        voter_predictions = {r["voter_idx"]: r["predicted_position"] for r in alt_results}
        
        # Create hypothetical profile where alt is at predicted position for each voter
        modified_profile = []
        for voter_idx in range(n_voters):
            voter_ranking = [preferences[rank][voter_idx] for rank in range(n_alts)]
            
            if voter_idx in voter_predictions:
                current_pos = voter_ranking.index(str(alt_idx))
                voter_ranking.pop(current_pos)
                
                predicted_pos = voter_predictions[voter_idx]
                predicted_pos = max(0, min(predicted_pos, len(voter_ranking)))
                voter_ranking.insert(predicted_pos, str(alt_idx))
            
            modified_profile.append(voter_ranking)
        
        # Convert to [rank][voter] format
        modified_preferences = []
        for rank in range(n_alts):
            rank_row = [modified_profile[voter][rank] for voter in range(n_voters)]
            modified_preferences.append(rank_row)
        
        # Compute epsilon on modified profile
        alternatives = [str(i) for i in range(n_alts)]
        try:
            predicted_epsilon = compute_critical_epsilon(
                modified_preferences, alternatives, str(alt_idx)
            )
        except Exception as e:
            logger.error(f"Error computing epsilon for alt {alt_idx}: {e}")
            predicted_epsilon = None
        
        if predicted_epsilon is not None:
            epsilon_results.append({
                "alt_idx": alt_idx,
                "original_epsilon": original_epsilon,
                "predicted_epsilon": predicted_epsilon,
                "epsilon_diff": predicted_epsilon - original_epsilon,
            })
    
    return epsilon_results


def plot_epsilon_original_vs_predicted(topics: List[str]) -> Dict[str, List[Dict]]:
    """Plot scatter of original epsilon vs epsilon at predicted position."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'abortion': '#2E86AB', 'environment': '#A23B72'}
    markers = {'abortion': 'o', 'environment': 's'}
    
    all_results = {}
    np.random.seed(42)
    jitter_amount = 0.008
    
    for topic in topics:
        logger.info(f"Computing predicted epsilons for {topic}...")
        results = compute_epsilon_at_predicted_positions(topic)
        all_results[topic] = results
        
        original = np.array([r["original_epsilon"] for r in results])
        predicted = np.array([r["predicted_epsilon"] for r in results])
        
        # Add jitter to prevent overlapping points
        original_jittered = original + np.random.uniform(-jitter_amount, jitter_amount, len(original))
        predicted_jittered = predicted + np.random.uniform(-jitter_amount, jitter_amount, len(predicted))
        
        ax.scatter(original_jittered, predicted_jittered, s=120, alpha=0.4, 
                   label=f'{topic.capitalize()} (n={len(results)})',
                   color=colors.get(topic, 'gray'), 
                   marker=markers.get(topic, 'o'),
                   edgecolors='white', linewidth=1)
    
    # Perfect prediction line
    all_orig = [r["original_epsilon"] for t in all_results.values() for r in t]
    all_pred = [r["predicted_epsilon"] for t in all_results.values() for r in t]
    max_eps = max(max(all_orig), max(all_pred)) if all_orig else 0.3
    
    ax.plot([0, max_eps + 0.05], [0, max_eps + 0.05], 'r--', linewidth=2, 
            label='Perfect (pred=orig)')
    
    ax.set_xlabel('Original Epsilon (jittered)', fontsize=12)
    ax.set_ylabel('Predicted Epsilon (jittered)', fontsize=12)
    ax.set_title('Epsilon Comparison: Original vs Predicted Position\n(Points above line = predicted epsilon higher/worse; jittered for visibility)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, max_eps + 0.05)
    ax.set_ylim(-0.02, max_eps + 0.05)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "epsilon_original_vs_predicted.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved epsilon_original_vs_predicted.png")
    
    return all_results


def plot_error_by_original_position(topics: List[str]) -> None:
    """Plot error by original position bins."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'abortion': '#2E86AB', 'environment': '#A23B72'}
    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    bin_labels = ['1-20', '21-40', '41-60', '61-80', '81-100']
    
    width = 0.35
    x = np.arange(len(bins))
    
    for i, topic in enumerate(topics):
        data = load_results(topic)
        results = data["results"]
        successful = [r for r in results if r.get("success", False)]
        
        bin_means = []
        bin_stds = []
        for low, high in bins:
            bin_errors = [r["position_error"] for r in successful 
                          if low <= r["original_position"] < high]
            if bin_errors:
                bin_means.append(np.mean(bin_errors))
                bin_stds.append(np.std(bin_errors))
            else:
                bin_means.append(0)
                bin_stds.append(0)
        
        offset = width * (i - 0.5)
        ax.bar(x + offset, bin_means, width, yerr=bin_stds, capsize=3,
               label=topic.capitalize(), color=colors.get(topic, 'gray'), alpha=0.7)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel('Original Position Bin', fontsize=12)
    ax.set_ylabel('Mean Position Error', fontsize=12)
    ax.set_title('Position Error by Original Rank\n(Negative = predicted more preferred; Positive = predicted less preferred)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_by_original_position.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved error_by_original_position.png")


def main():
    """Run analysis and generate visualizations."""
    # Create figures directory
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find available topics
    topics = []
    for topic_dir in OUTPUT_DIR.iterdir():
        if topic_dir.is_dir() and topic_dir.name != "figures" and topic_dir.name != "analysis":
            if (topic_dir / "rep0" / "summary.json").exists():
                topics.append(topic_dir.name)
    
    if not topics:
        logger.error("No test results found!")
        return
    
    logger.info(f"Found results for topics: {topics}")
    
    # Generate comparison report
    logger.info("Generating comparison report...")
    report = create_comparison_report(topics)
    
    # Save report
    analysis_dir = OUTPUT_DIR / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with open(analysis_dir / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved comparison_report.json")
    
    # Print summary
    print("\n" + "="*60)
    print("CHUNKED BORDA INSERTION TEST RESULTS")
    print("="*60)
    
    for topic in topics:
        t = report["topics"][topic]
        print(f"\n{topic.upper()}:")
        print(f"  Tests: {t['successful_tests']}/{t['n_tests']}")
        print(f"  Position Error: mean={t['position_error_mean']:.2f}, std={t['position_error_std']:.2f}, median={t['position_error_median']:.1f}")
        print(f"  Absolute Error: mean={t['absolute_error_mean']:.2f}")
    
    print(f"\nOVERALL (both topics):")
    o = report["overall"]
    print(f"  Total Tests: {o['total_tests']}")
    print(f"  Position Error: mean={o['position_error_mean']:.2f}, std={o['position_error_std']:.2f}, median={o['position_error_median']:.1f}")
    print(f"  Absolute Error: mean={o['absolute_error_mean']:.2f}")
    print("="*60)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    plot_position_error_distribution(topics)
    plot_predicted_vs_original_scatter(topics)
    plot_predicted_vs_original_scatter_binned(topics)
    plot_epsilon_comparison(topics)
    plot_error_by_original_position(topics)
    
    # Compute and plot epsilon at predicted positions
    logger.info("Computing epsilon at predicted positions...")
    epsilon_results = plot_epsilon_original_vs_predicted(topics)
    
    # Save epsilon comparison results
    with open(analysis_dir / "epsilon_comparison_results.json", "w") as f:
        json.dump(epsilon_results, f, indent=2)
    logger.info("Saved epsilon_comparison_results.json")
    
    # Print epsilon summary
    print("\nEPSILON COMPARISON:")
    for topic, results in epsilon_results.items():
        diffs = [r["epsilon_diff"] for r in results]
        print(f"  {topic.capitalize()}: mean diff={np.mean(diffs):+.4f}, std={np.std(diffs):.4f}")
    
    logger.info(f"\nAll visualizations saved to {FIGURES_DIR}")
    print(f"\nFigures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
