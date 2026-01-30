"""
Compare chunked Borda vs original insertion methods.

This script:
1. Loads test results from both methods
2. Generates side-by-side comparison analysis
3. Creates comparison visualization plots

Usage:
    uv run python -m src.experiment_utils.compare_insertion_methods
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CHUNKED_DIR = PROJECT_ROOT / "outputs" / "chunked_insertion_test"
ORIGINAL_DIR = PROJECT_ROOT / "outputs" / "original_insertion_test"
COMPARISON_DIR = PROJECT_ROOT / "outputs" / "insertion_comparison"
FIGURES_DIR = COMPARISON_DIR / "figures"


def load_results(method: str, topic: str, rep_id: int = 0) -> Dict:
    """Load test results for a method and topic."""
    if method == "chunked":
        base_dir = CHUNKED_DIR
    else:
        base_dir = ORIGINAL_DIR
    
    results_path = base_dir / topic / f"rep{rep_id}" / "test_results.json"
    summary_path = base_dir / topic / f"rep{rep_id}" / "summary.json"
    
    with open(results_path) as f:
        results = json.load(f)
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    return {"results": results, "summary": summary}


def create_comparison_report(topics: List[str]) -> Dict:
    """Generate comparison report across methods and topics."""
    report = {
        "methods": {},
        "comparison": {},
    }
    
    methods = ["chunked", "original"]
    
    for method in methods:
        report["methods"][method] = {}
        for topic in topics:
            try:
                data = load_results(method, topic)
                summary = data["summary"]
                report["methods"][method][topic] = {
                    "position_error_mean": summary["position_error_mean"],
                    "position_error_std": summary["position_error_std"],
                    "position_error_median": summary["position_error_median"],
                    "absolute_error_mean": summary["absolute_error_mean"],
                    "successful_tests": summary["successful_tests"],
                    "elapsed_time": summary["elapsed_time"],
                }
            except Exception as e:
                logger.warning(f"Could not load {method}/{topic}: {e}")
    
    # Calculate improvement (chunked vs original)
    for topic in topics:
        if topic in report["methods"]["chunked"] and topic in report["methods"]["original"]:
            chunked_error = report["methods"]["chunked"][topic]["position_error_mean"]
            original_error = report["methods"]["original"][topic]["position_error_mean"]
            
            report["comparison"][topic] = {
                "chunked_error": chunked_error,
                "original_error": original_error,
                "improvement": chunked_error - original_error,  # positive = chunked is better (less negative)
                "improvement_pct": ((abs(original_error) - abs(chunked_error)) / abs(original_error)) * 100 if original_error != 0 else 0,
            }
    
    return report


def plot_method_comparison_bar(topics: List[str]) -> None:
    """Create bar chart comparing methods across topics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ["chunked", "original"]
    colors = {"chunked": "#2E86AB", "original": "#A23B72"}
    labels = {"chunked": "Chunked Borda", "original": "Original (single-call)"}
    
    x = np.arange(len(topics))
    width = 0.35
    
    for i, method in enumerate(methods):
        means = []
        stds = []
        for topic in topics:
            try:
                data = load_results(method, topic)
                means.append(data["summary"]["position_error_mean"])
                stds.append(data["summary"]["position_error_std"])
            except:
                means.append(0)
                stds.append(0)
        
        offset = width * (i - 0.5)
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                     label=labels[method], color=colors[method], alpha=0.7)
        
        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
                   f'{mean:.1f}', ha='center', va='top', fontsize=10, fontweight='bold',
                   color='white')
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Perfect (error=0)')
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in topics])
    ax.set_xlabel('Topic', fontsize=12)
    ax.set_ylabel('Mean Position Error', fontsize=12)
    ax.set_title('Insertion Method Comparison\n(Negative = predicted too preferred; closer to 0 is better)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "method_comparison_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved method_comparison_bar.png")


def plot_predicted_vs_original_comparison(topics: List[str]) -> None:
    """Create side-by-side scatter plots for both topics, with methods as lines."""
    fig, axes = plt.subplots(1, len(topics), figsize=(16, 7))
    if len(topics) == 1:
        axes = [axes]
    
    methods = ["chunked", "original"]
    colors = {"chunked": "#2E86AB", "original": "#A23B72"}
    labels = {"chunked": "Chunked Borda", "original": "Original"}
    
    for ax_idx, topic in enumerate(topics):
        ax = axes[ax_idx]
        
        for method in methods:
            try:
                data = load_results(method, topic)
                results = data["results"]
                successful = [r for r in results if r.get("success", False)]
                
                original = np.array([r["original_position"] for r in successful])
                predicted = np.array([r["predicted_position"] for r in successful])
                
                # Bin the data
                bin_size = 5
                bins = np.arange(0, 101, bin_size)
                bin_means = []
                bin_centers = []
                
                for i in range(len(bins) - 1):
                    low, high = bins[i], bins[i+1]
                    mask = (original >= low) & (original < high)
                    if mask.sum() > 0:
                        bin_centers.append((low + high) / 2)
                        bin_means.append(np.mean(predicted[mask]))
                
                ax.plot(bin_centers, bin_means, 'o-', markersize=6,
                       label=f'{labels[method]}',
                       color=colors.get(method, 'gray'), alpha=0.8, linewidth=2)
            except Exception as e:
                logger.warning(f"Could not plot {method}/{topic}: {e}")
        
        ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect')
        ax.set_xlabel('Original Position (binned)', fontsize=11)
        ax.set_ylabel('Mean Predicted Position', fontsize=11)
        ax.set_title(f'{topic.capitalize()}', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')
    
    fig.suptitle('Predicted vs Original Position by Topic\n(Below diagonal = predicted too preferred)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "method_comparison_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved method_comparison_scatter.png")


def plot_error_distribution_comparison(topics: List[str]) -> None:
    """Create overlaid histograms comparing error distributions."""
    fig, axes = plt.subplots(1, len(topics), figsize=(14, 5))
    if len(topics) == 1:
        axes = [axes]
    
    methods = ["chunked", "original"]
    colors = {"chunked": "#2E86AB", "original": "#A23B72"}
    labels = {"chunked": "Chunked Borda", "original": "Original"}
    
    for ax, topic in zip(axes, topics):
        for method in methods:
            try:
                data = load_results(method, topic)
                results = data["results"]
                successful = [r for r in results if r.get("success", False)]
                errors = [r["position_error"] for r in successful]
                
                ax.hist(errors, bins=30, alpha=0.5, label=f'{labels[method]} (μ={np.mean(errors):.1f})',
                       color=colors[method], edgecolor='white')
            except Exception as e:
                logger.warning(f"Could not plot {method}/{topic}: {e}")
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Position Error', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{topic.capitalize()}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Position Error Distribution by Method\n(Negative = predicted too preferred)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "method_comparison_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved method_comparison_histogram.png")


def main():
    """Run comparison analysis and generate visualizations."""
    # Create output directories
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find available topics (must exist in both directories)
    topics = []
    for topic_dir in CHUNKED_DIR.iterdir():
        if topic_dir.is_dir() and topic_dir.name not in ["figures", "analysis"]:
            original_path = ORIGINAL_DIR / topic_dir.name / "rep0" / "summary.json"
            chunked_path = CHUNKED_DIR / topic_dir.name / "rep0" / "summary.json"
            if original_path.exists() and chunked_path.exists():
                topics.append(topic_dir.name)
    
    if not topics:
        logger.error("No topics with both chunked and original results found!")
        return
    
    logger.info(f"Found results for topics: {topics}")
    
    # Generate comparison report
    logger.info("Generating comparison report...")
    report = create_comparison_report(topics)
    
    # Save report
    with open(COMPARISON_DIR / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved comparison_report.json")
    
    # Print summary
    print("\n" + "="*70)
    print("INSERTION METHOD COMPARISON")
    print("="*70)
    
    print("\n{:<12} {:>20} {:>20} {:>15}".format(
        "Topic", "Chunked Borda", "Original", "Improvement"))
    print("-"*70)
    
    for topic in topics:
        if topic in report["comparison"]:
            c = report["comparison"][topic]
            print("{:<12} {:>20.2f} {:>20.2f} {:>15.2f}".format(
                topic.capitalize(),
                c["chunked_error"],
                c["original_error"],
                c["improvement"]))
    
    print("-"*70)
    
    # Overall improvement
    if topics:
        avg_chunked = np.mean([report["comparison"][t]["chunked_error"] for t in topics])
        avg_original = np.mean([report["comparison"][t]["original_error"] for t in topics])
        avg_improvement = np.mean([report["comparison"][t]["improvement"] for t in topics])
        print("{:<12} {:>20.2f} {:>20.2f} {:>15.2f}".format(
            "AVERAGE", avg_chunked, avg_original, avg_improvement))
    
    print("="*70)
    print("\nPositive improvement = Chunked Borda is better (less 'too preferred' bias)")
    
    # Generate visualizations
    logger.info("\nGenerating comparison visualizations...")
    plot_method_comparison_bar(topics)
    plot_predicted_vs_original_comparison(topics)
    plot_error_distribution_comparison(topics)
    
    logger.info(f"\nAll visualizations saved to {FIGURES_DIR}")
    print(f"\nFigures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
