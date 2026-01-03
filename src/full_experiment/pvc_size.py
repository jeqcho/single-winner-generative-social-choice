"""
Compute PVC sizes for each rep and generate strip plots by topic.

This module computes the Proportional Veto Core (PVC) size for each rep
across all topics and generates strip plots showing the distribution.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pvc_toolbox import compute_pvc

from .config import ALL_TOPICS, TOPIC_DISPLAY_NAMES, OUTPUT_DIR

logger = logging.getLogger(__name__)


def collect_pvc_sizes_for_topic(
    topic_slug: str,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full"
) -> List[int]:
    """
    Collect PVC sizes for all reps of a topic.
    
    Args:
        topic_slug: Topic slug
        output_dir: Output directory
        ablation: Ablation type ('full', 'no_filtering', 'no_bridging')
    
    Returns:
        List of PVC sizes (one per rep)
    """
    pvc_sizes = []
    
    topic_dir = output_dir / "data" / topic_slug
    
    if not topic_dir.exists():
        logger.warning(f"Topic directory not found: {topic_dir}")
        return pvc_sizes
    
    # Iterate through all rep directories
    for rep_dir in sorted(topic_dir.glob("rep*")):
        # Handle ablation subdirectory
        if ablation == "full":
            prefs_file = rep_dir / "filtered_preferences.json"
        elif ablation == "no_filtering":
            prefs_file = rep_dir / "full_preferences.json"
        elif ablation == "no_bridging":
            prefs_file = rep_dir / "ablation_no_bridging" / "full_preferences.json"
        else:
            prefs_file = rep_dir / "filtered_preferences.json"
        
        if not prefs_file.exists():
            logger.warning(f"Preferences file not found: {prefs_file}")
            continue
        
        try:
            with open(prefs_file, 'r') as f:
                preferences = json.load(f)
            
            n_statements = len(preferences)
            alternatives = [str(i) for i in range(n_statements)]
            
            pvc = compute_pvc(preferences, alternatives)
            pvc_size = len(pvc)
            pvc_sizes.append(pvc_size)
            
            logger.debug(f"  {rep_dir.name}: PVC size = {pvc_size}/{n_statements}")
        
        except Exception as e:
            logger.error(f"Failed to compute PVC for {rep_dir}: {e}")
            continue
    
    return pvc_sizes


def collect_all_pvc_sizes(
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full",
    topics: Optional[List[str]] = None
) -> Dict[str, List[int]]:
    """
    Collect PVC sizes for all topics.
    
    Args:
        output_dir: Output directory
        ablation: Ablation type
        topics: List of topics (None = all)
    
    Returns:
        Dict mapping topic_slug to list of PVC sizes
    """
    if topics is None:
        topics = ALL_TOPICS
    
    results = {}
    
    for topic_slug in topics:
        logger.info(f"Computing PVC sizes for: {topic_slug}")
        pvc_sizes = collect_pvc_sizes_for_topic(topic_slug, output_dir, ablation)
        if pvc_sizes:
            results[topic_slug] = pvc_sizes
            logger.info(f"  Found {len(pvc_sizes)} reps, sizes: {pvc_sizes}")
    
    return results


def save_pvc_sizes(
    pvc_sizes: Dict[str, List[int]],
    output_path: Path
) -> None:
    """
    Save PVC sizes to JSON file.
    
    Args:
        pvc_sizes: Dict mapping topic_slug to list of PVC sizes
        output_path: Path to output JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(pvc_sizes, f, indent=2)
    logger.info(f"Saved PVC sizes to {output_path}")


def load_pvc_sizes(input_path: Path) -> Dict[str, List[int]]:
    """
    Load PVC sizes from JSON file.
    
    Args:
        input_path: Path to input JSON file
    
    Returns:
        Dict mapping topic_slug to list of PVC sizes
    """
    with open(input_path, 'r') as f:
        return json.load(f)


def plot_pvc_size_stripplot(
    pvc_sizes: Dict[str, List[int]],
    title: str = "PVC Size Distribution by Topic",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot horizontal strip plot of PVC sizes with topics as rows.
    
    Args:
        pvc_sizes: Dict mapping topic_slug to list of PVC sizes
        title: Plot title
        output_path: Path to save figure (None = show)
    """
    # Prepare data for seaborn
    data = []
    for topic_slug, sizes in pvc_sizes.items():
        display_name = TOPIC_DISPLAY_NAMES.get(topic_slug, topic_slug[:30])
        for size in sizes:
            data.append({"Topic": display_name, "PVC Size": size})
    
    if not data:
        logger.warning("No PVC size data to plot")
        return
    
    df = pd.DataFrame(data)
    
    # Create figure - adjust height based on number of topics
    n_topics = df["Topic"].nunique()
    fig_height = max(4, n_topics * 0.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    
    # Get topic order (sorted alphabetically by display name)
    topic_order = sorted(df["Topic"].unique())
    
    # Plot strip plot with jitter
    sns.stripplot(
        data=df,
        x="PVC Size",
        y="Topic",
        order=topic_order,
        color="#1f77b4",
        jitter=0.2,
        alpha=0.7,
        size=8,
        ax=ax
    )
    
    # Add mean markers
    for i, topic in enumerate(topic_order):
        topic_data = df[df["Topic"] == topic]["PVC Size"]
        if len(topic_data) > 0:
            mean_val = topic_data.mean()
            ax.scatter([mean_val], [i], color='black', s=100, marker='|', linewidths=2, zorder=10)
    
    # Add horizontal lines to separate rows
    for i in range(len(topic_order) - 1):
        ax.axhline(y=i + 0.5, color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
    
    ax.set_xlabel("PVC Size", fontsize=12)
    ax.set_ylabel("Topic", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add note about sample size
    n_samples = len(df)
    ax.text(0.98, 0.02, f"n={n_samples} reps | bars = means",
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            horizontalalignment='right', color='gray')
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved PVC size strip plot to {output_path}")
        plt.close()
    else:
        plt.show()


def generate_pvc_size_analysis(
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full",
    topics: Optional[List[str]] = None
) -> None:
    """
    Generate PVC size analysis: compute sizes, save to JSON, and create strip plot.
    
    Args:
        output_dir: Output directory
        ablation: Ablation type
        topics: List of topics (None = all)
    """
    logger.info(f"Generating PVC size analysis for ablation: {ablation}")
    
    # Collect PVC sizes
    pvc_sizes = collect_all_pvc_sizes(output_dir, ablation, topics)
    
    if not pvc_sizes:
        logger.warning("No PVC sizes collected")
        return
    
    # Save to JSON
    json_path = output_dir / "data" / "pvc_sizes.json"
    save_pvc_sizes(pvc_sizes, json_path)
    
    # Create figures directory structure
    figures_dir = output_dir / "figures" / ablation / "aggregate"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate strip plot
    ablation_label = f" ({ablation.replace('_', ' ')})" if ablation != "full" else ""
    plot_pvc_size_stripplot(
        pvc_sizes,
        title=f"PVC Size Distribution by Topic{ablation_label}",
        output_path=figures_dir / "pvc_size_stripplot.png"
    )
    
    # Print summary statistics
    all_sizes = []
    for sizes in pvc_sizes.values():
        all_sizes.extend(sizes)
    
    if all_sizes:
        logger.info(f"\nPVC Size Summary:")
        logger.info(f"  Total reps: {len(all_sizes)}")
        logger.info(f"  Mean: {np.mean(all_sizes):.1f}")
        logger.info(f"  Std: {np.std(all_sizes):.1f}")
        logger.info(f"  Min: {min(all_sizes)}")
        logger.info(f"  Max: {max(all_sizes)}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    output_dir = Path("outputs/full_experiment")
    
    # Check for command line arguments
    ablation = "full"
    if len(sys.argv) > 1:
        ablation = sys.argv[1]
    
    generate_pvc_size_analysis(output_dir=output_dir, ablation=ablation)

