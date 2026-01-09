#!/usr/bin/env python3
"""
Generate CDF plots for epsilon-100 values.

This script creates cumulative distribution function plots showing epsilon-100
distributions for all voting methods overlaid on a single plot.

Generates:
- Linear scale CDF plot
- Log scale CDF plot

Run with:
    uv run python scripts/generate_epsilon_100_cdf.py
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.full_experiment.epsilon_100 import collect_all_epsilon_100
from src.full_experiment.visualizer import (
    METHOD_COLORS,
    METHOD_NAMES,
    BARPLOT_METHOD_ORDER,
)
from src.full_experiment.config import OUTPUT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_epsilon_100_cdf(
    results: Dict[str, List[float]],
    title: str = "Epsilon (100 Personas) CDF by Voting Method",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot CDF of epsilon-100 values with all voting methods overlaid.
    
    Args:
        results: Dict mapping method to list of epsilon-100 values
        title: Plot title
        output_path: Path to save figure (None = show)
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for method in BARPLOT_METHOD_ORDER:
        values = results.get(method, [])
        if not values:
            continue
        
        # Filter out None and negative values
        values = [v for v in values if v is not None and v >= 0]
        if not values:
            continue
        
        # Compute empirical CDF
        sorted_values = np.sort(values)
        cdf = np.linspace(0, 1, len(sorted_values))
        
        # Get display name and color
        display_name = METHOD_NAMES.get(method, method)
        color = METHOD_COLORS.get(method, "#333333")
        
        # Plot CDF line
        ax.plot(sorted_values, cdf, label=display_name, color=color, linewidth=2)
    
    ax.set_xlabel("Epsilon (ε)", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved epsilon-100 CDF to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_epsilon_100_cdf_log(
    results: Dict[str, List[float]],
    title: str = "Epsilon (100 Personas) CDF by Voting Method (Log Scale)",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot CDF of epsilon-100 values with log-scale x-axis.
    
    Args:
        results: Dict mapping method to list of epsilon-100 values
        title: Plot title
        output_path: Path to save figure (None = show)
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Track min non-zero value for x-axis limit
    min_nonzero = 1.0
    
    for method in BARPLOT_METHOD_ORDER:
        values = results.get(method, [])
        if not values:
            continue
        
        # Filter out None and non-positive values (can't log zero)
        values = [v for v in values if v is not None and v > 0]
        if not values:
            continue
        
        # Track minimum non-zero value
        min_nonzero = min(min_nonzero, min(values))
        
        # Compute empirical CDF
        sorted_values = np.sort(values)
        cdf = np.linspace(0, 1, len(sorted_values))
        
        # Get display name and color
        display_name = METHOD_NAMES.get(method, method)
        color = METHOD_COLORS.get(method, "#333333")
        
        # Plot CDF line
        ax.plot(sorted_values, cdf, label=display_name, color=color, linewidth=2)
    
    ax.set_xscale('log')
    ax.set_xlabel("Epsilon (ε) [log scale]", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Set x-axis limits: from slightly below min to 1
    ax.set_xlim(min_nonzero * 0.5, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved epsilon-100 log CDF to {output_path}")
        plt.close()
    else:
        plt.show()


def generate_epsilon_100_cdf_plots(
    output_dir: Path = OUTPUT_DIR,
    topics: Optional[List[str]] = None,
    ablation: str = "full"
) -> None:
    """
    Generate CDF plots for epsilon-100 values.
    
    Args:
        output_dir: Output directory
        topics: List of topics (None = auto-detect)
        ablation: Ablation type ('full', 'no_bridging', 'no_filtering')
    """
    logger.info(f"Generating epsilon-100 CDF plots for ablation: {ablation}")
    
    # Collect epsilon-100 results
    flat_results = collect_all_epsilon_100(output_dir, ablation, topics)
    
    # Log stats
    total_samples = sum(len(v) for v in flat_results.values())
    logger.info(f"Collected {total_samples} epsilon-100 values")
    
    if total_samples == 0:
        logger.warning("No data found, skipping plot generation")
        return
    
    # Create output directory
    ablation_label = f" ({ablation.replace('_', ' ')})" if ablation != "full" else ""
    figures_dir = output_dir / "figures" / ablation / "aggregate"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate linear CDF
    plot_epsilon_100_cdf(
        flat_results,
        title=f"Epsilon (100 Personas) CDF by Voting Method{ablation_label}",
        output_path=figures_dir / "epsilon_100_cdf.png"
    )
    
    # Generate log-scale CDF
    plot_epsilon_100_cdf_log(
        flat_results,
        title=f"Epsilon (100 Personas) CDF by Voting Method (Log Scale){ablation_label}",
        output_path=figures_dir / "epsilon_100_cdf_log.png"
    )
    
    logger.info("CDF plot generation complete!")


if __name__ == "__main__":
    output_dir = Path("outputs/full_experiment")
    data_dir = output_dir / "data"
    
    # Auto-discover topics from folder names
    if data_dir.exists():
        topics = [d.name for d in sorted(data_dir.iterdir()) if d.is_dir()]
        print(f"Found {len(topics)} topics: {topics}")
    else:
        topics = None
        print("Data directory not found, will auto-detect topics")
    
    # Generate CDF plots for the 'full' ablation
    generate_epsilon_100_cdf_plots(
        output_dir=output_dir,
        topics=topics,
        ablation="full"
    )
    
    print("Done generating epsilon-100 CDF plots!")

