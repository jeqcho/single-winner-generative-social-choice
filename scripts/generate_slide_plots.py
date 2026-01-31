#!/usr/bin/env python3
"""
Generate slide-quality plots for research presentation.

Creates CDF plots of critical epsilon (zoomed) for persona_no_context + uniform condition.
- Figure 1: abortion, electoral, healthcare (1x3 subplots)
- Figure 2: policing, environment, trust (1x3 subplots)

Each subplot shows Traditional Methods (Schulze, Borda, IRV, Plurality, VBC) 
with a Random baseline.

Usage:
    uv run python scripts/generate_slide_plots.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.sample_alt_voters.results_aggregator import collect_all_results

# Output directory for slides
SLIDES_OUTPUT_DIR = project_root / "outputs" / "slides"

# Set style for slide-quality plots
plt.style.use('seaborn-v0_8-whitegrid')

# Traditional methods in desired legend order: VBC, Borda, Schulze, IRV, Plurality
TRADITIONAL_METHODS = ["veto_by_consumption", "borda", "schulze", "irv", "plurality"]

# Method display names
METHOD_LABELS = {
    'veto_by_consumption': 'VBC',
    'borda': 'Borda',
    'schulze': 'Schulze',
    'irv': 'IRV',
    'plurality': 'Plurality',
}

# Standard contrasting colors (matching existing plots)
CONTRAST_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Topic display names
TOPIC_DISPLAY_NAMES = {
    'abortion': 'Abortion',
    'electoral': 'Electoral College',
    'healthcare': 'Healthcare',
    'policing': 'Policing',
    'environment': 'Environment',
    'trust': 'Trust in Institutions',
}

# Alternative distribution labels (in order)
ALT_DISTRIBUTIONS = ["persona_no_context", "persona_context", "no_persona_context", "no_persona_no_context"]
ALT_DIST_LABELS = {
    "persona_no_context": "Alt1: Persona Only",
    "persona_context": "Alt2: Persona + Bridging Round",
    "no_persona_context": "Alt3: Bridging Round Only",
    "no_persona_no_context": "Alt4: Blind",
}

# Voter distribution labels (in order)
VOTER_DISTRIBUTIONS = ["uniform", "progressive_liberal", "conservative_traditional"]
VOTER_DIST_LABELS = {
    "uniform": "Uniform",
    "progressive_liberal": "Progressive/Liberal",
    "conservative_traditional": "Conservative/Traditional",
}

# Method categories for 4-panel plots
CHATGPT_METHODS = ["chatgpt", "chatgpt_rankings", "chatgpt_personas"]
CHATGPT_STAR_METHODS = ["chatgpt_star", "chatgpt_star_rankings", "chatgpt_star_personas"]
CHATGPT_DOUBLE_STAR_METHODS = ["chatgpt_double_star", "chatgpt_double_star_rankings", "chatgpt_double_star_personas"]
CHATGPT_TRIPLE_STAR_METHODS = ["chatgpt_triple_star"]
NEW_RANDOM_METHODS = ["new_random"]

# Extended method labels for GPT methods
METHOD_LABELS_EXTENDED = {
    'veto_by_consumption': 'VBC',
    'borda': 'Borda',
    'schulze': 'Schulze',
    'irv': 'IRV',
    'plurality': 'Plurality',
    'chatgpt': 'GPT',
    'chatgpt_rankings': 'GPT+Rank',
    'chatgpt_personas': 'GPT+Pers',
    'chatgpt_star': 'GPT*',
    'chatgpt_star_rankings': 'GPT*+Rank',
    'chatgpt_star_personas': 'GPT*+Pers',
    'chatgpt_double_star': 'GPT**',
    'chatgpt_double_star_rankings': 'GPT**+Rank',
    'chatgpt_double_star_personas': 'GPT**+Pers',
    'chatgpt_triple_star': 'GPT***',
    'new_random': 'New Random',
}


def plot_cdf_by_method_category(
    df: pd.DataFrame,
    topic: str,
    output_path: Path,
    voter_dist_label: str = "Uniform",
    x_max: float = 0.5,
    y_min: float = 0.5,
    zoomed: bool = True,
) -> None:
    """
    Create a 1x4 CDF plot for a single topic with 4 method categories.
    
    Subplots: Traditional, GPT, GPT*, GPT**/***
    Fixed to Alt1 (persona_no_context).
    
    Args:
        df: DataFrame with epsilon values (should be filtered to Alt1 + specific voter dist)
        topic: Topic short name
        output_path: Path to save the plot
        voter_dist_label: Label for voter distribution (for title)
        x_max: Maximum value for x-axis
        y_min: Minimum value for y-axis
        zoomed: Whether this is a zoomed plot (affects title)
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    topic_display = TOPIC_DISPLAY_NAMES.get(topic, topic.title())
    
    # Filter to this topic
    topic_df = df[df["topic"] == topic]
    topic_df = topic_df[topic_df["epsilon"].notna()]
    
    if topic_df.empty:
        print(f"Warning: No data for topic {topic}")
        plt.close()
        return
    
    # Get all epsilons for random baseline
    all_epsilons = topic_df["epsilon"].values
    
    # Method groups (ordered for legend)
    double_triple_and_random = CHATGPT_DOUBLE_STAR_METHODS + CHATGPT_TRIPLE_STAR_METHODS + NEW_RANDOM_METHODS
    method_groups = [
        ('Traditional Methods', TRADITIONAL_METHODS),
        ('ChatGPT Methods', CHATGPT_METHODS),
        ('ChatGPT* Methods', CHATGPT_STAR_METHODS),
        ('GPT**/*** + New Random', double_triple_and_random),
    ]
    
    for group_idx, (group_name, methods) in enumerate(method_groups):
        ax = axes[group_idx]
        
        color_idx = 0
        has_data = False
        
        for method in methods:
            method_data = topic_df[topic_df["method"] == method]["epsilon"].values
            if len(method_data) == 0:
                continue
            
            has_data = True
            sorted_data = np.sort(method_data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            color = CONTRAST_COLORS[color_idx % len(CONTRAST_COLORS)]
            label = METHOD_LABELS_EXTENDED.get(method, method)
            ax.step(sorted_data, cdf, where='post', label=label, color=color, linewidth=2.5)
            color_idx += 1
        
        # Add Random baseline (black line)
        if len(all_epsilons) > 0:
            sorted_random = np.sort(all_epsilons)
            cdf_random = np.arange(1, len(sorted_random) + 1) / len(sorted_random)
            ax.step(sorted_random, cdf_random, where='post', 
                    label='Random', color='black', linewidth=2.5)
        
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min, 1.05)
        ax.grid(True, alpha=0.3)
        
        ax.set_title(group_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Epsilon', fontsize=10)
        
        if group_idx == 0:
            ax.set_ylabel('Cumulative Probability', fontsize=10)
        
        if has_data or len(all_epsilons) > 0:
            ax.legend(loc='lower right', fontsize=8)
    
    zoom_suffix = ", Zoomed" if zoomed else ""
    fig.suptitle(f"{topic_display}: CDF of Critical Epsilon by Method Category\n(Alt1: Persona Only × {voter_dist_label}{zoom_suffix})", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_cdf_by_voter_dist(
    df: pd.DataFrame,
    topic: str,
    output_path: Path,
    x_max: float = 0.5,
    y_min: float = 0.5,
    zoomed: bool = True,
) -> None:
    """
    Create a 1x3 CDF plot for a single topic across 3 voter distributions.
    
    Fixed to Alt1 (persona_no_context). Each subplot shows Traditional Methods + Random.
    
    Args:
        df: DataFrame with epsilon values (should be filtered to persona_no_context)
        topic: Topic short name
        output_path: Path to save the plot
        x_max: Maximum value for x-axis
        y_min: Minimum value for y-axis
        zoomed: Whether this is a zoomed plot (affects title)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    topic_display = TOPIC_DISPLAY_NAMES.get(topic, topic.title())
    
    for voter_idx, voter_dist in enumerate(VOTER_DISTRIBUTIONS):
        ax = axes[voter_idx]
        
        # Filter to this topic and voter_dist
        subset_df = df[(df["topic"] == topic) & (df["voter_dist"] == voter_dist)]
        subset_df = subset_df[subset_df["epsilon"].notna()]
        
        if subset_df.empty:
            ax.set_title(VOTER_DIST_LABELS.get(voter_dist, voter_dist), fontsize=11, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, x_max)
            ax.set_ylim(y_min, 1.05)
            continue
        
        # Get all epsilons for random baseline
        all_epsilons = subset_df["epsilon"].values
        
        # Plot traditional methods
        color_idx = 0
        for method in TRADITIONAL_METHODS:
            method_data = subset_df[subset_df["method"] == method]["epsilon"].values
            if len(method_data) == 0:
                continue
            
            sorted_data = np.sort(method_data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            color = CONTRAST_COLORS[color_idx % len(CONTRAST_COLORS)]
            label = METHOD_LABELS.get(method, method)
            ax.step(sorted_data, cdf, where='post', label=label, color=color, linewidth=2.5)
            color_idx += 1
        
        # Add Random baseline (black line)
        if len(all_epsilons) > 0:
            sorted_random = np.sort(all_epsilons)
            cdf_random = np.arange(1, len(sorted_random) + 1) / len(sorted_random)
            ax.step(sorted_random, cdf_random, where='post', 
                    label='Random', color='black', linewidth=2.5)
        
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min, 1.05)
        ax.grid(True, alpha=0.3)
        
        ax.set_title(VOTER_DIST_LABELS.get(voter_dist, voter_dist), fontsize=11, fontweight='bold')
        ax.set_xlabel('Epsilon', fontsize=10)
        
        if voter_idx == 0:
            ax.set_ylabel('Cumulative Probability', fontsize=10)
        
        ax.legend(loc='lower right', fontsize=9)
    
    zoom_suffix = ", Zoomed" if zoomed else ""
    fig.suptitle(f"{topic_display}: Traditional Methods by Voter Distribution\n(Alt1: Persona Only{zoom_suffix})", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_cdf_by_alt_dist(
    df: pd.DataFrame,
    topic: str,
    output_path: Path,
    x_max: float = 0.5,
    y_min: float = 0.5,
    zoomed: bool = True,
) -> None:
    """
    Create a 1x4 CDF plot for a single topic across 4 alternative distributions.
    
    Each subplot shows Traditional Methods (VBC, Borda, Schulze, IRV, Plurality) + Random.
    
    Args:
        df: DataFrame with epsilon values (should be filtered to uniform voter dist)
        topic: Topic short name
        output_path: Path to save the plot
        x_max: Maximum value for x-axis
        y_min: Minimum value for y-axis
        zoomed: Whether this is a zoomed plot (affects title)
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    topic_display = TOPIC_DISPLAY_NAMES.get(topic, topic.title())
    
    for alt_idx, alt_dist in enumerate(ALT_DISTRIBUTIONS):
        ax = axes[alt_idx]
        
        # Filter to this topic and alt_dist
        subset_df = df[(df["topic"] == topic) & (df["alt_dist"] == alt_dist)]
        subset_df = subset_df[subset_df["epsilon"].notna()]
        
        if subset_df.empty:
            ax.set_title(ALT_DIST_LABELS.get(alt_dist, alt_dist), fontsize=11, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Get all epsilons for random baseline
        all_epsilons = subset_df["epsilon"].values
        
        # Plot traditional methods
        color_idx = 0
        for method in TRADITIONAL_METHODS:
            method_data = subset_df[subset_df["method"] == method]["epsilon"].values
            if len(method_data) == 0:
                continue
            
            sorted_data = np.sort(method_data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            color = CONTRAST_COLORS[color_idx % len(CONTRAST_COLORS)]
            label = METHOD_LABELS.get(method, method)
            ax.step(sorted_data, cdf, where='post', label=label, color=color, linewidth=2.5)
            color_idx += 1
        
        # Add Random baseline (black line)
        if len(all_epsilons) > 0:
            sorted_random = np.sort(all_epsilons)
            cdf_random = np.arange(1, len(sorted_random) + 1) / len(sorted_random)
            ax.step(sorted_random, cdf_random, where='post', 
                    label='Random', color='black', linewidth=2.5)
        
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min, 1.05)
        ax.grid(True, alpha=0.3)
        
        ax.set_title(ALT_DIST_LABELS.get(alt_dist, alt_dist), fontsize=11, fontweight='bold')
        ax.set_xlabel('Epsilon', fontsize=10)
        
        if alt_idx == 0:
            ax.set_ylabel('Cumulative Probability', fontsize=10)
        
        ax.legend(loc='lower right', fontsize=8)
    
    zoom_suffix = ", Zoomed" if zoomed else ""
    fig.suptitle(f"{topic_display}: Traditional Methods by Alternative Distribution\n(Uniform Voters{zoom_suffix})", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_cdf_traditional_methods(
    df: pd.DataFrame,
    topics: list[str],
    output_path: Path,
    suptitle: str = "Traditional Methods: CDF of Critical Epsilon",
    x_max: float = 0.5,
    y_min: float = 0.5,
    zoomed: bool = True,
) -> None:
    """
    Create a 1x3 CDF plot showing Traditional Methods only for 3 topics.
    
    Each subplot shows: Schulze, Borda, IRV, Plurality, VBC + Random baseline.
    
    Args:
        df: DataFrame with epsilon values filtered for persona_no_context + uniform
        topics: List of 3 topic short names
        output_path: Path to save the plot
        suptitle: Overall figure title (without zoom suffix)
        x_max: Maximum value for x-axis
        y_min: Minimum value for y-axis
        zoomed: Whether this is a zoomed plot (affects title)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for topic_idx, topic in enumerate(topics):
        ax = axes[topic_idx]
        topic_df = df[df["topic"] == topic]
        topic_df = topic_df[topic_df["epsilon"].notna()]
        
        if topic_df.empty:
            print(f"Warning: No data for topic {topic}")
            continue
        
        # Get all epsilons for random baseline
        all_epsilons = topic_df["epsilon"].values
        
        # Plot traditional methods
        color_idx = 0
        for method in TRADITIONAL_METHODS:
            method_data = topic_df[topic_df["method"] == method]["epsilon"].values
            if len(method_data) == 0:
                continue
            
            sorted_data = np.sort(method_data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            color = CONTRAST_COLORS[color_idx % len(CONTRAST_COLORS)]
            label = METHOD_LABELS.get(method, method)
            ax.step(sorted_data, cdf, where='post', label=label, color=color, linewidth=2.5)
            color_idx += 1
        
        # Add Random baseline (black line)
        if len(all_epsilons) > 0:
            sorted_random = np.sort(all_epsilons)
            cdf_random = np.arange(1, len(sorted_random) + 1) / len(sorted_random)
            ax.step(sorted_random, cdf_random, where='post', 
                    label='Random', color='black', linewidth=2.5)
        
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min, 1.05)
        ax.grid(True, alpha=0.3)
        
        topic_display = TOPIC_DISPLAY_NAMES.get(topic, topic.title())
        ax.set_title(topic_display, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epsilon', fontsize=10)
        
        if topic_idx == 0:
            ax.set_ylabel('Cumulative Probability', fontsize=10)
        
        ax.legend(loc='lower right', fontsize=9)
    
    zoom_suffix = " (Zoomed)" if zoomed else ""
    fig.suptitle(f"{suptitle}{zoom_suffix}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    """Generate all slide plots (both zoomed and full-range versions)."""
    print("Loading results...")
    df = collect_all_results()
    print(f"Loaded {len(df)} results")
    
    # Filter to uniform voters only (for alt_dist comparison)
    uniform_df = df[df["voter_dist"] == "uniform"]
    print(f"Filtered to {len(uniform_df)} results (uniform voters)")
    
    # Filter to persona_no_context + uniform only (for topic comparison)
    filtered_df = df[
        (df["alt_dist"] == "persona_no_context") & 
        (df["voter_dist"] == "uniform")
    ]
    print(f"Filtered to {len(filtered_df)} results (persona_no_context + uniform)")
    
    # Check available topics
    available_topics = uniform_df["topic"].unique()
    print(f"Available topics: {list(available_topics)}")
    
    # All 6 topics
    all_topics = ["abortion", "electoral", "healthcare", "policing", "environment", "trust"]
    
    # Topic groups for 1x3 plots
    group1_topics = ["abortion", "electoral", "healthcare"]
    group2_topics = ["policing", "environment", "trust"]
    
    # Verify topics exist
    for topic in all_topics:
        if topic not in available_topics:
            print(f"Warning: Topic '{topic}' not found in data!")
    
    SLIDES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Filter to Alt1 only (for voter distribution comparison)
    alt1_df = df[df["alt_dist"] == "persona_no_context"]
    print(f"Filtered to {len(alt1_df)} results (Alt1: persona_no_context)")
    
    # Prepare filtered DataFrames for Progressive and Conservative
    alt1_progressive_df = df[
        (df["alt_dist"] == "persona_no_context") & 
        (df["voter_dist"] == "progressive_liberal")
    ]
    alt1_conservative_df = df[
        (df["alt_dist"] == "persona_no_context") & 
        (df["voter_dist"] == "conservative_traditional")
    ]
    
    # Generate both zoomed and full-range versions
    zoom_configs = [
        (True, 0.5, 0.5, ""),       # Zoomed: x_max=0.5, y_min=0.5, no suffix
        (False, 1.0, 0.0, "_full"), # Full: x_max=1.0, y_min=0.0, _full suffix
    ]
    
    for zoomed, x_max, y_min, dir_suffix in zoom_configs:
        mode_label = "ZOOMED" if zoomed else "FULL-RANGE"
        print(f"\n{'='*60}")
        print(f"Generating {mode_label} plots (x_max={x_max}, y_min={y_min})")
        print(f"{'='*60}")
        
        # Create subfolders with suffix
        by_topic_dir = SLIDES_OUTPUT_DIR / f"by_topic{dir_suffix}"
        by_alt_dist_dir = SLIDES_OUTPUT_DIR / f"by_alt_dist{dir_suffix}"
        by_voter_dist_dir = SLIDES_OUTPUT_DIR / f"by_voter_dist{dir_suffix}"
        by_method_category_dir = SLIDES_OUTPUT_DIR / f"by_method_category{dir_suffix}"
        by_method_category_progressive_dir = SLIDES_OUTPUT_DIR / f"by_method_category_progressive{dir_suffix}"
        by_method_category_conservative_dir = SLIDES_OUTPUT_DIR / f"by_method_category_conservative{dir_suffix}"
        
        by_topic_dir.mkdir(parents=True, exist_ok=True)
        by_alt_dist_dir.mkdir(parents=True, exist_ok=True)
        by_voter_dist_dir.mkdir(parents=True, exist_ok=True)
        by_method_category_dir.mkdir(parents=True, exist_ok=True)
        by_method_category_progressive_dir.mkdir(parents=True, exist_ok=True)
        by_method_category_conservative_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate per-topic plots with 4 method categories (1x4 layout) - Uniform
        print(f"\n--- Generating per-topic CDF plots (1x4, by method category) - Uniform ---")
        for topic in all_topics:
            if topic in available_topics:
                plot_cdf_by_method_category(
                    filtered_df,
                    topic=topic,
                    output_path=by_method_category_dir / f"cdf_{topic}_by_method.png",
                    voter_dist_label="Uniform",
                    x_max=x_max,
                    y_min=y_min,
                    zoomed=zoomed,
                )
        
        # Generate per-topic plots with 4 method categories (1x4 layout) - Progressive
        print(f"\n--- Generating per-topic CDF plots (1x4, by method category) - Progressive ---")
        for topic in all_topics:
            if topic in available_topics:
                plot_cdf_by_method_category(
                    alt1_progressive_df,
                    topic=topic,
                    output_path=by_method_category_progressive_dir / f"cdf_{topic}_by_method.png",
                    voter_dist_label="Progressive/Liberal",
                    x_max=x_max,
                    y_min=y_min,
                    zoomed=zoomed,
                )
        
        # Generate per-topic plots with 4 method categories (1x4 layout) - Conservative
        print(f"\n--- Generating per-topic CDF plots (1x4, by method category) - Conservative ---")
        for topic in all_topics:
            if topic in available_topics:
                plot_cdf_by_method_category(
                    alt1_conservative_df,
                    topic=topic,
                    output_path=by_method_category_conservative_dir / f"cdf_{topic}_by_method.png",
                    voter_dist_label="Conservative/Traditional",
                    x_max=x_max,
                    y_min=y_min,
                    zoomed=zoomed,
                )
        
        # Generate per-topic plots with 3 voter distributions (1x3 layout)
        print(f"\n--- Generating per-topic CDF plots (1x3, by voter distribution) ---")
        for topic in all_topics:
            if topic in available_topics:
                plot_cdf_by_voter_dist(
                    alt1_df,
                    topic=topic,
                    output_path=by_voter_dist_dir / f"cdf_traditional_{topic}_by_voter.png",
                    x_max=x_max,
                    y_min=y_min,
                    zoomed=zoomed,
                )
        
        # Generate per-topic plots with 4 alt distributions (1x4 layout)
        print(f"\n--- Generating per-topic CDF plots (1x4, by alt distribution) ---")
        for topic in all_topics:
            if topic in available_topics:
                plot_cdf_by_alt_dist(
                    uniform_df,
                    topic=topic,
                    output_path=by_alt_dist_dir / f"cdf_traditional_{topic}_by_alt.png",
                    x_max=x_max,
                    y_min=y_min,
                    zoomed=zoomed,
                )
        
        # Generate Traditional Methods CDF plots (1x3 layout)
        print(f"\n--- Generating Traditional Methods CDF plots (1x3, by topic) ---")
        plot_cdf_traditional_methods(
            filtered_df,
            topics=group1_topics,
            output_path=by_topic_dir / "cdf_traditional_group1.png",
            suptitle="Traditional Methods: Abortion, Electoral, Healthcare\n(Alt1: Persona Only × Uniform)",
            x_max=x_max,
            y_min=y_min,
            zoomed=zoomed,
        )
        
        plot_cdf_traditional_methods(
            filtered_df,
            topics=group2_topics,
            output_path=by_topic_dir / "cdf_traditional_group2.png",
            suptitle="Traditional Methods: Policing, Environment, Trust\n(Alt1: Persona Only × Uniform)",
            x_max=x_max,
            y_min=y_min,
            zoomed=zoomed,
        )
    
    print(f"\nAll plots saved to: {SLIDES_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
