"""
Visualization functions for experiment results.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np

from .config import VOTING_METHODS, OUTPUT_DIR

logger = logging.getLogger(__name__)

# Color palette for voting methods
METHOD_COLORS = {
    "schulze": "#1f77b4",
    "veto_by_consumption": "#ff7f0e",
    "borda": "#2ca02c",
    "irv": "#d62728",
    "plurality": "#9467bd",
    "chatgpt": "#8c564b",
    "chatgpt_with_rankings": "#e377c2",
}

# Display names for methods
METHOD_NAMES = {
    "schulze": "Schulze",
    "veto_by_consumption": "Veto by Consumption",
    "borda": "Borda",
    "irv": "IRV",
    "plurality": "Plurality",
    "chatgpt": "ChatGPT",
    "chatgpt_with_rankings": "ChatGPT + Rankings",
}

# Short names for topics (for filenames)
TOPIC_SHORT_NAMES = {
    "how-should-we-increase-the-general-publics-trust-i": "trust",
    "what-are-the-best-policies-to-prevent-littering-in": "littering",
    "what-are-your-thoughts-on-the-way-university-campu": "campus_speech",
    "what-balance-should-be-struck-between-environmenta": "environment",
    "what-balance-should-exist-between-gun-safety-laws-": "guns",
    "what-limits-if-any-should-exist-on-free-speech-reg": "free_speech",
    "what-principles-should-guide-immigration-policy-an": "immigration",
    "what-reforms-if-any-should-replace-or-modify-the-e": "electoral",
    "what-responsibilities-should-tech-companies-have-w": "tech_privacy",
    "what-role-should-artificial-intelligence-play-in-s": "ai",
    "what-role-should-the-government-play-in-ensuring-u": "healthcare",
    "what-should-guide-laws-concerning-abortion": "abortion",
    "what-strategies-should-guide-policing-to-address-b": "policing",
}

# Display names for topics (for plot titles)
TOPIC_DISPLAY_NAMES = {
    "how-should-we-increase-the-general-publics-trust-i": "Public Trust in Institutions",
    "what-are-the-best-policies-to-prevent-littering-in": "Littering Prevention",
    "what-are-your-thoughts-on-the-way-university-campu": "Campus Free Speech",
    "what-balance-should-be-struck-between-environmenta": "Environment vs Economy",
    "what-balance-should-exist-between-gun-safety-laws-": "Gun Safety",
    "what-limits-if-any-should-exist-on-free-speech-reg": "Free Speech Limits",
    "what-principles-should-guide-immigration-policy-an": "Immigration Policy",
    "what-reforms-if-any-should-replace-or-modify-the-e": "Electoral Reform",
    "what-responsibilities-should-tech-companies-have-w": "Tech Privacy",
    "what-role-should-artificial-intelligence-play-in-s": "AI in Society",
    "what-role-should-the-government-play-in-ensuring-u": "Healthcare",
    "what-should-guide-laws-concerning-abortion": "Abortion Laws",
    "what-strategies-should-guide-policing-to-address-b": "Policing & Civil Rights",
}


def collect_results_for_topic(
    topic_slug: str,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full"
) -> Dict[str, List[float]]:
    """
    Collect all epsilon results for a topic across all repetitions and samples.
    
    Args:
        topic_slug: Topic slug
        output_dir: Output directory
        ablation: Ablation type
    
    Returns:
        Dict mapping method name to list of epsilon values
    """
    results = {method: [] for method in VOTING_METHODS}
    
    topic_dir = output_dir / "data" / topic_slug
    
    if not topic_dir.exists():
        logger.warning(f"Topic directory not found: {topic_dir}")
        return results
    
    # Iterate through all rep directories
    for rep_dir in sorted(topic_dir.glob("rep*")):
        # Handle ablation subdirectory
        if ablation != "full":
            data_dir = rep_dir / f"ablation_{ablation}"
        else:
            data_dir = rep_dir
        
        if not data_dir.exists():
            continue
        
        # Iterate through all sample directories
        for sample_dir in sorted(data_dir.glob("sample*")):
            results_file = sample_dir / "results.json"
            if not results_file.exists():
                continue
            
            with open(results_file, 'r') as f:
                sample_results = json.load(f)
            
            for method in VOTING_METHODS:
                if method in sample_results:
                    epsilon = sample_results[method].get("epsilon")
                    if epsilon is not None:
                        results[method].append(epsilon)
    
    return results


def collect_all_results(
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full",
    topics: Optional[List[str]] = None
) -> Dict[str, List[float]]:
    """
    Collect all epsilon results across all topics.
    
    Args:
        output_dir: Output directory
        ablation: Ablation type
        topics: List of topics to include (None = all)
    
    Returns:
        Dict mapping method name to list of epsilon values
    """
    all_results = {method: [] for method in VOTING_METHODS}
    
    data_dir = output_dir / "data"
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return all_results
    
    # Get all topic directories
    if topics is None:
        topic_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    else:
        topic_dirs = [data_dir / t for t in topics if (data_dir / t).exists()]
    
    for topic_dir in topic_dirs:
        topic_results = collect_results_for_topic(
            topic_dir.name, output_dir, ablation
        )
        for method in VOTING_METHODS:
            all_results[method].extend(topic_results[method])
    
    return all_results


def plot_epsilon_histogram(
    results: Dict[str, List[float]],
    title: str = "Epsilon Distribution by Voting Method",
    output_path: Optional[Path] = None,
    bins: int = 20
) -> None:
    """
    Plot histogram of epsilon values for each voting method.
    
    Args:
        results: Dict mapping method to list of epsilon values
        title: Plot title
        output_path: Path to save figure (None = show)
        bins: Number of histogram bins
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get all epsilon values to determine range
    all_values = []
    for values in results.values():
        all_values.extend([v for v in values if v is not None])
    
    if not all_values:
        logger.warning("No epsilon values to plot")
        return
    
    min_val = min(all_values)
    max_val = max(all_values)
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    # Collect data for stacked histogram
    data_to_plot = []
    labels = []
    colors = []
    for method in VOTING_METHODS:
        values = [v for v in results.get(method, []) if v is not None]
        if values:
            data_to_plot.append(values)
            labels.append(METHOD_NAMES.get(method, method))
            colors.append(METHOD_COLORS.get(method, None))
    
    # Plot stacked histogram
    if data_to_plot:
        ax.hist(
            data_to_plot,
            bins=bin_edges,
            label=labels,
            color=colors,
            stacked=True,
            edgecolor='white',
            linewidth=0.5
        )
    
    ax.set_xlabel("Epsilon (ε)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved histogram to {output_path}")
        plt.close()
    else:
        plt.show()


def plot_epsilon_barplot(
    results: Dict[str, List[float]],
    title: str = "Average Epsilon by Voting Method",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot bar chart of average epsilon with error bars.
    
    Args:
        results: Dict mapping method to list of epsilon values
        title: Plot title
        output_path: Path to save figure (None = show)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    means = []
    stds = []
    colors = []
    
    for method in VOTING_METHODS:
        values = [v for v in results.get(method, []) if v is not None]
        if values:
            methods.append(METHOD_NAMES.get(method, method))
            means.append(np.mean(values))
            stds.append(np.std(values))
            colors.append(METHOD_COLORS.get(method, "#333333"))
    
    if not methods:
        logger.warning("No data to plot")
        return
    
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    
    ax.set_xlabel("Voting Method", fontsize=12)
    ax.set_ylabel("Average Epsilon (ε)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.annotate(
            f'{mean:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=9
        )
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved bar plot to {output_path}")
        plt.close()
    else:
        plt.show()


def collect_likert_for_topic(
    topic_slug: str,
    output_dir: Path = OUTPUT_DIR,
    ablation: str = "full"
) -> Dict[str, List[float]]:
    """
    Collect average Likert ratings for winners of each voting method.
    
    Args:
        topic_slug: Topic slug
        output_dir: Output directory
        ablation: Ablation type
    
    Returns:
        Dict mapping method name to list of average Likert ratings
    """
    results = {method: [] for method in VOTING_METHODS}
    
    topic_dir = output_dir / "data" / topic_slug
    
    if not topic_dir.exists():
        return results
    
    for rep_dir in sorted(topic_dir.glob("rep*")):
        if ablation != "full":
            data_dir = rep_dir / f"ablation_{ablation}"
        else:
            data_dir = rep_dir
        
        if not data_dir.exists():
            continue
        
        # Load filtered Likert ratings
        likert_file = data_dir / "filtered_likert.json"
        if not likert_file.exists():
            continue
        
        with open(likert_file, 'r') as f:
            likert = json.load(f)
        
        # Iterate through samples
        for sample_dir in sorted(data_dir.glob("sample*")):
            results_file = sample_dir / "results.json"
            if not results_file.exists():
                continue
            
            with open(results_file, 'r') as f:
                sample_results = json.load(f)
            
            # Load sampled persona indices
            persona_file = sample_dir / "persona_indices.json"
            if not persona_file.exists():
                continue
            
            with open(persona_file, 'r') as f:
                persona_indices = json.load(f)
            
            for method in VOTING_METHODS:
                if method in sample_results:
                    winner = sample_results[method].get("winner")
                    if winner is not None:
                        winner_idx = int(winner)
                        # Average Likert rating for winner across sampled personas
                        ratings = [likert[p_idx][winner_idx] for p_idx in persona_indices]
                        avg_rating = np.mean(ratings)
                        results[method].append(avg_rating)
    
    return results


def plot_likert_barplot(
    results: Dict[str, List[float]],
    title: str = "Average Likert Rating by Voting Method",
    output_path: Optional[Path] = None
) -> None:
    """
    Plot bar chart of average Likert ratings with error bars.
    
    Args:
        results: Dict mapping method to list of average Likert ratings
        title: Plot title
        output_path: Path to save figure (None = show)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = []
    means = []
    stds = []
    colors = []
    
    for method in VOTING_METHODS:
        values = [v for v in results.get(method, []) if v is not None]
        if values:
            methods.append(METHOD_NAMES.get(method, method))
            means.append(np.mean(values))
            stds.append(np.std(values))
            colors.append(METHOD_COLORS.get(method, "#333333"))
    
    if not methods:
        logger.warning("No data to plot")
        return
    
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    
    ax.set_xlabel("Voting Method", fontsize=12)
    ax.set_ylabel("Average Likert Rating (1-5)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylim(1, 5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.annotate(
            f'{mean:.2f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=9
        )
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved Likert bar plot to {output_path}")
        plt.close()
    else:
        plt.show()


def generate_all_plots(
    output_dir: Path = OUTPUT_DIR,
    topics: Optional[List[str]] = None,
    ablations: Optional[List[str]] = None
) -> None:
    """
    Generate all plots for the experiment.
    
    Args:
        output_dir: Output directory
        topics: List of topics (None = auto-detect)
        ablations: List of ablations to plot
    """
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    if ablations is None:
        ablations = ["full"]
    
    for ablation in ablations:
        suffix = f"_{ablation}" if ablation != "full" else ""
        
        # Aggregate plots
        all_results = collect_all_results(output_dir, ablation, topics)
        
        plot_epsilon_histogram(
            all_results,
            title=f"Epsilon Distribution by Voting Method{suffix.replace('_', ' ').title()}",
            output_path=figures_dir / f"epsilon_histogram_aggregate{suffix}.png"
        )
        
        plot_epsilon_barplot(
            all_results,
            title=f"Average Epsilon by Voting Method{suffix.replace('_', ' ').title()}",
            output_path=figures_dir / f"epsilon_barplot_aggregate{suffix}.png"
        )
        
        # Per-topic plots
        data_dir = output_dir / "data"
        if topics is None and data_dir.exists():
            topics_to_plot = [d.name for d in data_dir.iterdir() if d.is_dir()]
        else:
            topics_to_plot = topics or []
        
        for topic in topics_to_plot:
            # Use short name for filename, display name for title
            short_name = TOPIC_SHORT_NAMES.get(topic, topic[:20])
            display_name = TOPIC_DISPLAY_NAMES.get(topic, topic[:50])
            ablation_label = suffix.replace('_', ' ').title() if suffix else ""
            
            topic_results = collect_results_for_topic(topic, output_dir, ablation)
            
            plot_epsilon_histogram(
                topic_results,
                title=f"Epsilon Distribution: {display_name}{ablation_label}",
                output_path=figures_dir / f"epsilon_histogram_{short_name}{suffix}.png"
            )
            
            plot_epsilon_barplot(
                topic_results,
                title=f"Average Epsilon: {display_name}{ablation_label}",
                output_path=figures_dir / f"epsilon_barplot_{short_name}{suffix}.png"
            )
            
            # Likert plots
            likert_results = collect_likert_for_topic(topic, output_dir, ablation)
            
            plot_likert_barplot(
                likert_results,
                title=f"Average Likert Rating: {display_name}{ablation_label}",
                output_path=figures_dir / f"likert_barplot_{short_name}{suffix}.png"
            )
    
    logger.info(f"Generated all plots in {figures_dir}")

