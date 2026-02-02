"""
What-If Scatter Analysis: Recompute epsilons with m=101.

This script recomputes epsilons for randomly inserted statements using m=101
instead of m=100. By including the new statement as a full participant in the
alternative pool, we can see how epsilon changes.

Output:
- data/: JSON files with recomputed epsilons
- plots/: Scatter plots comparing precomputed vs random insertion with m=101
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from pvc_toolbox import compute_critical_epsilon

from .config import PROJECT_ROOT, PHASE2_DATA_DIR
from src.experiment_utils.epsilon_calculator import compute_critical_epsilon_custom

logger = logging.getLogger(__name__)

# Set style for slide-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Slide-quality figure sizes
FIGURE_SIZE_WIDE = (14, 6)

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "random_insertion_test" / "scatter_what_if"
DATA_DIR = OUTPUT_DIR / "data"
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_OVERRIDE_DIR = OUTPUT_DIR / "plots_override"

# Topics to process (short names used in directory structure)
TOPICS = ["abortion", "electoral", "environment", "healthcare", "policing", "trust"]

# Number of reps and mini_reps
N_REPS = 10
N_MINI_REPS = 4

# Methods with insertion_positions to process (new statements, m=101)
INSERTION_METHODS = {
    "random_insertion": {"label": "Random", "marker": "x", "color": "red"},
    "chatgpt_double_star": {"label": "GPT**", "marker": "s", "color": "green"},
    "chatgpt_double_star_rankings": {"label": "GPT**+Rank", "marker": "^", "color": "purple"},
    "chatgpt_double_star_personas": {"label": "GPT**+Pers", "marker": "v", "color": "orange"},
    "chatgpt_triple_star": {"label": "GPT***", "marker": "D", "color": "brown"},
}

# Traditional methods (select from existing alternatives, use precomputed epsilon m=100)
TRADITIONAL_METHODS = {
    "plurality": {"label": "Plurality", "marker": "o", "color": "cyan"},
}


def construct_101_preferences(
    preferences: List[List[str]], 
    insertion_positions: List[int]
) -> List[List[str]]:
    """
    Construct 101-alternative preference profile by inserting new statement.
    
    Args:
        preferences: Original 100x100 preference matrix [rank][voter]
        insertion_positions: List of positions where "100" should be inserted for each voter
        
    Returns:
        New preference matrix with 101 alternatives [rank][voter]
    """
    n_ranks = len(preferences)
    n_voters = len(preferences[0]) if preferences else 0
    
    # Convert to voter-centric format, insert "100", then convert back
    # preferences[rank][voter] -> voter_rankings[voter][rank]
    voter_rankings = []
    for voter_idx in range(n_voters):
        ranking = [preferences[rank][voter_idx] for rank in range(n_ranks)]
        voter_rankings.append(ranking)
    
    # Insert "100" at specified position for each voter
    for voter_idx, pos in enumerate(insertion_positions):
        if pos is not None and voter_idx < len(voter_rankings):
            # Clamp position to valid range
            pos = max(0, min(pos, len(voter_rankings[voter_idx])))
            voter_rankings[voter_idx].insert(pos, "100")
    
    # Convert back to [rank][voter] format
    n_new_ranks = 101
    new_preferences = []
    for rank in range(n_new_ranks):
        rank_row = []
        for voter_idx in range(n_voters):
            if rank < len(voter_rankings[voter_idx]):
                rank_row.append(voter_rankings[voter_idx][rank])
            else:
                rank_row.append("100")  # Fallback
        new_preferences.append(rank_row)
    
    return new_preferences


def compute_epsilon_m101(preferences_101: List[List[str]]) -> Optional[float]:
    """
    Compute epsilon for the new statement "100" with m=101.
    
    Args:
        preferences_101: Preference matrix with 101 alternatives
        
    Returns:
        Epsilon value or None if computation fails
    """
    alternatives = [str(i) for i in range(101)]
    winner = "100"
    
    try:
        epsilon = compute_critical_epsilon(preferences_101, alternatives, winner)
        return epsilon
    except Exception as e:
        logger.error(f"Epsilon computation failed: {e}")
        return None


def compute_epsilon_m101_override(preferences_101: List[List[str]]) -> Optional[float]:
    """
    Compute epsilon for the new statement "100" using compute_critical_epsilon_custom
    with explicit m_override=101.
    
    This serves as a sanity check that the custom implementation matches
    the standard pvc_toolbox implementation when m_override equals the
    actual number of alternatives.
    
    Args:
        preferences_101: Preference matrix with 101 alternatives
        
    Returns:
        Epsilon value or None if computation fails
    """
    alternatives = [str(i) for i in range(101)]
    winner = "100"
    
    try:
        epsilon = compute_critical_epsilon_custom(
            preferences_101, alternatives, winner, m_override=101
        )
        return epsilon
    except Exception as e:
        logger.error(f"Epsilon computation (override) failed: {e}")
        return None


def compute_average_ranks(preferences: List[List[str]]) -> Dict[str, float]:
    """
    Compute average rank for each statement across all voters.
    
    Args:
        preferences: 2D list where preferences[rank][voter] = statement_id
        
    Returns:
        Dict mapping statement_id to average rank
    """
    statement_ranks: Dict[str, List[int]] = {}
    
    for rank_idx, rank_row in enumerate(preferences):
        for statement_id in rank_row:
            if statement_id not in statement_ranks:
                statement_ranks[statement_id] = []
            statement_ranks[statement_id].append(rank_idx)
    
    return {sid: np.mean(ranks) for sid, ranks in statement_ranks.items()}


def compute_and_save_whatif_data(topic: str) -> Dict[str, List[Dict]]:
    """
    Compute what-if epsilons (m=101) for all methods with insertion_positions in a topic.
    
    Args:
        topic: Short topic name
        
    Returns:
        Dict mapping method name to list of data points with epsilon and avg_position
    """
    # Initialize data dict for each method
    method_data: Dict[str, List[Dict]] = {method: [] for method in INSERTION_METHODS}
    
    for rep_idx in range(N_REPS):
        base_path = PHASE2_DATA_DIR / topic / "uniform" / "persona_no_context" / f"rep{rep_idx}"
        
        # Load original preferences
        preferences_path = base_path / "preferences.json"
        if not preferences_path.exists():
            logger.warning(f"Missing preferences file: {preferences_path}")
            continue
            
        with open(preferences_path, 'r') as f:
            preferences = json.load(f)
        
        for mini_rep_idx in range(N_MINI_REPS):
            results_path = base_path / f"mini_rep{mini_rep_idx}" / "results.json"
            
            if not results_path.exists():
                logger.warning(f"Missing results file: {results_path}")
                continue
                
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            if "results" not in results:
                continue
            
            # Process each method with insertion_positions
            for method_name in INSERTION_METHODS:
                if method_name not in results["results"]:
                    continue
                    
                method_result = results["results"][method_name]
                insertion_positions = method_result.get("insertion_positions", [])
                original_epsilon = method_result.get("epsilon")
                
                if not insertion_positions:
                    continue
                
                # Filter None values
                valid_positions = [p if p is not None else 50 for p in insertion_positions]
                
                if not valid_positions:
                    continue
                
                # Construct 101-alternative preferences
                preferences_101 = construct_101_preferences(preferences, valid_positions)
                
                # Compute epsilon with m=101 (natural)
                epsilon_m101 = compute_epsilon_m101(preferences_101)
                
                if epsilon_m101 is not None:
                    avg_position = np.mean([p for p in insertion_positions if p is not None])
                    method_data[method_name].append({
                        "rep": rep_idx,
                        "mini_rep": mini_rep_idx,
                        "avg_position": avg_position,
                        "epsilon_m101": epsilon_m101,
                        "epsilon_m100": original_epsilon,
                    })
    
    return method_data


def load_precomputed_data(topic: str) -> List[Tuple[float, float]]:
    """
    Load precomputed statements with their average ranks and epsilons.
    (Same as in plot_rank_epsilon_scatter.py)
    
    Args:
        topic: Short topic name
        
    Returns:
        List of (average_rank, epsilon) tuples
    """
    data_points = []
    
    for rep_idx in range(N_REPS):
        base_path = PHASE2_DATA_DIR / topic / "uniform" / "persona_no_context" / f"rep{rep_idx}"
        
        # Load preferences
        preferences_path = base_path / "preferences.json"
        if not preferences_path.exists():
            continue
            
        with open(preferences_path, 'r') as f:
            preferences = json.load(f)
        
        # Compute average ranks
        avg_ranks = compute_average_ranks(preferences)
        
        # Load precomputed epsilons
        epsilons_path = base_path / "precomputed_epsilons.json"
        if not epsilons_path.exists():
            continue
            
        with open(epsilons_path, 'r') as f:
            epsilons = json.load(f)
        
        # Combine: (avg_rank, epsilon) for each statement
        for statement_id, epsilon in epsilons.items():
            if epsilon is not None and statement_id in avg_ranks:
                data_points.append((avg_ranks[statement_id], epsilon))
    
    return data_points


def load_traditional_method_data(topic: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    Load winners from traditional methods with their avg rank and precomputed epsilon.
    
    Traditional methods select from existing 100 alternatives, so we use
    precomputed epsilons (m=100) rather than computing m=101.
    
    Args:
        topic: Short topic name
        
    Returns:
        Dict mapping method name to list of (avg_rank, epsilon) tuples
    """
    method_data: Dict[str, List[Tuple[float, float]]] = {method: [] for method in TRADITIONAL_METHODS}
    
    for rep_idx in range(N_REPS):
        base_path = PHASE2_DATA_DIR / topic / "uniform" / "persona_no_context" / f"rep{rep_idx}"
        
        # Load preferences
        preferences_path = base_path / "preferences.json"
        if not preferences_path.exists():
            continue
            
        with open(preferences_path, 'r') as f:
            preferences = json.load(f)
        
        # Compute average ranks
        avg_ranks = compute_average_ranks(preferences)
        
        # Load precomputed epsilons
        epsilons_path = base_path / "precomputed_epsilons.json"
        if not epsilons_path.exists():
            continue
            
        with open(epsilons_path, 'r') as f:
            epsilons = json.load(f)
        
        # Process each mini-rep to get winners
        for mini_rep_idx in range(N_MINI_REPS):
            results_path = base_path / f"mini_rep{mini_rep_idx}" / "results.json"
            
            if not results_path.exists():
                continue
                
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            if "results" not in results:
                continue
            
            # Get winner for each traditional method
            for method_name in TRADITIONAL_METHODS:
                if method_name not in results["results"]:
                    continue
                    
                method_result = results["results"][method_name]
                # Get full_winner_idx which is the index in the 100-statement pool
                winner_idx = method_result.get("full_winner_idx")
                
                if winner_idx is None:
                    continue
                
                # Look up avg rank and epsilon for this winner
                winner_str = str(winner_idx)
                if winner_str in avg_ranks and winner_str in epsilons:
                    epsilon = epsilons[winner_str]
                    if epsilon is not None:
                        method_data[method_name].append((avg_ranks[winner_str], epsilon))
    
    return method_data


def plot_whatif_scatter(
    precomputed: List[Tuple[float, float]],
    method_data: Dict[str, List[Dict]],
    traditional_data: Dict[str, List[Tuple[float, float]]],
    topic: str,
    output_path: Path
) -> None:
    """
    Create scatter plot with what-if epsilon (m=101) for all methods.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    
    # Plot precomputed (blue)
    if precomputed:
        pre_x, pre_y = zip(*precomputed)
        ax.scatter(pre_x, pre_y, alpha=0.4, label=f"Precomputed (n={len(precomputed):,})", s=15, color='blue')
    
    # Plot traditional methods (select from existing alternatives)
    for method_name, config in TRADITIONAL_METHODS.items():
        data = traditional_data.get(method_name, [])
        if data:
            x, y = zip(*data)
            ax.scatter(x, y, alpha=0.7, 
                      label=f"{config['label']} (n={len(data):,})", 
                      s=60, marker=config['marker'], color=config['color'], edgecolors='black', linewidths=0.5)
    
    # Plot each method with insertion_positions (new statements)
    for method_name, config in INSERTION_METHODS.items():
        data = method_data.get(method_name, [])
        if data:
            x = [d["avg_position"] for d in data]
            y = [d["epsilon_m101"] for d in data]
            ax.scatter(x, y, alpha=0.7, 
                      label=f"{config['label']} (n={len(data):,})", 
                      s=50, marker=config['marker'], color=config['color'])
    
    # Formatting
    ax.set_xlabel("Average Rank Position", fontsize=12)
    ax.set_ylabel("Epsilon (lower = better consensus)", fontsize=12)
    ax.set_title(f"Rank vs Epsilon - {topic.title()}", fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved what-if scatter plot to {output_path}")


def plot_combined_whatif_scatter(
    all_precomputed: List[Tuple[float, float]],
    all_method_data: Dict[str, List[Dict]],
    all_traditional_data: Dict[str, List[Tuple[float, float]]],
    output_path: Path
) -> None:
    """
    Create combined scatter plot with data from all topics.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    
    # Plot precomputed (blue)
    if all_precomputed:
        pre_x, pre_y = zip(*all_precomputed)
        ax.scatter(pre_x, pre_y, alpha=0.3, label=f"Precomputed (n={len(all_precomputed):,})", s=10, color='blue')
    
    # Plot traditional methods (select from existing alternatives)
    for method_name, config in TRADITIONAL_METHODS.items():
        data = all_traditional_data.get(method_name, [])
        if data:
            x, y = zip(*data)
            ax.scatter(x, y, alpha=0.7, 
                      label=f"{config['label']} (n={len(data):,})", 
                      s=60, marker=config['marker'], color=config['color'], edgecolors='black', linewidths=0.5)
    
    # Plot each method with insertion_positions (new statements)
    for method_name, config in INSERTION_METHODS.items():
        data = all_method_data.get(method_name, [])
        if data:
            x = [d["avg_position"] for d in data]
            y = [d["epsilon_m101"] for d in data]
            ax.scatter(x, y, alpha=0.7, 
                      label=f"{config['label']} (n={len(data):,})", 
                      s=50, marker=config['marker'], color=config['color'])
    
    # Formatting
    ax.set_xlabel("Average Rank Position", fontsize=12)
    ax.set_ylabel("Epsilon (lower = better consensus)", fontsize=12)
    ax.set_title("Rank vs Epsilon - All Topics Combined", fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved combined what-if scatter plot to {output_path}")


def plot_whatif_scatter_override(
    precomputed: List[Tuple[float, float]],
    whatif_data: List[Dict],
    topic: str,
    output_path: Path
) -> None:
    """
    Create scatter plot with what-if epsilon using m_override=101.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    
    # Plot precomputed (blue)
    if precomputed:
        pre_x, pre_y = zip(*precomputed)
        ax.scatter(pre_x, pre_y, alpha=0.5, label=f"Precomputed (n={len(precomputed):,})", s=20)
    
    # Plot random insertion with m_override=101 (red)
    if whatif_data:
        ri_x = [d["avg_position"] for d in whatif_data]
        ri_y = [d["epsilon_m101_override"] for d in whatif_data if d.get("epsilon_m101_override") is not None]
        ri_x = [d["avg_position"] for d in whatif_data if d.get("epsilon_m101_override") is not None]
        ax.scatter(ri_x, ri_y, alpha=0.7, label=f"Random Insertion m_override=101 (n={len(ri_x):,})", s=50, marker='x')
    
    # Formatting
    ax.set_xlabel("Average Rank Position", fontsize=12)
    ax.set_ylabel("Epsilon (lower = better consensus)", fontsize=12)
    ax.set_title(f"Rank vs Epsilon (m_override=101) - {topic.title()}", fontsize=14)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved what-if override scatter plot to {output_path}")


def plot_combined_whatif_scatter_override(
    all_precomputed: List[Tuple[float, float]],
    all_whatif_data: List[Dict],
    output_path: Path
) -> None:
    """
    Create combined scatter plot with m_override=101 data from all topics.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
    
    # Plot precomputed (blue)
    if all_precomputed:
        pre_x, pre_y = zip(*all_precomputed)
        ax.scatter(pre_x, pre_y, alpha=0.3, label=f"Precomputed (n={len(all_precomputed):,})", s=15)
    
    # Plot random insertion with m_override=101 (red)
    valid_data = [d for d in all_whatif_data if d.get("epsilon_m101_override") is not None]
    if valid_data:
        ri_x = [d["avg_position"] for d in valid_data]
        ri_y = [d["epsilon_m101_override"] for d in valid_data]
        ax.scatter(ri_x, ri_y, alpha=0.7, label=f"Random Insertion m_override=101 (n={len(valid_data):,})", s=50, marker='x')
    
    # Formatting
    ax.set_xlabel("Average Rank Position", fontsize=12)
    ax.set_ylabel("Epsilon (lower = better consensus)", fontsize=12)
    ax.set_title("Rank vs Epsilon (m_override=101) - All Topics Combined", fontsize=14)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved combined what-if override scatter plot to {output_path}")


def main():
    """Generate what-if scatter plots with m=101 epsilon computation."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create output directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directories: {DATA_DIR}, {PLOTS_DIR}")
    
    # Collect data for combined plot
    combined_precomputed = []
    combined_method_data: Dict[str, List[Dict]] = {method: [] for method in INSERTION_METHODS}
    combined_traditional_data: Dict[str, List[Tuple[float, float]]] = {method: [] for method in TRADITIONAL_METHODS}
    
    # Process each topic
    for topic in tqdm(TOPICS, desc="Processing topics"):
        logger.info(f"Processing topic: {topic}")
        
        # Compute what-if epsilons for all methods (this may take a while)
        method_data = compute_and_save_whatif_data(topic)
        
        # Save data
        data_path = DATA_DIR / f"{topic}.json"
        with open(data_path, 'w') as f:
            json.dump(method_data, f, indent=2)
        
        total_points = sum(len(data) for data in method_data.values())
        logger.info(f"Saved {total_points} data points to {data_path}")
        
        # Load precomputed data
        precomputed = load_precomputed_data(topic)
        
        # Load traditional method data (plurality, etc.)
        traditional_data = load_traditional_method_data(topic)
        
        has_method_data = any(len(data) > 0 for data in method_data.values())
        if not precomputed and not has_method_data:
            logger.warning(f"Skipping topic '{topic}' due to missing data")
            continue
        
        # Plot scatter for this topic
        plot_path = PLOTS_DIR / f"scatter_{topic}.png"
        plot_whatif_scatter(precomputed, method_data, traditional_data, topic, plot_path)
        
        # Accumulate for combined plot
        combined_precomputed.extend(precomputed)
        for method_name in INSERTION_METHODS:
            combined_method_data[method_name].extend(method_data.get(method_name, []))
        for method_name in TRADITIONAL_METHODS:
            combined_traditional_data[method_name].extend(traditional_data.get(method_name, []))
    
    # Generate combined scatter plot
    has_combined_data = any(len(data) > 0 for data in combined_method_data.values())
    if combined_precomputed or has_combined_data:
        plot_path = PLOTS_DIR / "scatter_all_topics.png"
        plot_combined_whatif_scatter(combined_precomputed, combined_method_data, combined_traditional_data, plot_path)
    
    # Log summary
    logger.info("\nSummary of data points by method:")
    logger.info("  Traditional methods (m=100):")
    for method_name, config in TRADITIONAL_METHODS.items():
        count = len(combined_traditional_data[method_name])
        logger.info(f"    {config['label']}: {count} points")
    logger.info("  Generative methods (m=101):")
    for method_name, config in INSERTION_METHODS.items():
        count = len(combined_method_data[method_name])
        logger.info(f"    {config['label']}: {count} points")
    
    logger.info("Done generating all what-if scatter plots")


if __name__ == "__main__":
    main()
