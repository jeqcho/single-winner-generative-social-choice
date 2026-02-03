"""
Analyze ranking stability test results.

Computes three metrics:
1. Kendall Tau Correlation - pairwise rank agreement between runs
2. Position Variance - how much each alternative's position varies across runs
3. Exact Match Rate - percentage of identical rankings across runs

Usage:
    uv run python -m src.experiment_utils.analyze_ranking_stability
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ranking_stability_test"
FIGURES_DIR = OUTPUT_DIR / "figures"


def load_results(topic: str) -> Dict:
    """Load raw rankings for a topic."""
    rankings_path = OUTPUT_DIR / topic / "raw_rankings.json"
    summary_path = OUTPUT_DIR / topic / "summary.json"
    
    with open(rankings_path) as f:
        results_by_voter = json.load(f)
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    return {"results_by_voter": results_by_voter, "summary": summary}


def compute_kendall_tau(ranking1: List[int], ranking2: List[int]) -> float:
    """
    Compute Kendall tau correlation between two rankings.
    
    Rankings should be lists of alternative IDs in preference order.
    Returns tau value in [-1, 1] where 1 = identical, -1 = reversed.
    """
    if len(ranking1) != len(ranking2):
        return np.nan
    
    # Convert rankings to position arrays
    n = len(ranking1)
    pos1 = {alt: i for i, alt in enumerate(ranking1)}
    pos2 = {alt: i for i, alt in enumerate(ranking2)}
    
    # Create position vectors for common alternatives
    common_alts = set(ranking1) & set(ranking2)
    if len(common_alts) < 2:
        return np.nan
    
    alts = sorted(common_alts)
    vec1 = [pos1[alt] for alt in alts]
    vec2 = [pos2[alt] for alt in alts]
    
    tau, _ = kendalltau(vec1, vec2)
    return tau


def compute_position_variance(rankings: List[List[int]]) -> Dict:
    """
    Compute position variance for each alternative across multiple rankings.
    
    Returns dict mapping alternative -> position variance.
    """
    if not rankings:
        return {}
    
    # Build position matrix: alt -> list of positions
    alt_positions = {}
    for ranking in rankings:
        for pos, alt in enumerate(ranking):
            if alt not in alt_positions:
                alt_positions[alt] = []
            alt_positions[alt].append(pos)
    
    # Compute variance for each alternative
    variances = {}
    for alt, positions in alt_positions.items():
        if len(positions) > 1:
            variances[alt] = np.var(positions)
        else:
            variances[alt] = 0.0
    
    return variances


def compute_exact_matches(rankings: List[List[int]]) -> Tuple[int, int]:
    """
    Count exact matches among pairs of rankings.
    
    Returns (n_matches, n_pairs).
    """
    n_rankings = len(rankings)
    if n_rankings < 2:
        return 0, 0
    
    n_matches = 0
    n_pairs = 0
    
    for i, j in combinations(range(n_rankings), 2):
        n_pairs += 1
        if rankings[i] == rankings[j]:
            n_matches += 1
    
    return n_matches, n_pairs


def analyze_voter(voter_data: Dict) -> Dict:
    """Analyze stability for a single voter."""
    rankings_data = voter_data.get("rankings", [])
    
    # Extract just the ranking lists (filtering out any with errors)
    rankings = [r["ranking"] for r in rankings_data if r.get("ranking")]
    
    if len(rankings) < 2:
        return {"error": "Not enough valid rankings"}
    
    # 1. Kendall tau - pairwise between all runs
    tau_values = []
    for i, j in combinations(range(len(rankings)), 2):
        tau = compute_kendall_tau(rankings[i], rankings[j])
        if not np.isnan(tau):
            tau_values.append(tau)
    
    # 2. Position variance
    position_variances = compute_position_variance(rankings)
    
    # 3. Exact matches
    n_matches, n_pairs = compute_exact_matches(rankings)
    
    return {
        "n_rankings": len(rankings),
        "kendall_tau_mean": np.mean(tau_values) if tau_values else np.nan,
        "kendall_tau_std": np.std(tau_values) if tau_values else np.nan,
        "kendall_tau_min": np.min(tau_values) if tau_values else np.nan,
        "kendall_tau_max": np.max(tau_values) if tau_values else np.nan,
        "position_variance_mean": np.mean(list(position_variances.values())) if position_variances else np.nan,
        "position_variance_max": np.max(list(position_variances.values())) if position_variances else np.nan,
        "exact_matches": n_matches,
        "total_pairs": n_pairs,
        "exact_match_rate": n_matches / n_pairs if n_pairs > 0 else 0.0,
        "position_variances": position_variances,
    }


def analyze_topic(topic: str) -> Dict:
    """Analyze stability for a topic."""
    data = load_results(topic)
    results_by_voter = data["results_by_voter"]
    
    voter_analyses = {}
    all_tau_values = []
    all_position_variances = []
    total_matches = 0
    total_pairs = 0
    
    for voter_idx, voter_data in results_by_voter.items():
        analysis = analyze_voter(voter_data)
        voter_analyses[voter_idx] = analysis
        
        if "error" not in analysis:
            if not np.isnan(analysis["kendall_tau_mean"]):
                all_tau_values.append(analysis["kendall_tau_mean"])
            all_position_variances.append(analysis["position_variance_mean"])
            total_matches += analysis["exact_matches"]
            total_pairs += analysis["total_pairs"]
    
    # Overall metrics
    overall = {
        "n_voters": len(voter_analyses),
        "kendall_tau_mean": np.mean(all_tau_values) if all_tau_values else np.nan,
        "kendall_tau_std": np.std(all_tau_values) if all_tau_values else np.nan,
        "position_variance_mean": np.nanmean(all_position_variances) if all_position_variances else np.nan,
        "position_variance_std": np.nanstd(all_position_variances) if all_position_variances else np.nan,
        "exact_match_rate": total_matches / total_pairs if total_pairs > 0 else 0.0,
        "total_exact_matches": total_matches,
        "total_pairs": total_pairs,
    }
    
    return {
        "topic": topic,
        "overall": overall,
        "per_voter": voter_analyses,
    }


def plot_kendall_tau_distribution(topics: List[str]) -> None:
    """Plot distribution of Kendall tau values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'abortion': '#2E86AB', 'environment': '#A23B72'}
    
    for topic in topics:
        analysis = analyze_topic(topic)
        
        # Collect all tau means per voter
        tau_means = [v["kendall_tau_mean"] for v in analysis["per_voter"].values() 
                    if "error" not in v and not np.isnan(v["kendall_tau_mean"])]
        
        ax.hist(tau_means, bins=20, alpha=0.6, label=topic.capitalize(),
                color=colors.get(topic, 'gray'), edgecolor='white')
    
    ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Perfect (tau=1.0)')
    ax.set_xlabel('Mean Kendall Tau per Voter', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Ranking Stability: Kendall Tau Distribution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kendall_tau_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved kendall_tau_distribution.png")


def plot_position_variance_heatmap(topics: List[str]) -> None:
    """Plot heatmap of position variances."""
    fig, axes = plt.subplots(1, len(topics), figsize=(14, 6))
    if len(topics) == 1:
        axes = [axes]
    
    for ax, topic in zip(axes, topics):
        analysis = analyze_topic(topic)
        
        # Collect position variances for all voters
        voter_ids = sorted(analysis["per_voter"].keys(), key=int)
        n_voters = len(voter_ids)
        
        # Get all alternatives
        all_alts = set()
        for v in analysis["per_voter"].values():
            if "position_variances" in v:
                all_alts.update(v["position_variances"].keys())
        all_alts = sorted(all_alts, key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else (x if isinstance(x, int) else 0))
        
        # Build matrix
        matrix = np.zeros((len(all_alts), n_voters))
        for j, voter_id in enumerate(voter_ids):
            voter_data = analysis["per_voter"][voter_id]
            if "position_variances" in voter_data:
                for i, alt in enumerate(all_alts):
                    matrix[i, j] = voter_data["position_variances"].get(alt, 0)
        
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')
        ax.set_xlabel('Voter', fontsize=11)
        ax.set_ylabel('Alternative', fontsize=11)
        ax.set_title(f'{topic.capitalize()}\nMean Var: {analysis["overall"]["position_variance_mean"]:.1f}', fontsize=12)
        
        # Color bar
        plt.colorbar(im, ax=ax, label='Position Variance')
    
    fig.suptitle('Position Variance Across Voters and Alternatives', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "position_variance_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved position_variance_heatmap.png")


def plot_stability_by_voter(topics: List[str]) -> None:
    """Plot stability metrics by voter."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = {'abortion': '#2E86AB', 'environment': '#A23B72'}
    
    for topic in topics:
        analysis = analyze_topic(topic)
        voter_ids = sorted(analysis["per_voter"].keys(), key=int)
        
        tau_means = []
        var_means = []
        match_rates = []
        
        for vid in voter_ids:
            v = analysis["per_voter"][vid]
            if "error" not in v:
                tau_means.append(v["kendall_tau_mean"])
                var_means.append(v["position_variance_mean"])
                match_rates.append(v["exact_match_rate"] * 100)
            else:
                tau_means.append(np.nan)
                var_means.append(np.nan)
                match_rates.append(np.nan)
        
        x = range(len(voter_ids))
        
        # Kendall tau
        axes[0].bar([i + (0.4 if topic == "environment" else 0) for i in x], 
                   tau_means, width=0.4, label=topic.capitalize(),
                   color=colors.get(topic, 'gray'), alpha=0.7)
        
        # Position variance
        axes[1].bar([i + (0.4 if topic == "environment" else 0) for i in x],
                   var_means, width=0.4, label=topic.capitalize(),
                   color=colors.get(topic, 'gray'), alpha=0.7)
        
        # Exact match rate
        axes[2].bar([i + (0.4 if topic == "environment" else 0) for i in x],
                   match_rates, width=0.4, label=topic.capitalize(),
                   color=colors.get(topic, 'gray'), alpha=0.7)
    
    axes[0].set_xlabel('Voter Index')
    axes[0].set_ylabel('Mean Kendall Tau')
    axes[0].set_title('Kendall Tau by Voter')
    axes[0].legend()
    axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    
    axes[1].set_xlabel('Voter Index')
    axes[1].set_ylabel('Mean Position Variance')
    axes[1].set_title('Position Variance by Voter')
    axes[1].legend()
    
    axes[2].set_xlabel('Voter Index')
    axes[2].set_ylabel('Exact Match Rate (%)')
    axes[2].set_title('Exact Match Rate by Voter')
    axes[2].legend()
    axes[2].axhline(y=100, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stability_by_voter.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved stability_by_voter.png")


def compute_position_errors(topic: str) -> Dict:
    """
    Compute position errors for each alternative across runs.
    
    For each voter and each alternative, compute:
    - Mean position across all runs (reference)
    - Error = actual position - mean position for each run
    
    Returns dict with all errors and per-alternative stats.
    """
    data = load_results(topic)
    results_by_voter = data["results_by_voter"]
    
    all_errors = []
    per_alt_errors = {}
    
    for voter_idx, voter_data in results_by_voter.items():
        rankings_data = voter_data.get("rankings", [])
        rankings = [r["ranking"] for r in rankings_data if r.get("ranking")]
        
        if len(rankings) < 2:
            continue
        
        # For each alternative, compute positions across runs
        alt_positions = {}
        for ranking in rankings:
            for pos, alt in enumerate(ranking):
                if alt not in alt_positions:
                    alt_positions[alt] = []
                alt_positions[alt].append(pos)
        
        # Compute errors (deviation from mean position)
        for alt, positions in alt_positions.items():
            if len(positions) > 1:
                mean_pos = np.mean(positions)
                for pos in positions:
                    error = pos - mean_pos
                    all_errors.append({
                        "voter_idx": voter_idx,
                        "alt": alt,
                        "position": pos,
                        "mean_position": mean_pos,
                        "error": error,
                    })
                    
                    if alt not in per_alt_errors:
                        per_alt_errors[alt] = []
                    per_alt_errors[alt].append(error)
    
    return {
        "all_errors": all_errors,
        "per_alt_errors": per_alt_errors,
    }


def plot_position_error_histogram(topics: List[str]) -> None:
    """Plot histogram of position errors (deviation from mean)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'abortion': '#2E86AB', 'environment': '#A23B72'}
    
    for topic in topics:
        error_data = compute_position_errors(topic)
        errors = [e["error"] for e in error_data["all_errors"]]
        
        ax.hist(errors, bins=50, alpha=0.5, label=f'{topic.capitalize()} (n={len(errors)})',
                color=colors.get(topic, 'gray'), edgecolor='white')
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect (error=0)')
    ax.set_xlabel('Position Error (actual - mean position)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Ranking Stability: Position Error Distribution\n(How much each alternative deviates from its mean position across runs)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add stats
    for i, topic in enumerate(topics):
        error_data = compute_position_errors(topic)
        errors = [e["error"] for e in error_data["all_errors"]]
        text = f'{topic.capitalize()}:\nmean={np.mean(errors):.1f}\nstd={np.std(errors):.1f}'
        ax.text(0.02 + i*0.15, 0.98, text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "position_error_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved position_error_histogram.png")


def plot_mean_position_scatter(topics: List[str]) -> None:
    """Plot scatter of mean position vs position variance."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'abortion': '#2E86AB', 'environment': '#A23B72'}
    markers = {'abortion': 'o', 'environment': 's'}
    
    for topic in topics:
        error_data = compute_position_errors(topic)
        
        # Compute mean position and variance per alternative
        alt_stats = {}
        for e in error_data["all_errors"]:
            alt = e["alt"]
            if alt not in alt_stats:
                alt_stats[alt] = {"positions": [], "mean": e["mean_position"]}
            alt_stats[alt]["positions"].append(e["position"])
        
        mean_positions = []
        variances = []
        for alt, stats in alt_stats.items():
            mean_positions.append(stats["mean"])
            variances.append(np.var(stats["positions"]))
        
        ax.scatter(mean_positions, variances, alpha=0.4, 
                   label=f'{topic.capitalize()} (n={len(mean_positions)})',
                   color=colors.get(topic, 'gray'), 
                   marker=markers.get(topic, 'o'), s=30)
    
    ax.set_xlabel('Mean Position (across runs)', fontsize=12)
    ax.set_ylabel('Position Variance', fontsize=12)
    ax.set_title('Ranking Stability: Mean Position vs Variance\n(Higher variance = less stable ranking for that alternative)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mean_position_vs_variance.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved mean_position_vs_variance.png")


def plot_run_comparison_scatter(topics: List[str]) -> None:
    """Plot scatter comparing positions between first run and other runs."""
    fig, axes = plt.subplots(1, len(topics), figsize=(14, 6))
    if len(topics) == 1:
        axes = [axes]
    
    for ax, topic in zip(axes, topics):
        data = load_results(topic)
        results_by_voter = data["results_by_voter"]
        
        run1_positions = []
        other_positions = []
        
        for voter_idx, voter_data in results_by_voter.items():
            rankings_data = voter_data.get("rankings", [])
            rankings = [r["ranking"] for r in rankings_data if r.get("ranking")]
            
            if len(rankings) < 2:
                continue
            
            # Use first run as reference
            first_ranking = rankings[0]
            first_pos = {alt: pos for pos, alt in enumerate(first_ranking)}
            
            # Compare other runs
            for ranking in rankings[1:]:
                for pos, alt in enumerate(ranking):
                    if alt in first_pos:
                        run1_positions.append(first_pos[alt])
                        other_positions.append(pos)
        
        ax.scatter(run1_positions, other_positions, alpha=0.1, s=10, color='#2E86AB')
        ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect agreement')
        
        ax.set_xlabel('Position in Run 1', fontsize=11)
        ax.set_ylabel('Position in Other Runs', fontsize=11)
        ax.set_title(f'{topic.capitalize()}\n(n={len(run1_positions)} comparisons)', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_aspect('equal')
    
    fig.suptitle('Position Comparison: Run 1 vs Other Runs\n(Points off diagonal = position changed)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "run_comparison_scatter.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved run_comparison_scatter.png")


def plot_position_change_by_rank(topics: List[str]) -> None:
    """Plot mean absolute position change by original rank bin."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'abortion': '#2E86AB', 'environment': '#A23B72'}
    bins_range = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    bin_labels = ['1-20', '21-40', '41-60', '61-80', '81-100']
    
    width = 0.35
    x = np.arange(len(bins_range))
    
    for i, topic in enumerate(topics):
        data = load_results(topic)
        results_by_voter = data["results_by_voter"]
        
        # Collect position changes by original position bin
        bin_changes = {b: [] for b in bins_range}
        
        for voter_idx, voter_data in results_by_voter.items():
            rankings_data = voter_data.get("rankings", [])
            rankings = [r["ranking"] for r in rankings_data if r.get("ranking")]
            
            if len(rankings) < 2:
                continue
            
            # Use first run as reference
            first_ranking = rankings[0]
            first_pos = {alt: pos for pos, alt in enumerate(first_ranking)}
            
            for ranking in rankings[1:]:
                for pos, alt in enumerate(ranking):
                    if alt in first_pos:
                        orig_pos = first_pos[alt]
                        change = abs(pos - orig_pos)
                        for low, high in bins_range:
                            if low <= orig_pos < high:
                                bin_changes[(low, high)].append(change)
                                break
        
        bin_means = []
        bin_stds = []
        for b in bins_range:
            changes = bin_changes[b]
            if changes:
                bin_means.append(np.mean(changes))
                bin_stds.append(np.std(changes))
            else:
                bin_means.append(0)
                bin_stds.append(0)
        
        offset = width * (i - 0.5)
        ax.bar(x + offset, bin_means, width, yerr=bin_stds, capsize=3,
               label=topic.capitalize(), color=colors.get(topic, 'gray'), alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel('Original Position Bin (Run 1)', fontsize=12)
    ax.set_ylabel('Mean Absolute Position Change', fontsize=12)
    ax.set_title('Position Instability by Original Rank\n(How much positions change between runs)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "position_change_by_rank.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved position_change_by_rank.png")


def plot_run_comparison_scatter_binned(topics: List[str], bin_size: int = 5) -> None:
    """Plot binned scatter comparing positions between first run and other runs."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = {'abortion': '#2E86AB', 'environment': '#A23B72'}
    markers = {'abortion': 'o', 'environment': 's'}
    
    bins = np.arange(0, 101, bin_size)
    
    # Collect errors per topic for mean error calculation
    errors_by_topic = {}
    
    for topic in topics:
        data = load_results(topic)
        results_by_voter = data["results_by_voter"]
        
        run1_positions = []
        other_positions = []
        
        for voter_idx, voter_data in results_by_voter.items():
            rankings_data = voter_data.get("rankings", [])
            rankings = [r["ranking"] for r in rankings_data if r.get("ranking")]
            
            if len(rankings) < 2:
                continue
            
            # Use first run as reference
            first_ranking = rankings[0]
            first_pos = {alt: pos for pos, alt in enumerate(first_ranking)}
            
            # Compare other runs
            for ranking in rankings[1:]:
                for pos, alt in enumerate(ranking):
                    if alt in first_pos:
                        run1_positions.append(first_pos[alt])
                        other_positions.append(pos)
        
        run1_positions = np.array(run1_positions)
        other_positions = np.array(other_positions)
        
        # Collect errors per topic
        errors_by_topic[topic] = other_positions - run1_positions
        
        # Bin the data
        bin_centers = []
        bin_means = []
        
        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i+1]
            mask = (run1_positions >= low) & (run1_positions < high)
            if mask.sum() > 0:
                bin_centers.append((low + high) / 2)
                bin_means.append(np.mean(other_positions[mask]))
        
        ax.plot(bin_centers, bin_means, '-', markersize=10, linewidth=2.5,
                label=topic.capitalize(),
                color=colors.get(topic, 'gray'),
                marker=markers.get(topic, 'o'))
    
    ax.plot([0, 100], [0, 100], 'r--', linewidth=2.5, label='Perfect', alpha=0.7)
    
    # Add mean error text per topic
    error_text = '\n'.join([f'{t.capitalize()} Mean Error: {np.mean(e):.2f}' 
                           for t, e in errors_by_topic.items()])
    ax.text(0.05, 0.95, error_text, 
            transform=ax.transAxes, fontsize=14, verticalalignment='top')
    
    ax.set_xlabel(f'Original Position (binned by {bin_size} ranks)', fontsize=16)
    ax.set_ylabel('Mean Reconstructed Position', fontsize=16)
    ax.set_title('Iterative Ranking: Binned Reconstructed vs Original', fontsize=18)
    ax.legend(fontsize=14, loc='upper left', bbox_to_anchor=(1.02, 1))
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "iterative_scatter_binned.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved iterative_scatter_binned.png")


def plot_position_change_by_rank_signed(topics: List[str]) -> None:
    """Plot mean signed position change by original rank bin (positive = moved down/less preferred)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'abortion': '#2E86AB', 'environment': '#A23B72'}
    bins_range = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    bin_labels = ['1-20\n(top)', '21-40', '41-60', '61-80', '81-100\n(bottom)']
    
    width = 0.35
    x = np.arange(len(bins_range))
    
    for i, topic in enumerate(topics):
        data = load_results(topic)
        results_by_voter = data["results_by_voter"]
        
        # Collect signed position changes by original position bin
        bin_changes = {b: [] for b in bins_range}
        
        for voter_idx, voter_data in results_by_voter.items():
            rankings_data = voter_data.get("rankings", [])
            rankings = [r["ranking"] for r in rankings_data if r.get("ranking")]
            
            if len(rankings) < 2:
                continue
            
            # Use first run as reference
            first_ranking = rankings[0]
            first_pos = {alt: pos for pos, alt in enumerate(first_ranking)}
            
            for ranking in rankings[1:]:
                for pos, alt in enumerate(ranking):
                    if alt in first_pos:
                        orig_pos = first_pos[alt]
                        change = pos - orig_pos  # Positive = moved to higher index = less preferred
                        for low, high in bins_range:
                            if low <= orig_pos < high:
                                bin_changes[(low, high)].append(change)
                                break
        
        bin_means = []
        bin_stds = []
        for b in bins_range:
            changes = bin_changes[b]
            if changes:
                bin_means.append(np.mean(changes))
                bin_stds.append(np.std(changes) / np.sqrt(len(changes)))  # SEM
            else:
                bin_means.append(0)
                bin_stds.append(0)
        
        offset = width * (i - 0.5)
        ax.bar(x + offset, bin_means, width, yerr=bin_stds, capsize=3,
               label=topic.capitalize(), color=colors.get(topic, 'gray'), alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel('Original Position Bin (Run 1)', fontsize=12)
    ax.set_ylabel('Mean Position Change (± SEM)', fontsize=12)
    ax.set_title('Signed Position Change by Original Rank\n(Positive = moved down/less preferred, Negative = moved up/more preferred)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "position_change_by_rank_signed.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved position_change_by_rank_signed.png")


def main():
    """Run analysis and generate visualizations."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find available topics
    topics = []
    for topic_dir in OUTPUT_DIR.iterdir():
        if topic_dir.is_dir() and topic_dir.name != "figures":
            if (topic_dir / "raw_rankings.json").exists():
                topics.append(topic_dir.name)
    
    if not topics:
        logger.error("No test results found!")
        return
    
    logger.info(f"Found results for topics: {topics}")
    
    # Analyze each topic
    all_analyses = {}
    for topic in topics:
        logger.info(f"\nAnalyzing {topic}...")
        analysis = analyze_topic(topic)
        all_analyses[topic] = analysis
        
        # Save metrics
        with open(OUTPUT_DIR / topic / "stability_metrics.json", "w") as f:
            # Convert numpy values to Python types for JSON
            def convert(obj):
                if isinstance(obj, np.floating):
                    return float(obj) if not np.isnan(obj) else None
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(v) for v in obj]
                return obj
            
            json.dump(convert(analysis), f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("RANKING STABILITY TEST RESULTS")
    print("="*70)
    
    print("\n{:<15} {:>15} {:>15} {:>15}".format(
        "Topic", "Kendall Tau", "Pos. Variance", "Exact Match %"))
    print("-"*70)
    
    for topic in topics:
        o = all_analyses[topic]["overall"]
        tau = o["kendall_tau_mean"]
        var = o["position_variance_mean"]
        match = o["exact_match_rate"] * 100
        
        print("{:<15} {:>15.3f} {:>15.1f} {:>15.1f}%".format(
            topic.capitalize(),
            tau if not np.isnan(tau) else 0,
            var if not np.isnan(var) else 0,
            match))
    
    print("="*70)
    print("\nInterpretation:")
    print("  - Kendall Tau: 1.0 = perfect agreement, 0 = no correlation")
    print("  - Position Variance: 0 = perfect stability, higher = more variation")
    print("  - Exact Match %: 100% = all runs identical")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    plot_kendall_tau_distribution(topics)
    plot_position_variance_heatmap(topics)
    plot_stability_by_voter(topics)
    plot_position_error_histogram(topics)
    plot_mean_position_scatter(topics)
    plot_run_comparison_scatter(topics)
    plot_run_comparison_scatter_binned(topics)
    plot_position_change_by_rank(topics)
    plot_position_change_by_rank_signed(topics)
    
    logger.info(f"\nAll visualizations saved to {FIGURES_DIR}")
    print(f"\nFigures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
