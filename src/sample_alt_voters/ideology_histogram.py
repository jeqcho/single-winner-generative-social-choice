"""
Generate ideology ranking histograms for Phase 2 sample-alt-voters experiment data.

Shows how voter ideology groups rank statements from different author ideology groups,
with programmatic mean/median statistics output.

Usage:
    # Single condition
    uv run python -m src.sample_alt_voters.ideology_histogram \
        --topic abortion --alt-dist persona_no_context --rep 0

    # All conditions (uniform voters only)
    uv run python -m src.sample_alt_voters.ideology_histogram --all
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .config import (
    PHASE2_DATA_DIR,
    PHASE2_FIGURES_DIR,
    DATA_DIR,
    TOPIC_SHORT_NAMES,
    ALT_DISTRIBUTIONS,
    N_REPS_UNIFORM,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

IDEOLOGY_COLORS = {
    "progressive_liberal": "blue",
    "conservative_traditional": "red",
    "other": "gray",
}

IDEOLOGY_LABELS = {
    "progressive_liberal": "Progressive",
    "conservative_traditional": "Conservative",
    "other": "Other",
}

# Topics (short names)
TOPICS = ["abortion", "electoral"]


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_ideology_clusters() -> Dict[str, List[int]]:
    """Load ideology cluster assignments from JSON file."""
    clusters_path = DATA_DIR / "sample-alt-voters" / "ideology_clusters.json"
    with open(clusters_path) as f:
        return json.load(f)


def build_ideology_lookup(clusters: Dict[str, List[int]]) -> Dict[int, str]:
    """
    Create reverse lookup: persona_id -> ideology.
    
    Args:
        clusters: Dict mapping ideology names to lists of persona indices
        
    Returns:
        Dict mapping persona index to ideology name
    """
    lookup = {}
    for ideology, ids in clusters.items():
        for pid in ids:
            lookup[pid] = ideology
    return lookup


def load_voter_ids(rep_dir: Path) -> List[int]:
    """
    Load voter_indices from voters.json.
    
    Args:
        rep_dir: Path to replication directory
        
    Returns:
        List of voter persona indices
    """
    voters_path = rep_dir / "voters.json"
    with open(voters_path) as f:
        return json.load(f)["voter_indices"]


def get_statement_author_ids(topic: str, rep_id: int) -> List[int]:
    """
    Load context_persona_ids from sampled-context file.
    
    These are the persona IDs of the statement authors (100 per rep).
    
    Args:
        topic: Topic short name (e.g., "abortion", "electoral")
        rep_id: Replication ID
        
    Returns:
        List of author persona indices
    """
    context_path = DATA_DIR / "sample-alt-voters" / "sampled-context" / topic / f"rep{rep_id}.json"
    with open(context_path) as f:
        return [int(pid) for pid in json.load(f)["context_persona_ids"]]


def load_rankings_from_preferences(rep_dir: Path) -> Optional[List[List[int]]]:
    """
    Load preferences.json and convert to rankings format.
    
    Input format: preferences[rank][voter] = stmt_idx (as string)
    Output format: rankings[voter][rank] = stmt_idx (as int)
    
    Args:
        rep_dir: Path to replication directory
        
    Returns:
        List of rankings, where rankings[voter][rank] = statement index
        Returns None if file not found
    """
    pref_path = rep_dir / "preferences.json"
    if not pref_path.exists():
        logger.warning(f"Preferences file not found: {pref_path}")
        return None

    with open(pref_path) as f:
        preferences = json.load(f)  # [rank][voter] = "stmt_idx"

    n_voters = len(preferences[0])
    n_ranks = len(preferences)

    # Transpose and convert to int
    rankings = []
    for voter in range(n_voters):
        voter_ranking = [int(preferences[rank][voter]) for rank in range(n_ranks)]
        rankings.append(voter_ranking)

    return rankings


def get_rep_dirs(topic: str, alt_dist: str) -> List[Tuple[Path, int]]:
    """
    Get all rep directories for a given condition (uniform voters only).
    
    Args:
        topic: Topic short name
        alt_dist: Alternative distribution name
        
    Returns:
        List of (rep_dir, rep_id) tuples
    """
    base_dir = PHASE2_DATA_DIR / topic / "uniform" / alt_dist

    if not base_dir.exists():
        logger.warning(f"Directory not found: {base_dir}")
        return []

    rep_dirs = []
    for rep_id in range(N_REPS_UNIFORM):
        rep_dir = base_dir / f"rep{rep_id}"
        if rep_dir.exists():
            rep_dirs.append((rep_dir, rep_id))

    return rep_dirs


# =============================================================================
# Analysis Functions
# =============================================================================

def collect_rank_distributions(
    rankings: List[List[int]],
    voter_ids: List[int],
    author_ids: List[int],
    ideology_lookup: Dict[int, str]
) -> Dict[Tuple[str, str], List[int]]:
    """
    Collect ranks grouped by (voter_ideology, author_ideology).
    
    For each voter-statement pair, records the 1-indexed rank (1=most preferred, 100=least).
    
    Args:
        rankings: List where rankings[voter][rank] = statement index
        voter_ids: List of voter persona IDs (global indices)
        author_ids: List of author persona IDs (global indices, indexed by statement position)
        ideology_lookup: Dict mapping persona ID to ideology
        
    Returns:
        Dict mapping (voter_ideology, author_ideology) to list of ranks
    """
    distributions = defaultdict(list)

    for voter_pos, voter_ranking in enumerate(rankings):
        voter_id = voter_ids[voter_pos]
        voter_ideology = ideology_lookup.get(voter_id, "other")

        for rank_0indexed, stmt_idx in enumerate(voter_ranking):
            author_id = author_ids[stmt_idx]
            author_ideology = ideology_lookup.get(author_id, "other")
            rank_1indexed = rank_0indexed + 1

            distributions[(voter_ideology, author_ideology)].append(rank_1indexed)

    return distributions


def compute_statistics(
    rank_distributions: Dict[Tuple[str, str], List[int]]
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Compute mean, median, and count for each (voter, author) ideology pair.
    
    Args:
        rank_distributions: Dict mapping (voter_ideo, author_ideo) to rank lists
        
    Returns:
        Dict mapping same keys to statistics dicts with mean, median, count
    """
    stats = {}
    for key, ranks in rank_distributions.items():
        if ranks:
            stats[key] = {
                "mean": float(np.mean(ranks)),
                "median": float(np.median(ranks)),
                "count": len(ranks)
            }
        else:
            stats[key] = {"mean": 0.0, "median": 0.0, "count": 0}
    return stats


def print_statistics(stats: Dict[Tuple[str, str], Dict[str, float]], label: str) -> None:
    """Print statistics to console in a formatted way."""
    print(f"\n=== {label} ===")

    for author_ideology in ["progressive_liberal", "conservative_traditional", "other"]:
        author_label = IDEOLOGY_LABELS.get(author_ideology, author_ideology)
        print(f"\nRankings of {author_label}-Authored Statements:")

        for voter_ideology in ["progressive_liberal", "conservative_traditional", "other"]:
            key = (voter_ideology, author_ideology)
            if key in stats:
                s = stats[key]
                voter_label = IDEOLOGY_LABELS.get(voter_ideology, voter_ideology)
                print(f"  {voter_label} voters: mean={s['mean']:.1f}, median={s['median']:.1f}, n={s['count']}")


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_ideology_histograms(
    rank_distributions: Dict[Tuple[str, str], List[int]],
    output_path: Path,
    title: str
) -> None:
    """
    Generate 2-panel histogram plot.
    
    Left panel: Rankings of Progressive-authored statements
    Right panel: Rankings of Conservative-authored statements
    
    Args:
        rank_distributions: Dict mapping (voter_ideo, author_ideo) to rank lists
        output_path: Path to save the plot
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.arange(0, 101, 10)

    # Left: Progressive-authored statements
    for voter_ideology in ["progressive_liberal", "conservative_traditional", "other"]:
        key = (voter_ideology, "progressive_liberal")
        if key in rank_distributions and rank_distributions[key]:
            ranks = rank_distributions[key]
            voter_label = IDEOLOGY_LABELS.get(voter_ideology, voter_ideology)
            color = IDEOLOGY_COLORS.get(voter_ideology, "gray")
            ax1.hist(
                ranks, bins=bins, alpha=0.5,
                color=color,
                label=f"{voter_label} voters (n={len(ranks)})",
                edgecolor='black', linewidth=0.5
            )

    ax1.set_xlabel("Rank (1=most preferred, 100=least preferred)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Rankings of Progressive-Authored Statements")
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, 100)

    # Right: Conservative-authored statements
    for voter_ideology in ["progressive_liberal", "conservative_traditional", "other"]:
        key = (voter_ideology, "conservative_traditional")
        if key in rank_distributions and rank_distributions[key]:
            ranks = rank_distributions[key]
            voter_label = IDEOLOGY_LABELS.get(voter_ideology, voter_ideology)
            color = IDEOLOGY_COLORS.get(voter_ideology, "gray")
            ax2.hist(
                ranks, bins=bins, alpha=0.5,
                color=color,
                label=f"{voter_label} voters (n={len(ranks)})",
                edgecolor='black', linewidth=0.5
            )

    ax2.set_xlabel("Rank (1=most preferred, 100=least preferred)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Rankings of Conservative-Authored Statements")
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, 100)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved histogram to {output_path}")


# =============================================================================
# Main Processing Functions
# =============================================================================

def process_single_condition(
    topic: str,
    alt_dist: str,
    rep_dir: Path,
    rep_id: int,
    ideology_lookup: Dict[int, str],
    output_dir: Path
) -> Optional[Dict[Tuple[str, str], Dict[str, float]]]:
    """
    Process a single replication condition.
    
    Args:
        topic: Topic short name
        alt_dist: Alternative distribution name
        rep_dir: Path to replication directory
        rep_id: Replication ID
        ideology_lookup: Dict mapping persona ID to ideology
        output_dir: Base output directory for figures
        
    Returns:
        Statistics dict, or None if processing failed
    """
    label = f"{topic}/uniform/{alt_dist}/rep{rep_id}"
    logger.info(f"Processing {label}...")

    # Load rankings
    rankings = load_rankings_from_preferences(rep_dir)
    if rankings is None:
        return None

    # Load voter IDs
    voter_ids = load_voter_ids(rep_dir)

    # Load author IDs
    author_ids = get_statement_author_ids(topic, rep_id)

    # Collect distributions
    distributions = collect_rank_distributions(
        rankings, voter_ids, author_ids, ideology_lookup
    )

    # Compute statistics
    stats = compute_statistics(distributions)

    # Print to console
    print_statistics(stats, label)

    # Generate histogram for this rep
    plot_path = output_dir / topic / f"{alt_dist}_uniform" / f"rep{rep_id}_ideology_rankings.png"
    plot_title = f"Ideology Rankings: {topic}/{alt_dist}/uniform/rep{rep_id}"
    plot_ideology_histograms(distributions, plot_path, plot_title)

    return stats


def aggregate_distributions(
    all_distributions: List[Dict[Tuple[str, str], List[int]]]
) -> Dict[Tuple[str, str], List[int]]:
    """
    Aggregate distributions from multiple replications.
    
    Args:
        all_distributions: List of distribution dicts from each rep
        
    Returns:
        Combined distribution dict
    """
    combined = defaultdict(list)
    for dist in all_distributions:
        for key, ranks in dist.items():
            combined[key].extend(ranks)
    return combined


def run_all_conditions(
    topics: Optional[List[str]] = None,
    alt_dists: Optional[List[str]] = None,
    output_dir: Optional[Path] = None
) -> None:
    """
    Run histogram generation for all specified conditions (uniform voters only).
    
    Args:
        topics: List of topic short names (defaults to all)
        alt_dists: List of alt distribution names (defaults to all)
        output_dir: Output directory (defaults to PHASE2_FIGURES_DIR)
    """
    if output_dir is None:
        output_dir = PHASE2_FIGURES_DIR
    if topics is None:
        topics = TOPICS
    if alt_dists is None:
        alt_dists = ALT_DISTRIBUTIONS

    # Load ideology data
    logger.info("Loading ideology clusters...")
    clusters = load_ideology_clusters()
    ideology_lookup = build_ideology_lookup(clusters)
    
    # Print cluster stats
    for name, ids in clusters.items():
        logger.info(f"  {name}: {len(ids)} personas")

    # Process all conditions
    all_stats = {}
    total_processed = 0

    for topic in topics:
        for alt_dist in alt_dists:
            rep_dirs = get_rep_dirs(topic, alt_dist)
            
            if not rep_dirs:
                logger.warning(f"No reps found for {topic}/{alt_dist}")
                continue

            # Collect distributions across all reps for aggregated plot
            all_rep_distributions = []

            for rep_dir, rep_id in rep_dirs:
                # Process single rep
                stats = process_single_condition(
                    topic, alt_dist,
                    rep_dir, rep_id,
                    ideology_lookup, output_dir
                )
                if stats is not None:
                    key = f"{topic}/uniform/{alt_dist}/rep{rep_id}"
                    all_stats[key] = stats
                    total_processed += 1
                    
                    # Also collect raw distributions for aggregation
                    rankings = load_rankings_from_preferences(rep_dir)
                    voter_ids = load_voter_ids(rep_dir)
                    author_ids = get_statement_author_ids(topic, rep_id)
                    dist = collect_rank_distributions(rankings, voter_ids, author_ids, ideology_lookup)
                    all_rep_distributions.append(dist)

            # Generate aggregated plot for this topic/alt_dist combination
            if all_rep_distributions:
                aggregated = aggregate_distributions(all_rep_distributions)
                agg_stats = compute_statistics(aggregated)
                
                agg_label = f"{topic}/uniform/{alt_dist} (aggregated, {len(all_rep_distributions)} reps)"
                print_statistics(agg_stats, agg_label)
                
                agg_plot_path = output_dir / topic / f"{alt_dist}_uniform" / "ideology_rankings.png"
                agg_title = f"Ideology Rankings: {topic}/{alt_dist}/uniform (All Reps)"
                plot_ideology_histograms(aggregated, agg_plot_path, agg_title)

    # Save JSON summary
    json_path = output_dir / "ideology_histogram_statistics.json"
    json_stats = {
        k: {f"{v[0]},{v[1]}": s for v, s in stats.items()}
        for k, stats in all_stats.items()
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(json_stats, f, indent=2)
    logger.info(f"Saved JSON statistics to {json_path}")

    print(f"\n=== Done ===")
    print(f"Processed {total_processed} conditions")
    print(f"Histogram plots saved to {output_dir}")


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description="Generate ideology ranking histograms for Phase 2 data (uniform voters only)"
    )
    parser.add_argument("--topic", choices=TOPICS, help="Topic to process")
    parser.add_argument("--alt-dist", choices=ALT_DISTRIBUTIONS, help="Alternative distribution")
    parser.add_argument("--rep", type=int, help="Replication ID")
    parser.add_argument("--all", action="store_true", help="Process all conditions")
    parser.add_argument("--output-dir", type=str, help="Output directory for plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else PHASE2_FIGURES_DIR

    if args.all:
        topics = [args.topic] if args.topic else None
        alt_dists = [args.alt_dist] if args.alt_dist else None
        run_all_conditions(
            topics=topics,
            alt_dists=alt_dists,
            output_dir=output_dir
        )
    elif args.topic and args.alt_dist and args.rep is not None:
        # Single condition
        clusters = load_ideology_clusters()
        ideology_lookup = build_ideology_lookup(clusters)

        rep_dir = PHASE2_DATA_DIR / args.topic / "uniform" / args.alt_dist / f"rep{args.rep}"

        if not rep_dir.exists():
            print(f"Error: Directory not found: {rep_dir}")
            return

        process_single_condition(
            args.topic, args.alt_dist,
            rep_dir, args.rep,
            ideology_lookup, output_dir
        )
    else:
        print("Please specify either --all or all of --topic, --alt-dist, --rep")
        parser.print_help()


if __name__ == "__main__":
    main()
