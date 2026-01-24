"""
Results aggregator for Phase 2 experiment.

Collects and summarizes results across all experimental conditions:
- Topics: abortion, electoral
- Alt distributions: Alt1-4
- Voter distributions: uniform, clustered
- Reps and mini-reps

Provides functions to:
- Load all results into a unified DataFrame
- Compute summary statistics by condition
- Export results for visualization
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .config import (
    PHASE2_DATA_DIR,
    TOPICS,
    TOPIC_SHORT_NAMES,
    ALT_DISTRIBUTIONS,
    N_REPS_UNIFORM,
    N_REPS_CLUSTERED,
    N_SAMPLES_PER_REP,
    IDEOLOGY_CLUSTERS,
)

logger = logging.getLogger(__name__)


def load_mini_rep_results(mini_rep_dir: Path) -> Optional[Dict]:
    """Load results from a single mini-rep directory."""
    results_path = mini_rep_dir / "results.json"
    if not results_path.exists():
        return None
    
    with open(results_path) as f:
        return json.load(f)


def load_rep_results(rep_dir: Path) -> Dict:
    """Load all results from a replication directory."""
    results = {
        "preferences_exist": (rep_dir / "preferences.json").exists(),
        "epsilons_exist": (rep_dir / "precomputed_epsilons.json").exists(),
        "mini_reps": []
    }
    
    # Load summary if exists
    summary_path = rep_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            results["summary"] = json.load(f)
    
    # Load mini-rep results
    for i in range(N_SAMPLES_PER_REP):
        mini_rep_dir = rep_dir / f"mini_rep{i}"
        if mini_rep_dir.exists():
            mini_rep_data = load_mini_rep_results(mini_rep_dir)
            if mini_rep_data:
                results["mini_reps"].append(mini_rep_data)
    
    return results


def collect_all_results() -> pd.DataFrame:
    """
    Collect all results into a single DataFrame.
    
    Returns:
        DataFrame with columns:
        - topic: "abortion" or "electoral"
        - alt_dist: alternative distribution name
        - voter_dist: "uniform" or cluster name
        - rep_id: replication index
        - mini_rep_id: mini-replication index
        - method: voting method name
        - winner: winning alternative index
        - epsilon: epsilon value for the winner
    """
    rows = []
    
    for topic_slug in TOPICS:
        topic_short = TOPIC_SHORT_NAMES.get(topic_slug, topic_slug)
        
        # Uniform voter distribution
        uniform_dir = PHASE2_DATA_DIR / topic_short / "uniform"
        if uniform_dir.exists():
            for alt_dist in ALT_DISTRIBUTIONS:
                alt_dir = uniform_dir / alt_dist
                if not alt_dir.exists():
                    continue
                
                for rep_id in range(N_REPS_UNIFORM):
                    rep_dir = alt_dir / f"rep{rep_id}"
                    if not rep_dir.exists():
                        continue
                    
                    rep_results = load_rep_results(rep_dir)
                    
                    for mini_rep_data in rep_results.get("mini_reps", []):
                        mini_rep_id = mini_rep_data.get("mini_rep_id", 0)
                        
                        for method, result in mini_rep_data.get("results", {}).items():
                            rows.append({
                                "topic": topic_short,
                                "alt_dist": alt_dist,
                                "voter_dist": "uniform",
                                "rep_id": rep_id,
                                "mini_rep_id": mini_rep_id,
                                "method": method,
                                "winner": result.get("winner"),
                                "epsilon": result.get("epsilon"),
                                "full_winner_idx": result.get("full_winner_idx"),
                                "error": result.get("error"),
                            })
        
        # Clustered voter distribution
        clustered_dir = PHASE2_DATA_DIR / topic_short / "clustered"
        if clustered_dir.exists():
            for alt_dist in ALT_DISTRIBUTIONS:
                alt_dir = clustered_dir / alt_dist
                if not alt_dir.exists():
                    continue
                
                for rep_id in range(N_REPS_CLUSTERED):
                    cluster_name = IDEOLOGY_CLUSTERS[rep_id] if rep_id < len(IDEOLOGY_CLUSTERS) else f"cluster{rep_id}"
                    rep_dir = alt_dir / f"rep{rep_id}_{cluster_name}"
                    
                    if not rep_dir.exists():
                        continue
                    
                    rep_results = load_rep_results(rep_dir)
                    
                    for mini_rep_data in rep_results.get("mini_reps", []):
                        mini_rep_id = mini_rep_data.get("mini_rep_id", 0)
                        
                        for method, result in mini_rep_data.get("results", {}).items():
                            rows.append({
                                "topic": topic_short,
                                "alt_dist": alt_dist,
                                "voter_dist": cluster_name,
                                "rep_id": rep_id,
                                "mini_rep_id": mini_rep_id,
                                "method": method,
                                "winner": result.get("winner"),
                                "epsilon": result.get("epsilon"),
                                "full_winner_idx": result.get("full_winner_idx"),
                                "error": result.get("error"),
                            })
    
    return pd.DataFrame(rows)


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics grouped by method.
    
    Returns:
        DataFrame with mean, std, min, max epsilon per method
    """
    if df.empty:
        return pd.DataFrame()
    
    # Filter to valid epsilons
    valid_df = df[df["epsilon"].notna()]
    
    if valid_df.empty:
        return pd.DataFrame()
    
    summary = valid_df.groupby("method")["epsilon"].agg([
        ("mean", "mean"),
        ("std", "std"),
        ("min", "min"),
        ("max", "max"),
        ("count", "count")
    ]).round(4)
    
    return summary.sort_values("mean")


def compute_stats_by_condition(
    df: pd.DataFrame,
    group_cols: List[str] = None
) -> pd.DataFrame:
    """
    Compute summary statistics grouped by experimental conditions.
    
    Args:
        df: Results DataFrame
        group_cols: Columns to group by (default: ["alt_dist", "voter_dist", "method"])
        
    Returns:
        DataFrame with mean epsilon per condition
    """
    if df.empty:
        return pd.DataFrame()
    
    if group_cols is None:
        group_cols = ["alt_dist", "voter_dist", "method"]
    
    valid_df = df[df["epsilon"].notna()]
    
    if valid_df.empty:
        return pd.DataFrame()
    
    summary = valid_df.groupby(group_cols)["epsilon"].agg([
        ("mean_epsilon", "mean"),
        ("std_epsilon", "std"),
        ("n_samples", "count")
    ]).round(4)
    
    return summary.reset_index()


def get_method_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank voting methods by mean epsilon.
    
    Lower epsilon is better.
    
    Returns:
        DataFrame with methods ranked by mean epsilon
    """
    summary = compute_summary_stats(df)
    if summary.empty:
        return pd.DataFrame()
    
    summary["rank"] = range(1, len(summary) + 1)
    return summary.reset_index()


def compare_voter_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare performance across voter distributions.
    
    Returns:
        Pivot table: method × voter_dist → mean epsilon
    """
    if df.empty:
        return pd.DataFrame()
    
    valid_df = df[df["epsilon"].notna()]
    
    pivot = valid_df.pivot_table(
        values="epsilon",
        index="method",
        columns="voter_dist",
        aggfunc="mean"
    ).round(4)
    
    return pivot


def compare_alt_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare performance across alternative distributions.
    
    Returns:
        Pivot table: method × alt_dist → mean epsilon
    """
    if df.empty:
        return pd.DataFrame()
    
    valid_df = df[df["epsilon"].notna()]
    
    pivot = valid_df.pivot_table(
        values="epsilon",
        index="method",
        columns="alt_dist",
        aggfunc="mean"
    ).round(4)
    
    return pivot


def export_results_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Export results DataFrame to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Exported results to {output_path}")


def print_summary_report(df: pd.DataFrame) -> None:
    """Print a summary report to stdout."""
    print("\n" + "=" * 60)
    print("PHASE 2 EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    
    if df.empty:
        print("\nNo results found.")
        return
    
    # Overall stats
    valid_df = df[df["epsilon"].notna()]
    print(f"\nTotal results: {len(df)}")
    print(f"Valid epsilons: {len(valid_df)}")
    
    # By topic
    print("\n--- Results by Topic ---")
    for topic in df["topic"].unique():
        topic_df = valid_df[valid_df["topic"] == topic]
        print(f"  {topic}: {len(topic_df)} results, "
              f"mean ε = {topic_df['epsilon'].mean():.4f}")
    
    # By voter distribution
    print("\n--- Results by Voter Distribution ---")
    for voter_dist in df["voter_dist"].unique():
        dist_df = valid_df[valid_df["voter_dist"] == voter_dist]
        if len(dist_df) > 0:
            print(f"  {voter_dist}: {len(dist_df)} results, "
                  f"mean ε = {dist_df['epsilon'].mean():.4f}")
    
    # By alt distribution
    print("\n--- Results by Alternative Distribution ---")
    for alt_dist in df["alt_dist"].unique():
        alt_df = valid_df[valid_df["alt_dist"] == alt_dist]
        if len(alt_df) > 0:
            print(f"  {alt_dist}: {len(alt_df)} results, "
                  f"mean ε = {alt_df['epsilon'].mean():.4f}")
    
    # Method ranking
    print("\n--- Voting Method Ranking (by mean epsilon) ---")
    ranking = get_method_ranking(df)
    if not ranking.empty:
        for _, row in ranking.iterrows():
            print(f"  {row['rank']:2d}. {row['method']:30s} "
                  f"mean ε = {row['mean']:.4f} ± {row['std']:.4f}")
    
    print("\n" + "=" * 60)


def main():
    """CLI entry point for results aggregation."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description="Aggregate and summarize Phase 2 experiment results"
    )
    parser.add_argument(
        "--export-csv",
        type=str,
        help="Export results to CSV file"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary report"
    )
    args = parser.parse_args()
    
    # Collect results
    logger.info("Collecting results...")
    df = collect_all_results()
    logger.info(f"Collected {len(df)} result rows")
    
    if args.summary or not args.export_csv:
        print_summary_report(df)
    
    if args.export_csv:
        export_results_csv(df, Path(args.export_csv))


if __name__ == "__main__":
    main()
