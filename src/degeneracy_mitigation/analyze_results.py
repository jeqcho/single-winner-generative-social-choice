"""
Analyze results from degeneracy mitigation tests.

Computes:
- Degeneracy rates per condition
- Unique rankings count
- Correlation between Approach A and B
- Comparison across reasoning effort levels
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

from .config import OUTPUT_DIR, N_STATEMENTS, REASONING_EFFORTS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_rankings(results_dir: Path) -> list[list[int]]:
    """Load rankings from a results directory."""
    rankings_path = results_dir / 'rankings.json'
    if not rankings_path.exists():
        return []
    
    with open(rankings_path, 'r') as f:
        return json.load(f)


def load_stats(results_dir: Path) -> dict:
    """Load stats from a results directory."""
    stats_path = results_dir / 'stats.json'
    if not stats_path.exists():
        return {}
    
    with open(stats_path, 'r') as f:
        return json.load(f)


def is_sequential_ranking(ranking: list[int], n: int) -> bool:
    """Check if ranking is sequential [0, 1, 2, ..., n-1]."""
    if len(ranking) != n:
        return False
    return ranking == list(range(n))


def is_reverse_ranking(ranking: list[int], n: int) -> bool:
    """Check if ranking is reverse [n-1, n-2, ..., 0]."""
    if len(ranking) != n:
        return False
    return ranking == list(range(n - 1, -1, -1))


def compute_degeneracy_stats(rankings: list[list[int]], n: int = N_STATEMENTS) -> dict:
    """
    Compute degeneracy statistics for a set of rankings.
    
    Note: For Approach A (iterative), we check if the FINAL ranking
    matches sequential/reverse of ORIGINAL indices. This is different
    from per-round degeneracy which is checked against presentation order.
    
    Args:
        rankings: List of rankings (each is list of statement IDs)
        n: Expected number of statements
    
    Returns:
        Dictionary with degeneracy statistics.
    """
    if not rankings:
        return {
            'total': 0,
            'sequential_count': 0,
            'reverse_count': 0,
            'degenerate_count': 0,
            'degenerate_rate': 0.0,
            'unique_rankings': 0,
        }
    
    sequential_count = sum(1 for r in rankings if is_sequential_ranking(r, n))
    reverse_count = sum(1 for r in rankings if is_reverse_ranking(r, n))
    degenerate_count = sequential_count + reverse_count
    
    # Count unique rankings
    ranking_tuples = [tuple(r) for r in rankings if len(r) == n]
    unique_rankings = len(set(ranking_tuples))
    
    return {
        'total': len(rankings),
        'sequential_count': sequential_count,
        'reverse_count': reverse_count,
        'degenerate_count': degenerate_count,
        'degenerate_rate': degenerate_count / len(rankings) if rankings else 0.0,
        'unique_rankings': unique_rankings,
        'valid_rankings': len(ranking_tuples),
    }


def compute_spearman_correlation(
    rankings_a: list[list[int]], 
    rankings_b: list[list[int]]
) -> dict:
    """
    Compute Spearman correlation between two sets of rankings.
    
    Rankings are compared voter-by-voter.
    
    Args:
        rankings_a: Rankings from Approach A
        rankings_b: Rankings from Approach B
    
    Returns:
        Dictionary with correlation statistics.
    """
    if not rankings_a or not rankings_b:
        return {'mean_correlation': None, 'correlations': []}
    
    n_voters = min(len(rankings_a), len(rankings_b))
    correlations = []
    
    for i in range(n_voters):
        r_a = rankings_a[i]
        r_b = rankings_b[i]
        
        # Skip if rankings are incomplete
        if not r_a or not r_b or len(r_a) != len(r_b):
            continue
        
        # Convert rankings to rank arrays
        # ranking[i] = statement ID at position i (most to least preferred)
        # We want: for each statement, what rank does it have?
        n = len(r_a)
        
        try:
            # Build rank arrays
            ranks_a = [0] * n
            ranks_b = [0] * n
            
            for rank, stmt_id in enumerate(r_a):
                if 0 <= stmt_id < n:
                    ranks_a[stmt_id] = rank
            
            for rank, stmt_id in enumerate(r_b):
                if 0 <= stmt_id < n:
                    ranks_b[stmt_id] = rank
            
            # Compute Spearman correlation
            corr, _ = stats.spearmanr(ranks_a, ranks_b)
            if not np.isnan(corr):
                correlations.append(corr)
        except Exception as e:
            logger.warning(f"Error computing correlation for voter {i}: {e}")
    
    if not correlations:
        return {'mean_correlation': None, 'correlations': []}
    
    return {
        'mean_correlation': float(np.mean(correlations)),
        'std_correlation': float(np.std(correlations)),
        'min_correlation': float(np.min(correlations)),
        'max_correlation': float(np.max(correlations)),
        'n_compared': len(correlations),
        'correlations': correlations,
    }


def analyze_condition(results_dir: Path) -> dict:
    """Analyze results for a single condition (approach + reasoning effort)."""
    rankings = load_rankings(results_dir)
    stats = load_stats(results_dir)
    
    degeneracy = compute_degeneracy_stats(rankings)
    
    return {
        'path': str(results_dir),
        'stats': stats,
        'degeneracy': degeneracy,
    }


def analyze_all(output_dir: Path = OUTPUT_DIR) -> dict:
    """
    Analyze all results in the output directory.
    
    Args:
        output_dir: Base output directory
    
    Returns:
        Complete analysis dictionary.
    """
    results = {
        'approach_a': {},
        'approach_b': {},
        'correlations': {},
    }
    
    # Analyze Approach A (iterative ranking)
    approach_a_dir = output_dir / 'approach_a'
    if approach_a_dir.exists():
        for effort in REASONING_EFFORTS:
            effort_dir = approach_a_dir / effort
            if effort_dir.exists():
                results['approach_a'][effort] = analyze_condition(effort_dir)
                logger.info(f"Analyzed Approach A ({effort})")
    
    # Analyze Approach B (scoring)
    approach_b_dir = output_dir / 'approach_b'
    if approach_b_dir.exists():
        for effort in REASONING_EFFORTS:
            effort_dir = approach_b_dir / effort
            if effort_dir.exists():
                results['approach_b'][effort] = analyze_condition(effort_dir)
                logger.info(f"Analyzed Approach B ({effort})")
    
    # Compute correlations between A and B at each reasoning level
    for effort in REASONING_EFFORTS:
        if effort in results['approach_a'] and effort in results['approach_b']:
            rankings_a = load_rankings(output_dir / 'approach_a' / effort)
            rankings_b = load_rankings(output_dir / 'approach_b' / effort)
            
            corr = compute_spearman_correlation(rankings_a, rankings_b)
            results['correlations'][effort] = corr
            logger.info(f"Computed correlation for {effort}: {corr.get('mean_correlation', 'N/A')}")
    
    return results


def print_summary(analysis: dict) -> None:
    """Print a summary of the analysis."""
    print("\n" + "=" * 70)
    print("DEGENERACY MITIGATION TEST RESULTS")
    print("=" * 70)
    
    # Approach A summary
    print("\n### Approach A: Iterative Top-K/Bottom-K Ranking")
    print("-" * 50)
    print(f"{'Effort':<10} {'Degen %':<10} {'Unique':<10} {'Valid':<10} {'Retries':<10}")
    print("-" * 50)
    
    for effort in REASONING_EFFORTS:
        if effort in analysis['approach_a']:
            data = analysis['approach_a'][effort]
            degen = data['degeneracy']
            stats = data.get('stats', {})
            
            degen_pct = f"{degen['degenerate_rate']*100:.1f}%"
            unique = str(degen['unique_rankings'])
            valid = str(degen.get('valid_rankings', degen['total']))
            retries = str(stats.get('total_retries', 'N/A'))
            
            print(f"{effort:<10} {degen_pct:<10} {unique:<10} {valid:<10} {retries:<10}")
    
    # Approach B summary
    print("\n### Approach B: Scoring-Based Ranking")
    print("-" * 50)
    print(f"{'Effort':<10} {'Degen %':<10} {'Unique':<10} {'Unresolved':<10} {'Dedup':<10}")
    print("-" * 50)
    
    for effort in REASONING_EFFORTS:
        if effort in analysis['approach_b']:
            data = analysis['approach_b'][effort]
            degen = data['degeneracy']
            stats = data.get('stats', {})
            
            degen_pct = f"{degen['degenerate_rate']*100:.1f}%"
            unique = str(degen['unique_rankings'])
            unresolved = str(stats.get('unresolved_duplicates_count', 'N/A'))
            dedup = str(stats.get('voters_needing_dedup', 'N/A'))
            
            print(f"{effort:<10} {degen_pct:<10} {unique:<10} {unresolved:<10} {dedup:<10}")
    
    # Correlation summary
    print("\n### Correlation: Approach A vs B")
    print("-" * 50)
    print(f"{'Effort':<10} {'Mean Corr':<15} {'Std':<10} {'N':<10}")
    print("-" * 50)
    
    for effort in REASONING_EFFORTS:
        if effort in analysis.get('correlations', {}):
            corr = analysis['correlations'][effort]
            mean = corr.get('mean_correlation')
            std = corr.get('std_correlation')
            n = corr.get('n_compared', 0)
            
            mean_str = f"{mean:.3f}" if mean is not None else "N/A"
            std_str = f"{std:.3f}" if std is not None else "N/A"
            
            print(f"{effort:<10} {mean_str:<15} {std_str:<10} {n:<10}")
    
    print("\n" + "=" * 70)
    
    # Success criteria check
    print("\n### Success Criteria Check")
    print("-" * 50)
    
    target_degen = 0.05  # <5%
    
    for approach in ['approach_a', 'approach_b']:
        for effort in REASONING_EFFORTS:
            if effort in analysis.get(approach, {}):
                degen_rate = analysis[approach][effort]['degeneracy']['degenerate_rate']
                status = "PASS" if degen_rate < target_degen else "FAIL"
                print(f"{approach} ({effort}): {degen_rate*100:.1f}% degeneracy - {status}")
    
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze degeneracy mitigation test results"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=OUTPUT_DIR,
        help=f'Output directory to analyze (default: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Save analysis to comparison.json'
    )
    
    args = parser.parse_args()
    
    logger.info(f"Analyzing results in {args.output_dir}")
    
    analysis = analyze_all(args.output_dir)
    
    # Print summary
    print_summary(analysis)
    
    # Save if requested
    if args.save:
        output_path = args.output_dir / 'comparison.json'
        
        # Remove non-serializable data (full correlation lists)
        save_analysis = analysis.copy()
        for effort in save_analysis.get('correlations', {}):
            if 'correlations' in save_analysis['correlations'][effort]:
                # Keep only summary stats, not full list
                del save_analysis['correlations'][effort]['correlations']
        
        with open(output_path, 'w') as f:
            json.dump(save_analysis, f, indent=2)
        
        logger.info(f"Analysis saved to {output_path}")


if __name__ == '__main__':
    main()
