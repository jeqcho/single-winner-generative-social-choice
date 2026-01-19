#!/usr/bin/env python3
"""
Script to check if preference profiles are degenerate (all voters have the same preferences).
"""

import json
import os
from pathlib import Path
from collections import Counter

DATA_DIR = Path("/home/ec2-user/single-winner-generative-social-choice/outputs/full_sampling_experiment/data")

def check_preferences_degeneracy(preferences):
    """
    Check if preferences are degenerate (all identical).
    Returns (is_degenerate, unique_count, total_count)
    """
    if not preferences:
        return True, 0, 0
    
    # Convert each preference list to a tuple for hashing
    pref_tuples = [tuple(p) for p in preferences]
    unique_prefs = set(pref_tuples)
    
    return len(unique_prefs) == 1, len(unique_prefs), len(preferences)


def check_likert_degeneracy(scores):
    """
    Check if Likert scores are degenerate (all identical).
    Returns (is_degenerate, unique_count, total_count)
    """
    if not scores:
        return True, 0, 0
    
    # Convert each score list to a tuple for hashing
    score_tuples = [tuple(s) for s in scores]
    unique_scores = set(score_tuples)
    
    return len(unique_scores) == 1, len(unique_scores), len(scores)


def analyze_variance(scores):
    """
    Analyze the variance in Likert scores across voters.
    Returns statistics about the diversity of opinions.
    """
    if not scores or len(scores) == 0:
        return {}
    if len(scores[0]) == 0:
        return {}
    
    n_voters = len(scores)
    n_statements = len(scores[0])
    
    # For each statement, calculate variance across voters
    statement_variances = []
    for stmt_idx in range(n_statements):
        stmt_scores = [scores[v][stmt_idx] for v in range(n_voters)]
        mean = sum(stmt_scores) / len(stmt_scores)
        variance = sum((s - mean) ** 2 for s in stmt_scores) / len(stmt_scores)
        statement_variances.append(variance)
    
    avg_variance = sum(statement_variances) / len(statement_variances) if statement_variances else 0
    max_variance = max(statement_variances) if statement_variances else 0
    min_variance = min(statement_variances) if statement_variances else 0
    
    # Count statements with zero variance (all voters agree)
    zero_variance_count = sum(1 for v in statement_variances if v == 0)
    
    return {
        "avg_variance": avg_variance,
        "max_variance": max_variance,
        "min_variance": min_variance,
        "zero_variance_statements": zero_variance_count,
        "total_statements": n_statements,
        "pct_unanimous": (zero_variance_count / n_statements * 100) if n_statements > 0 else 0
    }


def main():
    results = []
    degenerate_profiles = []
    
    print("=" * 80)
    print("PREFERENCE PROFILE DEGENERACY CHECK")
    print("=" * 80)
    
    # Iterate through all topics and repetitions
    for topic_dir in sorted(DATA_DIR.iterdir()):
        if not topic_dir.is_dir():
            continue
        
        topic_name = topic_dir.name
        
        for rep_dir in sorted(topic_dir.iterdir()):
            if not rep_dir.is_dir():
                continue
            
            rep_name = rep_dir.name
            full_prefs_path = rep_dir / "full_preferences.json"
            likert_path = rep_dir / "likert_scores.json"
            
            result = {
                "topic": topic_name,
                "rep": rep_name,
                "prefs_degenerate": None,
                "likert_degenerate": None,
            }
            
            # Check full_preferences.json
            if full_prefs_path.exists():
                with open(full_prefs_path) as f:
                    prefs = json.load(f)
                is_degen, unique, total = check_preferences_degeneracy(prefs)
                result["prefs_degenerate"] = is_degen
                result["prefs_unique"] = unique
                result["prefs_total"] = total
            
            # Check likert_scores.json
            if likert_path.exists():
                with open(likert_path) as f:
                    data = json.load(f)
                # Handle dictionary format with "scores" key
                scores = data.get("scores", data) if isinstance(data, dict) else data
                is_degen, unique, total = check_likert_degeneracy(scores)
                result["likert_degenerate"] = is_degen
                result["likert_unique"] = unique
                result["likert_total"] = total
                
                # Analyze variance
                variance_stats = analyze_variance(scores)
                result.update(variance_stats)
            
            results.append(result)
            
            # Track degenerate profiles
            if result["prefs_degenerate"] or result["likert_degenerate"]:
                degenerate_profiles.append(result)
    
    # Print summary
    print(f"\nTotal profiles checked: {len(results)}")
    print(f"Degenerate profiles found: {len(degenerate_profiles)}")
    
    if degenerate_profiles:
        print("\n" + "=" * 80)
        print("DEGENERATE PROFILES:")
        print("=" * 80)
        for d in degenerate_profiles:
            print(f"\n{d['topic']}/{d['rep']}:")
            if d['prefs_degenerate']:
                print(f"  - Preferences: DEGENERATE (only {d['prefs_unique']} unique out of {d['prefs_total']} voters)")
            if d['likert_degenerate']:
                print(f"  - Likert scores: DEGENERATE (only {d['likert_unique']} unique out of {d['likert_total']} voters)")
    else:
        print("\n✅ No degenerate profiles found!")
    
    # Print diversity statistics
    print("\n" + "=" * 80)
    print("DIVERSITY STATISTICS (Likert Score Variance)")
    print("=" * 80)
    
    # Group by topic
    from collections import defaultdict
    by_topic = defaultdict(list)
    for r in results:
        by_topic[r["topic"]].append(r)
    
    for topic in sorted(by_topic.keys()):
        reps = by_topic[topic]
        avg_vars = [r.get("avg_variance", 0) for r in reps if "avg_variance" in r]
        pct_unan = [r.get("pct_unanimous", 0) for r in reps if "pct_unanimous" in r]
        
        if avg_vars:
            print(f"\n{topic[:50]}...")
            print(f"  Avg variance across reps: {sum(avg_vars)/len(avg_vars):.3f}")
            print(f"  Avg % unanimous statements: {sum(pct_unan)/len(pct_unan):.1f}%")
            
            # Check unique preference counts
            unique_counts = [r.get("prefs_unique", 0) for r in reps]
            print(f"  Unique preference profiles per rep: {unique_counts}")


if __name__ == "__main__":
    main()
