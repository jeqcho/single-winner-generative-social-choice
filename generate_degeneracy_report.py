#!/usr/bin/env python3
"""
Generate a markdown report on preference profile degeneracy.
"""

import json
from pathlib import Path
from collections import Counter
from datetime import datetime

DATA_DIR = Path("/home/ec2-user/single-winner-generative-social-choice/outputs/full_sampling_experiment/data")
REPORTS_DIR = Path("/home/ec2-user/single-winner-generative-social-choice/reports")

def analyze_profile(prefs_path: Path) -> dict:
    """Analyze a single preference profile for degeneracy."""
    with open(prefs_path) as f:
        prefs = json.load(f)
    
    n_ranks = len(prefs)
    n_voters = len(prefs[0])
    
    # Define sequential and reverse rankings
    sequential = tuple(str(i) for i in range(n_ranks))
    reverse_seq = tuple(str(n_ranks - 1 - i) for i in range(n_ranks))
    
    # Get all rankings
    rankings = []
    for voter in range(n_voters):
        ranking = tuple(prefs[rank][voter] for rank in range(n_ranks))
        rankings.append(ranking)
    
    ranking_counts = Counter(rankings)
    
    unique = len(ranking_counts)
    seq_count = ranking_counts.get(sequential, 0)
    rev_count = ranking_counts.get(reverse_seq, 0)
    
    # Calculate percentages
    seq_pct = (seq_count / n_voters) * 100
    rev_pct = (rev_count / n_voters) * 100
    degenerate_pct = seq_pct + rev_pct  # Combined degenerate %
    
    return {
        "n_voters": n_voters,
        "n_unique": unique,
        "seq_count": seq_count,
        "rev_count": rev_count,
        "seq_pct": seq_pct,
        "rev_pct": rev_pct,
        "degenerate_pct": degenerate_pct,
        "unique_pct": (unique / n_voters) * 100
    }


def generate_report():
    """Generate the full markdown report."""
    results = {}
    
    # Collect data
    for topic_dir in sorted(DATA_DIR.iterdir()):
        if not topic_dir.is_dir():
            continue
        
        topic_name = topic_dir.name
        results[topic_name] = {}
        
        for rep_dir in sorted(topic_dir.iterdir()):
            if not rep_dir.is_dir():
                continue
            
            rep_name = rep_dir.name
            prefs_path = rep_dir / "full_preferences.json"
            
            if prefs_path.exists():
                results[topic_name][rep_name] = analyze_profile(prefs_path)
    
    # Generate markdown
    md_lines = []
    md_lines.append("# Preference Profile Degeneracy Report")
    md_lines.append("")
    md_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("")
    md_lines.append("## Summary")
    md_lines.append("")
    md_lines.append("A **degenerate** preference profile is one where voters have trivial rankings:")
    md_lines.append("- **Sequential**: `[0, 1, 2, 3, ..., 99]` (alternatives ranked in index order)")
    md_lines.append("- **Reverse**: `[99, 98, 97, ..., 0]` (alternatives ranked in reverse index order)")
    md_lines.append("")
    md_lines.append("These rankings indicate the LLM failed to produce meaningful preference orderings.")
    md_lines.append("")
    
    # Overall statistics
    all_seq_pcts = []
    all_degenerate_pcts = []
    all_unique_counts = []
    
    for topic_name, reps in results.items():
        for rep_name, data in reps.items():
            all_seq_pcts.append(data["seq_pct"])
            all_degenerate_pcts.append(data["degenerate_pct"])
            all_unique_counts.append(data["n_unique"])
    
    md_lines.append("### Overall Statistics")
    md_lines.append("")
    md_lines.append(f"- **Total profiles analyzed:** {len(all_seq_pcts)}")
    md_lines.append(f"- **Average sequential ranking %:** {sum(all_seq_pcts)/len(all_seq_pcts):.1f}%")
    md_lines.append(f"- **Average degenerate (seq + rev) %:** {sum(all_degenerate_pcts)/len(all_degenerate_pcts):.1f}%")
    md_lines.append(f"- **Average unique rankings:** {sum(all_unique_counts)/len(all_unique_counts):.1f} / 100")
    md_lines.append(f"- **Min unique rankings:** {min(all_unique_counts)}")
    md_lines.append(f"- **Max unique rankings:** {max(all_unique_counts)}")
    md_lines.append("")
    
    # Detailed tables per topic
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Detailed Results by Topic")
    md_lines.append("")
    
    for topic_name in sorted(results.keys()):
        reps = results[topic_name]
        
        # Create readable topic name
        readable_topic = topic_name.replace("-", " ").title()
        if len(readable_topic) > 60:
            readable_topic = readable_topic[:57] + "..."
        
        md_lines.append(f"### {readable_topic}")
        md_lines.append("")
        
        # Calculate topic averages
        topic_seq_pcts = [d["seq_pct"] for d in reps.values()]
        topic_degen_pcts = [d["degenerate_pct"] for d in reps.values()]
        topic_unique = [d["n_unique"] for d in reps.values()]
        
        md_lines.append(f"**Topic Average:** {sum(topic_degen_pcts)/len(topic_degen_pcts):.1f}% degenerate, {sum(topic_unique)/len(topic_unique):.1f} unique rankings")
        md_lines.append("")
        
        # Table header
        md_lines.append("| Rep | Sequential % | Reverse % | **Total Degenerate %** | Unique Rankings |")
        md_lines.append("|-----|-------------|-----------|------------------------|-----------------|")
        
        for rep_name in sorted(reps.keys(), key=lambda x: int(x.replace("rep", ""))):
            data = reps[rep_name]
            
            # Add warning emoji for high degeneracy
            degen_indicator = "🔴" if data["degenerate_pct"] > 80 else ("🟡" if data["degenerate_pct"] > 60 else "🟢")
            
            md_lines.append(
                f"| {rep_name} | {data['seq_pct']:.1f}% | {data['rev_pct']:.1f}% | "
                f"**{data['degenerate_pct']:.1f}%** {degen_indicator} | {data['n_unique']} / 100 |"
            )
        
        md_lines.append("")
    
    # Heatmap-style summary table
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Summary Heatmap (Degenerate %)")
    md_lines.append("")
    md_lines.append("Quick reference showing degenerate percentage for all topic-rep combinations.")
    md_lines.append("")
    
    # Header row
    header = "| Topic | " + " | ".join([f"rep{i}" for i in range(10)]) + " | Avg |"
    md_lines.append(header)
    separator = "|" + "---|" * 12
    md_lines.append(separator)
    
    for topic_name in sorted(results.keys()):
        reps = results[topic_name]
        
        # Short topic name
        short_topic = topic_name[:40] + "..." if len(topic_name) > 40 else topic_name
        
        row_values = []
        for i in range(10):
            rep_name = f"rep{i}"
            if rep_name in reps:
                pct = reps[rep_name]["degenerate_pct"]
                row_values.append(f"{pct:.0f}%")
            else:
                row_values.append("-")
        
        avg_pct = sum(d["degenerate_pct"] for d in reps.values()) / len(reps)
        row_values.append(f"**{avg_pct:.0f}%**")
        
        row = f"| {short_topic} | " + " | ".join(row_values) + " |"
        md_lines.append(row)
    
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## Legend")
    md_lines.append("")
    md_lines.append("- 🔴 **Critical** (>80% degenerate): Most voters have trivial rankings")
    md_lines.append("- 🟡 **Warning** (60-80% degenerate): Majority of voters have trivial rankings")
    md_lines.append("- 🟢 **OK** (<60% degenerate): Less than majority have trivial rankings")
    md_lines.append("")
    md_lines.append("**Note:** Even 'OK' profiles may still have significant degeneracy issues for experimental validity.")
    
    # Write report
    report_path = REPORTS_DIR / "preference_degeneracy_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(md_lines))
    
    print(f"Report generated: {report_path}")
    return report_path


if __name__ == "__main__":
    generate_report()
