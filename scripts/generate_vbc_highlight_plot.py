#!/usr/bin/env python3
"""
Generate the special Likert barplot with red line at VBC lower CI for no_bridging ablation.
"""

from pathlib import Path
from src.full_experiment.visualizer import (
    collect_all_likert_clustered,
    plot_likert_barplot,
    OUTPUT_DIR
)

output_dir = OUTPUT_DIR
ablation = "no_bridging"

# Collect clustered results
all_likert_results_clustered = collect_all_likert_clustered(
    output_dir, 
    ablation=ablation, 
    topics=None  # All topics
)

# Generate the special plot
aggregate_dir = output_dir / "figures" / ablation / "aggregate"
plot_likert_barplot(
    all_likert_results_clustered,
    title=f"Average Likert Rating by Voting Method (no bridging)",
    output_path=aggregate_dir / "likert_barplot_vbc_highlight.png",
    highlight_vbc_lower_ci=True
)

print(f"Generated special plot at: {aggregate_dir / 'likert_barplot_vbc_highlight.png'}")

