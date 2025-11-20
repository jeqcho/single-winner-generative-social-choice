# Experiment Outputs

This directory contains all generated reports and visualizations from the large-scale social choice experiments.

## Directory Structure

```
outputs/
├── tables/          # LaTeX and CSV tables
│   ├── pvc_winner_table.tex
│   ├── pvc_winner_table.csv
│   ├── pvc_size_table.tex
│   └── pvc_size_table.csv
└── figures/         # Histogram plots
    ├── histogram_plurality.png
    ├── histogram_borda.png
    ├── histogram_irv.png
    ├── histogram_rankedpairs.png
    ├── histogram_chatgpt.png
    ├── histogram_chatgpt_rankings.png
    ├── histogram_chatgpt_profiles.png
    └── histogram_chatgpt_rankings_profiles.png
```

## Tables

### PVC Winner Table
Shows which voting methods selected winners that are in the Proportional Veto Core (PVC) for each topic.

- **LaTeX**: `tables/pvc_winner_table.tex`
- **CSV**: `tables/pvc_winner_table.csv`

### PVC Size Table
Reports the size of the PVC for each topic (both absolute numbers and percentages).

- **LaTeX**: `tables/pvc_size_table.tex`
- **CSV**: `tables/pvc_size_table.csv`

## Figures

### Histograms
Each histogram shows the distribution of evaluative Likert scores (1-5) for the winner selected by each voting method, aggregated across all topics.

- `figures/histogram_plurality.png` - Plurality voting
- `figures/histogram_borda.png` - Borda count
- `figures/histogram_irv.png` - Instant Runoff Voting
- `figures/histogram_rankedpairs.png` - Ranked Pairs (Condorcet)
- `figures/histogram_chatgpt.png` - ChatGPT consensus selection
- `figures/histogram_chatgpt_rankings.png` - ChatGPT with preference rankings
- `figures/histogram_chatgpt_profiles.png` - ChatGPT with voter profiles
- `figures/histogram_chatgpt_rankings_profiles.png` - ChatGPT with both rankings and profiles

## Regenerating Outputs

To regenerate all outputs after running experiments:

```bash
# PVC winner comparison table
python -m src.large_scale.generate_pvc_table \
  --results-dir data/large_scale/results \
  --output-dir outputs/tables

# PVC size table
python -m src.large_scale.generate_pvc_size_table \
  --results-dir data/large_scale/results \
  --output-dir outputs/tables

# Histogram plots
python -m src.large_scale.generate_technique_histograms \
  --results-dir data/large_scale/results \
  --output-dir outputs/figures
```

## Notes

- All tables are available in both LaTeX (.tex) and CSV (.csv) formats for easy integration into papers or analysis tools.
- Histograms are saved as PNG images at 300 DPI for publication quality.
- The outputs are automatically generated from the results in `data/large_scale/results/`.

