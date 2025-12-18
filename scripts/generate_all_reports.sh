#!/bin/bash
# Generate all reports from experiment results

set -e

echo "📊 Generating all reports from experiment results..."
echo ""

# Create output directories
mkdir -p outputs/tables outputs/figures

# Generate PVC winner comparison table
echo "1. Generating PVC winner comparison table..."
python -m src.large_scale.generate_pvc_table \
  --results-dir data/large_scale/results \
  --latex-output outputs/tables/pvc_winner_table.tex \
  --csv-output outputs/tables/pvc_winner_table.csv
echo ""

# Generate PVC size table
echo "2. Generating PVC size table..."
python -m src.large_scale.generate_pvc_size_table \
  --results-dir data/large_scale/results \
  --latex-output outputs/tables/pvc_size_table.tex \
  --csv-output outputs/tables/pvc_size_table.csv
echo ""

# Generate histogram plots
echo "3. Generating histogram plots..."
python -m src.large_scale.generate_technique_histograms \
  --results-dir data/large_scale/results \
  --output-dir outputs/figures
echo ""

echo "✅ All reports generated successfully!"
echo ""
echo "📁 Output files:"
echo "   Tables:    outputs/tables/"
echo "   Figures:   outputs/figures/"
echo ""
echo "View outputs:"
echo "   ls outputs/tables/"
echo "   ls outputs/figures/"

