#!/bin/bash
# Generate reports in timestamped output directory

set -e

# Parse arguments
RUN_TYPE="${1:-test}"  # test or prod
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="outputs/${RUN_TYPE}_run_${TIMESTAMP}"
RESULTS_DIR="data/large_scale/${RUN_TYPE}/results"

echo "📊 Generating reports for ${RUN_TYPE} run..."
echo "   Output directory: $OUTPUT_BASE"
echo ""

# Create output directories
mkdir -p "$OUTPUT_BASE/tables" "$OUTPUT_BASE/figures"

# Generate PVC winner comparison table
echo "1. Generating PVC winner comparison table..."
python -m src.large_scale.generate_pvc_table \
  --results-dir "$RESULTS_DIR" \
  --latex-output "$OUTPUT_BASE/tables/pvc_winner_table.tex" \
  --csv-output "$OUTPUT_BASE/tables/pvc_winner_table.csv"
echo ""

# Generate PVC size table
echo "2. Generating PVC size table..."
python -m src.large_scale.generate_pvc_size_table \
  --results-dir "$RESULTS_DIR" \
  --latex-output "$OUTPUT_BASE/tables/pvc_size_table.tex" \
  --csv-output "$OUTPUT_BASE/tables/pvc_size_table.csv"
echo ""

# Generate histogram plots
echo "3. Generating histogram plots..."
python -m src.large_scale.generate_technique_histograms \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$OUTPUT_BASE/figures"
echo ""

echo "✅ All reports generated successfully!"
echo ""
echo "📁 Output location: $OUTPUT_BASE/"
echo "   Tables:  $OUTPUT_BASE/tables/"
echo "   Figures: $OUTPUT_BASE/figures/"
echo ""
echo "View outputs:"
echo "   ls $OUTPUT_BASE/tables/"
echo "   ls $OUTPUT_BASE/figures/"

