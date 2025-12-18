#!/bin/bash
# Script to generate all tables once pipeline completes

echo "Generating tables for statements-only results..."
uv run python -m src.generate_table --results-dir data/results_statements_only --output results_table_statements_only.tex
uv run python -m src.generate_ratings_histogram --results-dir data/results_statements_only --output ratings_histogram_statements_only.png

echo ""
echo "All tables generated!"



