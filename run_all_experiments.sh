#!/bin/bash
# Run all multi-persona experiments and regenerate all figures
# This script is designed to be run in a tmux session for SSH disconnection safety

set -e

cd /home/ec2-user/single-winner-generative-social-choice

echo "========================================"
echo "Starting multi-persona experiments"
echo "========================================"
echo "Start time: $(date)"
echo ""

# Run multi-persona experiments for all ablations
echo "Running multi-persona experiments for all ablations..."
uv run python -m src.full_experiment.run_multi_persona --persona-counts 5 10 --ablations full no_bridging no_filtering

echo ""
echo "========================================"
echo "Experiments complete. Regenerating figures..."
echo "========================================"
echo ""

# Regenerate all figures using visualizer
echo "Running visualizer..."
uv run python -c "
import logging
logging.basicConfig(level=logging.INFO)
from src.full_experiment.visualizer import generate_all_plots
from pathlib import Path
generate_all_plots(Path('outputs/full_experiment'), ablations=['full', 'no_bridging', 'no_filtering'])
"

echo ""
echo "Running epsilon-100 plotter..."
uv run python -m src.full_experiment.epsilon_100_plotter

echo ""
echo "Running multi-persona plots generator..."
uv run python -m src.full_experiment.generate_multi_persona_plots

echo ""
echo "========================================"
echo "All experiments and plotting complete!"
echo "End time: $(date)"
echo "========================================"



