#!/bin/bash
# Run model comparison experiment
# 
# Usage with screen (recommended for SSH sessions):
#   screen -S model_comparison
#   ./scripts/run_model_comparison.sh
#   # Detach: Ctrl+A, D
#   # Reattach: screen -r model_comparison
#
# Or with tmux:
#   tmux new -s model_comparison
#   ./scripts/run_model_comparison.sh
#   # Detach: Ctrl+B, D
#   # Reattach: tmux attach -t model_comparison

set -e

cd /home/ec2-user/single-winner-generative-social-choice

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "Running Model Comparison Experiment"
echo "=========================================="
echo "Start time: $(date)"
echo ""

python3 -m src.large_scale.experiments.model_comparison_experiment

echo ""
echo "=========================================="
echo "Experiment completed at: $(date)"
echo "Outputs saved to: outputs/check-models/"
echo "=========================================="

