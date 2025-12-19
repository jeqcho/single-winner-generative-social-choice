#!/bin/bash
# Run insertion ranking model comparison experiment
# 
# Usage with tmux:
#   tmux new -s insertion-exp
#   ./scripts/run_insertion_model_comparison.sh
#   # Detach: Ctrl+B, D
#   # Reattach: tmux attach -t insertion-exp
#
# To monitor logs:
#   tail -f outputs/check-models-insertion/experiment_*.log

set -e

cd /home/ec2-user/single-winner-generative-social-choice

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================="
echo "Running Insertion Ranking Model Comparison"
echo "=========================================="
echo "Start time: $(date)"
echo "Models: gpt-5-nano-t1-{1,2}, gpt-5-mini-t1-{1,2}, gpt-5.2-t1-{1,2}, gpt-5.2-t0"
echo "Statements: 50, Personas: 10"
echo ""

# Use -u for unbuffered output
python3 -u -m src.large_scale.experiments.run_insertion_model_comparison

echo ""
echo "=========================================="
echo "Experiment completed at: $(date)"
echo "Outputs saved to: outputs/check-models-insertion/"
echo "=========================================="



