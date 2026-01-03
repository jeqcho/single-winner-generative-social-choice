#!/bin/bash
# Generate cluster size markdown report

cd /home/ec2-user/single-winner-generative-social-choice

echo "Generating cluster size report..."
uv run python -m src.full_experiment.generate_cluster_report "$@"
echo "Done!"

