#!/bin/bash
# Generate cluster size histogram plots

cd /home/ec2-user/single-winner-generative-social-choice

echo "Generating cluster size histograms..."
uv run python -m src.full_experiment.generate_cluster_size_histogram "$@"
echo "Done!"

