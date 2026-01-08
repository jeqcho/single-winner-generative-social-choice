#!/bin/bash
# Generate winner-in-biggest-cluster table plots

cd /home/ec2-user/single-winner-generative-social-choice

echo "Generating winner-in-biggest-cluster tables..."
uv run python -m src.full_experiment.generate_winner_biggest_cluster_table




