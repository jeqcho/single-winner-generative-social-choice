#!/bin/bash
# Generate all visualization plots for the full experiment

cd /home/ec2-user/single-winner-generative-social-choice

uv run python -c "
from src.full_experiment.visualizer import generate_all_plots
from pathlib import Path

output_dir = Path('outputs/full_experiment')
data_dir = output_dir / 'data'

# Auto-discover topics from folder names
topics = [d.name for d in sorted(data_dir.iterdir()) if d.is_dir()]
print(f'Found {len(topics)} topics: {topics}')

# Generate plots for all ablations
generate_all_plots(output_dir, topics=topics, ablations=['full', 'no_bridging', 'no_filtering'])
print('Done!')
"

