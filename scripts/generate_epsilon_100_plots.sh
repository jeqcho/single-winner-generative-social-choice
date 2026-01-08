#!/bin/bash
# Generate epsilon-100 plots (epsilon computed wrt 100 personas using winners from 20 personas)
# This generates bar plots and strip plots for all ablations: full, no_bridging, no_filtering

cd /home/ec2-user/single-winner-generative-social-choice

echo "Generating epsilon-100 plots..."

uv run python -c "
from src.full_experiment.epsilon_100_plotter import generate_epsilon_100_plots
from pathlib import Path

output_dir = Path('outputs/full_experiment')
data_dir = output_dir / 'data'

# Auto-discover topics from folder names
if data_dir.exists():
    topics = [d.name for d in sorted(data_dir.iterdir()) if d.is_dir()]
    print(f'Found {len(topics)} topics: {topics}')
else:
    topics = None
    print('Data directory not found, will auto-detect topics')

# Generate epsilon-100 plots for all ablations
generate_epsilon_100_plots(
    output_dir=output_dir,
    topics=topics,
    ablations=['full', 'no_bridging', 'no_filtering']
)
print('Done generating epsilon-100 plots!')
"

echo "Epsilon-100 plots generated!"



