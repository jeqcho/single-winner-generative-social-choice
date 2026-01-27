# Single Winner Generative Social Choice

This repository contains implementations for social choice experiments using generative AI and the Proportional Veto Core (PVC).

## Overview

The project explores how different voting methods select consensus statements from AI-generated personas, comparing their results against the Proportional Veto Core. The experiment uses the critical epsilon (ε\*) metric to evaluate how well each method finds consensus.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable package management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (creates virtual environment automatically)
uv sync

# Or install in development mode
uv pip install -e .
```

<details>
<summary>Alternative: Using pip</summary>

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -e .
```
</details>

## Setup

Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
src/
├── sample_alt_voters/      # Main experiment module
│   ├── run_experiment.py   # Main experiment runner
│   ├── run_triple_star.py  # GPT*** method runner
│   ├── fix_star_epsilons.py # Fix GPT** epsilon values
│   ├── visualizer.py       # Generate plots and figures
│   ├── config.py           # Configuration settings
│   ├── alternative_generators/  # Statement generation strategies
│   └── voter_samplers/     # Voter sampling strategies
├── sampling_experiment/    # Shared utilities
│   ├── epsilon_calculator.py
│   ├── voting_methods.py
│   └── config.py
├── degeneracy_mitigation/  # Ranking utilities
│   ├── iterative_ranking.py
│   └── config.py
└── compute_pvc.py          # PVC computation

data/
├── sample-alt-voters/      # Generated statements and context
├── personas/               # 1000 personas from SynthLabsAI/PERSONA
├── topics.txt              # List of discussion topics
└── topic_mappings.json     # Topic name mappings

outputs/sample_alt_voters/
├── data/                   # Experiment results (JSON)
└── figures/                # Visualization plots (PNG)
```

## Running the Experiment

### Main Experiment

```bash
# Run the full experiment
uv run python -m src.sample_alt_voters.run_experiment
```

### GPT Methods

```bash
# Run GPT*** method (generates blind consensus statements)
uv run python -m src.sample_alt_voters.run_triple_star

# Fix GPT** epsilon values (computes epsilon for generated statements)
uv run python -m src.sample_alt_voters.fix_star_epsilons --double-star-only
```

### Visualization

```bash
# Generate all visualization plots
uv run python -m src.sample_alt_voters.visualizer --all
```

## Voting Methods

The experiment evaluates multiple voting methods:

### Traditional Methods
- **Plurality**: First-past-the-post voting
- **Borda**: Borda count (positional voting)
- **IRV**: Instant Runoff Voting
- **Schulze**: Schulze method (Condorcet-compliant)
- **VBC**: Veto by Consumption

### GPT-Based Methods
- **GPT**: Baseline ChatGPT selection
- **GPT+Rank**: ChatGPT with preference rankings
- **GPT+Pers**: ChatGPT with persona descriptions
- **GPT\***: GPT methods with A\*-low iterative ranking
- **GPT\*\***: GPT methods that generate new statements
- **GPT\*\*\***: Blind consensus generation (no voter preferences)

## Data Structure

Results are organized hierarchically:

```
outputs/sample_alt_voters/data/{topic}/{voter_dist}/{alt_dist}/rep{N}/
├── preferences.json         # Voter preference rankings
├── precomputed_epsilons.json # Epsilon values for all statements
├── voters.json              # Sampled voter personas
├── summary.json             # Experiment summary
└── mini_rep{0-4}/
    └── results.json         # Voting method results
```

## Topics

The experiment covers multiple discussion topics:
- Trust in US elections
- Abortion laws
- Electoral College reform
- And more (see `data/topics.txt`)

## Key Metrics

- **Critical Epsilon (ε\*)**: Measures consensus quality - lower is better
- **PVC Membership**: Whether the selected winner is in the Proportional Veto Core

## Citation

If you use this code, please cite the relevant papers on Proportional Veto Core and generative social choice.

## License

[Add license information here]
