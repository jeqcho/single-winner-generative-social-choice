# Single Winner Generative Social Choice

This repository contains implementations for social choice experiments using generative AI and the Proportional Veto Core (PVC).

## Overview

The project explores how different voting methods select consensus statements from AI-generated personas, comparing their results against the Proportional Veto Core. It includes both a small-scale baseline experiment and a large-scale experiment using 1000 unique personas from HuggingFace's SynthLabsAI/PERSONA dataset.

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

# Or manually install required packages
pip install votekit python-dotenv openai datasets matplotlib numpy
```
</details>

## Setup

1. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

2. (Optional) If running the large-scale experiment, you may need to authenticate with HuggingFace:
   ```bash
   huggingface-cli login
   ```

## Project Structure

- `src/` - Original small-scale experiment code (10 personas per group)
- `src/large_scale/` - Large-scale experiment code (up to 900/50/50 personas)
- `data/results/` - Results from small-scale experiments
- `data/large_scale/` - Data and results from large-scale experiments
- `data/topics.txt` - List of 13 discussion topics

## Large-Scale Experiment

The large-scale experiment uses three groups of personas:
- **Generative personas**: Generate statements on each topic (20 for testing, 900 for production)
- **Discriminative personas**: Rank statements using pairwise comparisons (5 for testing, 50 for production)
- **Evaluative personas**: Rate statements on a Likert scale 1-5 (5 for testing, 50 for production)

### Running the Large-Scale Experiment

#### Step 1: Load and Split Personas

```bash
# Test mode (20/5/5 personas)
uv run python -m src.large_scale.persona_loader --n-generative 20 --n-discriminative 5 --n-evaluative 5

# Production mode (900/50/50 personas)
uv run python -m src.large_scale.persona_loader --n-generative 900 --n-discriminative 50 --n-evaluative 50
```

This loads personas from HuggingFace's SynthLabsAI/PERSONA dataset and saves them to `data/personas/`.

#### Step 2: Run Main Experiment Pipeline

```bash
# Test mode: Run on first topic only
uv run python -m src.large_scale.main --test-mode --load-personas --topic-index 0

# Test mode: Run on all topics
uv run python -m src.large_scale.main --test-mode --load-personas

# Production mode: Run on all topics (WARNING: This will be expensive!)
uv run python -m src.large_scale.main --load-personas
```

The pipeline performs:
1. **Statement Generation**: Each generative persona creates a statement for each topic
2. **Discriminative Ranking**: Each discriminative persona ranks statements using pairwise comparisons (merge sort, O(n log n) comparisons)
3. **Evaluative Scoring**: Each evaluative persona rates all statements on a Likert scale (1-5)
4. **PVC Computation**: Compute the Proportional Veto Core
5. **Voting Methods**: Apply 8 different voting methods to select winners

Results are saved to `data/large_scale/results/`.

### Voting Methods

The experiment evaluates 8 voting methods:

1. **Plurality**: First-past-the-post voting
2. **Borda**: Borda count (positional voting)
3. **IRV**: Instant Runoff Voting
4. **RankedPairs**: Ranked Pairs method (Condorcet-compliant)
5. **ChatGPT**: Baseline ChatGPT selection (only sees statements)
6. **ChatGPT+Rankings**: ChatGPT sees preference rankings from discriminative personas
7. **ChatGPT+Profiles**: ChatGPT sees discriminative persona descriptions
8. **ChatGPT+Rankings+Profiles**: ChatGPT sees both rankings and profiles

### Generating Reports

After running experiments, generate tables and visualizations:

```bash
# Generate PVC winner comparison table (LaTeX + CSV)
uv run python -m src.large_scale.generate_pvc_table --results-dir data/large_scale/results

# Generate PVC size table (LaTeX + CSV)
uv run python -m src.large_scale.generate_pvc_size_table --results-dir data/large_scale/results

# Generate histogram plots for each voting method
uv run python -m src.large_scale.generate_technique_histograms --results-dir data/large_scale/results --output-dir .
```

Output files:
- `pvc_winner_table.tex` / `pvc_winner_table.csv`: Shows which methods selected PVC elements
- `pvc_size_table.tex` / `pvc_size_table.csv`: Shows PVC size for each topic
- `histogram_*.png`: Distribution of evaluative ratings for each method's winners

## Original Small-Scale Experiment

The original experiment uses 30 personas (3 groups of 10):

```bash
# Run on all topics
uv run python -m src.main

# Run on specific topic
uv run python -m src.main --topic-index 0

# Skip summary generation (use statements directly)
uv run python -m src.main --skip-summaries
```

Generate reports:

```bash
# Generate LaTeX tables
uv run python -m src.generate_table
uv run python -m src.generate_bridging_table

# Generate histogram visualization
uv run python -m src.generate_ratings_histogram
```

## Key Differences: Test vs Production Mode

| Aspect | Test Mode | Production Mode |
|--------|-----------|-----------------|
| Generative Personas | 20 | 900 |
| Discriminative Personas | 5 | 50 |
| Evaluative Personas | 5 | 50 |
| Pairwise Comparisons (per persona) | ~87 | ~8,966 |
| Evaluative Ratings (per persona) | 20 | 900 |
| API Cost | Low (~$5-10/topic) | High (~$500-1000/topic) |
| Runtime | ~10-20 min/topic | Several hours/topic |

## Computational Details

### Pairwise Comparison Complexity

The discriminative ranking uses merge sort with pairwise comparisons:
- Complexity: O(n log n) comparisons per persona
- Test mode (20 statements): ~87 comparisons per persona
- Production mode (900 statements): ~8,966 comparisons per persona

### Data Storage

Results are stored hierarchically:
- `data/large_scale/statements/{topic}.json`: Generated statements
- `data/large_scale/preferences/{topic}.json`: Preference rankings
- `data/large_scale/evaluations/{topic}.json`: Likert ratings
- `data/large_scale/results/{topic}.json`: Complete results with PVC and method winners

## Understanding the Output

### PVC Winner Table

Shows whether each method's winner is in the PVC:
- ✓ (checkmark): Winner is in the PVC
- (empty): Winner is not in the PVC
- Proportion row: Overall frequency of PVC selection

### PVC Size Table

Reports the size of the PVC for each topic:
- Absolute number (e.g., 5)
- Percentage (e.g., 5/20 = 25%)

### Technique Histograms

For each method, shows distribution of evaluative Likert ratings (1-5) across topics:
- One row per topic
- Red dashed line indicates mean rating
- Box shows sample size (n)

## Topics

The experiment covers 13 topics (see `data/topics.txt`):
1. Trust in US elections
2. Littering prevention policies
3. Israel/Gaza campus demonstrations
4. Abortion laws
5. Gun safety vs Second Amendment
6. Universal healthcare
7. Environmental protection vs economic growth
8. Immigration policy
9. Free speech and hate speech
10. Tech companies and user data
11. AI in society
12. Electoral College reform
13. Policing and use-of-force

## Citation

If you use this code, please cite the relevant papers on Proportional Veto Core and generative social choice.

## License

[Add license information here]

