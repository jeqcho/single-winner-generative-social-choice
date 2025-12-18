# Quick Start Guide - Large-Scale Experiment

## Prerequisites

1. Python 3.11+
2. OpenAI API key with sufficient credits
3. (Optional) HuggingFace account for dataset access

## Installation

```bash
# Clone/navigate to repository
cd /path/to/single-winner-generative-social-choice

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (creates .venv automatically)
uv sync

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or `.venv\Scripts\activate` on Windows
```

<details>
<summary>Alternative: Using pip</summary>

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Or install manually
pip install votekit python-dotenv openai datasets matplotlib numpy
```
</details>

## Setup

Create `.env` file in project root:

```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

## Test Run (20/5/5 personas)

### Step 1: Load Personas

```bash
uv run python -m src.large_scale.persona_loader \
  --n-generative 20 \
  --n-discriminative 5 \
  --n-evaluative 5
```

Expected output:
- Loads 1000 unique personas from HuggingFace
- Saves splits to `data/personas/`

### Step 2: Run Experiment

Run on first topic only (recommended for testing):

```bash
uv run python -m src.large_scale.main \
  --test-mode \
  --load-personas \
  --topic-index 0
```

Or run on all 13 topics:

```bash
uv run python -m src.large_scale.main \
  --test-mode \
  --load-personas
```

Expected runtime: ~10-20 minutes per topic  
Expected cost: ~$5-10 per topic

### Step 3: Generate Reports

```bash
# PVC winner comparison table (LaTeX + CSV)
uv run python -m src.large_scale.generate_pvc_table \
  --results-dir data/large_scale/results

# PVC size table (LaTeX + CSV)
uv run python -m src.large_scale.generate_pvc_size_table \
  --results-dir data/large_scale/results

# Histogram plots (one per method)
uv run python -m src.large_scale.generate_technique_histograms \
  --results-dir data/large_scale/results \
  --output-dir .
```

Output files:
- `pvc_winner_table.tex` / `.csv`
- `pvc_size_table.tex` / `.csv`
- `histogram_plurality.png`
- `histogram_borda.png`
- `histogram_irv.png`
- `histogram_rankedpairs.png`
- `histogram_chatgpt.png`
- `histogram_chatgpt_rankings.png`
- `histogram_chatgpt_profiles.png`
- `histogram_chatgpt_rankings_profiles.png`

## Production Run (900/50/50 personas)

⚠️ **WARNING**: This will be very expensive! Estimated $500-1000 per topic.

### Step 1: Load Personas

```bash
uv run python -m src.large_scale.persona_loader \
  --n-generative 900 \
  --n-discriminative 50 \
  --n-evaluative 50
```

### Step 2: Run Experiment

Test on one topic first:

```bash
uv run python -m src.large_scale.main \
  --load-personas \
  --topic-index 0
```

If satisfied, run all topics:

```bash
uv run python -m src.large_scale.main \
  --load-personas
```

Expected runtime: Several hours per topic  
Expected cost: ~$500-1000 per topic  
Total for 13 topics: ~$6,500-13,000

### Step 3: Generate Reports

Same commands as test mode, but with more data.

## Monitoring Progress

The pipeline saves intermediate results, allowing you to monitor progress and resume if interrupted:

```bash
# Check statement generation progress
ls -lh data/large_scale/statements/

# Check preference ranking progress
ls -lh data/large_scale/preferences/

# Check evaluation progress
ls -lh data/large_scale/evaluations/

# Check final results
ls -lh data/large_scale/results/
```

## Resume After Interruption

The pipeline automatically skips completed steps:

```bash
# Just re-run the same command
uv run python -m src.large_scale.main --load-personas
```

It will:
- Skip topics with existing result files
- Skip statement generation if statements file exists
- Skip preference ranking if preferences file exists
- Skip evaluations if evaluations file exists

## Troubleshooting

### API Rate Limits

If you hit rate limits, the script will fail. You may need to:
1. Add retry logic (not yet implemented)
2. Run topics in smaller batches
3. Add delays between API calls

### Out of Memory

If running on a machine with limited RAM:
1. Run topics one at a time using `--topic-index`
2. Close other applications
3. Consider using a machine with more RAM

### Token Limits

ChatGPT variants show only first 10 voters/personas to avoid hitting token limits. This is intentional and should not affect results significantly.

## Cost Optimization Tips

1. **Start Small**: Always test with `--test-mode` first
2. **One Topic at a Time**: Use `--topic-index` to run individual topics
3. **Monitor Costs**: Check your OpenAI usage dashboard frequently
4. **Use Caching**: The pipeline caches intermediate results automatically
5. **Batch Topics**: Run 2-3 topics at a time and check results before continuing

## Expected Output Structure

```
data/large_scale/results/{topic}.json
{
  "topic": "...",
  "n_generative_personas": 20 or 900,
  "n_discriminative_personas": 5 or 50,
  "n_evaluative_personas": 5 or 50,
  "n_statements": 20 or 900,
  "statements": [...],
  "preference_matrix": [...],
  "evaluations": [...],
  "pvc": ["0", "3", "7", ...],
  "pvc_size": 5,
  "pvc_percentage": 25.0,
  "method_results": {
    "plurality": {"winner": "0", "in_pvc": true},
    "borda": {"winner": "3", "in_pvc": true},
    "irv": {"winner": "7", "in_pvc": true},
    "rankedpairs": {"winner": "0", "in_pvc": true},
    "chatgpt": {"winner": "5", "in_pvc": false},
    "chatgpt_rankings": {"winner": "3", "in_pvc": true},
    "chatgpt_profiles": {"winner": "0", "in_pvc": true},
    "chatgpt_rankings_profiles": {"winner": "0", "in_pvc": true}
  }
}
```

## Getting Help

1. Check `README.md` for full documentation
2. Check `IMPLEMENTATION_NOTES.md` for technical details
3. Review error messages carefully - they often indicate missing dependencies or API issues
4. Ensure `.env` file is properly configured
5. Verify you have sufficient API credits

