# Implementation Notes for Large-Scale Experiment

## Completed Implementation

All modules for the large-scale persona experiment have been successfully implemented:

### Core Modules

1. **persona_loader.py** - Loads 1000 unique personas from HuggingFace SynthLabsAI/PERSONA dataset and splits them into three groups
2. **pairwise_ranking.py** - Implements merge sort with pairwise comparisons for efficient O(n log n) ranking
3. **generate_statements.py** - Generates statements from generative personas on each topic
4. **discriminative_ranking.py** - Gets preference rankings from discriminative personas using pairwise comparisons
5. **evaluative_scoring.py** - Gets Likert scale ratings (1-5) from evaluative personas
6. **voting_methods.py** - Implements 8 voting methods including traditional methods and ChatGPT variants
7. **main.py** - Main orchestration pipeline

### Reporting Modules

1. **generate_pvc_table.py** - Generates LaTeX and CSV tables showing which methods selected PVC elements
2. **generate_technique_histograms.py** - Generates histogram plots for each technique's winners
3. **generate_pvc_size_table.py** - Generates LaTeX and CSV tables showing PVC size for each topic

## Key Design Decisions

### Voting Method Change

**Original Plan**: Use Schulze method  
**Actual Implementation**: Use RankedPairs method

**Reason**: VoteKit does not include a Schulze implementation, but includes RankedPairs which is also a Condorcet-compliant method with similar properties.

### 8 Voting Methods

1. **Plurality** - First-past-the-post
2. **Borda** - Borda count (positional)
3. **IRV** - Instant Runoff Voting
4. **RankedPairs** - Ranked Pairs (Condorcet)
5. **ChatGPT** - Baseline (only sees statements)
6. **ChatGPT+Rankings** - Sees discriminative preference rankings
7. **ChatGPT+Profiles** - Sees discriminative persona descriptions
8. **ChatGPT+Rankings+Profiles** - Sees both rankings and profiles

## Testing Status

### Persona Loading

✅ Successfully tested loading 1000 unique personas from HuggingFace
✅ Successfully split into 20/5/5 personas for test mode
✅ Saved to `data/personas/` directory

### Pipeline Test

A test run has been initiated in the background with:
- 20 generative personas
- 5 discriminative personas  
- 5 evaluative personas
- Topic index 1: "What are the best policies to prevent littering in public spaces?"

The test will verify:
- Statement generation (20 statements)
- Pairwise ranking (~87 comparisons per discriminative persona)
- Evaluative scoring (100 total ratings: 5 personas × 20 statements)
- PVC computation
- All 8 voting methods
- Result file generation

## Running Production Mode

**WARNING**: Production mode (900/50/50) will be very expensive and time-consuming!

### Cost Estimates (per topic)

- **Statement generation**: ~900 API calls
- **Pairwise comparisons**: ~8,966 comparisons/persona × 50 personas = ~448,300 API calls
- **Evaluative ratings**: 900 statements × 50 personas = 45,000 API calls
- **Total**: ~494,200 API calls per topic
- **Estimated cost**: $500-1000 per topic (depending on API pricing)
- **Total for 13 topics**: $6,500-13,000

### Runtime Estimates

- Test mode (20/5/5): ~10-20 minutes per topic
- Production mode (900/50/50): Several hours per topic
- Total for 13 topics: Could take 24-72 hours of runtime

### Commands for Production

```bash
# Step 1: Load and split personas (only need to do once)
python -m src.large_scale.persona_loader --n-generative 900 --n-discriminative 50 --n-evaluative 50

# Step 2: Run experiment on all topics
python -m src.large_scale.main --load-personas

# Step 3: Generate reports
python -m src.large_scale.generate_pvc_table --results-dir data/large_scale/results
python -m src.large_scale.generate_pvc_size_table --results-dir data/large_scale/results
python -m src.large_scale.generate_technique_histograms --results-dir data/large_scale/results --output-dir .
```

## File Structure

```
data/
├── personas/
│   ├── generative.json (20 or 900 personas)
│   ├── discriminative.json (5 or 50 personas)
│   └── evaluative.json (5 or 50 personas)
└── large_scale/
    ├── statements/
    │   └── {topic_slug}.json (generated statements)
    ├── preferences/
    │   └── {topic_slug}.json (preference rankings)
    ├── evaluations/
    │   └── {topic_slug}.json (Likert ratings)
    └── results/
        └── {topic_slug}.json (complete results)
```

## Known Issues and Limitations

1. **API Rate Limits**: May need to add rate limiting or retry logic for production runs
2. **Memory Usage**: Large preference matrices (900×50) may consume significant memory
3. **Resume Capability**: Pipeline checks for existing files and skips completed steps, allowing resumption after failures
4. **Token Limits**: ChatGPT variants may hit token limits with 900 statements; currently showing first 10 voters/personas to avoid this

## Next Steps for Production Run

1. Ensure sufficient API credits
2. Consider running topics in batches to manage costs
3. Monitor first few topics to estimate actual costs and runtime
4. Consider implementing rate limiting if API errors occur
5. Save intermediate results frequently (already implemented)

