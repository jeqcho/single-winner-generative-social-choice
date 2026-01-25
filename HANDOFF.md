# Handoff: Phase 2 Sample-Alt-Voters Experiment

> Last updated: 2026-01-25
> Session focus: Created voter preference reports and ideology ranking analysis

## Objective

Run Phase 2 of the sample-alt-voters experiment: compare traditional voting methods (Schulze, Borda, IRV, Plurality, VBC) using A*-low iterative ranking to generate preference profiles. Evaluate using critical epsilon (ε*) as the consensus metric.

## Current Status

**State**: Complete - Data analysis and reporting in progress
**Branch**: main
**Key results**: 
- 480 valid results per traditional voting method
- 84 visualization figures across all conditions
- Voter preference reports created for qualitative analysis
- Ideology ranking histograms show minimal cross-ideology sorting

## Progress Summary

### Completed (This Session - 2026-01-25)
- Created voter preference reports with detailed personas and ranked statements
  - `reports/no_persona_context/` - 3 voters (progressive, conservative, other)
  - `reports/persona_no_context/` - 3 voters (progressive, conservative, other)
- Explained preference score derivation (LLM ranking → score = 99 - rank)
- Generated ideology ranking histograms: `plots/persona_no_context_rep0_ideology_rankings.png`
- Key finding: Minimal ideological sorting in persona_no_context (mean ranks ~50 for all combos)

### Previous Sessions Completed
- Fixed 117 invalid voters (had `-1` placeholders from failed LLM ranking calls)
- Re-ran 28 failed mini-reps that had crashed on invalid data
- Generated comprehensive visualizations (84 figures total)
- Implemented full Phase 2 pipeline (`run_experiment.py`)
- Created ideology-based voter sampling (progressive/liberal, conservative/traditional)

## Technical Context

### Entry Points
- Main experiment: `src/sample_alt_voters/run_experiment.py`
- Visualizer: `src/sample_alt_voters/visualizer.py`
- Results aggregator: `src/sample_alt_voters/results_aggregator.py`
- Data location: `outputs/sample_alt_voters/data/`
- Figures location: `outputs/sample_alt_voters/figures/`
- Voter reports: `reports/no_persona_context/`, `reports/persona_no_context/`
- Ideology plots: `plots/persona_no_context_rep0_ideology_rankings.png`

### Key Commands
```bash
# Generate all visualizations
uv run python -m src.sample_alt_voters.visualizer --all

# Check results summary
uv run python -c "
from src.sample_alt_voters.results_aggregator import collect_all_results, get_method_ranking
df = collect_all_results()
print(get_method_ranking(df))
"

# Re-run invalid voters (if needed again)
uv run python -m src.sample_alt_voters.rerun_invalid_voters

# Re-evaluate mini-rep epsilons (after fixing preferences)
uv run python -m src.sample_alt_voters.reeval_minireps

# Re-run failed mini-reps (after fixing preferences)
uv run python -m src.sample_alt_voters.rerun_failed_minireps
```

### Data Structure
```
outputs/sample_alt_voters/data/{topic}/{voter_dist}/{alt_dist}/rep{N}/
├── preferences.json          # 100×100 matrix: preferences[stmt_idx][voter_idx] = score (0-99)
├── precomputed_epsilons.json # ε* for each alternative
├── voters.json               # Sampled voter indices (voter_indices maps position → original ID)
├── mini_rep{0-4}/
│   └── results.json          # Voting method winners + epsilons
```

### Preference Score Derivation
- LLM ranks all 100 statements via iterative top-K/bottom-K selection (A*-low method)
- Ranking converted to scores: `score = 99 - rank` (rank 0 → score 99, rank 99 → score 0)
- Matrix format: `preferences[stmt_idx][voter_position] = score`

### Ideology Clusters
- Defined in `data/sample-alt-voters/ideology_clusters.json`
- Maps voter IDs to: `progressive_liberal`, `conservative_traditional`, `other`
- Persona ideology (from `data/personas/prod/adult.json`) may differ from cluster assignment

## Final Results

### Method Ranking (Lower ε* = Better)

| Rank | Method | Mean ε | Count |
|------|--------|--------|-------|
| 1 | Plurality | 0.0006 | 480 |
| 2 | Borda | 0.0008 | 480 |
| 3 | IRV | 0.0008 | 480 |
| 4 | Schulze | 0.0009 | 480 |
| 5 | VBC | 0.0010 | 480 |

**Note**: This ranking is REVERSED from previous experiments (where VBC~Borda were best). This may be due to A*-low producing different preference structures than previous methods.

### Visualization Structure
```
figures/
├── 6 overall figures
├── abortion/
│   ├── 3 topic-level figures
│   └── 12 combo subfolders (alt_dist × voter_dist) × 3 figures
└── electoral/
    ├── 3 topic-level figures
    └── 12 combo subfolders × 3 figures
Total: 84 figures
```

## Decision Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Re-run invalid voters (not random fallback) | User explicitly requested no random data in dataset | Generate random rankings as fallback |
| Max 10 retries per invalid voter | Balance between fixing data and cost | Unlimited retries, fewer retries |
| Use A*-low (not A-low) | User specified A* variant | Original A approach |
| Parallel voter re-runs | Significant speedup for 117 voters | Sequential processing |

## What Worked

- **Targeted re-run**: Only re-ran the 117 invalid voters, not all 9,600
- **Parallel processing**: Used ThreadPoolExecutor for concurrent voter re-runs
- **Staged fix**: (1) Fix preferences → (2) Recompute epsilons → (3) Re-eval mini-reps → (4) Re-run failed mini-reps
- **Validation without modification**: `validate_preferences()` identifies issues without injecting random data

## What Didn't Work

> ⚠️ **Do not retry these approaches without new information**

- **Random fallback for invalid voters**: User explicitly rejected - corrupts data integrity
- **Silent epsilon filtering**: Original code filtered out invalid rankings, causing all epsilons to be null
- **ChatGPT voting methods**: Failed with `reasoning.effort='minimal'` not supported by gpt-5.2 - needs fix in `voting_methods.py`

## Known Issues

1. **ChatGPT methods not working**: All 3 ChatGPT voting methods fail with API error:
   - Error: `Unsupported value: 'minimal' is not supported with 'gpt-5.2'`
   - Fix needed in: `src/sampling_experiment/voting_methods.py`
   - Change `reasoning.effort` from `'minimal'` to `'low'`

2. **Reversed method ranking**: Results show Plurality best, VBC worst - opposite of previous experiments. May warrant investigation.

## Recommended Next Steps

1. **Compare no_persona_context vs persona_no_context**:
   - Create similar ideology ranking histograms for `no_persona_context/rep0`
   - Compare ideological sorting between conditions

2. **Investigate cluster-persona mismatches**:
   - Check how ideology_clusters were computed vs persona self-reported ideology
   - Voter 754: Libertarian persona in conservative_traditional cluster

3. **Generate reports for other conditions**:
   - `persona_context`, `no_persona_no_context` conditions
   - Electoral topic (currently only abortion analyzed)

4. **Commit current results**:
   - Many modified files in `outputs/sample_alt_voters/`
   - New reports in `reports/` and plot in `plots/`

5. **Fix ChatGPT voting methods** (if needed):
   - Update `src/sampling_experiment/voting_methods.py`
   - Change `reasoning={'effort': 'minimal'}` to `reasoning={'effort': 'low'}`

## Session Notes

### User Preferences
- User has OpenAI grant - must use OpenAI models only (gpt-5-mini)
- User prefers data integrity over completeness - invalid data should fail loudly, not be silently fixed

### Experiment Structure
- Voter distributions: uniform (10 reps), progressive_liberal (1 rep), conservative_traditional (1 rep)
- Each rep has 5 mini-reps (20×20 subsamples of 100×100 preference matrix)
- Total: 96 reps × 5 mini-reps × 8 methods = 3840 potential results (2400 valid for traditional methods)

### Key Findings from Ideology Analysis (2026-01-25)
- **Minimal ideological sorting**: Mean ranks ~50 for all voter-author ideology combinations
- **Cross-ideology appeal**: Conservative voter (754) had top 2 preferred statements authored by progressive voters
- **Cluster-persona mismatch**: Voter 754 in `conservative_traditional` cluster but persona shows "Libertarian/Independent"
- **Bridging statements**: All statements are compromise/bridging language, explaining uniform rankings across ideologies

## Files Modified This Session (2026-01-25)

```
reports/
├── no_persona_context/
│   ├── voter_654_preferences.md     # Progressive voter (Liberal/Democrat)
│   ├── voter_32_preferences.md      # Conservative voter (Conservative/Catholic)
│   └── voter_25_preferences.md      # Other voter (Independent)
├── persona_no_context/
│   ├── voter_654_preferences.md     # Progressive voter
│   ├── voter_754_preferences.md     # Conservative cluster (but Libertarian persona!)
│   └── voter_25_preferences.md      # Other voter

plots/
└── persona_no_context_rep0_ideology_rankings.png  # Ideology ranking histograms
```

## Files Modified Previous Sessions

```
src/sample_alt_voters/
├── preference_builder_iterative.py  # validate_preferences() (no random fallback)
├── rerun_invalid_voters.py          # Fix invalid voter rankings
├── reeval_minireps.py               # Re-evaluate epsilon lookups
├── rerun_failed_minireps.py         # Re-run crashed mini-reps
├── visualizer.py                    # Added CDF plots, per-topic combo folders

outputs/sample_alt_voters/
├── data/                            # Fixed preferences + epsilons
└── figures/                         # 84 visualization figures
```
