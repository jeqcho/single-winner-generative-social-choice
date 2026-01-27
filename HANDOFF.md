# Handoff: Phase 2 Sample-Alt-Voters Experiment

> Last updated: 2026-01-27
> Session focus: Fixed GPT** epsilons, running GPT*** - CDF plots now show all methods

## Objective

Run Phase 2 of the sample-alt-voters experiment: compare traditional voting methods AND GPT-based methods using A*-low iterative ranking to generate preference profiles. Evaluate using critical epsilon (ε*) as the consensus metric.

## Current Status

**State**: In Progress - GPT*** running in background
**Branch**: main
**Running tmux sessions**:
- `triple_star`: 79% complete (76/96 reps), ~5 hours remaining
- `full_pipeline`: Running new experiment

**Key results**: 
- 7365 total results (including GPT** now)
- GPT** epsilons fixed (1440 fixes applied)
- All visualization plots regenerated with GPT** data

## Progress Summary

### Completed (This Session - 2026-01-27)
- Investigated why GPT** wasn't showing in CDF plots
- **Root cause found**: GPT** methods generate NEW statements (`winner=100`), so `epsilon=None` until computed
- Ran `fix_star_epsilons.py --double-star-only` - fixed 1440 GPT** epsilons
- Regenerated all visualizations with `visualizer.py --all`
- **CDF plots now show GPT** methods** (GPT**, GPT**+Rank, GPT**+Pers)

### In Progress
- `triple_star` tmux session: Running GPT*** method (76/96 reps complete)
- `full_pipeline` tmux session: Running new experiment

### Previous Sessions Completed
- Created `src/sample_alt_voters/ideology_histogram.py` module
- Generated 88 ideology ranking histogram plots
- Fixed 117 invalid voters, re-ran 28 failed mini-reps
- Created voter preference reports

## Technical Context

### Entry Points
- Main experiment: `src/sample_alt_voters/run_experiment.py`
- Fix GPT** epsilons: `src/sample_alt_voters/fix_star_epsilons.py`
- Run GPT***: `src/sample_alt_voters/run_triple_star.py`
- Visualizer: `src/sample_alt_voters/visualizer.py`
- Ideology histogram: `src/sample_alt_voters/ideology_histogram.py`

### Key Commands
```bash
# Check running tmux sessions
tmux list-sessions

# Monitor triple_star progress
tmux capture-pane -t triple_star -p | tail -20
cat logs/triple_star*.log | grep "Processing reps:" | tail -1

# Regenerate all visualizations (after triple_star completes)
uv run python -m src.sample_alt_voters.visualizer --all

# Fix GPT** epsilons (already done)
uv run python -m src.sample_alt_voters.fix_star_epsilons --double-star-only

# Run GPT*** (already running in tmux)
uv run python -m src.sample_alt_voters.run_triple_star
```

### Data Structure
```
outputs/sample_alt_voters/
├── data/{topic}/{voter_dist}/{alt_dist}/rep{N}/
│   ├── preferences.json
│   ├── precomputed_epsilons.json
│   ├── voters.json
│   ├── chatgpt_triple_star.json  # GPT*** results (after run_triple_star)
│   └── mini_rep{0-4}/results.json  # Contains GPT** with epsilon now
├── figures/
│   └── cdf_epsilon.png  # Now shows GPT** methods
```

## GPT Method Status

| Method Category | Status | Notes |
|-----------------|--------|-------|
| Traditional (Schulze, Borda, IRV, Plurality, VBC) | ✅ Complete | 480 results each |
| GPT (base) | ✅ Complete | Results in mini-reps |
| GPT* | ✅ Complete | Results in mini-reps |
| GPT** | ✅ Fixed | 1440 epsilons computed, now in plots |
| GPT*** | 🔄 Running | 76/96 reps (79%), ~5 hours remaining |

## Decision Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Run fix_star_epsilons for GPT** | GPT** generates new statements that need epsilon computation | Leave as null (rejected) |
| Run triple_star in tmux | Long-running (~10 hours), needs background execution | Run in foreground (rejected) |

## What Worked

- **fix_star_epsilons.py**: Successfully computed 1440 GPT** epsilons
- **Visualizer regeneration**: Plots now show GPT** methods after fixing epsilons
- **tmux for long jobs**: Allows monitoring progress while continuing work

## What Didn't Work

> ⚠️ **Do not retry these approaches without new information**

- **Random fallback for invalid voters**: User explicitly rejected - corrupts data integrity
- **Expecting GPT** to have precomputed epsilons**: They generate NEW statements, so epsilon must be computed separately

### GPT Method Epsilon Issue (RESOLVED)

**Problem**: GPT** methods return `winner=100` (index of newly generated statement) and `epsilon=None`.

**Solution**: Run `fix_star_epsilons.py --double-star-only` which:
1. Takes the generated statement text
2. Inserts it into each voter's rankings (requires API calls)
3. Computes epsilon for the new statement position

**Status**: FIXED - GPT** now shows in CDF plots

## Recommended Next Steps

1. **Wait for triple_star to complete** (~5 hours):
   - Monitor: `tmux capture-pane -t triple_star -p | tail -20`
   - Log: `cat logs/triple_star*.log | grep "Processing reps:" | tail -1`

2. **Regenerate plots after triple_star completes**:
   ```bash
   uv run python -m src.sample_alt_voters.visualizer --all
   ```

3. **Analyze GPT*** results**:
   - Compare GPT*** epsilon to traditional methods
   - GPT*** generates blind consensus statements without seeing voter preferences

4. **Commit all results** (after triple_star):
   - Fixed GPT** data in mini-rep results.json files
   - GPT*** data in chatgpt_triple_star.json files
   - Updated visualization figures

## Session Notes

### User Preferences
- User has OpenAI grant - must use OpenAI models only (gpt-5-mini)
- User prefers data integrity over completeness
- Slide-quality plots preferred

### Experiment Structure
- 96 total reps (80 uniform + 16 clustered across topics)
- 5 mini-reps per rep (20×20 subsamples)
- Methods: 5 traditional + 9 GPT variants + GPT***

### Log Files
```
logs/
├── fix_double_star_20260127_062412.log  # GPT** fix log (completed)
├── triple_star_20260127_061657.log       # GPT*** running log
```

## Files Modified This Session (2026-01-27)

```
outputs/sample_alt_voters/data/
└── **/mini_rep*/results.json  # GPT** now has epsilon values

outputs/sample_alt_voters/figures/
└── **/*.png  # All plots regenerated with GPT** data

logs/
├── fix_double_star_*.log
└── triple_star_*.log
```
