# Handoff: Sample-Alt-Voters Experiment

> Last updated: 2026-01-22
> Session focus: Implemented Phase 1 statement generation infrastructure and ran pre-generation for all 4 alternative distributions

## Objective

Build a factorial experiment comparing 4 alternative (statement) distributions × 2 voter distributions × 2 topics (abortion, electoral college). The goal is to evaluate different methods of generating bridging statements and how they perform under different voter sampling strategies.

## Current Status

**State**: Phase 1 Complete - Ready for Phase 2 (Voting & Analysis)
**Branch**: main
**Key files created this session**:
- `src/sample_alt_voters/config.py` - Experiment configuration
- `src/sample_alt_voters/alternative_generators/*.py` - Four statement generators (Alt1-4)
- `src/sample_alt_voters/generate_statements.py` - CLI for pre-generation
- `src/sample_alt_voters/generate_per_rep_statements.py` - CLI for per-rep generation

## Progress Summary

### Completed
- **Alt1** (`persona_no_context`): Persona generates statement WITHOUT context - 815 statements × 2 topics pre-generated
- **Alt2** (`persona_context`): Persona sees 100 statements then generates (Ben's bridging) - 100 statements × 10 reps × 2 topics
- **Alt3** (`no_persona_context`): No persona, sees 100 statements, verbalized sampling - 100 statements × 10 reps × 2 topics
- **Alt4** (`no_persona_no_context`): No persona, no context, verbalized sampling - 815 statements × 2 topics pre-generated
- **Sampled context indices** saved for each rep to ensure reproducibility
- Total: ~4,356 API calls completed (gpt-5.2)

### In Progress
- Phase 2: Voter sampling, preference building, and voting evaluation (not yet implemented)

## Technical Context

### Entry Points
- Configuration: `src/sample_alt_voters/config.py`
- Pre-generation CLI: `src/sample_alt_voters/generate_statements.py`
- Per-rep generation CLI: `src/sample_alt_voters/generate_per_rep_statements.py`
- Plan document: `.cursor/plans/sample_alt_voters_experiment_e02e7ecb.plan.md`

### Key Commands
```bash
# Pre-generate Alt1 and Alt4 (already done)
uv run python -m src.sample_alt_voters.generate_statements --all

# Generate Alt2 and Alt3 for all reps (already done)
uv run python -m src.sample_alt_voters.generate_per_rep_statements --all

# Test imports
uv run python -c "from src.sample_alt_voters import config; print(config.MODEL)"
```

### Data Structure
```
data/sample-alt-voters/
├── sampled-statements/
│   ├── persona_no_context/          # Alt1: 815 per topic (pre-generated)
│   │   ├── abortion.json
│   │   └── electoral.json
│   ├── no_persona_no_context/       # Alt4: 815 per topic (pre-generated)
│   │   ├── abortion.json
│   │   └── electoral.json
│   ├── persona_context/             # Alt2: 100 per rep (per-rep generated)
│   │   ├── abortion/rep{0-9}.json
│   │   └── electoral/rep{0-9}.json
│   └── no_persona_context/          # Alt3: 100 per rep (per-rep generated)
│       ├── abortion/rep{0-9}.json
│       └── electoral/rep{0-9}.json
└── sampled-context/                 # Context indices per rep
    ├── abortion/rep{0-9}.json
    └── electoral/rep{0-9}.json
```

### Dependencies/Environment Notes
- OpenAI API key in `.env` file (OPENAI_API_KEY)
- Model: gpt-5.2, Temperature: 1.0
- Uses `uv` for package management

## Decision Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Alt4 standardized to 815 statements | Matches Alt1 pool size for fair comparison | Originally planned ~1000 |
| Context statements shared across voter_dists | Reduces API calls, ensures comparability | Separate sampling per voter_dist |
| Alt2 uses personas who wrote context statements | Maintains logical consistency (same persona rewrites after seeing all) | Using sampled voters instead |
| Verbalized sampling returns all 5 statements | Increases diversity without extra API calls | Probabilistic sampling from 5 |

## What Worked

- Parallel API calls with ThreadPoolExecutor (~15-20 calls/sec for Alt1, slower for Alt2/3 due to longer prompts)
- Incremental saving every 50 statements for resumability
- XML parsing with fallback patterns for verbalized sampling responses
- Running generation in tmux with timestamped log files

## What Didn't Work

> ⚠️ **Do not retry these approaches without new information**

- Initial tmux run failed because `.env` wasn't sourced - fixed by adding `export $(cat .env | xargs)` to shell scripts

## Blockers & Open Questions

- [ ] Phase 2 implementation: voter samplers (uniform, clustered), preference building, voting evaluation
- [ ] How to handle cluster sizes that are < 100 personas (some clusters have only 39-68 members)
- [ ] Mini-rep sampling strategy: sample 20 from 100 for both voters and statements

## Recommended Next Steps

1. **Implement voter samplers**: Create `src/sample_alt_voters/voter_samplers/uniform.py` and `clustered.py` to sample 100 voters per rep
2. **Implement preference builder**: Wrap existing `src/sampling_experiment/preference_builder.py` to build 100×100 preference profiles
3. **Implement mini-rep sampling**: Sample 20 voters × 20 statements for each of 5 mini-reps per rep
4. **Implement voting evaluation**: Compute voting method performance on 20×20, evaluate using 100×100 as ground truth

## Session Notes

- User prefers `uv` for Python package management
- User wants experiments run in tmux with logs stored in `logs/` folder with timestamps
- The experiment structure has 3 levels: Global (815) → Rep (100) → Mini-rep (20)
- Statement distributions are independent of voter distributions - same statements used for both uniform and clustered voter sampling
