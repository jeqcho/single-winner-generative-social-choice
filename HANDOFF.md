# Handoff: Degeneracy Mitigation for LLM Preference Rankings

> Last updated: 2026-01-22
> Session focus: Implemented and tested mitigations for 81% preference ranking degeneracy

## Objective

Solve the preference degeneracy problem (81% of LLM-generated rankings were trivial sequential/reverse patterns) before proceeding with Phase 2 of the sample-alt-voters experiment. Target: reduce degeneracy to <5%.

## Current Status

**State**: Complete - Ready for Phase 2
**Branch**: main (uncommitted changes)
**Key files created this session**:
- `src/degeneracy_mitigation/` - New module with 7 Python files
- `outputs/degeneracy_mitigation/` - Experiment results for 6 conditions

## Progress Summary

### Completed
- Implemented full degeneracy mitigation test framework
- Ran experiments across 2 approaches × 3 reasoning levels (6 conditions)
- **Achieved 0% degeneracy** across all conditions (down from 81% baseline)
- Identified **B-low (scoring + low reasoning)** as best approach for Phase 2
- Discovered that Approach A and B produce uncorrelated rankings at low/medium reasoning

### Key Finding
The two approaches (iterative ranking vs scoring) produce **completely different preference orderings** despite using same voters/statements:
- A-low vs B-low correlation: **-0.012** (essentially zero)
- But within-approach consistency is high: B-low vs B-medium = **0.928**

## Technical Context

### Entry Points
- Main CLI: `src/degeneracy_mitigation/run_test.py`
- Analysis: `src/degeneracy_mitigation/analyze_results.py`
- Results: `outputs/degeneracy_mitigation/`

### Key Commands
```bash
# Run full experiment (both approaches, all reasoning levels)
uv run python -m src.degeneracy_mitigation.run_test --approach both --reasoning-effort all

# Run just scoring with low reasoning (recommended for Phase 2)
uv run python -m src.degeneracy_mitigation.run_test --approach scoring --reasoning-effort low

# Analyze results
uv run python -m src.degeneracy_mitigation.analyze_results --save
```

### Module Structure
```
src/degeneracy_mitigation/
├── __init__.py
├── config.py              # Model settings, paths, prompts
├── hash_identifiers.py    # Deterministic 4-char hashes (breaks index/rank association)
├── degeneracy_detector.py # Validation and retry logic
├── iterative_ranking.py   # Approach A: 5-round top-K/bottom-K
├── scoring_ranking.py     # Approach B: Single-call scoring (-100 to +100)
├── run_test.py            # CLI entry point
└── analyze_results.py     # Analysis and comparison
```

## Decision Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Use 4-char hash identifiers | Breaks association between statement index (0-99) and rank position | Longer hashes, random IDs |
| Scoring approach for Phase 2 | Simpler, cheaper, 100% valid at low reasoning | Iterative ranking (more complex, 98% valid) |
| "Low" reasoning effort | Sweet spot: much better than minimal, cheaper than medium | Minimal (poor), Medium (marginal gains) |
| Per-round shuffling | Prevents presentation-order bias in iterative approach | Fixed order |
| Iterative dedup for scoring | Resolves duplicate scores without failing | Reject on duplicates |

## What Worked

- **Hash identifiers**: Completely eliminated sequential pattern output
- **Explicit anti-sequential instruction**: "Do NOT simply list codes in the order they appear"
- **Validation + retry**: Catches invalid outputs (wrong counts, duplicates, bad hashes)
- **Scoring approach**: Simpler task for LLM, better results at lower reasoning
- **Decimals in scores**: Allowed model to differentiate similar preferences

## What Didn't Work

> ⚠️ **Do not retry these approaches without new information**

- **Minimal reasoning**: Too many invalid outputs (A-minimal: only 20% valid, B-minimal: 28% unresolved duplicates)
- **Single-call ranking of 100 items**: Original approach had 81% degeneracy (see `reports/preference_degeneracy_report.md`)

## Open Questions

- [ ] Why do Approach A and B produce uncorrelated rankings? Which reflects "true" preferences?
- [ ] Should Phase 2 use scoring (simpler) or iterative ranking (more deliberate)?
- [ ] Need to decide: accept different preference orderings or investigate further?

## Recommended Next Steps

1. **Commit the degeneracy mitigation module**: 
   ```bash
   git add src/degeneracy_mitigation/ outputs/degeneracy_mitigation/
   git commit -m "Add degeneracy mitigation module - achieves 0% degeneracy"
   ```

2. **Update Phase 2 plan** to use scoring approach with low reasoning:
   - Modify `src/sampling_experiment/` to use new `scoring_ranking.py`
   - Or create new preference builder based on `src/degeneracy_mitigation/scoring_ranking.py`

3. **Run Phase 2** with the validated approach (see plan at `.cursor/plans/phase_2_voting_experiment_74efac44.plan.md`)

## Session Notes

- User has OpenAI grant - must use OpenAI models only, max GPT-5-mini due to cost
- User prefers uv for package management
- Experiments run in tmux for long-running jobs
- Related plan files in `.cursor/plans/`
- The correlation finding (A vs B uncorrelated) is surprising and may warrant investigation before Phase 2

## Experiment Results Summary

| Condition | Valid % | Unique Rankings | Efficiency |
|-----------|---------|-----------------|------------|
| A-minimal | 79% | 79/100 | 807 retries |
| A-low | 98% | 98/100 | 67 retries |
| A-medium | 100% | 100/100 | 5 retries |
| **B-low** | **100%** | **100/100** | **0 dedup** |
| B-minimal | 72% | 99/100 | 28 unresolved |
| B-medium | 100% | 98/100 | 1 dedup |

**Recommendation**: Use B-low (scoring + low reasoning) for Phase 2.
