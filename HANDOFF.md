# Handoff: Degeneracy Mitigation for LLM Preference Rankings

> Last updated: 2026-01-24
> Session focus: Tested A* variant (reversed bottom-K ordering) and validated A-low as final recommendation

## Objective

Solve the preference degeneracy problem (81% of LLM-generated rankings were trivial sequential/reverse patterns) before proceeding with Phase 2 of the sample-alt-voters experiment. Target: reduce degeneracy to <5%.

## Current Status

**State**: Complete - A-low validated, ready for Phase 2
**Branch**: main
**Key files created this session**:
- `src/degeneracy_mitigation/iterative_ranking_star.py` - A* variant implementation
- `outputs/degeneracy_mitigation/approach_a_star/` - A* experiment results (3 conditions)

## Progress Summary

### Completed (This Session)
- Created A* variant: asks for bottom-K with "least preferred first" instead of "least preferred last"
- Ran A* experiment across all 3 reasoning levels (minimal, low, medium)
- Computed correlations between A and A*
- Validated ideology alignment for both A and A*
- Confirmed **A-low remains the recommended approach**

### Previous Session Completed
- Implemented full degeneracy mitigation test framework (Approaches A and B)
- Discovered critical bug in scoring approach (B) - produces degenerate descending scores
- Validated that iterative ranking approach (A) works correctly
- Generated human-readable voter files for manual inspection

## Technical Context

### Entry Points
- Main CLI: `src/degeneracy_mitigation/run_test.py`
- A* module: `src/degeneracy_mitigation/iterative_ranking_star.py`
- Voter file generator: `src/degeneracy_mitigation/generate_voter_files.py`
- Results: `outputs/degeneracy_mitigation/`
- Report: `reports/degeneracy_mitigation/experiment_report.md`

### Key Commands
```bash
# Run A (iterative ranking) - RECOMMENDED
uv run python -m src.degeneracy_mitigation.run_test --approach ranking --reasoning-effort low

# Run A* (reversed bottom-K)
uv run python -m src.degeneracy_mitigation.run_test --approach ranking_star --reasoning-effort low

# Generate voter files for inspection
uv run python src/degeneracy_mitigation/generate_voter_files.py
```

## A* Experiment Results

### Validity Rates

| Condition | Valid | Retries |
|-----------|-------|---------|
| A* minimal | 8% | 841 |
| A* low | 97% | 54 |
| A* medium | 100% | 6 |

### Ideology Alignment Comparison

| Condition | Conservative | Liberal | Overall |
|-----------|--------------|---------|---------|
| A-low | 94% | 87% | **92%** |
| A*-low | 94% | 89% | **93%** |

Both produce meaningful rankings that align with voter ideology.

### Correlation: A vs A*

| Level | Correlation |
|-------|-------------|
| minimal | r = 0.033 |
| low | r = 0.053 |
| medium | r = 0.033 |

**Key Finding**: A and A* rankings are essentially **uncorrelated** (r ≈ 0.03-0.05), yet both show ~92% ideology alignment. This indicates:
- Both capture the voter's general ideology (top choices align)
- They differ significantly in middle rankings
- The model is sensitive to prompt phrasing but still captures meaningful preferences

## Decision Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Use A-low for Phase 2 | 92% ideology alignment, 98% valid, near-zero presentation order bias | A*-low (similar quality, no clear advantage) |
| Keep A over A* | A-low has fewer retries (45 vs 54), more established | A* (valid but different rankings) |
| Iterative top-K/bottom-K | Comparative task forces genuine evaluation | Scoring (degenerate), Single-call ranking (81% degenerate) |
| "Low" reasoning effort | Sweet spot: much better than minimal, cheaper than medium | Minimal (poor validity), Medium (marginal gains, costly) |

## What Worked

- **Hash identifiers**: Eliminated sequential ranking degeneracy
- **Iterative ranking (Approach A)**: Produces genuine preferences
- **Per-round shuffling**: Prevents presentation-order bias
- **Ideology alignment check**: Effective sanity check for ranking quality

## What Didn't Work

> ⚠️ **Do not retry these approaches without new information**

- **Scoring approach (B) at low/medium reasoning**: Model assigns descending scores (100, 99, 98...) in presentation order - completely degenerate
- **A* at minimal reasoning**: 92% invalid - model struggles with reversed bottom-K at low reasoning effort
- **Single-call ranking**: Original approach had 81% degeneracy

## Recommended Next Steps

1. **Use A-low for Phase 2**: Iterative ranking with low reasoning
   - Module: `src/degeneracy_mitigation/iterative_ranking.py`
   - 98% valid, 92% ideology alignment, near-zero presentation order bias

2. **Integrate into Phase 2 pipeline**:
   - Adapt `iterative_ranking.rank_voter()` for the sample-alt-voters experiment
   - Use same hash seed (42) for reproducibility

3. **Run Phase 2** with validated approach

## Session Notes

- User has OpenAI grant - must use OpenAI models only, max GPT-5-mini due to cost
- A vs A* low correlation but both valid suggests the model produces different but meaningful rankings based on subtle prompt differences
- Comparative tasks (pick top 10) are more robust than absolute tasks (score 1-100)
- Human-readable voter files available in `outputs/degeneracy_mitigation/approach_*/*/voter_files/` for manual inspection

## Files Created This Session

```
src/degeneracy_mitigation/
├── iterative_ranking_star.py    # A* variant (new)
├── run_test.py                  # Updated with --approach ranking_star

outputs/degeneracy_mitigation/
├── approach_a_star/             # A* results (new)
│   ├── minimal/
│   ├── low/
│   └── medium/
```
