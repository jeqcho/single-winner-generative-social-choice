# Handoff: Degeneracy Mitigation for LLM Preference Rankings

> Last updated: 2026-01-22
> Session focus: Implemented, tested, and debugged mitigations for 81% preference ranking degeneracy

## Objective

Solve the preference degeneracy problem (81% of LLM-generated rankings were trivial sequential/reverse patterns) before proceeding with Phase 2 of the sample-alt-voters experiment. Target: reduce degeneracy to <5%.

## Current Status

**State**: Complete - Critical bug found and documented
**Branch**: main
**Key files created this session**:
- `src/degeneracy_mitigation/` - New module with 7 Python files
- `outputs/degeneracy_mitigation/` - Experiment results for 6 conditions
- `reports/degeneracy_mitigation/` - Comprehensive experiment report

## Progress Summary

### Completed
- Implemented full degeneracy mitigation test framework
- Ran experiments across 2 approaches × 3 reasoning levels (6 conditions)
- **Discovered critical bug in scoring approach (B)** - produces degenerate descending scores
- Validated that **iterative ranking approach (A) works correctly**
- Documented all findings in `reports/degeneracy_mitigation/experiment_report.md`

### Critical Finding: Scoring Approach Bug

**Approach B (scoring) is degenerate at low/medium reasoning:**
- B-low: 86/100 voters have exactly descending scores (100, 99, 98...)
- B-low: 0.987 correlation with presentation order
- The model takes a lazy shortcut to satisfy "unique scores" requirement

**Approach A (iterative ranking) works correctly:**
- A-low: 0.043 correlation with presentation order (near-zero = good)
- Rankings are genuine preferences, not presentation-order artifacts

## Technical Context

### Entry Points
- Main CLI: `src/degeneracy_mitigation/run_test.py`
- Analysis: `src/degeneracy_mitigation/analyze_results.py`
- Results: `outputs/degeneracy_mitigation/`
- Report: `reports/degeneracy_mitigation/experiment_report.md`

### Key Commands
```bash
# Run recommended approach (iterative ranking with low reasoning)
uv run python -m src.degeneracy_mitigation.run_test --approach ranking --reasoning-effort low

# Analyze results
uv run python -m src.degeneracy_mitigation.analyze_results --save
```

## Decision Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| Use A-low for Phase 2 | Near-zero presentation order correlation, 98% valid | B-low (discovered to be degenerate) |
| Iterative top-K/bottom-K | Comparative task forces genuine evaluation | Single-call ranking, scoring |
| "Low" reasoning effort | Sweet spot: much better than minimal, cheaper than medium | Minimal (poor), Medium (marginal gains) |

## What Worked

- **Hash identifiers**: Eliminated sequential ranking degeneracy
- **Iterative ranking (Approach A)**: Produces genuine preferences
- **Per-round shuffling**: Prevents presentation-order bias

## What Didn't Work

> ⚠️ **Do not retry these approaches without new information**

- **Scoring approach (B) at low/medium reasoning**: Model assigns descending scores (100, 99, 98...) in presentation order - completely degenerate
- **Single-call ranking**: Original approach had 81% degeneracy

## Recommended Next Steps

1. **Use A-low for Phase 2**: Iterative ranking with low reasoning
   - Modify preference builder to use `src/degeneracy_mitigation/iterative_ranking.py`
   - 98% valid, near-zero presentation order bias

2. **Fix scoring approach** (optional, for future):
   - Add presentation order correlation check to validation
   - Consider randomizing the scale or requiring justifications

3. **Run Phase 2** with validated approach

## Session Notes

- User has OpenAI grant - must use OpenAI models only, max GPT-5-mini due to cost
- The correlation analysis revealed the bug: high within-approach correlation but zero cross-approach correlation was suspicious
- Comparative tasks (pick top 10) seem more robust than absolute tasks (score 1-100)

## Experiment Results Summary

### Corrected Assessment

| Condition | Valid % | Pres. Order Corr | Status |
|-----------|---------|------------------|--------|
| A-minimal | 79% | 0.030 | ✓ OK (high retry) |
| **A-low** | **98%** | **0.043** | **✓ Recommended** |
| A-medium | 100% | 0.033 | ✓ OK (costly) |
| B-minimal | 72% | -0.126 | ⚠️ OK (errors) |
| B-low | 100% | -0.987 | ✗ DEGENERATE |
| B-medium | 100% | -0.952 | ✗ DEGENERATE |
