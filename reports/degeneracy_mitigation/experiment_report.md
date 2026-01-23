# Degeneracy Mitigation Experiment Report

> Date: 2026-01-22
> Model: gpt-5-mini
> Scope: 100 voters × 100 statements, abortion topic, rep 0

## Executive Summary

This experiment tested mitigations for the 81% preference ranking degeneracy problem. We implemented two approaches across three reasoning levels (6 conditions total).

**Key Findings:**
1. **Approach A (iterative ranking)** successfully eliminates degeneracy - rankings show near-zero correlation with presentation order
2. **Approach B (scoring)** has a critical bug - at low/medium reasoning, the model assigns descending scores (100, 99, 98...) in presentation order, producing meaningless rankings
3. **Recommended approach for Phase 2: A-low** (iterative ranking with low reasoning)

---

## Problem Background

The original `single_call_ranking.py` produced **81% degenerate rankings** where LLM voters output `[0,1,2,...,99]` (sequential) or `[99,...,0]` (reverse) instead of meaningful preferences. This made voting experiments invalid.

**Hypothesized root causes:**
- Statement indices (0-99) conflated with rank positions (0-99)
- Task too complex (rank 100 items in one call)
- Reasoning effort too low
- No prompt guidance against sequential output

---

## Experimental Design

### Mitigations Tested

1. **4-letter hash identifiers** - Replace `0, 1, 2, ..., 99` with random hashes like `aB3x`, `kM9p` to break index/rank association
2. **Anti-sequential instruction** - Explicit warning: "Do NOT output statement IDs in the order presented"
3. **Early flagging + retry** - Detect degenerate output, retry same prompt up to 3 times
4. **Task decomposition** - Break ranking into smaller sub-tasks

### Approach A: Iterative Top-K/Bottom-K Ranking

5 rounds of stateless API calls:
- Round 1: Top 10 + Bottom 10 from 100 statements
- Round 2: Top 10 + Bottom 10 from remaining 80
- Round 3: Top 10 + Bottom 10 from remaining 60
- Round 4: Top 10 + Bottom 10 from remaining 40
- Round 5: Rank all remaining 20

**Key design choices:**
- Per-round shuffling of remaining statements
- Fresh API call each round (no multi-turn conversation)
- Validation: correct counts, no duplicates, valid hashes

### Approach B: Scoring-Based Ranking

Single API call to score all 100 statements from -100 to +100:
- Decimals allowed for differentiation
- Unique scores required
- Iterative dedup for any duplicate scores (max 3 rounds)
- Convert scores to ranking (highest = rank 1)

### Conditions Tested

| Condition | Approach | Reasoning Effort |
|-----------|----------|------------------|
| A-minimal | Iterative ranking | minimal |
| A-low | Iterative ranking | low |
| A-medium | Iterative ranking | medium |
| B-minimal | Scoring | minimal |
| B-low | Scoring | low |
| B-medium | Scoring | medium |

---

## Results

### Initial Statistics (Before Bug Discovery)

| Condition | Valid % | Unique Rankings | Retries/Dedup |
|-----------|---------|-----------------|---------------|
| A-minimal | 79% | 79/100 | 807 retries |
| A-low | 98% | 98/100 | 67 retries |
| A-medium | 100% | 100/100 | 5 retries |
| B-minimal | 72% | 99/100 | 28 unresolved dups |
| B-low | 100% | 100/100 | 0 dedup rounds |
| B-medium | 100% | 98/100 | 1 dedup round |

All conditions reported **0% degeneracy** (no sequential/reverse rankings detected).

### Cross-Approach Correlation Analysis

When comparing rankings between approaches for the same voters:

| Comparison | Mean Spearman Correlation |
|------------|---------------------------|
| A-low vs A-medium | 0.802 (high within A) |
| B-low vs B-medium | 0.928 (high within B) |
| **A-low vs B-low** | **-0.012 (near zero!)** |
| A-medium vs B-medium | 0.019 (near zero) |

**This was suspicious** - high correlation within each approach but zero correlation between approaches suggested something was wrong.

---

## Bug Discovery: Scoring Approach Degeneracy

### Investigation

We checked if rankings correlated with presentation order (the order statements were shown to the model):

**Approach B correlation with presentation order:**
| Condition | Mean Correlation | Exactly Descending Scores |
|-----------|------------------|---------------------------|
| B-minimal | -0.126 | 0/100 |
| **B-low** | **-0.987** | **86/100** |
| **B-medium** | **-0.952** | **50/100** |

**Approach A correlation with presentation order:**
| Condition | Mean Correlation | Top-10 = First-10 Presented |
|-----------|------------------|------------------------------|
| A-minimal | 0.029 | 0/100 |
| A-low | 0.043 | 0/100 |
| A-medium | 0.033 | 0/100 |

### Root Cause

At low/medium reasoning, the scoring approach produces **degenerate scores** disguised as valid output:

```
Position 0  → Score 100
Position 1  → Score 99
Position 2  → Score 98
...
Position 99 → Score 1
```

Example from Voter 0 (B-low):
```
Position | Hash | Statement ID | Score
---------|------|--------------|-------
0        | T4Uz | 23           | 100.0
1        | wMQ3 | 8            | 99.0
2        | zwTF | 11           | 98.0
3        | kTmb | 7            | 97.0
...
97       | kQkr | 53           | 3.0
98       | FZe5 | 97           | 2.0
99       | vRdg | 49           | 1.0
```

### Why Our Validation Missed This

1. We checked for **duplicate scores** - but descending integers are all unique
2. We checked for **sequential rankings** - but rankings derived from descending scores aren't `[0,1,2,...,99]`; they depend on the (shuffled) presentation order
3. The model found a loophole: produce unique scores that satisfy validation but encode no real preferences

### Why Higher Reasoning Made It Worse

- **Minimal reasoning**: Model doesn't have capacity to systematically produce descending scores; output is more varied (but has more validation errors)
- **Low/medium reasoning**: Model is "smart enough" to take the lazy path of assigning descending integers, satisfying all validation checks

---

## Corrected Results

### Approach A: Working Correctly ✓

| Condition | Valid % | Presentation Order Correlation | Status |
|-----------|---------|--------------------------------|--------|
| A-minimal | 79% | 0.029 | ✓ Valid (but high retry rate) |
| **A-low** | **98%** | **0.043** | **✓ Recommended** |
| A-medium | 100% | 0.033 | ✓ Valid (higher cost) |

### Approach B: Degenerate ✗

| Condition | Valid % | Presentation Order Correlation | Status |
|-----------|---------|--------------------------------|--------|
| B-minimal | 72% | -0.126 | ⚠️ OK but high error rate |
| B-low | 100% | -0.987 | ✗ **DEGENERATE** |
| B-medium | 100% | -0.952 | ✗ **DEGENERATE** |

---

## Recommendations

### For Phase 2

**Use Approach A with "low" reasoning:**
- 98% valid rankings
- Near-zero correlation with presentation order (genuine preferences)
- Reasonable cost (67 retries, ~11s avg per call)
- 5 API calls per voter (500 total for 100 voters)

### Improvements Needed

1. **Add presentation order correlation check** to degeneracy detection
2. **Fix scoring approach** - possibly by:
   - Randomizing the scale (not always -100 to +100)
   - Requiring justification for scores
   - Using anchored comparisons instead of absolute ratings
3. **Consider why iterative ranking works** - the comparative task ("pick top 10") may force genuine evaluation in a way that absolute scoring ("rate this -100 to +100") does not

---

## Technical Details

### Code Location

```
src/degeneracy_mitigation/
├── __init__.py
├── config.py              # Model settings, paths, prompts
├── hash_identifiers.py    # Deterministic 4-char hashes
├── degeneracy_detector.py # Validation and retry logic
├── iterative_ranking.py   # Approach A implementation
├── scoring_ranking.py     # Approach B implementation (buggy)
├── run_test.py            # CLI entry point
└── analyze_results.py     # Analysis and comparison
```

### Output Location

```
outputs/degeneracy_mitigation/
├── approach_a/{minimal,low,medium}/
│   ├── rankings.json
│   ├── stats.json
│   └── round_logs/
├── approach_b/{minimal,low,medium}/
│   ├── rankings.json
│   ├── scores.json
│   ├── stats.json
│   └── full_results.json
├── summary.json
└── comparison.json
```

### Commands to Reproduce

```bash
# Run full experiment
uv run python -m src.degeneracy_mitigation.run_test --approach both --reasoning-effort all

# Run only recommended approach
uv run python -m src.degeneracy_mitigation.run_test --approach ranking --reasoning-effort low

# Analyze results
uv run python -m src.degeneracy_mitigation.analyze_results --save
```

---

## Appendix: API Cost Summary

| Condition | API Calls | Avg Time/Call | Total Time |
|-----------|-----------|---------------|------------|
| A-minimal | 1,307 | 3.3s | 72 min |
| A-low | 567 | 11.1s | 105 min |
| A-medium | ~500 | ~30s | ~250 min |
| B-minimal | 178 | 8.9s | 26 min |
| B-low | 100 | 16.0s | 27 min |
| B-medium | ~100 | ~30s | ~50 min |

Note: A-minimal has many retries due to validation failures; A-low is the most efficient valid option.
