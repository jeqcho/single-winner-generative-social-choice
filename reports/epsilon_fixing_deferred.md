# Epsilon Fixing: Deferred Work

This report documents the epsilon computation issues for GPT\* and GPT\*\* methods that were deferred from the unified pipeline launch work.

## Problem Summary

When running experiments, epsilon values are not being correctly computed for:
- **GPT\* methods** (`chatgpt_star`, `chatgpt_star_rankings`, `chatgpt_star_personas`)
- **GPT\*\* methods** (`chatgpt_double_star`, `chatgpt_double_star_rankings`, `chatgpt_double_star_personas`)

Example from `outputs/sample_alt_voters/data/abortion/clustered/persona_context/rep0_progressive_liberal/mini_rep0/results.json`:

```json
"chatgpt_star": {
  "winner": "57"
  // Missing: epsilon, full_winner_idx
},
"chatgpt_star_rankings": {
  "winner": "19",
  "epsilon": null,
  "full_winner_idx": "48"
}
```

Note: Some GPT\* results have `epsilon: null` while others are missing the field entirely.

## Root Cause Analysis

### Location

The issue is in `src/sample_alt_voters/run_experiment.py`, function `run_mini_rep()`, lines 352-363:

```python
# Current code
alt_mapping = {str(i): str(alt_indices[i]) for i in range(len(alt_indices))}

for method_name, result in results.items():
    winner = result.get("winner")
    if winner is not None and winner in alt_mapping:  # <-- Problem here
        full_winner = alt_mapping[winner]
        epsilon = lookup_epsilon(full_epsilons, full_winner)
        result["epsilon"] = epsilon
        result["full_winner_idx"] = full_winner
```

### Why It Fails

1. **Traditional methods + base ChatGPT**: Winner is in **sample index space** (0-19), which maps correctly via `alt_mapping` to full index (0-99).

2. **GPT\* methods**: Winner is already in **full index space** (0-99) because they select from all 100 statements. The winner (e.g., "57") is NOT in `alt_mapping` (which only has keys "0"-"19"), so the lookup is skipped.

3. **GPT\*\* methods**: Winner is a **NEW statement** (index 100) that doesn't exist in `precomputed_epsilons.json`. These require inserting the new statement into all voter rankings via API calls, then computing epsilon.

## Proposed Fixes

### Fix 1: GPT\* Epsilon Lookup (No API Calls)

Update `run_mini_rep()` to handle GPT\* methods separately:

```python
for method_name, result in results.items():
    winner = result.get("winner")
    
    # Traditional + base ChatGPT: winner in sample space (0-19) -> map to full
    if winner is not None and winner in alt_mapping:
        full_winner = alt_mapping[winner]
        result["epsilon"] = lookup_epsilon(full_epsilons, full_winner)
        result["full_winner_idx"] = full_winner
    
    # GPT* methods: winner already in full space (0-99) -> direct lookup
    elif method_name.startswith("chatgpt_star") and winner is not None:
        result["epsilon"] = lookup_epsilon(full_epsilons, winner)
        result["full_winner_idx"] = winner
    
    # GPT** methods: new statement -> epsilon computed later by fix_star_epsilons
    # (no action needed here)
```

This fix requires **no API calls** - just a code change in the epsilon lookup logic.

### Fix 2: GPT\*\* Epsilon Computation (Requires API Calls)

GPT\*\* methods generate new statements that don't exist in the precomputed epsilons. To compute epsilon for these:

1. Insert the new statement into all 100 voters' rankings (100 API calls per method)
2. Compute epsilon for the new statement using the updated rankings

This is expensive (~300 API calls per mini-rep for all 3 GPT\*\* variants), so it should remain as a separate pass.

## Current Workaround: fix_star_epsilons.py

The script `src/sample_alt_voters/fix_star_epsilons.py` was created to fix these epsilon values post-hoc:

### What It Does

1. **GPT\* fixes** (no API calls):
   - Looks up epsilon directly from `precomputed_epsilons.json` using the winner index
   - Function: `fix_gpt_star_epsilons()`

2. **GPT\*\* fixes** (requires API calls):
   - Inserts new statement into all 100 voter rankings
   - Computes epsilon for the new statement
   - Function: `fix_gpt_double_star_epsilons()`

### CLI Usage

```bash
# Fix both GPT* and GPT** (default)
uv run python -m src.sample_alt_voters.fix_star_epsilons

# Fix only GPT* (no API calls)
uv run python -m src.sample_alt_voters.fix_star_epsilons --star-only

# Fix only GPT** (requires API key)
uv run python -m src.sample_alt_voters.fix_star_epsilons --double-star-only
```

### ~~Current Issue with fix_star_epsilons.py~~ (RESOLVED)

~~The script has a hardcoded topic mapping that only covers 2 topics.~~

**Fixed 2026-01-27**: Now uses dynamic reverse mapping from `TOPIC_SHORT_NAMES`:
```python
topic_slug_map = {v: k for k, v in TOPIC_SHORT_NAMES.items()}
```

## Action Items for Later Session

### Priority 1: Fix GPT\* Epsilon Lookup in run_experiment.py

- [ ] Update `run_mini_rep()` to directly lookup epsilon for GPT\* methods
- [ ] Test with a new condition to verify fix works
- [x] Run fix_star_epsilons.py to backfill existing results (51 fixed, 213 blocked by corrupted data)

### Priority 2: Fix topic_slug_map in fix_star_epsilons.py

- [x] Replace hardcoded topic mapping with full TOPIC_SHORT_NAMES from config
- [x] Use reverse mapping: `{v: k for k, v in TOPIC_SHORT_NAMES.items()}`

### Priority 3: Decide on GPT\*\* Epsilon Strategy

Options:
1. **Keep separate pass**: Run `fix_star_epsilons.py --double-star-only` after experiments (current approach)
2. **Integrate into run_experiment.py**: Compute epsilon inline during experiment (more API calls during experiment)

Recommendation: Keep as separate pass to:
- Avoid slowing down the main experiment
- Allow batch processing of epsilon fixes
- Maintain flexibility to re-run just the epsilon computation

### Priority 4: Backfill Existing Results

After implementing fixes:
```bash
# First, fix GPT* epsilons (fast, no API)
uv run python -m src.sample_alt_voters.fix_star_epsilons --star-only

# Then, fix GPT** epsilons (slow, requires API)
uv run python -m src.sample_alt_voters.fix_star_epsilons --double-star-only
```

## Data Structures Reference

### precomputed_epsilons.json
```json
{
  "0": 0.1234,
  "1": 0.2345,
  ...
  "99": 0.5678
}
```
Maps full alternative index (0-99) to epsilon value.

### results.json (mini_rep)
```json
{
  "mini_rep_id": 0,
  "voter_indices": [81, 14, 3, ...],  // 20 indices into full 100 voters
  "alt_indices": [25, 91, 83, ...],    // 20 indices into full 100 alts
  "results": {
    "schulze": {"winner": "5", "epsilon": 0.15, "full_winner_idx": "53"},
    "chatgpt_star": {"winner": "57"},  // Missing epsilon!
    "chatgpt_double_star": {"winner": "100", "new_statement": "...", "is_new": true}
  }
}
```

## Files to Modify

| File | Change |
|------|--------|
| `src/sample_alt_voters/run_experiment.py` | Fix GPT\* epsilon lookup in `run_mini_rep()` |
| `src/sample_alt_voters/fix_star_epsilons.py` | ~~Expand topic_slug_map to all 13 topics~~ **DONE** |

---

## Backfill Session (2026-01-27)

### Completed Fixes

1. **Fixed `fix_star_epsilons.py` topic mapping** - Replaced hardcoded 2-topic mapping with dynamic reverse mapping from `TOPIC_SHORT_NAMES`.

2. **Fixed pre-existing import bugs**:
   - `run_experiment.py`: `REASONING_EFFORT` → `RANKING_REASONING`
   - `voting_methods.py`: `GENERATOR_MODEL/GENERATOR_REASONING` → `GENERATIVE_VOTING_MODEL/GENERATIVE_VOTING_REASONING`

3. **Ran backfill**: `uv run python -m src.sample_alt_voters.fix_star_epsilons --star-only`
   - **51 GPT\* epsilons fixed** in reps with valid precomputed_epsilons

### Remaining Issue: Corrupted Preference Data

**15 reps have all-null `precomputed_epsilons.json`** because their preference data contains duplicate alternatives in voter rankings. This prevents epsilon computation entirely.

| Topic | Valid Reps | Corrupted Reps |
|-------|------------|----------------|
| abortion | 38 | 10 |
| electoral | 45 | 3 |
| trust | 6 | 2 |
| **Total** | **89** | **15** |

### Corrupted Reps Detail

Each rep has 1-4 voters with duplicate alternatives in their rankings:

```
abortion/clustered/persona_context/rep0_progressive_liberal
  Voters with duplicates: [51]

abortion/uniform/persona_context/rep1
  Voters with duplicates: [44]

abortion/uniform/persona_context/rep2
  Voters with duplicates: [26]

abortion/uniform/persona_context/rep3
  Voters with duplicates: [2]

abortion/uniform/persona_context/rep4
  Voters with duplicates: [25]

abortion/uniform/persona_context/rep6
  Voters with duplicates: [1, 46, 60, 70]

abortion/uniform/persona_context/rep9
  Voters with duplicates: [61, 63, 67]

abortion/uniform/persona_no_context/rep4
  Voters with duplicates: [30]

abortion/uniform/persona_no_context/rep7
  Voters with duplicates: [10]

abortion/uniform/persona_no_context/rep9
  Voters with duplicates: [26]

electoral/uniform/no_persona_context/rep1
  Voters with duplicates: [65]

electoral/uniform/no_persona_context/rep7
  Voters with duplicates: [49]

electoral/uniform/persona_context/rep8
  Voters with duplicates: [31]

trust/uniform/persona_no_context/rep5
  Voters with duplicates: [6, 30]

trust/uniform/persona_no_context/rep7
  Voters with duplicates: [63]
```

### Root Cause (Suspected)

The duplicate alternatives likely originated from the iterative ranking phase (Phase 2) where the LLM occasionally returns invalid rankings with repeated alternatives. The degeneracy detector should catch this, but may have edge cases.

### Impact

- **213 missing GPT\* epsilons** remain in these 15 corrupted reps
- These reps also have `epsilon: null` for **all methods** (traditional + GPT), not just GPT\*

### Recommended Next Steps

1. **Investigate degeneracy detector** - Check why duplicate alternatives weren't caught during preference building
2. **Consider repair strategy** - Options:
   - Deduplicate rankings and append missing alternatives at bottom (approximation)
   - Re-run preference building for affected voters only
   - Exclude these reps from analysis
3. **Add validation** - Add a post-hoc validation step to catch corrupted preferences before epsilon computation
