# Fix Epsilon Computation for GPT\* and GPT\*\* Methods

## Overview

Fix epsilon computation for GPT\* and GPT\*\* methods by inlining computation (following GPT\*\*\* pattern), fixing topic mappings via config.py imports, and removing the fix-epsilons stage from the default pipeline.

## Problem Summary

Epsilon values are not being correctly computed for GPT\* and GPT\*\* methods:

1. **GPT\* index space mismatch**: Winners are in full index space (0-99), but lookup expects sample space (0-19)
2. **GPT\*\* deferred computation**: Epsilon computation deferred to fix_star_epsilons.py instead of inline
3. **Hardcoded topic mapping**: Scripts only cover 2 of 13 topics

## Design Pattern

Follow the **GPT\*\*\* pattern**: compute epsilon inline when generating new statements.

---

## Fix 1: GPT\* Epsilon Lookup in run_experiment.py

**File**: `src/sample_alt_voters/run_experiment.py`

**Location**: `run_mini_rep()` function, lines 352-362

Add branch for GPT\* methods that directly looks up epsilon:

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
    
    # GPT** methods: epsilon computed inline in voting_methods.py
```

---

## Fix 2: GPT\*\* Inline Epsilon Computation

**File**: `src/experiment_utils/voting_methods.py`

Update GPT\*\* methods to compute epsilon inline (like GPT\*\*\*):

1. Add parameter for all 100 voter personas
2. After generating statement, insert into all 100 voters' rankings (100 API calls, parallelized)
3. Compute epsilon and return in result

**File**: `src/sample_alt_voters/run_experiment.py`

Update `run_chatgpt_voting_methods()` to pass all 100 voter personas to GPT\*\* methods:

- Currently passes `sample_personas` (20)
- Should pass `voter_personas` (all 100) for epsilon computation

---

## Fix 3: Topic Mapping (config.py as source of truth)

**File**: `src/sample_alt_voters/fix_star_epsilons.py` (lines 257-261)

Replace hardcoded map:

```python
# Before (only 2 topics)
topic_slug_map = {
    "abortion": "what-should-guide-laws-concerning-abortion",
    "electoral": "what-reforms-if-any-should-replace-or-modify-the-e",
}

# After (import from config)
from .config import TOPIC_SHORT_NAMES
topic_slug_map = {v: k for k, v in TOPIC_SHORT_NAMES.items()}
```

**File**: `src/sample_alt_voters/run_triple_star.py` (lines 135-138)

Same fix - replace hardcoded map with import from config.py.

---

## Fix 4: Remove fix-epsilons Stage

**File**: `src/sample_alt_voters/__main__.py`

Remove `fix-epsilons` from default pipeline stages:

```python
# Before
stages = [
    "generate-statements",
    "run-experiment",
    "fix-epsilons",      # Remove this
    "run-triple-star",
    "visualize",
]

# After
stages = [
    "generate-statements",
    "run-experiment",
    "run-triple-star",
    "visualize",
]
```

Keep `fix-epsilons` as CLI option for backfilling old results.

---

## Cleanup

- Delete `reports/epsilon_fixing_deferred.md` (issue resolved)
- Delete `data/topic_mappings.json` (not used by any code)

---

## Updated Pipeline

```
generate-statements → run-experiment → run-triple-star → visualize
```

(4 stages instead of 5)

---

## Data Quality Issue Discovered

During investigation, found **15 out of 104 reps** have completely broken `precomputed_epsilons.json` (all null values):

- `abortion/clustered/persona_context/rep0_progressive_liberal`
- `abortion/uniform/persona_context/rep1,2,3,4,6,9`
- `abortion/uniform/persona_no_context/rep4,7,9`
- `electoral/uniform/no_persona_context/rep1,7`
- `electoral/uniform/persona_context/rep8`
- `trust/uniform/persona_no_context/rep5,7`

These require either:
1. Delete entire rep directories and regenerate (~7,500 API calls)
2. Or add pipeline logic to regenerate epsilons from existing preferences.json

---

## Files to Modify

| File | Change |
|------|--------|
| `src/sample_alt_voters/run_experiment.py` | Fix GPT\* lookup, pass all 100 personas to GPT\*\* |
| `src/experiment_utils/voting_methods.py` | GPT\*\* methods compute epsilon inline |
| `src/sample_alt_voters/fix_star_epsilons.py` | Import topic mapping from config.py |
| `src/sample_alt_voters/run_triple_star.py` | Import topic mapping from config.py |
| `src/sample_alt_voters/__main__.py` | Remove fix-epsilons from default pipeline |
| `README.md` | Update pipeline description (4 stages), remove topic_mappings.json reference |

## Files to Delete

| File | Reason |
|------|--------|
| `reports/epsilon_fixing_deferred.md` | Issue resolved |
| `data/topic_mappings.json` | Not used by any code |

---

## TODOs

- [ ] Fix GPT\* epsilon lookup in run_experiment.py run_mini_rep() - direct lookup from full_epsilons
- [ ] Update GPT\*\* methods in voting_methods.py to compute epsilon inline (like GPT\*\*\*)
- [ ] Pass all 100 voter personas to GPT\*\* methods in run_experiment.py
- [ ] Replace hardcoded topic_slug_map in fix_star_epsilons.py with import from config.py
- [ ] Replace hardcoded topic_slug_map in run_triple_star.py with import from config.py
- [ ] Remove fix-epsilons from default pipeline in __main__.py
- [ ] Run fix_star_epsilons.py to backfill existing results (GPT\* only - GPT\*\* requires API)
- [ ] Delete reports/epsilon_fixing_deferred.md and data/topic_mappings.json
- [ ] Update README to reflect new 4-stage pipeline
