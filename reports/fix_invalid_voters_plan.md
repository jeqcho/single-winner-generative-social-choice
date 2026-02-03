# Fix Invalid Voter Preferences Plan

**Date:** 2025-02-02  
**Status:** Pending implementation

## Problem

Two preference profiles in uniform voters have specific voters with `-1` values (failed rankings):

| Rep | Invalid Voter Indices |
|-----|----------------------|
| `environment/uniform/persona_no_context/rep4` | 48 |
| `trust/uniform/persona_no_context/rep7` | 34, 63, 90 |

The `-1` values indicate that the iterative ranking process failed for these voters, leaving their entire preference column as -1.

### Impact

1. **precomputed_epsilons.json** - Epsilon values are incorrect since they depend on the full preference matrix
2. **mini_rep\*/results.json** - All voting methods fail with errors like:
   ```
   "error": "Candidate c-1 found in ballot ... but not in candidate list"
   ```
3. **Downstream analysis** - Any aggregation using these reps will be affected

## Solution

Create a targeted fix script `src/sample_alt_voters/fix_invalid_voters.py` that:

1. **Re-ranks only invalid voters** (4 total, not 200)
   - Load existing `preferences.json` and identify columns with -1 values
   - Load `voters.json` to get persona indices for those voters
   - Load personas from `data/personas/prod/adult.json`
   - Load statements using `load_statements_for_rep()`
   - Re-rank using `rank_voter()` (5 API calls per voter)
   - Update preferences matrix in place

2. **Recompute epsilons**
   - Use `precompute_all_epsilons(preferences)`
   - Save to `precomputed_epsilons.json`

3. **Re-run mini_rep evaluations**
   - Re-run all 4 mini_reps using `run_mini_rep()`
   - Regenerate `mini_rep{0-3}/results.json` files

### Script Usage

```bash
uv run python -m src.sample_alt_voters.fix_invalid_voters \
  --topic environment --rep 4

uv run python -m src.sample_alt_voters.fix_invalid_voters \
  --topic trust --rep 7
```

### Key Functions to Reuse

| Function | Location | Purpose |
|----------|----------|---------|
| `load_statements_for_rep()` | `src/sample_alt_voters/run_experiment.py` | Load 100 statements for topic/rep |
| `rank_voter()` | `src/degeneracy_mitigation/iterative_ranking.py` | Rank all alternatives for one voter |
| `precompute_all_epsilons()` | `src/experiment_utils/epsilon_calculator.py` | Compute epsilon for all alternatives |
| `run_mini_rep()` | `src/sample_alt_voters/run_experiment.py` | Evaluate voting methods on 20x20 subsample |

### Data Locations

| Data | Path |
|------|------|
| Personas | `data/personas/prod/adult.json` (815 adult personas) |
| Preferences | `outputs/sample_alt_voters/data/{topic}/uniform/persona_no_context/rep{N}/preferences.json` |
| Voters | `outputs/sample_alt_voters/data/{topic}/uniform/persona_no_context/rep{N}/voters.json` |
| Mini-rep results | `outputs/sample_alt_voters/data/{topic}/uniform/persona_no_context/rep{N}/mini_rep{0-3}/results.json` |

### Topic Mapping

- "environment" -> slug: `what-balance-should-be-struck-between-environmenta`
- "trust" -> slug: `how-should-we-increase-the-general-publics-trust-i`

### Estimated Runtime

- Re-ranking 4 invalid voters: ~20 API calls (~1-2 min)
- Epsilon computation: ~30 seconds
- Mini-rep evaluations (8 total): ~10-15 minutes (ChatGPT methods)
- **Total:** ~15-20 minutes

### Verification

```bash
# Check no -1 values in preferences
grep '"-1"' outputs/sample_alt_voters/data/environment/uniform/persona_no_context/rep4/preferences.json
grep '"-1"' outputs/sample_alt_voters/data/trust/uniform/persona_no_context/rep7/preferences.json

# Check mini_rep results have no errors
grep '"error"' outputs/sample_alt_voters/data/environment/uniform/persona_no_context/rep4/mini_rep*/results.json
grep '"error"' outputs/sample_alt_voters/data/trust/uniform/persona_no_context/rep7/mini_rep*/results.json
# All should return empty (no matches)
```

---

## Open Question: Corrupted Voters in GPT Double Star + Rankings

**TODO:** Need to investigate what happens to corrupted voters for the `chatgpt_double_star_rankings` method.

This method uses the preference rankings to compute insertion positions for a generated consensus statement. If a voter has all -1 rankings:

1. **How does the insertion position calculation handle -1 values?**
   - Does it skip the voter?
   - Does it treat -1 as a valid position (causing errors)?
   - Does it default to some position?

2. **Are the existing `chatgpt_double_star_rankings` results in results.json affected?**
   - The results show `"epsilon": null` which suggests the lookup failed
   - But the `insertion_positions` array may contain invalid data

3. **Should we re-run the double star generation or just recompute epsilon?**
   - If the new_statement was generated with corrupted preference data, it may need regeneration
   - If only the epsilon lookup failed, we just need to recompute

**Action needed:** Review the double star ranking code in `src/experiment_utils/voting_methods.py` to understand how it handles -1 values before implementing the fix.
