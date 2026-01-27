# Bug Report: GPT\*\* Epsilon Computed Over Different Population Size

## Summary

GPT\*\* (double star) epsilon values are computed over 20 voters, while traditional methods and GPT\*\*\* use 100 voters. This makes GPT\*\* epsilon values not directly comparable to other methods.

## Severity

**High** - This affects the validity of comparisons between GPT\*\* and other voting methods in the experimental results.

## Affected Code

- `src/sample_alt_voters/fix_star_epsilons.py` (lines 134-145)
- `src/experiment_utils/voting_methods.py` (`insert_new_statement_into_rankings`)

## Description

### Expected Behavior

All epsilon values should be computed over the same voter population (n=100) to ensure fair comparison across methods.

### Actual Behavior

| Method | Voters for Epsilon Calculation | Source |
|--------|-------------------------------|--------|
| Traditional (Schulze, Borda, etc.) | 100 | Precomputed on full 100×100 matrix |
| GPT, GPT\* | 100 | Lookup from precomputed epsilons |
| **GPT\*\*** | **20** | Computed on 101×20 matrix |
| GPT\*\*\* | 100 | Computed on 101×100 matrix |

### Root Cause

In `fix_star_epsilons.py`, the `fix_gpt_double_star_epsilons` function:

1. Gets only the 20 mini-rep voter indices (line 182: `voter_indices = data['voter_indices']`)
2. Inserts the new statement into only these 20 voters' rankings (lines 134-142)
3. Computes epsilon on the resulting 101×20 matrix (line 145)

```python
# fix_star_epsilons.py lines 134-145
updated_prefs = insert_new_statement_into_rankings(
    new_statement=new_statement,
    all_statements=all_statements,
    voter_personas=selected_personas,  # Only 20 personas
    voter_indices=voter_indices,        # Only 20 indices
    full_preferences=full_preferences,
    topic=topic_question,
    openai_client=openai_client,
)

# Epsilon computed over 20 voters, not 100
epsilon = compute_epsilon_for_new_statement(updated_prefs, len(all_statements))
```

### Why This Matters

The critical epsilon formula depends on `n` (number of voters):

```
ε* = (S_a / (m × n)) - 1.0
```

Where `S_a` is computed from a max-flow network that scales with `n`. Computing epsilon with n=20 vs n=100 produces values that are not on the same scale, making direct comparison invalid.

### Code Flow Comparison

```
Traditional: select from 20 alts → map winner to full index → lookup epsilon (n=100)

GPT**:       generate new stmt → insert into 20 voters → compute epsilon (n=20)  ← BUG

GPT***:      generate new stmt → insert into 100 voters → compute epsilon (n=100)
```

## Evidence

### Mini-rep results.json structure

```json
{
  "mini_rep_id": 0,
  "voter_indices": [81, 14, 3, 94, ...],  // 20 indices into the 100-voter pool
  ...
}
```

### insert_new_statement_into_rankings output

The function returns a matrix of shape `[n_ranks][n_voters]` where `n_voters = len(voter_indices)`:

```python
# voting_methods.py line 859
n_voters = len(voter_indices)  # This is 20 for GPT**, 100 for GPT***
```

## Proposed Fix

Modify `fix_star_epsilons.py` to insert the new statement into **all 100 voters'** rankings instead of just the 20 mini-rep voters:

```python
def fix_gpt_double_star_epsilons(
    results: Dict,
    all_statements: List[Dict],
    full_preferences: List[List[str]],
    voter_indices: List[int],  # Keep for persona selection context
    all_personas: List[str],
    global_voter_indices: List[int],
    topic_question: str,
    openai_client: OpenAI,
) -> int:
    ...
    # Instead of using mini-rep voter_indices, use ALL 100 voters
    all_voter_indices = list(range(100))
    all_selected_personas = [all_personas[idx] for idx in global_voter_indices]
    
    updated_prefs = insert_new_statement_into_rankings(
        new_statement=new_statement,
        all_statements=all_statements,
        voter_personas=all_selected_personas,  # All 100 personas
        voter_indices=all_voter_indices,        # All 100 indices
        full_preferences=full_preferences,
        topic=topic_question,
        openai_client=openai_client,
    )
    ...
```

### Impact of Fix

- API calls for GPT\*\* insertion increase from 20 to 100 per method (×5)
- Total API calls per topic increase
- Epsilon values become comparable across all methods

## Alternative Fix

If the 20-voter epsilon is intentional (e.g., to measure consensus within the mini-rep sample), then:

1. Document this explicitly in README
2. Consider computing traditional method epsilons on the 20-voter subsample as well for fair comparison
3. Add a separate metric for "mini-rep epsilon" vs "full-rep epsilon"

## Files to Update

1. `src/sample_alt_voters/fix_star_epsilons.py` - Main fix
2. `README.md` - Update API call counts and documentation
3. Potentially re-run experiments to get corrected epsilon values

## Related

- GPT\*\*\* correctly uses all 100 voters (see `src/experiment_utils/voting_methods.py` `run_chatgpt_triple_star`)
- The precomputed epsilons in `precomputed_epsilons.json` are computed over 100 voters
