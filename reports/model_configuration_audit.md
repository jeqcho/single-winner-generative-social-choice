# Model Configuration Audit Report

**Date:** January 27, 2026  
**Auditor:** Automated Code Review

## Summary

This report documents discrepancies between the README.md documentation and the actual model/reasoning effort configuration in the codebase. One **critical bug** was identified that would cause a runtime error.

---

## Findings

### 1. CRITICAL BUG: Undefined Variable in `persona_context.py`

**File:** `src/sample_alt_voters/alternative_generators/persona_context.py`  
**Severity:** Critical (code will crash if executed)

The file imports `STATEMENT_MODEL` and `STATEMENT_REASONING` on line 25:

```python
from src.experiment_utils.config import STATEMENT_MODEL, STATEMENT_REASONING
```

But on line 96, it uses `GENERATOR_MODEL` which is **not defined or imported**:

```python
response = client.responses.create(
    model=GENERATOR_MODEL,  # <-- NameError: undefined variable
    input=[...],
    ...
)
```

**Expected Fix:** Change `GENERATOR_MODEL` to `STATEMENT_MODEL` on line 96.

---

### 2. README Model Configuration Section is Incorrect

**File:** `README.md`, lines 123-127  
**Severity:** Documentation error

The README states:

> ### Model Configuration
> 
> All model settings are centralized in `src/experiment_utils/config.py`:
> 
> - **GENERATOR_MODEL** (`gpt-5.2`, reasoning=none): Used for all content generation tasks (statement generation, GPT selection/generation)
> - **RANKING_MODEL** (`gpt-5-mini`, reasoning=low): Used for all preference/ranking tasks (iterative ranking, epsilon insertion)

**Issues:**

1. **`GENERATOR_MODEL` doesn't exist** - The actual code uses two separate constants:
   - `STATEMENT_MODEL` for statement generation
   - `GENERATIVE_VOTING_MODEL` for GPT voting methods

2. **Statement generation uses different settings** - The README incorrectly implies statement generation uses `gpt-5.2` with `reasoning=none`. The actual code uses `gpt-5-mini` with `reasoning=minimal`.

**Actual Code (from `src/experiment_utils/config.py`):**

```python
# STATEMENT_MODEL: Used for statement/alternative generation (Phase 1)
STATEMENT_MODEL = "gpt-5-mini"
STATEMENT_REASONING = "minimal"

# GENERATIVE_VOTING_MODEL: Used for GPT-based voting methods (Phase 3)
GENERATIVE_VOTING_MODEL = "gpt-5.2"
GENERATIVE_VOTING_REASONING = "none"

# RANKING_MODEL: Used for all preference/ranking tasks
RANKING_MODEL = "gpt-5-mini"
RANKING_REASONING = "low"
```

---

### 3. Inconsistency Between README Sections

The README has internally inconsistent information:

**Flowchart (lines 55-76)** correctly shows:
- Statement Generation: `gpt-5-mini`, `reasoning=minimal`
- Iterative Ranking: `gpt-5-mini`, `reasoning=low`
- GPT/GPT\*/GPT\*\*/GPT\*\*\* Methods: `gpt-5.2`, `reasoning=none`
- Epsilon via Insertion: `gpt-5-mini`, `reasoning=low`

**Model Configuration section (lines 123-127)** incorrectly shows:
- `GENERATOR_MODEL` (`gpt-5.2`, `reasoning=none`) for "all content generation"
- `RANKING_MODEL` (`gpt-5-mini`, `reasoning=low`) for ranking tasks

The flowchart is **correct**. The Model Configuration text section is **incorrect**.

---

## Actual Model Configuration (from code)

| Task | Model | Reasoning Effort | Source Variable |
|------|-------|------------------|-----------------|
| Statement Generation (Alt1-4) | `gpt-5-mini` | `minimal` | `STATEMENT_MODEL`, `STATEMENT_REASONING` |
| Iterative Ranking (Phase 2) | `gpt-5-mini` | `low` | `RANKING_MODEL`, `RANKING_REASONING` |
| GPT/GPT\* Selection | `gpt-5.2` | `none` | `GENERATIVE_VOTING_MODEL`, `GENERATIVE_VOTING_REASONING` |
| GPT\*\*/GPT\*\*\* Generation | `gpt-5.2` | `none` | `GENERATIVE_VOTING_MODEL`, `GENERATIVE_VOTING_REASONING` |
| Epsilon Insertion | `gpt-5-mini` | `low` | `RANKING_MODEL`, `RANKING_REASONING` |

---

## Verified Correct Implementations

The following files correctly import and use the model constants:

| File | Task | Correctly Uses |
|------|------|----------------|
| `persona_no_context.py` | Alt1 statements | `STATEMENT_MODEL`, `STATEMENT_REASONING` |
| `no_persona_context.py` | Alt3 statements | `STATEMENT_MODEL`, `STATEMENT_REASONING` |
| `no_persona_no_context.py` | Alt4 statements | `STATEMENT_MODEL`, `STATEMENT_REASONING` |
| `voting_methods.py` | GPT methods | `GENERATIVE_VOTING_MODEL`, `GENERATIVE_VOTING_REASONING` |
| `single_call_ranking.py` | Ranking/Insertion | `RANKING_MODEL`, `RANKING_REASONING` |
| `iterative_ranking.py` | Preference building | `RANKING_MODEL`, `RANKING_REASONING` |
| `run_experiment.py` | Orchestration | `RANKING_REASONING` |
| `preference_builder_iterative.py` | Preference building | `RANKING_REASONING` |

---

## Recommendations

### Critical (Must Fix)

1. **Fix `persona_context.py`**: Replace `GENERATOR_MODEL` with `STATEMENT_MODEL` on line 96.

### Documentation Updates

2. **Update README Model Configuration section** (lines 123-127) to accurately reflect the three model constants:

```markdown
### Model Configuration

All model settings are centralized in `src/experiment_utils/config.py`:

- **STATEMENT_MODEL** (`gpt-5-mini`, reasoning=minimal): Used for statement/alternative generation (Phase 1)
- **GENERATIVE_VOTING_MODEL** (`gpt-5.2`, reasoning=none): Used for GPT-based voting methods (Phase 3 selection/generation)
- **RANKING_MODEL** (`gpt-5-mini`, reasoning=low): Used for all preference/ranking tasks (iterative ranking, epsilon insertion)
```

---

## Appendix: Reasoning Effort Support by Model

From the code comments in `src/experiment_utils/config.py`:

```
gpt-5-mini: minimal, low, medium, high (NOT none or xhigh)
gpt-5.2:    none, low, medium, high, xhigh (NOT minimal)
```

The current configuration correctly respects these constraints:
- `gpt-5-mini` uses `minimal` or `low` (both supported)
- `gpt-5.2` uses `none` (supported)
