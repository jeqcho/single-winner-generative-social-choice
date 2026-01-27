# Prompt Documentation

This document details all LLM prompts and API requests made in the Single Winner Generative Social Choice project.

## Overview

The project uses OpenAI's API with two primary models:
- **gpt-5-mini**: Used for statement generation and ranking tasks
- **gpt-5.2**: Used for generative voting methods

All API calls use the `client.responses.create()` method.

---

## Table of Contents

1. [Model Configuration Summary](#model-configuration-summary)
2. [Phase 1: Statement Generation](#phase-1-statement-generation)
   - [Alt1: Persona, No Context](#alt1-persona-no-context)
   - [Alt2: Persona + Context](#alt2-persona--context)
   - [Alt3: No Persona, With Context](#alt3-no-persona-with-context)
   - [Alt4: No Persona, No Context](#alt4-no-persona-no-context)
3. [Phase 2: Preference Building](#phase-2-preference-building)
   - [Iterative Ranking (Top-K/Bottom-K)](#iterative-ranking-top-kbottom-k)
   - [Single-Call Ranking (Alternative)](#single-call-ranking-alternative)
4. [Phase 3: Winner Selection](#phase-3-winner-selection)
   - [GPT: Select from P Alternatives](#gpt-select-from-p-alternatives)
   - [GPT+Rank: Select with Rankings](#gptrank-select-with-rankings)
   - [GPT+Pers: Select with Personas](#gptpers-select-with-personas)
   - [GPT\*: Select from All 100](#gpt-select-from-all-100)
   - [GPT\*\*: Generate New Statement](#gpt-generate-new-statement)
   - [GPT\*\*\*: Blind Bridging Generation](#gpt-blind-bridging-generation)
5. [Epsilon Computation](#epsilon-computation)
   - [Statement Insertion into Rankings](#statement-insertion-into-rankings)

---

## Model Configuration Summary

| Configuration | Model | Reasoning Effort | Temperature | Purpose |
|--------------|-------|------------------|-------------|---------|
| `STATEMENT_MODEL` | gpt-5-mini | minimal | 1.0 | Statement/alternative generation (Phase 1) |
| `RANKING_MODEL` | gpt-5-mini | low | 1.0 | Preference ranking, epsilon insertion |
| `GENERATIVE_VOTING_MODEL` | gpt-5.2 | none | 1.0 | GPT voting methods (Phase 3) |

**Source**: `src/experiment_utils/config.py`

```python
# Statement generation
STATEMENT_MODEL = "gpt-5-mini"
STATEMENT_REASONING = "minimal"

# Preference ranking
RANKING_MODEL = "gpt-5-mini"
RANKING_REASONING = "low"

# Generative voting
GENERATIVE_VOTING_MODEL = "gpt-5.2"
GENERATIVE_VOTING_REASONING = "none"

# Temperature (all tasks)
TEMPERATURE = 1.0
```

---

## Phase 1: Statement Generation

### Alt1: Persona, No Context

**Purpose**: Generate bridging statements where each persona writes based on their characteristics alone, without any context from existing statements.

**File**: `src/sample_alt_voters/alternative_generators/persona_no_context.py`

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5-mini` |
| Reasoning Effort | `minimal` |
| Temperature | `1.0` |
| API Calls per Topic | 815 (one per persona) |

**System Prompt**:
```
You are writing a statement that reflects your perspective on a topic.
```

**User Prompt**:
```
You are a person with the following characteristics:
{persona}

Topic: "{topic}"

Write a bridging statement expressing your views on this topic. Your statement should:
- Reflect your background, values, and life experiences
- Aim to find common ground or bridge different viewpoints
- Be 2-4 sentences long
- NOT write in first-person
- NOT explicitly reference your identity or demographics (avoid "As a [X]...")

Write only the statement:
```

---

### Alt2: Persona + Context

**Purpose**: Generate bridging statements where each persona reads 100 existing statements first, then writes a new bridging statement synthesizing what they read.

**File**: `src/sample_alt_voters/alternative_generators/persona_context.py`

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5-mini` |
| Reasoning Effort | `minimal` |
| Temperature | `1.0` |
| API Calls per Topic | 1,200 (100 per rep × 12 reps) |

**System Prompt**:
```
You are writing a statement that reflects your perspective on a topic.
```

**User Prompt**:
```
You are a person with the following characteristics:
{persona}

Topic: "{topic}"

Here are 100 statements from people with diverse perspectives on this topic:

{statements_list}

Write a NEW bridging statement expressing your views on this topic. Your statement should:
- Reflect your background, values, and life experiences
- Synthesize key themes you observed across the discussion
- Aim to find common ground or bridge different viewpoints
- Be 2-4 sentences long
- NOT write in first-person
- NOT explicitly reference your identity or demographics (avoid "As a [X]...")
- Be self-contained (do not reference "the statements above" or "other people")

Write only the statement:
```

---

### Alt3: No Persona, With Context

**Purpose**: Generate bridging statements without a persona but after reading 100 existing statements, using verbalized sampling for diversity.

**File**: `src/sample_alt_voters/alternative_generators/no_persona_context.py`

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5-mini` |
| Reasoning Effort | `minimal` |
| Temperature | `1.0` |
| API Calls per Topic | 240 (20 per rep × 12 reps) |

**System Prompt** (Verbalized Sampling):
```
You are a helpful assistant that generates statements. Return only the statement text, no JSON or additional commentary. For each query, please generate a set of five possible responses, each within a separate <response> tag. Each <response> must include a <text> and a numeric <probability>.
Please sample at random from the tails of the distribution, such that the probability of each response is less than 0.10.
```

**User Prompt**:
```
Topic: "{topic}"

Here are 100 statements from people with diverse perspectives on this topic:

{statements_list}

Write a NEW bridging statement on this topic. Your statement should:
- Synthesize key themes across the different viewpoints
- Aim to find common ground or bridge different viewpoints
- Be 2-4 sentences long
- Be self-contained (do not reference "the statements above" or "other people")
```

---

### Alt4: No Persona, No Context

**Purpose**: Generate bridging statements without any persona or context, using verbalized sampling for diversity.

**File**: `src/sample_alt_voters/alternative_generators/no_persona_no_context.py`

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5-mini` |
| Reasoning Effort | `minimal` |
| Temperature | `1.0` |
| API Calls per Topic | 163 (5 statements per call × 163 = 815 statements) |

**System Prompt** (Verbalized Sampling):
```
You are a helpful assistant that generates statements. Return only the statement text, no JSON or additional commentary. For each query, please generate a set of five possible responses, each within a separate <response> tag. Each <response> must include a <text> and a numeric <probability>.
Please sample at random from the tails of the distribution, such that the probability of each response is less than 0.10.
```

**User Prompt**:
```
Topic: "{topic}"

Write a bridging statement on this topic. Your statement should:
- Aim to find common ground or bridge different viewpoints
- Be 2-4 sentences long
```

---

## Phase 2: Preference Building

### Iterative Ranking (Top-K/Bottom-K)

**Purpose**: Build a full ranking of 100 statements through 5 rounds of iterative selection, using 4-letter hash identifiers to avoid index/rank conflation.

**File**: `src/degeneracy_mitigation/iterative_ranking.py`

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5-mini` |
| Reasoning Effort | `low` |
| Temperature | `1.0` |
| API Calls per Topic | 24,000 (500 per rep × 48 reps) |

**System Prompt**:
```
You are simulating a single, internally consistent person defined by the following persona:
{persona}

You must evaluate each statement solely through the lens of this persona's values, background, beliefs, and preferences.

Your task is to rank statements by preference and return valid JSON only.
Do not include explanations, commentary, or extra text.
```

#### Rounds 1-4: Top-K/Bottom-K Selection

**User Prompt**:
```
Topic: "{topic}"

Here are {n} statements (identified by 4-letter codes):
{stmt_lines}

From these {n} statements, identify:
1. Your TOP {k} most preferred (in order, most preferred first)
2. Your BOTTOM {k} least preferred (in order, least preferred last)

IMPORTANT: Do NOT simply list codes in the order they appear above.
Your preferences should reflect your persona's values and background.

Return JSON: {"top_{k}": ["code1", "code2", ...], "bottom_{k}": ["code1", "code2", ...]}
```

Where `{stmt_lines}` is formatted as:
```
XXXX: "Statement text here..."
YYYY: "Another statement..."
```

#### Round 5: Final Ranking

**User Prompt**:
```
Topic: "{topic}"

Here are {n} statements (identified by 4-letter codes):
{stmt_lines}

Rank ALL of these statements from most to least preferred.

IMPORTANT: Do NOT simply list codes in the order they appear above.
Your preferences should reflect your persona's values and background.

Return JSON: {"ranking": ["most_preferred", "second", ..., "least_preferred"]}
```

---

## Phase 3: Winner Selection

### GPT: Select from P Alternatives

**Purpose**: Select the best consensus statement from P subsampled alternatives.

**File**: `src/experiment_utils/voting_methods.py`

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5.2` |
| Reasoning Effort | `none` |
| Temperature | `1.0` |
| API Calls per Topic | 240 (1 per mini-rep × 240 mini-reps) |

**System Prompt**:
```
You are a helpful assistant that selects consensus statements. Return ONLY valid JSON.
```

**User Prompt**:
```
Here are {n} statements from a discussion:

{statements_text}

Which statement would be the best choice as a consensus/bridging statement?
Consider which one best represents a reasonable middle ground that could satisfy diverse perspectives.

Return your choice as JSON: {"selected_statement_index": <index>}
Where the value is the index (0-{n-1}) of the statement you select.
```

Where `{statements_text}` is formatted as:
```
Statement 0: Statement text here...

Statement 1: Another statement...
```

---

### GPT+Rank: Select with Rankings

**Purpose**: Select the best consensus statement with additional preference ranking information.

**File**: `src/experiment_utils/voting_methods.py`

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5.2` |
| Reasoning Effort | `none` |
| Temperature | `1.0` |
| API Calls per Topic | 240 (1 per mini-rep × 240 mini-reps) |

**System Prompt**:
```
You are a helpful assistant that selects consensus statements. Return ONLY valid JSON.
```

**User Prompt**:
```
Here are {n} statements from a discussion:

{statements_text}

Here are preference rankings from {n_voters} voters:

{rankings_text}

Based on both the statements and the preference rankings, which statement would be the best choice as a consensus/bridging statement?

Return your choice as JSON: {"selected_statement_index": <index>}
Where the value is the index (0-{n-1}) of the statement you select.
```

Where `{rankings_text}` is formatted as:
```
Voter 1: 5 > 3 > 1 > 8 > 2...
Voter 2: 3 > 5 > 8 > 1 > 2...
... and {n_voters - 10} more voters
```

---

### GPT+Pers: Select with Personas

**Purpose**: Select the best consensus statement with additional voter persona information.

**File**: `src/experiment_utils/voting_methods.py`

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5.2` |
| Reasoning Effort | `none` |
| Temperature | `1.0` |
| API Calls per Topic | 240 (1 per mini-rep × 240 mini-reps) |

**System Prompt**:
```
You are a helpful assistant that selects consensus statements. Return ONLY valid JSON.
```

**User Prompt**:
```
Here are {n} statements from a discussion:

{statements_text}

Here are the {n_voters} voters who will be voting on these statements:

{personas_text}

Based on both the statements and the voter personas, which statement would be the best choice as a consensus/bridging statement?

Return your choice as JSON: {"selected_statement_index": <index>}
Where the value is the index (0-{n-1}) of the statement you select.
```

Where `{personas_text}` is formatted as:
```
Voter 1: age: 35, sex: Female, race: White...

Voter 2: age: 52, sex: Male, race: Black...

... and {n_voters - 10} more voters
```

---

### GPT\*: Select from All 100

**Purpose**: Select from all 100 alternatives (given P alternatives as context).

**File**: `src/experiment_utils/voting_methods.py`

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5.2` |
| Reasoning Effort | `none` |
| Temperature | `1.0` |
| API Calls per Topic | 240 (1 per mini-rep × 240 mini-reps) |

**System Prompt**:
```
You are a helpful assistant that selects consensus statements. Return ONLY valid JSON.
```

**User Prompt**:
```
Here are some sample statements from a discussion:

{sample_text}

Below are ALL {n_all} available statements you can choose from:

{all_text}

Which statement (from 0-{n_all-1}) would be the best choice as a consensus/bridging statement?
You may choose any statement, not just the samples shown above.

Return your choice as JSON: {"selected_statement_index": <index>}
```

**Variants**: GPT\*+Rank and GPT\*+Pers follow similar patterns with additional context.

---

### GPT\*\*: Generate New Statement

**Purpose**: Generate a NEW consensus statement (given P alternatives as context).

**File**: `src/experiment_utils/voting_methods.py`

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5.2` |
| Reasoning Effort | `none` |
| Temperature | `1.0` |
| API Calls per Topic | 720 generation + 72,000 insertion = 72,720 total |

**System Prompt**:
```
You are a helpful assistant that generates consensus statements. Return ONLY valid JSON.
```

**User Prompt**:
```
Topic: {topic}

Here are some existing statements from a discussion:

{statements_text}

Generate a NEW statement that could serve as a better consensus/bridging statement.
The statement should:
- Represent a reasonable middle ground that could satisfy diverse perspectives
- Be different from the existing statements but address the same topic
- Be clear and substantive (2-4 sentences)

Return your new statement as JSON: {"new_statement": "<your statement>"}
```

**Variants**:
- **GPT\*\*+Rank**: Adds preference rankings to the prompt
- **GPT\*\*+Pers**: Adds voter personas to the prompt

---

### GPT\*\*\*: Blind Bridging Generation

**Purpose**: Generate a bridging statement given ONLY the topic (no existing statements, no context).

**File**: `src/experiment_utils/voting_methods.py`

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5.2` |
| Reasoning Effort | `none` |
| Temperature | `1.0` |
| API Calls per Topic | 48 generation + 4,800 insertion = 4,848 total |

**System Prompt**:
```
You are a helpful assistant that generates bridging statements. Return ONLY valid JSON.
```

**User Prompt**:
```
Given the topic: "{topic}"

Generate a bridging statement that could serve as a consensus position on this topic.
The statement should:
- Represent a reasonable middle ground that could satisfy diverse perspectives
- Acknowledge different viewpoints while finding common ground
- Be clear and substantive (2-4 sentences)

Return your statement as JSON: {"bridging_statement": "<your statement>"}
```

---

## Epsilon Computation

### Statement Insertion into Rankings

**Purpose**: Insert a newly generated statement (from GPT\*\* or GPT\*\*\*) into existing voter rankings to compute epsilon.

**File**: `src/experiment_utils/statement_insertion.py`

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5-mini` |
| Reasoning Effort | `low` |
| Temperature | `1.0` |
| API Calls per Topic | 76,800 (100 voters × 768 new statements from GPT\*\*/GPT\*\*\*) |

**System Prompt**:
```
You are inserting a new statement into your preference ranking. Return ONLY valid JSON.
```

**User Prompt**:
```
You are a person with the following characteristics:
{persona}

Given the topic: "{topic}"

You previously ranked these statements from most to least preferred:

{ranked_text}

NEW STATEMENT (ID {new_idx}): {new_statement}

Where should this new statement be inserted in your ranking?
- Return 0 to make it your MOST preferred (before rank 1)
- Return {n} to make it your LEAST preferred (after rank {n})
- Return any position 1-{n-1} to insert it between existing ranks

Return JSON: {"insert_position": <number>}
```

Where `{ranked_text}` is formatted as:
```
Rank 1 (ID 5): Statement text here...
Rank 2 (ID 12): Another statement...
...
```

---

## API Call Volume Summary

| Component | Model | Reasoning | Calls per Topic | Purpose |
|-----------|-------|-----------|-----------------|---------|
| Alt1 Statement Gen | gpt-5-mini | minimal | 815 | Generate persona-based statements |
| Alt4 Statement Gen | gpt-5-mini | minimal | 163 | Generate blind statements (5 per call) |
| Preference Building | gpt-5-mini | low | 24,000 | 5 rounds × 100 voters × 48 reps |
| GPT/GPT\* Selection | gpt-5.2 | none | 1,440 | Select consensus (240 mini-reps × 6 methods) |
| GPT\*\* Generation | gpt-5.2 | none | 720 | Generate new statements |
| GPT\*\* Insertion | gpt-5-mini | low | 72,000 | Insert into 100 rankings × 720 |
| GPT\*\*\* Generation | gpt-5.2 | none | 48 | Blind bridging (1 per rep) |
| GPT\*\*\* Insertion | gpt-5-mini | low | 4,800 | Insert into 100 rankings × 48 |

**Total per topic: ~104,000 API calls**

---

## Response Format Notes

1. **Ranking responses** return JSON like `{"top_10": [...], "bottom_10": [...]}` or `{"ranking": [...]}`
2. **Selection responses** return JSON like `{"selected_statement_index": N}`
3. **Generation responses** return JSON like `{"new_statement": "..."}` or `{"bridging_statement": "..."}`
4. **Insertion responses** return JSON like `{"insert_position": N}`
5. **Verbalized sampling** returns XML-style tags:
   ```xml
   <response>
   <text>Statement text here...</text>
   <probability>0.08</probability>
   </response>
   ```

---

## Notes on Reasoning Effort

| Level | Models Supported | Usage |
|-------|------------------|-------|
| `none` | gpt-5.2 only | Generative voting methods (fast, no reasoning needed) |
| `minimal` | gpt-5-mini only | Statement generation (lightweight reasoning) |
| `low` | gpt-5-mini, gpt-5.2 | Ranking tasks (moderate reasoning) |
| `medium` | gpt-5-mini, gpt-5.2 | Optional for complex ranking |
| `high` | gpt-5-mini, gpt-5.2 | Not used in this project |
| `xhigh` | gpt-5.2 only | Not used in this project |

---

*Generated from codebase analysis on 2026-01-27*
