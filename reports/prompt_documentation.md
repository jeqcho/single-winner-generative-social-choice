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
2. [API Metadata Schema](#api-metadata-schema)
3. [Phase 1: Statement Generation](#phase-1-statement-generation)
   - [Alt1: Persona, No Context](#alt1-persona-no-context)
   - [Alt2: Persona + Context](#alt2-persona--context)
   - [Alt3: No Persona, With Context](#alt3-no-persona-with-context)
   - [Alt4: No Persona, No Context](#alt4-no-persona-no-context)
4. [Phase 2: Preference Building](#phase-2-preference-building)
   - [Iterative Ranking (Top-K/Bottom-K)](#iterative-ranking-top-kbottom-k)
   - [Single-Call Ranking (Alternative)](#single-call-ranking-alternative)
5. [Phase 3: Winner Selection](#phase-3-winner-selection)
   - [GPT: Select from P Alternatives](#gpt-select-from-p-alternatives)
   - [GPT+Rank: Select with Rankings](#gptrank-select-with-rankings)
   - [GPT+Pers: Select with Personas](#gptpers-select-with-personas)
   - [GPT\*: Select from All 100](#gpt-select-from-all-100)
   - [GPT\*\*: Generate New Statement](#gpt-generate-new-statement)
   - [GPT\*\*\*: Blind Bridging Generation](#gpt-blind-bridging-generation)
6. [Epsilon Computation](#epsilon-computation)
   - [GPT\*\* Statement Insertion](#gpt-statement-insertion)
   - [GPT\*\*\* Statement Insertion](#gpt-statement-insertion-1)
   - [Insertion Prompt (Shared)](#insertion-prompt-shared)

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

## API Metadata Schema

All API calls include metadata for tracking, debugging, and cost analysis via the OpenAI dashboard. Metadata supports up to 16 key-value pairs (keys max 64 chars, values max 512 chars).

**Source**: `src/experiment_utils/config.py` - `build_api_metadata()` function

### Core Keys (Always Present)

| Key | Description | Example Values |
|-----|-------------|----------------|
| `project` | Fixed project identifier | `"gsc_single_winner"` |
| `run_id` | Timestamp-based run identifier for grouping calls | `"20260128_143052"` |
| `phase` | Experiment phase | `"1_statement_gen"`, `"2_preference"`, `"3_selection"`, `"4_insertion"` |
| `component` | Code-level identifier (function/module) | See component values below |

### Contextual Keys (Present When Applicable)

| Key | Description | Example Values |
|-----|-------------|----------------|
| `topic` | Topic slug being processed | `"abortion"`, `"gun_safety"`, `"healthcare"` |
| `voter_dist` | Voter distribution type | `"uniform"`, `"clustered"` |
| `alt_dist` | Alternative distribution type | `"persona_no_context"`, `"persona_context"`, `"no_persona_context"`, `"no_persona_no_context"` |
| `method` | Voting method name (Phase 3/4) | `"chatgpt"`, `"chatgpt_rankings"`, `"chatgpt_star"`, `"chatgpt_double_star"` |
| `rep` | Replication number | `"0"`, `"1"`, ..., `"9"` |
| `mini_rep` | Mini-rep index within a rep (Phase 3) | `"0"`, `"1"`, ..., `"4"` |
| `voter_idx` | Voter index for ranking/insertion | `"0"`, `"1"`, ..., `"99"` |
| `round` | Round number for iterative ranking | `"1"`, `"2"`, ..., `"5"` |

### Component Values by Phase

**Phase 1 - Statement Generation:**
- `alt1_persona_no_context` - Alt1 persona-based generation without context
- `alt2_persona_context` - Alt2 persona-based generation with context
- `alt3_no_persona_context` - Alt3 verbalized sampling with context
- `alt4_no_persona_no_context` - Alt4 blind verbalized sampling

**Phase 2 - Preference Building:**
- `iterative_ranking_topbottom` - Top-K/Bottom-K selection (rounds 1-4)
- `iterative_ranking_final` - Final ranking of remaining statements (round 5)

**Phase 3 - Winner Selection:**
- `gpt_select` - GPT baseline selection
- `gpt_select_rankings` - GPT+Rank selection
- `gpt_select_personas` - GPT+Pers selection
- `gpt_star_select` - GPT\* selection from all 100
- `gpt_star_select_rankings` - GPT\*+Rank selection
- `gpt_star_select_personas` - GPT\*+Pers selection
- `gpt_double_star_gen` - GPT\*\* statement generation
- `gpt_double_star_gen_rankings` - GPT\*\*+Rank generation
- `gpt_double_star_gen_personas` - GPT\*\*+Pers generation
- `gpt_triple_star_gen` - GPT\*\*\* blind bridging generation

**Phase 4 - Statement Insertion:**
- `statement_insertion` - Inserting new statement into voter ranking

### Dashboard Query Examples

With this schema, you can filter in the OpenAI dashboard by:
- All calls for a specific run: `run_id = "20260128_143052"`
- All Phase 2 calls: `phase = "2_preference"`
- All calls for a topic: `topic = "abortion"`
- All GPT\*\* voting method calls (generation + insertion): `method = "chatgpt_double_star"`
- Just GPT\*\* generation calls (not insertion): `component = "gpt_double_star_gen"`
- All insertion calls for voter 42: `component = "statement_insertion" AND voter_idx = "42"`

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
| Est. Cost per Topic | ~$0.34 |

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
| Est. Cost per Topic | ~$5.15 |

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
| Est. Cost per Topic | ~$1.30 |

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
| Est. Cost per Topic | ~$0.25 |

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
| Avg Output Tokens | ~770 (includes reasoning) |
| Est. Cost per Topic | ~$95.20 |

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

> **Note on Data Presentation**: Phase 3 methods use K=20 sampled voters and P=20 sampled alternatives per mini-rep:
> - **Rankings (+Rank variants)**: K=20 voters shown with full preference rankings over P=20 alternatives
> - **Personas (+Pers variants)**: K=20 voters shown with filtered personas (7 key fields: age, sex, race, education, occupation, political views, religion)

### GPT: Select from P Alternatives

**Purpose**: Select the best consensus statement from P subsampled alternatives.

**File**: `src/experiment_utils/voting_methods.py`

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5.2` |
| Reasoning Effort | `none` |
| Temperature | `1.0` |
| API Calls per Topic | 240 (1 per mini-rep × 240 mini-reps) |
| Est. Cost per Topic | ~$1.50 |

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
| Est. Cost per Topic | ~$1.79 |

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

Where `{rankings_text}` is formatted as (K=20 sampled voters, full rankings over P=20 alternatives):
```
Voter 1: 5 > 3 > 1 > 8 > 2 > 0 > 7 > 6 > 4 > 9 > 10 > 11 > 12 > 13 > 14 > 15 > 16 > 17 > 18 > 19
Voter 2: 3 > 5 > 8 > 1 > 2 > 0 > 7 > 6 > 4 > 9 > 10 > 11 > 12 > 13 > 14 > 15 > 16 > 17 > 18 > 19
...
Voter 20: 8 > 3 > 5 > 1 > 2 > 0 > 7 > 6 > 4 > 9 > 10 > 11 > 12 > 13 > 14 > 15 > 16 > 17 > 18 > 19
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
| Est. Cost per Topic | ~$2.88 |

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

Where `{personas_text}` is formatted as (K=20 sampled voters, filtered to 7 key fields):
```
Voter 1: age: 53
sex: Male
race: White alone
education: Master's degree
occupation: MGR-Education And Childcare Administrators
political views: Liberal
religion: Protestant

Voter 2: age: 54
sex: Male
race: White alone
education: Master's degree
occupation: ENT-News Analysts, Reporters, And Journalists
political views: Democrat
religion: Catholic

...

Voter 20: ...
```

> **Note**: K=20 voters are sampled per mini-rep. Personas are filtered to 7 key demographic fields: age, sex, race, education, occupation, political views, religion.

---

### GPT\*: Select from All 100

**Purpose**: Select from all 100 alternatives with topic context.

**File**: `src/experiment_utils/voting_methods.py`

#### Cost Breakdown by Variant

| Variant | Calls | Input Tokens | Output Tokens | Cost |
|---------|------:|-------------:|--------------:|-----:|
| GPT\* (base) | 240 | ~12,600 | ~30 | ~$5.29 |
| GPT\*+Rank | 240 | ~13,100 | ~30 | ~$5.50 |
| GPT\*+Pers | 240 | ~13,800 | ~30 | ~$5.80 |
| **Total** | **720** | | | **~$16.59** |

*Note: Uses K=20 sampled voters. Rankings ~500 tokens. Filtered personas ~1,200 tokens.*

#### Common Parameters

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5.2` |
| Reasoning Effort | `none` |
| Temperature | `1.0` |

**System Prompt**:
```
You are a helpful assistant that selects consensus statements. Return ONLY valid JSON.
```

**User Prompt (GPT\* base)**:
```
Topic: {topic}

A group of participants submitted the following {n_all} statements on this topic:

{all_text}

Select the statement that would best serve as a consensus or bridging position - one that:
- Engages substantively with the topic
- Could be acceptable to participants with diverse viewpoints
- Avoids extreme or polarizing framing

Return your choice as JSON: {"selected_statement_index": <index>}
Where the value is the index (0-{n_all-1}) of the statement you select.
```

> **Note**: All 100 statements are shown with full text in `{all_text}`. Topic provides context for evaluating relevance.

**GPT\*+Rank** adds preference rankings: `{rankings_text}` (K=20 sampled voters, full rankings)

**GPT\*+Pers** adds voter personas: `{personas_text}` (K=20 sampled voters, filtered to 7 key demographic fields)

---

### GPT\*\*: Generate New Statement

**Purpose**: Generate a NEW consensus statement (given P alternatives as context).

**File**: `src/experiment_utils/voting_methods.py`

#### Cost Breakdown by Variant

| Variant | Calls | Input Tokens | Output Tokens | Cost |
|---------|------:|-------------:|--------------:|-----:|
| GPT\*\* (base) | 240 | ~3,329 | ~151 | ~$1.91 |
| GPT\*\*+Rank | 240 | ~3,800 | ~151 | ~$2.10 |
| GPT\*\*+Pers | 240 | ~4,500 | ~151 | ~$2.20 |
| **Total** | **720** | | | **~$6.21** |

*Note: Uses K=20 sampled voters. Rankings ~500 tokens. Filtered personas ~1,200 tokens.*

#### Common Parameters

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5.2` |
| Reasoning Effort | `none` |
| Temperature | `1.0` |

**System Prompt**:
```
You are a helpful assistant that generates consensus statements. Return ONLY valid JSON.
```

**User Prompt (GPT\*\* base)**:
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

**GPT\*\*+Rank** adds preference rankings: `{rankings_text}` (all voters, full rankings)

**GPT\*\*+Pers** adds voter personas: `{personas_text}` (all voters, filtered to 7 key demographic fields)

---

### GPT\*\*\*: Blind Bridging Generation

**Purpose**: Generate a bridging statement given ONLY the topic (no existing statements, no context).

**File**: `src/experiment_utils/voting_methods.py`

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5.2` |
| Reasoning Effort | `none` |
| Temperature | `1.0` |
| API Calls per Topic | 48 (1 per rep × 48 reps) |
| Est. Cost per Topic | ~$0.12 |

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

Epsilon computation requires inserting newly generated statements into existing voter rankings. This is done separately for GPT\*\* and GPT\*\*\* generated statements.

**File**: `src/experiment_utils/statement_insertion.py`

### GPT\*\* Statement Insertion

**Purpose**: Insert GPT\*\*-generated statements into existing voter rankings to compute epsilon.

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5-mini` |
| Reasoning Effort | `low` |
| Temperature | `1.0` |
| GPT\*\* Statements | 720 (3 variants × 240 mini-reps) |
| API Calls per Topic | 72,000 (720 statements × 100 voters) |
| Est. Cost per Topic | ~$102.12 |

---

### GPT\*\*\* Statement Insertion

**Purpose**: Insert GPT\*\*\*-generated statements into existing voter rankings to compute epsilon.

| Parameter | Value |
|-----------|-------|
| Model | `gpt-5-mini` |
| Reasoning Effort | `low` |
| Temperature | `1.0` |
| GPT\*\*\* Statements | 48 (1 per rep × 48 reps) |
| API Calls per Topic | 4,800 (48 statements × 100 voters) |
| Est. Cost per Topic | ~$6.81 |

---

### Insertion Prompt (Shared)

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

> **Note**: The insertion prompt includes the full text of all 100 ranked statements, which is necessary for accurate preference insertion.

---

## API Call Volume and Cost Summary

> **Note**: Cost estimates below reflect Phase 3 methods with improved prompts:
> - **+Rank variants**: Full rankings for K=20 sampled voters over P=20 statements (~500 tokens)
> - **+Pers variants**: Filtered personas for K=20 sampled voters (~1,200 tokens)
> - **GPT\* methods**: All 100 statements with topic context (~12,600 tokens - sample statements removed)
>
> Run `python scripts/estimate_costs.py` for exact calculations.

### Calls per Topic

| Component | Model | Reasoning | Calls | Input Tokens | Output Tokens | Cost |
|-----------|-------|-----------|------:|-------------:|--------------:|-----:|
| **Statement Generation** | | | | | | |
| Alt1 Statement Gen | gpt-5-mini | minimal | 815 | ~468 | ~151 | $0.34 |
| Alt2 Statement Gen | gpt-5-mini | minimal | 1,200 | ~15,964 | ~151 | $5.15 |
| Alt3 Statement Gen | gpt-5-mini | minimal | 240 | ~15,656 | ~757 | $1.30 |
| Alt4 Statement Gen | gpt-5-mini | minimal | 163 | ~160 | ~757 | $0.25 |
| *Subtotal* | | | *2,418* | | | *$7.04* |
| **Preference Building** | | | | | | |
| Iterative Ranking | gpt-5-mini | low | 24,000 | ~9,706 | ~770 | $95.20 |
| *Subtotal* | | | *24,000* | | | *$95.20* |
| **GPT Selection** | | | | | | |
| GPT | gpt-5.2 | none | 240 | ~3,329 | ~30 | $1.50 |
| GPT+Rank | gpt-5.2 | none | 240 | ~4,000 | ~30 | ~$1.68 |
| GPT+Pers | gpt-5.2 | none | 240 | ~4,700 | ~30 | ~$1.97 |
| *Subtotal* | | | *720* | | | *~$5.15* |
| **GPT\* Selection** | | | | | | |
| GPT\* | gpt-5.2 | none | 240 | ~12,600 | ~30 | ~$5.29 |
| GPT\*+Rank | gpt-5.2 | none | 240 | ~13,100 | ~30 | ~$5.50 |
| GPT\*+Pers | gpt-5.2 | none | 240 | ~13,800 | ~30 | ~$5.80 |
| *Subtotal* | | | *720* | | | *~$16.59* |
| **GPT\*\* Generation** | | | | | | |
| GPT\*\* | gpt-5.2 | none | 240 | ~3,329 | ~151 | $1.91 |
| GPT\*\*+Rank | gpt-5.2 | none | 240 | ~3,800 | ~151 | ~$2.10 |
| GPT\*\*+Pers | gpt-5.2 | none | 240 | ~4,500 | ~151 | ~$2.20 |
| *Subtotal* | | | *720* | | | *~$6.21* |
| **GPT\*\*\* Generation** | | | | | | |
| GPT\*\*\* | gpt-5.2 | none | 48 | ~160 | ~151 | $0.12 |
| *Subtotal* | | | *48* | | | *$0.12* |
| **GPT\*\* Insertion** | | | | | | |
| GPT\*\* Insertion | gpt-5-mini | low | 72,000 | ~5,513 | ~20 | $102.12 |
| *Subtotal* | | | *72,000* | | | *$102.12* |
| **GPT\*\*\* Insertion** | | | | | | |
| GPT\*\*\* Insertion | gpt-5-mini | low | 4,800 | ~5,513 | ~20 | $6.81 |
| *Subtotal* | | | *4,800* | | | *$6.81* |
| | | | | | | |
| **GRAND TOTAL** | | | **~105,500** | | | **~$239** |

### Cost Summary

| Metric | Value |
|--------|------:|
| Total API calls per topic | ~105,500 |
| Total input tokens per topic | ~688M |
| Total output tokens per topic | ~21M |
| **Cost per topic** | **~$239** |
| **Cost for all 13 topics** | **~$3,107** |

*Note: GPT\* prompt improved to include topic and remove redundant sample statements.*

### Cost by Model

| Model | Input Tokens | Output Tokens | Input Cost | Output Cost | Total Cost |
|-------|-------------:|--------------:|-----------:|------------:|-----------:|
| gpt-5-mini | ~680M | ~21M | ~$170 | ~$42 | ~$212 |
| gpt-5.2 | ~18M | ~181K | ~$31 | ~$2.53 | ~$34 |

*Note: gpt-5.2 cost increased ~$8/topic due to GPT\* showing full statement text.*

### Pricing Reference (per 1M tokens)

| Model | Input | Cached Input | Output |
|-------|------:|-------------:|-------:|
| gpt-5.2 | $1.75 | $0.175 | $14.00 |
| gpt-5-mini | $0.25 | $0.025 | $2.00 |

*Note: Token estimates based on actual persona (~308 tokens full, ~60 tokens filtered) and statement (~151 tokens avg) data. Costs assume no caching.*

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

*Generated from codebase analysis on 2026-01-27. Truncations removed on 2026-01-28.*
