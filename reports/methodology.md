# Experiment Methodology: Single-Winner Generative Social Choice

> This document provides a comprehensive description of the experimental methodology for the
> Single-Winner Generative Social Choice project. It is intended to be fed into a scientific
> diagram generator and therefore includes all procedural details, parameter values,
> data flows, and evaluation criteria.

---

## 1. Research Question

**Can LLMs select or generate consensus statements as well as—or better than—traditional voting methods?**

The experiment compares traditional social choice mechanisms (Schulze, Borda, IRV, Plurality, Veto-by-Consumption) against LLM-based methods that either *select* an existing statement or *generate* a novel statement. The primary evaluation metric is **critical epsilon (epsilon\*)** from the **Proportional Veto Core (PVC)**, where lower epsilon\* indicates better consensus.

---

## 2. High-Level Experiment Flow

The experiment proceeds through six stages, each building on the outputs of the previous:

```
Input Data
  --> Phase 1: Data Generation (Voter Sampling + Statement Generation)
    --> Phase 2: Preference Building (Iterative Ranking --> 100x100 Preference Matrix)
      --> Phase 2.5: Precompute (epsilon* for all 100 alternatives)
        --> Phase 3: Winner Selection (Traditional Methods + LLM Methods)
          --> Phase 4: Evaluation (epsilon* Lookup or Batched Iterative Ranking --> Compare Methods)
```

---

## 3. Input Data

### 3.1 Personas

- **Source**: SynthLabsAI/PERSONA dataset on Hugging Face (gated)
- **Filtering**: Adults only (age >= 18)
- **Count**: 815 adult personas
- **Format**: Free-text persona descriptions including age, sex, race, education, occupation, political views, religion, and other characteristics
- **Storage**: `data/personas/prod/adult.json`

### 3.2 Topics

13 policy discussion topics covering diverse political and social issues:

| # | Topic (Short Name) | Full Question |
|---|-------------------|---------------|
| 1 | Trust | How should we increase the general public's trust in US elections? |
| 2 | Littering | What are the best policies to prevent littering in public spaces? |
| 3 | Campus Free Speech | What are your thoughts on the way university campus administrators should approach the issue of Israel/Gaza demonstrations? |
| 4 | Abortion | What should guide laws concerning abortion? |
| 5 | Gun Safety | What balance should exist between gun safety laws and Second Amendment rights? |
| 6 | Healthcare | What role should the government play in ensuring universal access to healthcare? |
| 7 | Environment | What balance should be struck between environmental protection and economic growth in climate policy? |
| 8 | Immigration | What principles should guide immigration policy and the path to citizenship? |
| 9 | Free Speech | What limits, if any, should exist on free speech regarding hate speech? |
| 10 | Tech Privacy | What responsibilities should tech companies have when collecting and monetizing user data? |
| 11 | AI in Society | What role should artificial intelligence play in society, and how should its risks be governed? |
| 12 | Electoral College | What reforms, if any, should replace or modify the Electoral College? |
| 13 | Policing | What strategies should guide policing to address bias and use-of-force concerns while maintaining public safety? |

---

## 4. Phase 1: Data Generation

### 4.1 Voter Sampling

Two voter distribution strategies determine how 100 voters are drawn from the 815-persona pool per replication.

#### 4.1.1 Uniform Sampling

- **Method**: Sample 100 personas uniformly at random (without replacement) from all 815 adult personas
- **Replications**: 10 per topic per alternative distribution
- **Seeding**: Deterministic; seed = `BASE_SEED (42) + rep_id`

#### 4.1.2 Clustered Sampling (Ideology-Based)

- **Pre-step — Ideology Classification**: Each persona is classified into one of three categories using keyword matching on the persona's ideology field:
  - **progressive\_liberal** (~431 personas): Keywords include "progressive", "liberal", "social justice", "feminist", "egalitarian", "environmentalist", "humanism", etc.
  - **conservative\_traditional** (~255 personas): Keywords include "conservative", "traditional", "libertarian", "fiscal conservatism", "christian values", "family values", "small government", etc.
  - **other** (~129 personas): Does not match either cluster
- **Sampling**: 100 personas sampled from the specified cluster. If cluster size < 100, sampling is with replacement.
- **Replications**: 10 per cluster per topic (so 20 total clustered reps per topic)
- **Seeding**: Same deterministic scheme as uniform

### 4.2 Statement (Alternative) Generation

Four alternative distributions define how the 100 candidate statements are generated per replication. They vary along two axes: **persona conditioning** (yes/no) and **context conditioning** (yes/no).

| Distribution | Persona | Context | Generation Mode | Pool Size |
|-------------|---------|---------|-----------------|-----------|
| Alt1 | Yes | No | Pre-generated | 815 per topic |
| Alt2 | Yes | Yes | Per-rep | 100 per rep |
| Alt3 | No | Yes | Per-rep | 100 per rep |
| Alt4 | No | No | Pre-generated | 815 per topic |

#### 4.2.1 Alt1: Persona, No Context (`persona_no_context`)

- **Description**: Each of the 815 personas generates one statement conditioned on their persona but without seeing other statements.
- **Pre-generated**: One pool of 815 statements per topic, shared across all reps.
- **Sampling per rep**: 100 statements drawn from the pool using context indices.
- **Model**: `gpt-5-mini`, reasoning=`minimal`, temperature=`1.0`
- **System prompt**: "You are writing a statement that reflects your perspective on a topic."
- **User prompt**: Contains the persona description + topic question + instructions to write a bridging statement (2–4 sentences, not first-person, avoid "As a [X]..." phrasing)
- **API calls**: 815 per topic

#### 4.2.2 Alt2: Persona + Context (`persona_context`)

- **Description**: Same 100 personas as Alt1 (per rep) each generate a statement, but now they also see 100 existing statements as context.
- **Per-rep generation**: Generated fresh for each replication because context differs.
- **Model**: `gpt-5-mini`, reasoning=`minimal`, temperature=`1.0`
- **System prompt**: Same as Alt1
- **User prompt**: Persona + topic + a numbered list of 100 context statements + instructions to synthesize themes into a NEW bridging statement (self-contained, no references to "the statements above")
- **API calls**: 100 per rep (parallelized with `MAX_WORKERS=50`)

#### 4.2.3 Alt3: No Persona, With Context (`no_persona_context`)

- **Description**: No persona conditioning; the model sees 100 existing statements and generates new ones using **verbalized sampling** (5 statements per API call).
- **Per-rep generation**: Context differs per rep.
- **Verbalized sampling prompt** (system): Instructs the LLM to generate 5 possible responses within `<response><text>...</text><probability>...</probability></response>` XML tags, each with probability < 0.10 (to encourage tail-of-distribution diversity).
- **User prompt**: Topic + 100 context statements + instructions to synthesize themes
- **Model**: `gpt-5-mini`, reasoning=`minimal`, temperature=`1.0`
- **API calls**: 20 per rep (20 calls x 5 statements = 100)
- **Parsing**: Regex extraction of `<text>` content; probabilities are ignored (used only to encourage diversity)

#### 4.2.4 Alt4: No Persona, No Context (`no_persona_no_context`)

- **Description**: Blind verbalized sampling with neither persona nor context.
- **Pre-generated**: Pool of 815 statements per topic, shared across reps.
- **Verbalized sampling**: Same technique as Alt3 but without context statements.
- **User prompt**: Topic question only + instructions to write a bridging statement
- **Model**: `gpt-5-mini`, reasoning=`minimal`, temperature=`1.0`
- **API calls**: 163 per topic (163 calls x 5 statements = 815)
- **Sampling per rep**: 100 statements drawn randomly from the pool

---

## 5. Phase 2: Preference Building

### 5.1 Iterative Ranking Algorithm

Each of the 100 voters ranks all 100 alternatives through a multi-round iterative process designed to mitigate the **81% degeneracy rate** observed with single-call ranking.

#### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| K\_TOP\_BOTTOM | 10 | Statements selected as top-K and bottom-K per round |
| MAX\_FINAL\_RANKING | 20 | Max statements in the final ranking round |
| Number of rounds (100 stmts) | 5 total | 4 top-K/bottom-K rounds + 1 final round |
| Number of rounds (117 stmts) | 6 total | 5 top-K/bottom-K rounds + 1 final round |
| MAX\_RETRIES | 20 | Maximum retries per round on invalid/degenerate output |
| HASH\_SEED | 42 | Deterministic seed for hash identifier generation |
| Model | gpt-5-mini | - |
| Reasoning | low | - |
| Temperature | 1.0 | - |

#### Round formula

```
n_rounds = ceil((n_statements - MAX_FINAL_RANKING) / (2 * K))
```

For 100 statements: `ceil((100 - 20) / (2 * 10))` = 4 selection rounds + 1 final round = 5 total.

#### Algorithm (per voter)

1. **Initialize**: All 100 statements are in the "remaining" pool.
2. **For each selection round (rounds 1 through 4)**:
   a. **Shuffle** remaining statements using seed `voter_seed * 10 + round_num` (breaks presentation order bias).
   b. **Assign 4-letter hash identifiers** to each statement. Hash alphabet excludes confusable characters (0/O, 1/l/I).
   c. **LLM call**: Present shuffled statements with hash IDs. Ask for top-10 (most agreed-with) and bottom-10 (least agreed-with) selections.
   d. **Validate**: Check structural correctness (exactly K items in each list, valid hashes, no duplicates).
   e. **Degeneracy detection**: Check if output suspiciously matches presentation order (indicating the model just echoed the order rather than reasoning).
   f. **Retry** on invalid/degenerate output (up to 20 retries with re-shuffling).
   g. **Record**: Top-10 appended to the ranking from the top; bottom-10 prepended to the ranking from the bottom. Remove these 20 from the remaining pool.
3. **Final round (round 5)**:
   a. Rank all remaining statements (<=20) from most to least agreed-with.
   b. Same validation, degeneracy detection, and retry logic.
   c. Insert final ranking in the middle of the overall ranking.
4. **Output**: A complete ranking of all 100 statements for this voter.

#### Degeneracy Mitigation Features

- **Per-round shuffling**: Different random order each round prevents the model from memorizing or echoing positions.
- **Hash identifiers**: 4-letter codes (e.g., "ABCD") replace numeric indices to prevent the model from conflating statement index with rank.
- **Degeneracy detector**: Flags outputs where the selected items suspiciously match their presentation order.
- **Retry with re-shuffle**: Failed rounds get new shuffles, giving the model a fresh perspective.

#### Result

- **Valid ranking rate**: ~97–100% (compared to 19% with single-call ranking)
- **Typical retries**: A few per voter at most

### 5.2 Preference Matrix Construction

- **Parallelization**: Up to 200 concurrent workers (one per voter).
- **Output format**: `preferences[rank][voter] = alternative_id` — a 100x100 matrix where `preferences[0][v]` is voter v's most preferred alternative and `preferences[99][v]` is their least preferred.
- **Validation**: Post-hoc check for duplicates and invalid values ("-1").
- **Storage**: Saved as `preferences.json` per replication.

### 5.3 Epsilon Precomputation

After building the full 100x100 preference matrix, critical epsilon (epsilon\*) is precomputed for **every** one of the 100 alternatives against the full profile:

- Uses `compute_critical_epsilon()` from the PVC toolbox
- Parallelized with up to 10 workers
- Saved as `precomputed_epsilons.json` (maps alternative ID to epsilon\*)
- **Purpose**: Traditional methods and GPT selection methods pick from these 100 alternatives, so their epsilon\* can be looked up instantly

---

## 6. Phase 3: Winner Selection

### 6.1 Mini-Rep Subsampling

Before running winner selection methods, the 100x100 preference matrix is subsampled into smaller "mini-reps":

- **Mini-rep size**: 20 voters x 20 alternatives (P=20, K=20)
- **Mini-reps per rep**: 4
- **Purpose**: Simulates realistic smaller-scale voting scenarios; provides multiple independent evaluations per preference profile
- **Subsampling**: Random selection of 20 voters and 20 alternatives from the 100x100 matrix

### 6.2 Traditional Voting Methods (No API)

These operate on the 20x20 mini-rep preference matrices:

| Method | Algorithm |
|--------|-----------|
| **Schulze** | Condorcet-compliant pairwise comparison using strongest paths |
| **Borda** | Positional voting — each alternative gets points based on rank position |
| **IRV** | Instant Runoff Voting — iterative elimination of least-preferred alternatives |
| **Plurality** | First-past-the-post — winner is the alternative ranked #1 by the most voters |
| **VBC** | Veto by Consumption — PVC-based method where voters proportionally veto disliked alternatives |

### 6.3 GPT Selection Methods (Select from Existing)

These methods ask an LLM to select a consensus statement from a set of existing alternatives. All use `gpt-5-mini`, reasoning=`minimal`, temperature=`1.0`.

#### GPT (select from P=20 subsampled alternatives)

| Variant | Input Context |
|---------|--------------|
| **GPT-Select** (`chatgpt`) | 20 statements from the mini-rep |
| **GPT-Sel+Rank** (`chatgpt_rankings`) | 20 statements + voter preference rankings |
| **GPT-Sel+Pers** (`chatgpt_personas`) | 20 statements + voter persona descriptions |

#### GPT\* (select from all 100 alternatives)

| Variant | Input Context |
|---------|--------------|
| **GPT-Full** (`chatgpt_star`) | All 100 statements + topic |
| **GPT-Full+Rank** (`chatgpt_star_rankings`) | All 100 statements + mini-rep rankings |
| **GPT-Full+Pers** (`chatgpt_star_personas`) | All 100 statements + mini-rep personas |

**Prompt structure**: System prompt instructs the model to be a helpful assistant selecting consensus statements. Output must be valid JSON containing the selected statement ID.

**Persona filtering**: When personas are provided as context, they are filtered to 7 key fields: age, sex, race, education, occupation, political views, religion.

### 6.4 GPT Generative Methods (Generate New Statements)

These methods ask an LLM to generate a **novel** consensus statement not in the existing pool. All use `gpt-5-mini`, reasoning=`minimal`, temperature=`1.0`.

#### GPT\*\* (generate given P=20 context statements)

| Variant | Input Context |
|---------|--------------|
| **GPT-Synthesize** (`chatgpt_double_star`) | 20 statements from the mini-rep |
| **GPT-Synth+Rank** (`chatgpt_double_star_rankings`) | 20 statements + voter rankings |
| **GPT-Synth+Pers** (`chatgpt_double_star_personas`) | 20 statements + voter personas |

#### GPT\*\*\* (generate blind — no context)

| Variant | Input Context |
|---------|--------------|
| **GPT-Blind** (`chatgpt_triple_star`) | Topic question only (no statements, no rankings, no personas) |

#### Random Insertion (Baseline)

- **Method**: Sample a random statement from the global pool (excluding statements already in the current replication)
- **Purpose**: Baseline for comparison with generative methods

**Generation counts per rep**: 4 mini-reps x (3 GPT\*\* variants + 1 GPT\*\*\* + 1 Random) = 20 new statements per rep.

---

## 7. Phase 4: Evaluation

### 7.1 Critical Epsilon (epsilon\*)

The primary evaluation metric. Epsilon\* measures the minimum "veto power relaxation" needed for a statement to be in the Proportional Veto Core:

- **epsilon\* = 0**: Statement is in the exact PVC (optimal consensus — no coalition can veto it)
- **Lower epsilon\* = better consensus**: The statement is more broadly acceptable
- **epsilon\* = 1**: Statement is heavily vetoed (poor consensus)

### 7.2 Proportional Veto Core (PVC)

- **Concept**: A fairness-based aggregation method grounded in veto power. Each voter receives proportional "veto tokens" to eliminate disliked alternatives. The PVC consists of alternatives that survive all vetoes.
- **Implementation**: Veto-by-consumption algorithm with successive elimination of least-preferred alternatives.
- **m\_override**: For newly generated statements (GPT\*\*, GPT\*\*\*, Random), the epsilon is computed with `m_override=100` — this means the new statement's epsilon is computed as if there are 100 alternatives (not 101), ensuring fair comparison with the pre-existing alternatives.

### 7.3 Epsilon Lookup (for Selection Methods)

For methods that select from the existing 100 alternatives (Traditional + GPT + GPT\*):

1. Winner is one of the 100 pre-existing alternatives
2. Epsilon\* is looked up directly from `precomputed_epsilons.json`
3. **No additional API calls needed**

### 7.4 Batched Iterative Ranking (for Generative Methods)

For methods that generate new statements (GPT\*\*, GPT\*\*\*, Random), the new statement's position within the existing preference order must be determined:

1. **Batch all new statements**: All 20 new statements from the rep (4 mini-reps x 5 methods) are collected.
2. **Combined set**: 100 original statements + up to 17–20 new statements = 117–120 total statements.
3. **Run ONE iterative ranking per voter**: Each of the 100 voters ranks all ~117 statements through the same iterative ranking algorithm (6 rounds for 117 statements instead of 5 rounds for 100).
4. **Extract positions relative to originals**: For each new statement, count how many of the 100 *original* statements are ranked above it (ignoring other new statements). This gives a position from 0 (most preferred) to 100 (least preferred).
5. **Construct preference profile**: Insert new statement at the determined position in each voter's ranking.
6. **Compute epsilon\***: Using the constructed 101-alternative profile with `m_override=100`.

**Cost savings**: Batching all new statements into one ranking per voter achieves ~16–20x cost savings compared to ranking each new statement individually.

**API calls for batched ranking**: 100 voters x 6 rounds = 600 API calls per rep.

---

## 8. Factorial Design Summary

### 8.1 Experimental Factors

| Factor | Levels | Values |
|--------|--------|--------|
| Alternative Distribution | 4 | Alt1 (Persona, No Context), Alt2 (Persona + Context), Alt3 (No Persona + Context), Alt4 (No Persona, No Context) |
| Voter Distribution | 3 | Uniform, Progressive/Liberal cluster, Conservative/Traditional cluster |
| Topic | 13 | See Section 3.2 |
| Replications | 10 | Per condition |
| Mini-reps per rep | 4 | 20x20 subsamples of 100x100 matrix |

### 8.2 Methods Compared

| Category | Methods | Count | Epsilon Source |
|----------|---------|-------|----------------|
| Traditional | Schulze, Borda, IRV, Plurality, VBC | 5 | Precomputed lookup |
| Selection (GPT) | GPT-Select, GPT-Sel+Rank, GPT-Sel+Pers | 3 | Precomputed lookup |
| Selection (GPT\*) | GPT-Full, GPT-Full+Rank, GPT-Full+Pers | 3 | Precomputed lookup |
| Generative (GPT\*\*) | GPT-Synthesize, GPT-Synth+Rank, GPT-Synth+Pers | 3 | Batched iterative ranking |
| Generative (GPT\*\*\*) | GPT-Blind | 1 | Batched iterative ranking |
| Baseline | Random Insertion | 1 | Batched iterative ranking |
| **Total** | | **16** | |

### 8.3 Scale

- **Total conditions**: 4 alt dists x 3 voter dists x 13 topics x 10 reps = 1,560 replications
- **Total mini-reps**: 1,560 x 4 = 6,240
- **Total method evaluations**: 6,240 x 16 = 99,840

### 8.4 API Call Budget (per topic)

| Component | Model | Reasoning | Calls per Unit | Total per Topic |
|-----------|-------|-----------|---------------|-----------------|
| Statement Generation (Alt1) | gpt-5-mini | minimal | 815 per topic | 815 |
| Statement Generation (Alt4) | gpt-5-mini | minimal | 163 per topic | 163 |
| Per-Rep Statements (Alt2) | gpt-5-mini | minimal | 100 per rep | ~1,200 |
| Per-Rep Statements (Alt3) | gpt-5-mini | minimal | 20 per rep | ~240 |
| Preference Building | gpt-5-mini | low | 500 per rep (100 voters x 5 rounds) | ~24,000 |
| GPT/GPT\* Selection | gpt-5-mini | minimal | 1 per method per mini-rep | ~1,440 |
| GPT\*\*/GPT\*\*\* Generation | gpt-5-mini | minimal | 5 per mini-rep | ~800 |
| Batched Iterative Ranking | gpt-5-mini | low | 600 per rep (100 voters x 6 rounds) | ~30,000 |
| **Approximate Total** | | | | **~58,000** |

---

## 9. Models and Hyperparameters

| Role | Model | Reasoning Level | Temperature |
|------|-------|----------------|-------------|
| Statement Generation | gpt-5-mini | minimal | 1.0 |
| Generative Voting (GPT methods) | gpt-5-mini | minimal | 1.0 |
| Preference / Iterative Ranking | gpt-5-mini | low | 1.0 |

All API calls include structured metadata for tracking and cost analysis via the OpenAI dashboard:

- Core fields: `project`, `run_id`, `phase`, `component`
- Contextual fields: `topic`, `voter_dist`, `alt_dist`, `method`, `rep`, `mini_rep`, `voter_idx`, `round`

---

## 10. Data Flow Diagram (Node-Edge Description)

For a scientific diagram generator, the following describes nodes and directed edges:

### Nodes

| ID | Label | Phase | Shape |
|----|-------|-------|-------|
| PERSONAS | 815 Adult Personas | Input | Database/cylinder |
| TOPICS | 13 Policy Topics | Input | Database/cylinder |
| VOTER\_SAMPLING | Voter Sampling (Uniform or Clustered) | Data Generation | Process box |
| STMT\_GEN | Statement Generation (gpt-5-mini) | Data Generation | Process box |
| ITER\_RANK | Iterative Ranking (gpt-5-mini, 5 rounds) | Preference Building | Process box |
| PREF\_MATRIX | 100x100 Preference Matrix | Preference Building | Data store |
| PRECOMPUTE | Precompute epsilon\* for all 100 alternatives | Precompute | Process box |
| MINI\_REP | Mini-Rep Subsampling (20x20 from 100x100) | Winner Selection | Process box |
| TRADITIONAL | Traditional Methods: Schulze, Borda, IRV, Plurality, VBC | Winner Selection | Process box |
| LLM\_SELECT | LLM Selection (GPT, GPT\*; gpt-5-mini) | Winner Selection | Process box |
| LLM\_GEN | LLM Generation (GPT\*\*, GPT\*\*\*; gpt-5-mini) | Winner Selection | Process box |
| EPS\_LOOKUP | epsilon\* Lookup (from precomputed) | Evaluation | Process box |
| BATCH\_RANK | Batched Iterative Ranking (100 orig + ~20 new; gpt-5-mini) | Evaluation | Process box |
| COMPARE | Compare Methods by epsilon\* | Evaluation | Terminal/output |

### Directed Edges

| From | To | Label/Description |
|------|----|-------------------|
| PERSONAS | VOTER\_SAMPLING | Persona pool for voter sampling |
| PERSONAS | STMT\_GEN | Persona conditioning (Alt1, Alt2) |
| TOPICS | STMT\_GEN | Topic question for generation |
| VOTER\_SAMPLING | ITER\_RANK | 100 sampled voters |
| STMT\_GEN | ITER\_RANK | 100 candidate statements |
| ITER\_RANK | PREF\_MATRIX | Full rankings from all voters |
| PREF\_MATRIX | PRECOMPUTE | Compute epsilon for existing alternatives |
| PREF\_MATRIX | MINI\_REP | Subsample 20x20 |
| MINI\_REP | TRADITIONAL | 20x20 profile |
| MINI\_REP | LLM\_SELECT | 20 or 100 statements + optional context |
| MINI\_REP | LLM\_GEN | 20 statements as context (or topic only) |
| TRADITIONAL | EPS\_LOOKUP | Winner alternative ID |
| LLM\_SELECT | EPS\_LOOKUP | Selected alternative ID |
| PRECOMPUTE | EPS\_LOOKUP | Precomputed epsilon values |
| LLM\_GEN | BATCH\_RANK | Newly generated statements |
| EPS\_LOOKUP | COMPARE | epsilon\* for selection methods |
| BATCH\_RANK | COMPARE | epsilon\* for generative methods |

---

## 11. Key Methodological Innovations

1. **Iterative Ranking**: Reduces LLM ranking degeneracy from 81% to ~3% through multi-round top-K/bottom-K selection with per-round shuffling and hash identifiers.

2. **Hash Identifiers**: 4-letter alphabetic codes (excluding confusable characters 0/O, 1/l/I) replace numeric indices to prevent the LLM from conflating statement position with rank.

3. **Batched Iterative Ranking**: Instead of evaluating each generative method's output separately (which would require 16+ separate rankings per voter), all new statements are batched into a single ranking per voter, achieving ~16-20x cost savings.

4. **Verbalized Sampling**: For non-persona alternative distributions (Alt3, Alt4), the LLM generates 5 diverse statements per API call using a probabilistic prompting technique, reducing generation costs by 5x.

5. **m\_override for Fair Comparison**: When computing epsilon\* for newly generated statements, `m_override=100` ensures the new statement doesn't get extra veto power from increasing the alternative set size from 100 to 101.

6. **Mini-Rep Subsampling**: Each 100x100 preference matrix yields 4 independent 20x20 evaluations, increasing statistical power without additional preference-building cost.

---

## 12. Output Structure

### Per-Replication Outputs

```
outputs/sample_alt_voters/data/{topic}/{voter_dist}/[{ideology}/]{alt_dist}/rep{N}/
  preferences.json           -- 100x100 preference matrix
  precomputed_epsilons.json  -- epsilon* for all 100 alternatives
  voters.json                -- sampled voter metadata
  summary.json               -- experiment metadata and timing
  mini_rep{0-3}/
    results.json             -- per-method results {winner, epsilon*, statement_text, ...}
```

### Evaluation Outputs

```
outputs/paper/
  plots/                     -- Cross-topic visualizations (heatmaps, CDFs, bar charts)
  tables/{voter_dist}/       -- CSV and LaTeX summary tables
    traditional/             -- Traditional method results
    selection/               -- GPT/GPT* selection method results
    generative/              -- GPT**/GPT*** generative method results
```

### Metrics Computed

| Metric | Description |
|--------|-------------|
| Mean epsilon\* | Average consensus quality across mini-reps |
| P90, P95, P99 epsilon\* | Tail behavior of consensus quality |
| % epsilon\* = 0 | Fraction achieving exact PVC membership |
| % epsilon\* < threshold | Fraction below various thresholds (0.0001, 0.001, 0.01, 0.05, 0.1) |
| Mann-Whitney U | Statistical comparison between methods |
| Win/Tie/Loss rates | Pairwise comparison with VBC baseline |

---

## 13. Reproducibility

- **Deterministic seeding**: `BASE_SEED = 42`; all random operations use derived seeds for exact reproducibility.
- **Incremental execution**: All pipeline stages skip completed work and resume from interruption points.
- **API metadata**: Every LLM call includes structured metadata for tracking and auditing.
- **Idempotent outputs**: Running the pipeline twice produces identical results (given same API responses).
