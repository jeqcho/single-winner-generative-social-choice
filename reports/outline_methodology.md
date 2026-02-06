# Experiment Methodology Outline: Single-Winner Generative Social Choice

## Research Question

Can LLMs select or generate consensus statements as well as traditional voting methods? Evaluated using critical epsilon (epsilon\*) from the Proportional Veto Core.

---

## Inputs

- **815 adult personas** from the SynthLabsAI/PERSONA dataset (filtered to age >= 18)
- **13 policy topics** spanning elections, healthcare, abortion, policing, immigration, environment, etc.

---

## Phase 1: Data Generation

### Voter Sampling
- **Uniform**: 100 voters sampled randomly from all 815 personas
- **Clustered**: 100 voters sampled from an ideology cluster (progressive/liberal or conservative/traditional)

### Statement Generation (4 distributions)
- **Alt1** — Persona-conditioned, no context (pre-generated pool)
- **Alt2** — Persona-conditioned + sees existing statements (per-rep)
- **Alt3** — No persona, sees existing statements, uses verbalized sampling (per-rep)
- **Alt4** — No persona, no context, verbalized sampling (pre-generated pool)

All generation uses gpt-5-mini. 100 statements sampled per replication.

---

## Phase 2: Preference Building

### Iterative Ranking
- Each of 100 voters ranks all 100 statements through 5 rounds of top-10/bottom-10 selection
- Uses hash identifiers and per-round shuffling to mitigate degeneracy
- Produces a full 100x100 preference matrix per replication

### Epsilon Precomputation
- Epsilon\* computed for all 100 alternatives against the full preference profile
- Stored for instant lookup by selection-based methods

---

## Phase 3: Winner Selection

### Mini-Rep Subsampling
- 4 independent 20-voter x 20-alternative subsamples drawn from each 100x100 matrix

### Methods (16 total)

**Traditional (5)**: Schulze, Borda, IRV, Plurality, VBC — operate on the 20x20 profile, no LLM

**LLM Selection (6)**: GPT and GPT\* variants — LLM selects one existing statement
- GPT: chooses from 20 subsampled statements
- GPT\*: chooses from all 100 statements
- Each has base, +Rankings, and +Personas variants

**LLM Generation (4)**: GPT\*\* and GPT\*\*\* — LLM generates a novel statement
- GPT\*\*: generates given context statements (+Rankings, +Personas variants)
- GPT\*\*\*: generates blind (topic only)

**Baseline (1)**: Random statement from the global pool

---

## Phase 4: Evaluation

### Epsilon Computation
- **Selection methods**: Epsilon\* looked up from precomputed values
- **Generative methods**: All new statements batched into a single iterative ranking per voter (100 originals + ~20 new), then epsilon\* computed from extracted positions

### Comparison
- Mean, P90/P95/P99 epsilon\*
- % achieving epsilon\* = 0 (exact PVC membership)
- Win/tie/loss rates vs. VBC baseline
- Mann-Whitney U tests for statistical significance

---

## Factorial Design

| Factor | Levels |
|--------|--------|
| Alternative Distribution | 4 (Alt1–Alt4) |
| Voter Distribution | 3 (Uniform, Progressive, Conservative) |
| Topics | 13 |
| Replications | 10 per condition |
| Mini-reps per rep | 4 |
| Methods | 16 |

---

## Key Innovations

1. **Iterative ranking** — multi-round top-K/bottom-K with hash IDs reduces degeneracy from 81% to ~3%
2. **Batched ranking** — all generative outputs ranked together for ~16-20x cost savings
3. **Verbalized sampling** — 5 diverse statements per LLM call for non-persona distributions
4. **m\_override** — fair epsilon comparison between existing and newly generated alternatives
