# Sampling Experiment Report: Voter and Alternative Sampling in Generative Social Choice

**Date:** January 2026  
**Topic:** Public Trust in Institutions (Test Run)

---

## 1. Executive Summary

This experiment investigates how voting methods perform when both voters and alternatives are sampled from larger pools. We compare traditional voting methods (Schulze, Borda, IRV, Plurality) with LLM-based methods (ChatGPT variants) that can either select from available alternatives or generate entirely new statements. The key metric is the **critical epsilon (ε*)**, which measures how close a winner is to being in the Proportional Veto Core (PVC).

**Key Finding:** The ChatGPT** methods, which generate new statements rather than selecting from existing ones, consistently achieve ε* ≈ 0, indicating these new statements are broadly acceptable and cannot be blocked by any voter coalition.

---

## 2. Experimental Setup

### 2.1 Pool Construction

For each replication, we construct two independent pools:

| Pool | Size | Description |
|------|------|-------------|
| **Voter Pool** | 100 personas | Synthetic personas with diverse backgrounds and perspectives |
| **Alternative Pool** | 100 statements | Policy position statements on the topic |

Both pools are sampled from a larger dataset of 1000 personas/statements using stratified sampling to ensure diversity.

### 2.2 Sampling Parameters

From the pools, we sample K voters and P alternatives for each experimental condition:

| Parameter | Values | Description |
|-----------|--------|-------------|
| **K** (voters) | {10, 20, 50, 100} | Number of voters sampled from the 100-voter pool |
| **P** (alternatives) | {10, 20, 50, 100} | Number of alternatives sampled from the 100-alternative pool |
| **Replications** | 5 | Independent pool constructions |

This creates a **4 × 4 × 5 = 80 sample** design (16 (K,P) combinations × 5 replications).

### 2.3 Preference Profile Construction

For each sample, we need voter preferences over alternatives. This is done in two stages:

#### Stage 1: Full 100×100 Preference Matrix

Before sampling, we construct the complete preference matrix where each of the 100 voters ranks all 100 alternatives. This is done using a **single-call ranking method**:

```
For each voter persona:
    1. Present all 100 statements to GPT-5.2
    2. Ask the model to rank statements from most to least preferred
    3. Model returns only statement IDs in ranked order
```

This approach replaces expensive pairwise comparison methods with a single API call per voter, reducing costs significantly while maintaining preference quality.

#### Stage 2: Subprofile Extraction

Given a sample with K voters and P alternatives, we extract the K×P subprofile from the full 100×100 matrix. This ensures consistency—the same voter always has the same preferences over the same alternatives across different samples.

### 2.4 Epsilon Calculation

The **critical epsilon (ε*)** measures how close an alternative is to the Proportional Veto Core (PVC):

- **ε* < 0**: Alternative is in the standard PVC (very safe)
- **ε* = 0**: Alternative is exactly on the PVC boundary
- **ε* > 0**: Alternative requires ε > ε* to be in the relaxed ε-PVC

**Important:** All epsilons are calculated with respect to the **full 100×100 preference profile**, not the sampled K×P subprofile. This ensures we're measuring how well the sampled voting method performs in terms of the full population's preferences.

For efficiency, we precompute ε* for all 100 alternatives against the full profile once per replication, then look up values when voting methods return winners.

---

## 3. Voting Methods

### 3.1 Traditional Methods

| Method | Description |
|--------|-------------|
| **Schulze** | Condorcet method using ranked pairs; selects the candidate that beats all others in pairwise comparisons |
| **Borda** | Positional scoring; awards points based on rank position |
| **IRV** | Instant Runoff Voting; eliminates lowest-ranked candidates iteratively |
| **Plurality** | Simple plurality; candidate with most first-place votes wins |
| **Veto by Consumption** | PVC-based method; implements the proportional veto core directly |

### 3.2 ChatGPT Methods (Select from P)

These methods use GPT-5.2 to select a winner from the P sampled alternatives:

| Method | Input | Description |
|--------|-------|-------------|
| **ChatGPT** | Statements only | Model sees only the P statement texts |
| **ChatGPT+Rankings** | Statements + Rankings | Model also sees how K voters ranked the statements |
| **ChatGPT+Personas** | Statements + Personas | Model also sees the K voter persona descriptions |

### 3.3 ChatGPT* Methods (Select from 100)

These methods can select from the **full 100-alternative pool**, not just the P sampled alternatives:

| Method | Input | Selection Space |
|--------|-------|-----------------|
| **ChatGPT*** | Statements only | All 100 alternatives |
| **ChatGPT*+Rankings** | Statements + Rankings | All 100 alternatives |
| **ChatGPT*+Personas** | Statements + Personas | All 100 alternatives |

The rankings and personas are still from the K×P sample, but the model can choose any of the 100 statements.

### 3.4 ChatGPT** Methods (Generate New)

These methods **generate entirely new statements** rather than selecting from existing ones:

| Method | Input | Output |
|--------|-------|--------|
| **ChatGPT**** | Statements only | New statement |
| **ChatGPT**+Rankings** | Statements + Rankings | New statement |
| **ChatGPT**+Personas** | Statements + Personas | New statement |

#### New Statement Integration

When ChatGPT** generates a new statement, we must integrate it into the preference profile for epsilon calculation:

1. **Statement Generation**: Model outputs a new policy statement
2. **Preference Update**: For each of the 100 voters, we query the model to insert the new statement into their existing ranking
3. **Epsilon Calculation**: Compute ε* for the new statement against the updated 101-alternative profile

**Critical Design Decision:** When computing epsilon for new statements, we use **m=100** (not 101) for the veto power calculation. This prevents the new statement from having inherent veto advantage simply by expanding the alternative space.

---

## 4. Results (Public Trust Topic)

### 4.1 Average Epsilon by Method

Results from 5 replications across 12 (K,P) combinations (47-60 samples per method):

| Method | Mean ε* | Std Dev | Interpretation |
|--------|---------|---------|----------------|
| **Veto by Consumption** | 0.000 | 0.000 | Always in PVC (by design) |
| **Borda** | 0.002 | 0.014 | Excellent |
| **Schulze** | 0.009 | 0.054 | Very good |
| **ChatGPT+Rankings** | 0.014 | 0.038 | Good |
| **Plurality** | 0.020 | 0.068 | Moderate |
| **IRV** | 0.026 | 0.091 | Moderate |
| **ChatGPT*+Rankings** | 0.009 | 0.034 | Very good |
| **ChatGPT*+Personas** | 0.047 | 0.079 | Moderate |
| **ChatGPT** | 0.053 | 0.111 | Moderate |
| **ChatGPT+Personas** | 0.055 | 0.100 | Moderate |
| **ChatGPT*** | 0.072 | 0.106 | Higher variance |
| **ChatGPT**** | **≈0.000** | 0.000 | In PVC |
| **ChatGPT**+Rankings** | **≈0.000** | 0.000 | In PVC |
| **ChatGPT**+Personas** | 0.000 | 0.002 | In PVC |

### 4.2 Key Observations

1. **ChatGPT** methods achieve ε* ≈ 0**: The LLM-generated statements consistently fall within or on the boundary of the Proportional Veto Core, suggesting the model synthesizes broadly acceptable compromises.

2. **Rankings improve ChatGPT performance**: Adding ranking information reduces epsilon for both ChatGPT and ChatGPT* variants (0.053 → 0.014 and 0.072 → 0.009 respectively).

3. **Traditional methods vary widely**: Borda and Schulze perform well, while Plurality and IRV show higher variance.

4. **Expanded selection (ChatGPT*) increases epsilon**: Having access to more alternatives doesn't necessarily improve outcomes—the model may select more extreme positions.

---

## 5. Visualization Summary

The experiment generates the following visualizations:

### 5.1 Bar Charts
- **Overall**: Mean epsilon across all methods
- **Per (K,P)**: Mean epsilon for each sample size combination

### 5.2 Heatmaps
- Epsilon values across the (K,P) grid for each method
- Shows how sample size affects method performance

### 5.3 Line Plots
- **By P**: X=K, lines=methods, one plot per P value
- **By K**: X=P, lines=methods, one plot per K value
- **By Method**: X=K, lines=P values, one plot per method

### 5.4 CDF Plots
- Cumulative distribution of epsilon values
- Grouped by method category (Traditional, ChatGPT, ChatGPT*, ChatGPT**)
- Log-scale versions for detailed comparison

---

## 6. Technical Implementation

### 6.1 API Usage

- **Model**: GPT-5.2 for all LLM calls
- **Temperature**: 1.0
- **Parallelization**: Up to 50 concurrent API calls
- **Retry Logic**: Exponential backoff with 3 attempts

### 6.2 Computational Efficiency

| Operation | Approach | Cost |
|-----------|----------|------|
| Preference construction | Single-call ranking | 100 API calls per replication |
| Traditional voting | Local computation | ~0ms |
| ChatGPT selection | 1 API call | ~1-2s |
| ChatGPT** + ranking update | 1 + K API calls | ~(1 + K×0.2)s |
| Epsilon calculation | Precomputed lookup | ~0ms |

### 6.3 Caching Strategy

- Pool data (personas, alternatives) cached per replication
- Full 100×100 preference matrix cached per replication  
- Precomputed epsilons cached per replication
- Results cached per (K,P) sample

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

1. **Single Topic**: Currently only tested on "Public Trust" topic
2. **Fixed Pool Size**: 100×100 pools; larger pools may show different patterns
3. **Model Dependency**: Results specific to GPT-5.2; other models may differ

### 7.2 Future Directions

1. **Multi-Topic Analysis**: Extend to all 13 policy topics
2. **Scaling Study**: Test with 200×200 or 500×500 pools
3. **Model Comparison**: Compare GPT-5.2 with Claude, Gemini, etc.
4. **Hybrid Methods**: Combine ChatGPT** generation with traditional refinement

---

## 8. Conclusion

This experiment demonstrates that LLM-based voting methods, particularly those that generate new statements (ChatGPT**), can identify alternatives that satisfy the Proportional Veto Core criterion. The ability to synthesize new positions rather than selecting from fixed alternatives appears to enable better compromise outcomes.

The sampling framework—where both voters and alternatives are sampled from larger pools—provides a realistic model for scenarios where full enumeration is infeasible. Evaluating winners against the full population's preferences (rather than just the sample) ensures we measure true representativeness.

---

## Appendix A: File Structure

```
outputs/sampling_experiment/
├── data/
│   └── {topic_slug}/
│       └── rep{i}/
│           ├── pool_data.json           # Sampled personas & alternatives
│           ├── full_preferences.json    # 100×100 preference matrix
│           ├── precomputed_epsilons.json # ε* for all 100 alternatives
│           └── k{K}_p{P}/
│               ├── sample_info.json     # Which voters/alts were sampled
│               └── results.json         # Voting method results
└── figures/
    └── {topic_slug}/
        ├── bar_*.png                    # Bar charts
        ├── heatmap_*.png                # Heatmaps
        ├── lines_by_p/                  # Line plots grouped by P
        ├── lines_by_k/                  # Line plots grouped by K
        ├── lines_by_method/             # Line plots grouped by method
        └── cdf/                         # CDF plots
```

## Appendix B: Configuration Reference

```python
# Pool sizes
N_VOTER_POOL = 100
N_ALT_POOL = 100

# Sampling parameters
K_VALUES = [10, 20, 50, 100]
P_VALUES = [10, 20, 50, 100]

# Replications
N_REPS = 5

# API settings
MODEL = "gpt-5.2"
TEMPERATURE = 1.0
```
