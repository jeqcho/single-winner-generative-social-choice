# Sample-Alt-Voters Experiment Setup

This plan creates a new modular experiment framework in `src/sample-alt-voters/` to systematically compare 4 alternative (statement) generation methods and 2 voter sampling strategies across 2 contentious topics.

## Key Files to Create

```
src/sample-alt-voters/
├── __init__.py
├── config.py                    # Experiment configuration and data paths
├── generate_statements.py       # Pre-generate Alt1 and Alt4 statements
├── run_experiment.py            # Main experiment entry point
├── verbalized_sampling.py       # Parser and sampler for Alt3/Alt4
├── alternative_generators/      # 4 alternative distribution methods
│   ├── __init__.py
│   ├── base.py                  # Abstract base class
│   ├── persona_context.py       # Alt1: Persona-conditioned, pre-generated
│   ├── persona_no_context.py    # Alt2: Persona + original statements (Ben's)
│   ├── no_persona_context.py    # Alt3: Verbalized sampling with context
│   └── no_persona_no_context.py # Alt4: Verbalized sampling, pre-generated
├── voter_samplers/              # Voter sampling methods
│   ├── __init__.py
│   ├── base.py                  # Abstract base class
│   ├── uniform.py               # Uniform sampling from all 815 personas
│   └── clustered.py             # Use pre-computed K=10 clusters
├── preference_builder.py        # Build preference profiles
├── epsilon_calculator.py        # Epsilon computation
└── results_aggregator.py        # Collect and summarize results
```

## Design Details

### 1. Alternative Distributions

Each generator class will implement a `generate(personas, topic, openai_client) -> List[Dict]` method:

**Alt1 - `persona_bridging.py`** (`persona_context`): Generate NEW bridging statements conditioned on a persona

- **Pre-generate for all 815 adult personas** (one API call per persona)
- Save to `data/sample-alt-voters/sampled-statements/persona_context/`
- Prompt improvement: Instruct model to avoid self-referential phrases

**Alt2 - `persona_with_context.py`** (`persona_no_context`): Persona sees original statements, then writes bridging (Ben's setup)

- **Uses original statements from Alt1 as context** (N_a=100 sampled per rep)
- Generated per rep (since context depends on which 100 statements are sampled)
- Save to `data/sample-alt-voters/sampled-statements/persona_no_context/{topic}/rep{i}/`

**Alt3 - `no_persona_with_context.py`** (`no_persona_context`): Generate statement seeing original statements but NO persona

- Like Alt2 but without persona conditioning
- Generated per rep (context-dependent)
- **Uses verbalized sampling** (5 responses with probabilities <0.10 each)
- Save to `data/sample-alt-voters/sampled-statements/no_persona_context/{topic}/rep{i}/`

**Alt4 - `no_persona_no_context.py`** (`no_persona_no_context`): Blind generation (topic only)

- **Pre-generate** (no persona, no context = can generate all upfront)
- **Uses verbalized sampling** (5 responses with probabilities <0.10 each)
- Save to `data/sample-alt-voters/sampled-statements/no_persona_no_context/`

#### Alternative Distribution Prompts

All prompts share these common requirements:

- Is 2-4 sentences long
- Aims to find common ground or bridge different viewpoints
- Is self-contained (does not reference other statements or people explicitly)
- Ends with "Write only the statement:"

**Alt1 - Persona-Conditioned Bridging (New, Improved)**

System prompt:

```
You are writing a bridging statement that reflects your perspective on a policy topic. Write in first person but do NOT explicitly reference your demographic characteristics or political identity (e.g., do not say "As a progressive" or "As a conservative" or "As someone who works in healthcare"). Just express your views naturally.
```

User prompt:

```
You are a person with the following characteristics:
{persona}

Topic: "{topic}"

Write a bridging statement expressing your views on this topic. Your statement should:
- Reflect your background, values, and life experiences
- Aim to find common ground or bridge different viewpoints
- Is 2-4 sentences long
- NOT explicitly reference your identity or demographics (avoid "As a [X]...")

Write only the statement:
```

**Alt2 - Persona with Context (Ben's Bridging Setup)**

System prompt:

```
You are writing a bridging statement that reflects your perspective on a policy topic after reading diverse viewpoints. Write in first person but do NOT explicitly reference your demographic characteristics or political identity (e.g., do not say "As a progressive" or "As a conservative" or "As someone who works in healthcare"). Just express your views naturally.
```

User prompt:

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
- Is 2-4 sentences long
- NOT explicitly reference your identity or demographics (avoid "As a [X]...")
- Is self-contained (do not reference "the statements above" or "other people")

Write only the statement:
```

**Alt3 - No Persona with Context (GPT\*\*)** - Uses Verbalized Sampling

System prompt:

```
You are a helpful assistant that generates statements. Return only the statement text, no JSON or additional commentary. For each query, please generate a set of five possible responses, each within a separate <response> tag. Each <response> must include a <text> and a numeric <probability>.
Please sample at random from the tails of the distribution, such that the probability of each response is less than 0.10.
```

User prompt:

```
Topic: "{topic}"

Here are 100 statements from people with diverse perspectives on this topic:

{statements_list}

Write a NEW bridging statement on this topic. Your statement should:
- Synthesize key themes across the different viewpoints
- Aim to find common ground or bridge different viewpoints
- Is 2-4 sentences long
- Is self-contained (do not reference "the statements above" or "other people")
```

Expected output format:

```xml
<response>
<text>Statement text here...</text>
<probability>0.08</probability>
</response>
<response>
<text>Another statement...</text>
<probability>0.07</probability>
</response>
... (5 total responses)
```

**Alt4 - No Persona, No Context (GPT\*\*\*)** - Uses Verbalized Sampling

System prompt:

```
You are a helpful assistant that generates statements. Return only the statement text, no JSON or additional commentary. For each query, please generate a set of five possible responses, each within a separate <response> tag. Each <response> must include a <text> and a numeric <probability>.
Please sample at random from the tails of the distribution, such that the probability of each response is less than 0.10.
```

User prompt:

```
Topic: "{topic}"

Write a bridging statement on this topic. Your statement should:
- Aim to find common ground or bridge different viewpoints
- Is 2-4 sentences long
```

Expected output format: Same as Alt3 (5 responses with <text> and <probability> tags)

#### Prompt Structure Summary

| Component | Alt1 (persona_context) | Alt2 (persona_no_context) | Alt3 (no_persona_context) | Alt4 (no_persona_no_context) |

|-----------|------------------------|---------------------------|---------------------------|------------------------------|

| Persona header | Yes | Yes | No | No |

| Statements context | No | Yes | Yes | No |

| "Reflect your background..." | Yes | Yes | No | No |

| "Synthesize key themes..." | No | Yes | Yes | No |

| "Avoid As a [X]..." | Yes | Yes | No | No |

| "Self-contained..." | No | Yes | Yes | No |

| "Aim to find common ground..." | Yes | Yes | Yes | Yes |

| "Is 2-4 sentences long" | Yes | Yes | Yes | Yes |

| **Verbalized sampling** | No | No | **Yes** | **Yes** |

| **Pre-generated** | **Yes** | No (per-rep) | No (per-rep) | **Yes** |

### 2. Voter Distributions

Each sampler class will implement a `sample(personas, k, seed) -> List[str]` method:

**Voter1 - `uniform.py`**: Sample k personas uniformly at random

- Current approach in [`data_loader.py`](src/sampling_experiment/data_loader.py)

**Voter2 - `clustered.py`**: Use pre-computed K=10 clusters

- Load cluster assignments from `data/persona_embeddings_adult/persona_clusters.json`
- For replication i (0-9), use cluster i
- Sample 100 voters uniformly from that cluster
- Note: Cluster sizes vary (39-155 personas), so some clusters may need all their members

### 3. Configuration (`config.py`)

Key constants to define:

```python
TOPICS = [
    "what-should-guide-laws-concerning-abortion",
    "what-reforms-if-any-should-replace-or-modify-the-e",
]

ALT_DISTRIBUTIONS = ["persona_bridging", "persona_with_context", "no_persona_with_context", "no_persona_no_context"]
VOTER_DISTRIBUTIONS = ["uniform", "clustered"]

N_ALTERNATIVES = 100  # Number of alternatives to generate
N_VOTERS = 100        # Size of voter pool
K_SAMPLE = 20         # Voters per sample
P_SAMPLE = 20         # Alternatives per sample
N_SAMPLES_PER_REP = 5 # Samples per replication

# Clustering configuration
N_CLUSTERS = 10       # Use K=10 clusters (each rep uses one cluster)
N_REPS_UNIFORM = 10   # Replications for uniform voter distribution
N_REPS_CLUSTERED = 10 # Replications for clustered (one per cluster)
```

### 3.1 Existing Data Paths

```python
# Adult-only data (pre-filtered)
PERSONAS_PATH = "data/personas/prod/adult.json"  # 815 adult personas
STATEMENTS_DIR = "data/large_scale/prod/statements_adult"  # Adult-only statements
CLUSTER_ASSIGNMENTS_PATH = "data/persona_embeddings_adult/persona_clusters.json"  # Pre-computed clusters
```

### 4. Experiment Flow

**Phase 1: Pre-generate Statements (`generate_statements.py`)**

```
Load adult personas from data/personas/prod/adult.json (815 personas)

# Alt1: Generate for all 815 personas (one API call per persona)
For each topic in [abortion, electoral]:
  For each persona in all_personas:
    Generate bridging statement (persona-conditioned, no context)
  Save to data/sample-alt-voters/sampled-statements/persona_context/{topic}.json

# Alt4: Generate blind statements (no persona, no context) - uses verbalized sampling
For each topic in [abortion, electoral]:
  Generate N statements using verbalized sampling (5 responses per call, sample one)
  Save to data/sample-alt-voters/sampled-statements/no_persona_no_context/{topic}.json
```

**Phase 2: Main Experiment (`run_experiment.py`)**

```
Load adult personas from data/personas/prod/adult.json (815 personas)
Load cluster assignments from data/persona_embeddings_adult/persona_clusters.json (K=10)
Load pre-generated Alt1 statements from data/sample-alt-voters/sampled-statements/persona_context/
Load pre-generated Alt4 statements from data/sample-alt-voters/sampled-statements/no_persona_no_context/

For each topic in [abortion, electoral]:
  
  For each voter_dist in [uniform, clustered]:
    For each replication:  # 10 reps for uniform, 10 clusters for clustered
      
      # Sample voters
      If uniform: Sample 100 voters from all 815 personas
      If clustered: Use all personas from cluster_id (rep_idx)
      Save to data/sample-alt-voters/sampled-personas/{voter_dist}/{topic}/rep{i}.json
      
      # Sample 100 original statements (from Alt1 pre-generated)
      Sample 100 statements from Alt1 pool (matching sampled personas or random)
      
      # Generate Alt2 statements (persona + context) - per rep
      For each of the 100 sampled personas:
        Generate bridging statement given the 100 sampled original statements
      Save to data/sample-alt-voters/sampled-statements/persona_no_context/{topic}/rep{i}.json
      
      # Generate Alt3 statements (no persona + context) - per rep, uses verbalized sampling
      Generate 100 statements given the 100 sampled original statements
      Save to data/sample-alt-voters/sampled-statements/no_persona_context/{topic}/rep{i}.json
      
      # For each alt_dist, build preferences and run voting...
      For each alt_dist in [persona_context, persona_no_context, no_persona_context, no_persona_no_context]:
        Load/use appropriate statements for this alt_dist
        Build 100×100 preference profile
        Precompute epsilons
        
        For each sample (5):
          Sample k=20 voters, p=20 alternatives
          Run voting methods
          Record epsilons
        
        Save to outputs/sample_alt_voters/{topic}/{alt_dist}_{voter_dist}/rep{i}/
```

### 5. Key Existing Code to Reuse

| Component | Source File |

|-----------|-------------|

| Load entries | [`src/sampling_experiment/data_loader.py`](src/sampling_experiment/data_loader.py) `load_all_entries()` |

| Build preferences | [`src/sampling_experiment/preference_builder.py`](src/sampling_experiment/preference_builder.py) |

| Epsilon calculation | [`src/sampling_experiment/epsilon_calculator.py`](src/sampling_experiment/epsilon_calculator.py) |

| Voting methods | [`src/sampling_experiment/voting_methods.py`](src/sampling_experiment/voting_methods.py) |

| Persona bridging (Ben's) | [`src/full_sampling_experiment/mini_variant.py`](src/full_sampling_experiment/mini_variant.py) |

| GPT*** | [`src/sampling_experiment/voting_methods.py`](src/sampling_experiment/voting_methods.py) `run_chatgpt_triple_star()` |

### 6. Sampled Data Structure

```
data/sample-alt-voters/
├── sampled-statements/
│   ├── persona_context/                    # Alt1: Pre-generated (815 per topic)
│   │   ├── abortion.json
│   │   └── electoral.json
│   ├── persona_no_context/                 # Alt2: Per-rep (100 per rep)
│   │   ├── abortion/
│   │   │   ├── uniform_rep0.json
│   │   │   ├── uniform_rep1.json
│   │   │   ├── clustered_cluster0.json
│   │   │   └── ...
│   │   └── electoral/
│   │       └── ...
│   ├── no_persona_context/                 # Alt3: Per-rep, verbalized sampling
│   │   ├── abortion/
│   │   │   └── ... (same structure as Alt2)
│   │   └── electoral/
│   │       └── ...
│   └── no_persona_no_context/              # Alt4: Pre-generated, verbalized sampling
│       ├── abortion.json
│       └── electoral.json
└── sampled-personas/
    ├── uniform/
    │   ├── abortion/
    │   │   ├── rep0.json                   # List of 100 persona indices
    │   │   └── ...
    │   └── electoral/
    │       └── ...
    └── clustered/
        ├── abortion/
        │   ├── cluster0.json               # All personas in cluster 0
        │   └── ...
        └── electoral/
            └── ...
```

### 7. Verbalized Sampling Parser

For Alt3 and Alt4, parse the XML-style response:

```python
import re

def parse_verbalized_response(response_text: str) -> list[dict]:
    """Parse verbalized sampling response with 5 responses."""
    responses = []
    pattern = r'<response>\s*<text>(.*?)</text>\s*<probability>([\d.]+)</probability>\s*</response>'
    matches = re.findall(pattern, response_text, re.DOTALL)
    for text, prob in matches:
        responses.append({"text": text.strip(), "probability": float(prob)})
    return responses

def sample_from_verbalized(responses: list[dict]) -> str:
    """Sample one response weighted by probability."""
    import random
    total = sum(r["probability"] for r in responses)
    r = random.random() * total
    cumsum = 0
    for resp in responses:
        cumsum += resp["probability"]
        if r <= cumsum:
            return resp["text"]
    return responses[-1]["text"]  # Fallback
```

### 8. Output Structure

```
outputs/sample_alt_voters/
├── abortion/
│   ├── persona_bridging_uniform/
│   │   ├── rep0/
│   │   │   ├── alternatives.json
│   │   │   ├── voter_pool.json
│   │   │   ├── full_preferences.json
│   │   │   ├── precomputed_epsilons.json
│   │   │   └── sample{0-4}/results.json
│   │   └── rep{1-9}/...
│   ├── persona_bridging_clustered/
│   │   ├── cluster0/                  # 93 personas
│   │   │   ├── alternatives.json
│   │   │   ├── voter_pool.json
│   │   │   ├── full_preferences.json
│   │   │   ├── precomputed_epsilons.json
│   │   │   └── sample{0-4}/results.json
│   │   ├── cluster1/                  # 95 personas
│   │   └── cluster{2-9}/...
│   ├── persona_with_context_uniform/
│   ├── persona_with_context_clustered/
│   ├── no_persona_with_context_uniform/
│   ├── no_persona_with_context_clustered/
│   ├── no_persona_no_context_uniform/
│   └── no_persona_no_context_clustered/
└── electoral/
    └── ... (same structure as abortion)
```

K=10 cluster sizes (from clustering analysis):

- Cluster 0: 93, Cluster 1: 95, Cluster 2: 39, Cluster 3: 123, Cluster 4: 90
- Cluster 5: 155, Cluster 6: 42, Cluster 7: 58, Cluster 8: 52, Cluster 9: 68

### 7. Pre-computed Persona Data (Already Exists)

```
data/personas/prod/adult.json                          # 815 adult personas
data/persona_embeddings_adult/
├── persona_embeddings.npy                             # Shape: (815, 1536)
└── persona_clusters.json                              # {"5": [...], "10": [...], "20": [...]}
data/large_scale/prod/statements_adult/                # Adult-only statements per topic
```