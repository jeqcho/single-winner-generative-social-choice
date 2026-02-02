# Alternative Distributions Table

Preview of the LaTeX table for quick review.

| Distribution   | Persona | Bridging Round | Verbalized | Cost  |
|----------------|:-------:|:--------------:|:----------:|------:|
| Baseline       | -       | -              | ✓          | $0.25 |
| Persona        | ✓       | -              | -          | $0.34 |
| Bridge         | -       | ✓              | ✓          | $1.08 |
| Persona+Bridge | ✓       | ✓              | -          | $4.29 |

## Caption

> Alternative distributions for statement generation. *Persona*: whether statements are conditioned on a synthetic persona's characteristics. *Bridging round*: whether the model reads 100 existing statements before generating. *Verbalized*: whether diversity is achieved via verbalized sampling (5 statements per API call) rather than one statement per persona. Cost shown for 10 replications per topic.

## Explanation

Methods without bridging will pre-generate 815 statements in total to correspond in numbers with 815 personas. Methods with bridging will use statements from Persona and regenerate statements in each rep. API calls and token estimates are as follows:

- **Baseline**: Pre-generated once per topic using verbalized sampling. One API call generates 5 statements. 815 / 5 = 163 calls. ~160 in, ~757 out tokens each, giving ~$0.25.
- **Persona**: Pre-generated once per topic, one statement per persona. 815 calls. ~468 in, ~151 out tokens each, giving ~$0.34.
- **Bridge**: Generated per replication using verbalized sampling. 100 / 5 = 20 calls/rep × 10 reps = 200 calls. ~15,656 in, ~757 out tokens each, giving ~$1.08.
- **Persona+Bridge**: Generated per replication, one statement per sampled persona. 100 × 10 = 1,000 calls. ~15,964 in, ~151 out tokens each, giving ~$4.29.

**Token estimates:** Average persona length ≈ 308 tokens. Average statement length ≈ 151 tokens. Prompt overhead ≈ 150 tokens. Bridging round context includes 100 statements (≈ 15,100 tokens). gpt-5-mini pricing: $0.25/1M input tokens, $2.00/1M output tokens.

## Code Name Mapping

| Descriptive Name | Code Name | Notes |
|------------------|-----------|-------|
| Persona | `persona_no_context` | Alt1 in original design |
| Persona+Bridge | `persona_context` | Alt2 in original design |
| Bridge | `no_persona_context` | Alt3 in original design |
| Baseline | `no_persona_no_context` | Alt4 in original design |

## Caption

> Alternative distributions for statement generation. Each distribution varies by whether statements are conditioned on a persona and whether they are generated after a bridging round (seeing 100 existing statements).
