#!/usr/bin/env python3
"""
Estimate API costs per topic based on actual persona and statement data.
Uses tiktoken for accurate token counting.
"""

import json
from pathlib import Path

# Try to use tiktoken for accurate counting, fallback to char-based estimate
try:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4")  # Use gpt-4 encoding as proxy
    def count_tokens(text: str) -> int:
        return len(enc.encode(text))
except ImportError:
    def count_tokens(text: str) -> int:
        # Rough estimate: ~4 chars per token
        return len(text) // 4

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PERSONAS_PATH = DATA_DIR / "personas" / "prod" / "adult.json"
STATEMENTS_DIR = DATA_DIR / "sample-alt-voters" / "sampled-statements"

# Pricing (per 1M tokens)
PRICING = {
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5.2": {"input": 1.75, "output": 14.00},
}

def load_personas():
    """Load all adult personas."""
    with open(PERSONAS_PATH) as f:
        return json.load(f)

def load_statements(topic: str = "abortion"):
    """Load statements for a topic."""
    path = STATEMENTS_DIR / "persona_no_context" / f"{topic}.json"
    with open(path) as f:
        data = json.load(f)
    return list(data["statements"].values())

def filter_persona(persona: str) -> str:
    """
    Extract key demographic fields from a persona string.
    Filters to: age, sex, race, education, occupation category, political views, religion.
    """
    fields_to_keep = {
        'age', 'sex', 'race', 'education', 
        'occupation category', 'political views', 'religion'
    }
    lines = []
    for line in persona.split('\n'):
        if ':' in line:
            key = line.split(':')[0].strip().lower()
            if key in fields_to_keep:
                if key == 'occupation category':
                    line = 'occupation:' + line.split(':', 1)[1]
                lines.append(line)
    return '\n'.join(lines)


def estimate_prompt_tokens():
    """Estimate token counts for different prompt types."""
    personas = load_personas()
    statements = load_statements()
    
    # Sample sizes for estimation
    sample_personas = personas[:100]
    sample_statements = statements[:100]
    
    # Average full persona length
    avg_persona_tokens = sum(count_tokens(p) for p in sample_personas) / len(sample_personas)
    
    # Average filtered persona length (7 key fields only)
    avg_filtered_persona_tokens = sum(count_tokens(filter_persona(p)) for p in sample_personas) / len(sample_personas)
    
    # Average statement length
    avg_statement_tokens = sum(count_tokens(s) for s in sample_statements) / len(sample_statements)
    
    # Topic question (typical length)
    topic_question = "What should guide laws concerning abortion?"
    topic_tokens = count_tokens(topic_question)
    
    print("=" * 60)
    print("TOKEN ESTIMATES")
    print("=" * 60)
    print(f"Average full persona: {avg_persona_tokens:.0f} tokens")
    print(f"Average filtered persona (7 fields): {avg_filtered_persona_tokens:.0f} tokens")
    print(f"Average statement: {avg_statement_tokens:.0f} tokens")
    print(f"Topic question: {topic_tokens} tokens")
    print()
    
    return {
        "persona": avg_persona_tokens,
        "filtered_persona": avg_filtered_persona_tokens,
        "statement": avg_statement_tokens,
        "topic": topic_tokens,
    }

def calculate_costs(token_estimates):
    """Calculate costs for each API call type."""
    persona_tokens = token_estimates["persona"]
    filtered_persona_tokens = token_estimates["filtered_persona"]
    stmt_tokens = token_estimates["statement"]
    topic_tokens = token_estimates["topic"]
    
    # Fixed prompt overhead (instructions, formatting)
    OVERHEAD_SMALL = 150  # Small prompts
    OVERHEAD_MEDIUM = 300  # Medium prompts  
    OVERHEAD_LARGE = 500  # Large prompts with context
    
    # Sample sizes for mini-rep evaluation
    K_VOTERS = 20  # Sampled voters per mini-rep
    P_ALTS = 20  # Sampled alternatives per mini-rep
    N_STATEMENTS = 100  # All statements for GPT* methods
    
    costs = []
    
    # =========================================================================
    # Phase 1: Statement Generation
    # =========================================================================
    
    # Alt1: Persona, No Context (815 calls/topic)
    alt1_input = OVERHEAD_SMALL + persona_tokens + topic_tokens
    alt1_output = stmt_tokens
    alt1_calls = 815
    costs.append({
        "name": "Alt1 Statement Gen",
        "model": "gpt-5-mini",
        "calls": alt1_calls,
        "input_tokens": alt1_input,
        "output_tokens": alt1_output,
    })
    
    # Alt2: Persona + Context (1,200 calls/topic = 100 per rep × 12 reps)
    alt2_input = OVERHEAD_LARGE + persona_tokens + topic_tokens + (100 * stmt_tokens)  # 100 statements as context
    alt2_output = stmt_tokens
    alt2_calls = 1200
    costs.append({
        "name": "Alt2 Statement Gen",
        "model": "gpt-5-mini",
        "calls": alt2_calls,
        "input_tokens": alt2_input,
        "output_tokens": alt2_output,
    })
    
    # Alt3: No Persona, With Context (240 calls/topic, 5 outputs each)
    alt3_input = OVERHEAD_LARGE + topic_tokens + (100 * stmt_tokens)
    alt3_output = 5 * stmt_tokens  # 5 statements per call
    alt3_calls = 240
    costs.append({
        "name": "Alt3 Statement Gen",
        "model": "gpt-5-mini",
        "calls": alt3_calls,
        "input_tokens": alt3_input,
        "output_tokens": alt3_output,
    })
    
    # Alt4: No Persona, No Context (163 calls/topic, 5 outputs each)
    alt4_input = OVERHEAD_SMALL + topic_tokens
    alt4_output = 5 * stmt_tokens
    alt4_calls = 163
    costs.append({
        "name": "Alt4 Statement Gen",
        "model": "gpt-5-mini",
        "calls": alt4_calls,
        "input_tokens": alt4_input,
        "output_tokens": alt4_output,
    })
    
    # =========================================================================
    # Phase 2: Preference Building
    # =========================================================================
    
    # Iterative Ranking (24,000 calls/topic = 500 per rep × 48 reps)
    # Each call shows ~60 statements on average (100->80->60->40->20, avg ~60)
    avg_stmts_per_round = 60
    ranking_input = OVERHEAD_MEDIUM + persona_tokens + topic_tokens + (avg_stmts_per_round * stmt_tokens)
    ranking_output = 50  # JSON with 20 codes
    ranking_calls = 24000
    costs.append({
        "name": "Iterative Ranking",
        "model": "gpt-5-mini",
        "calls": ranking_calls,
        "input_tokens": ranking_input,
        "output_tokens": ranking_output,
    })
    
    # =========================================================================
    # Phase 3: Winner Selection
    # =========================================================================
    
    # GPT Selection (240 calls/topic)
    gpt_input = OVERHEAD_MEDIUM + (20 * stmt_tokens)  # ~20 statements shown
    gpt_output = 30  # JSON response
    gpt_calls = 240
    costs.append({
        "name": "GPT Selection",
        "model": "gpt-5.2",
        "calls": gpt_calls,
        "input_tokens": gpt_input,
        "output_tokens": gpt_output,
    })
    
    # GPT+Rank Selection (240 calls/topic)
    # Full rankings for K=20 sampled voters over P=20 sampled statements
    # "Voter N: 5 > 3 > 1 > ... > 19" ≈ 100 chars ≈ 25 tokens per voter
    ranking_per_voter = 25  # Rankings are over P=20 sampled statements
    gpt_rank_input = OVERHEAD_LARGE + (P_ALTS * stmt_tokens) + (K_VOTERS * ranking_per_voter)
    gpt_rank_output = 30
    gpt_rank_calls = 240
    costs.append({
        "name": "GPT+Rank Selection",
        "model": "gpt-5.2",
        "calls": gpt_rank_calls,
        "input_tokens": gpt_rank_input,
        "output_tokens": gpt_rank_output,
    })
    
    # GPT+Pers Selection (240 calls/topic)
    # K=20 sampled voters with filtered personas (7 key fields each)
    gpt_pers_input = OVERHEAD_LARGE + (P_ALTS * stmt_tokens) + (K_VOTERS * filtered_persona_tokens)
    gpt_pers_output = 30
    gpt_pers_calls = 240
    costs.append({
        "name": "GPT+Pers Selection",
        "model": "gpt-5.2",
        "calls": gpt_pers_calls,
        "input_tokens": gpt_pers_input,
        "output_tokens": gpt_pers_output,
    })
    
    # GPT* Selection (240 calls/topic) - shows all 100 statements (full text, no truncation)
    gpt_star_input = OVERHEAD_LARGE + (N_STATEMENTS * stmt_tokens)
    gpt_star_output = 30
    gpt_star_calls = 240
    costs.append({
        "name": "GPT* Selection",
        "model": "gpt-5.2",
        "calls": gpt_star_calls,
        "input_tokens": gpt_star_input,
        "output_tokens": gpt_star_output,
    })
    
    # GPT*+Rank Selection (240 calls/topic) - all statements + K=20 sampled voters' rankings
    gpt_star_rank_input = OVERHEAD_LARGE + (N_STATEMENTS * stmt_tokens) + (K_VOTERS * ranking_per_voter)
    gpt_star_rank_calls = 240
    costs.append({
        "name": "GPT*+Rank Selection",
        "model": "gpt-5.2",
        "calls": gpt_star_rank_calls,
        "input_tokens": gpt_star_rank_input,
        "output_tokens": gpt_star_output,
    })
    
    # GPT*+Pers Selection (240 calls/topic) - all statements + K=20 sampled voters' filtered personas
    gpt_star_pers_input = OVERHEAD_LARGE + (N_STATEMENTS * stmt_tokens) + (K_VOTERS * filtered_persona_tokens)
    gpt_star_pers_calls = 240
    costs.append({
        "name": "GPT*+Pers Selection",
        "model": "gpt-5.2",
        "calls": gpt_star_pers_calls,
        "input_tokens": gpt_star_pers_input,
        "output_tokens": gpt_star_output,
    })
    
    # GPT** Generation (240 calls/topic) - base variant
    gpt_double_input = OVERHEAD_MEDIUM + (P_ALTS * stmt_tokens)
    gpt_double_output = stmt_tokens
    gpt_double_calls = 240
    costs.append({
        "name": "GPT** Generation",
        "model": "gpt-5.2",
        "calls": gpt_double_calls,
        "input_tokens": gpt_double_input,
        "output_tokens": gpt_double_output,
    })
    
    # GPT**+Rank Generation (240 calls/topic) - with K=20 sampled voters' rankings
    gpt_double_rank_input = OVERHEAD_MEDIUM + (P_ALTS * stmt_tokens) + (K_VOTERS * ranking_per_voter)
    gpt_double_rank_calls = 240
    costs.append({
        "name": "GPT**+Rank Generation",
        "model": "gpt-5.2",
        "calls": gpt_double_rank_calls,
        "input_tokens": gpt_double_rank_input,
        "output_tokens": gpt_double_output,
    })
    
    # GPT**+Pers Generation (240 calls/topic) - with K=20 sampled voters' filtered personas
    gpt_double_pers_input = OVERHEAD_MEDIUM + (P_ALTS * stmt_tokens) + (K_VOTERS * filtered_persona_tokens)
    gpt_double_pers_calls = 240
    costs.append({
        "name": "GPT**+Pers Generation",
        "model": "gpt-5.2",
        "calls": gpt_double_pers_calls,
        "input_tokens": gpt_double_pers_input,
        "output_tokens": gpt_double_output,
    })
    
    # GPT*** Generation (48 calls/topic)
    gpt_triple_input = OVERHEAD_SMALL + topic_tokens
    gpt_triple_output = stmt_tokens
    gpt_triple_calls = 48
    costs.append({
        "name": "GPT*** Generation",
        "model": "gpt-5.2",
        "calls": gpt_triple_calls,
        "input_tokens": gpt_triple_input,
        "output_tokens": gpt_triple_output,
    })
    
    # =========================================================================
    # Epsilon Computation: Statement Insertion
    # =========================================================================
    
    # GPT** Insertion (72,000 = 720 statements × 100 voters)
    insertion_input = OVERHEAD_LARGE + persona_tokens + topic_tokens + (100 * stmt_tokens * 0.3) + stmt_tokens
    insertion_output = 20  # JSON with position
    gpt_double_insertion_calls = 72000
    costs.append({
        "name": "GPT** Insertion",
        "model": "gpt-5-mini",
        "calls": gpt_double_insertion_calls,
        "input_tokens": insertion_input,
        "output_tokens": insertion_output,
    })
    
    # GPT*** Insertion (4,800 = 48 statements × 100 voters)
    gpt_triple_insertion_calls = 4800
    costs.append({
        "name": "GPT*** Insertion",
        "model": "gpt-5-mini",
        "calls": gpt_triple_insertion_calls,
        "input_tokens": insertion_input,
        "output_tokens": insertion_output,
    })
    
    return costs

def print_cost_report(costs):
    """Print formatted cost report."""
    print("=" * 100)
    print("COST BREAKDOWN PER TOPIC")
    print("=" * 100)
    print(f"{'Component':<25} {'Model':<12} {'Calls':>10} {'Input/Call':>12} {'Output/Call':>12} {'Total Cost':>12}")
    print("-" * 100)
    
    total_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0
    
    by_model = {"gpt-5-mini": {"input": 0, "output": 0, "cost": 0},
                "gpt-5.2": {"input": 0, "output": 0, "cost": 0}}
    
    for item in costs:
        model = item["model"]
        calls = item["calls"]
        input_tokens = item["input_tokens"]
        output_tokens = item["output_tokens"]
        
        total_input = calls * input_tokens
        total_output = calls * output_tokens
        
        input_cost = (total_input / 1_000_000) * PRICING[model]["input"]
        output_cost = (total_output / 1_000_000) * PRICING[model]["output"]
        cost = input_cost + output_cost
        
        total_cost += cost
        total_input_tokens += total_input
        total_output_tokens += total_output
        
        by_model[model]["input"] += total_input
        by_model[model]["output"] += total_output
        by_model[model]["cost"] += cost
        
        print(f"{item['name']:<25} {model:<12} {calls:>10,} {input_tokens:>12,} {output_tokens:>12,} ${cost:>10.2f}")
    
    print("-" * 100)
    print(f"{'TOTAL':<25} {'':<12} {sum(c['calls'] for c in costs):>10,} {'':<12} {'':<12} ${total_cost:>10.2f}")
    print()
    
    print("=" * 60)
    print("SUMMARY BY MODEL")
    print("=" * 60)
    for model, data in by_model.items():
        print(f"\n{model}:")
        print(f"  Input tokens:  {data['input']:>15,}")
        print(f"  Output tokens: {data['output']:>15,}")
        print(f"  Cost:          ${data['cost']:>14.2f}")
    
    print()
    print("=" * 60)
    print("GRAND TOTALS")
    print("=" * 60)
    print(f"Total input tokens:  {total_input_tokens:>15,}")
    print(f"Total output tokens: {total_output_tokens:>15,}")
    print(f"Total API calls:     {sum(c['calls'] for c in costs):>15,}")
    print(f"Total cost per topic: ${total_cost:>13.2f}")
    print(f"Total cost for 13 topics: ${total_cost * 13:>8.2f}")
    
    return total_cost

def main():
    token_estimates = estimate_prompt_tokens()
    costs = calculate_costs(token_estimates)
    total_cost = print_cost_report(costs)
    
    # Return summary for documentation
    return {
        "total_cost_per_topic": round(total_cost, 2),
        "total_calls_per_topic": sum(c["calls"] for c in costs),
    }

if __name__ == "__main__":
    main()
