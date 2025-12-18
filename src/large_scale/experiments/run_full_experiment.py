"""
Run full experiment across 13 topics with 6 voting methods.

Configuration:
- 200 statements per topic (from existing 900+ statements)
- 50 discriminative personas for preference rankings
- 100 evaluative personas for Likert ratings and pairwise comparisons
- 6 voting methods: Plurality, Borda, IRV, ChatGPT, Schulze, Veto by Consumption

Output directory: data/large_scale/gen-200-disc-50-eval-100-nano-low/

Resilience: Individual persona results saved to intermediate/ folder for restart capability.
"""

import csv
import json
import os
import re
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import openai
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import numpy as np

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI()

# Constants
OUTPUT_BASE = Path("data/large_scale/gen-200-disc-50-eval-50-nano-low")
STATEMENTS_DIR = Path("data/large_scale/prod/statements")
PERSONAS_DIR = Path("data/personas/prod")
TOPICS_FILE = Path("data/topics.txt")

NUM_STATEMENTS = 200
NUM_DISC_PERSONAS = 50
NUM_EVAL_PERSONAS = 50  # Using available 50 evaluative personas
MAX_WORKERS = 50

VOTING_METHODS = ["plurality", "borda", "irv", "chatgpt", "schulze", "veto_by_consumption"]


def setup_logging():
    """Setup timestamped logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = OUTPUT_BASE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"experiment_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_file}")
    return log_dir


def slugify(text: str) -> str:
    """Convert text to filesystem-safe slug."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text[:50]


def load_topics() -> List[str]:
    """Load topics from file."""
    with open(TOPICS_FILE, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_statements(topic_slug: str, limit: int = NUM_STATEMENTS) -> List[Dict]:
    """Load statements for a topic."""
    # Find matching file
    for f in STATEMENTS_DIR.glob("*.json"):
        if f.stem.startswith(topic_slug[:20]):
            with open(f, 'r') as fp:
                data = json.load(fp)
            return data[:limit]
    raise FileNotFoundError(f"No statements file found for {topic_slug}")


def load_personas(persona_type: str, limit: int) -> List[str]:
    """Load personas of specified type."""
    filepath = PERSONAS_DIR / f"{persona_type}.json"
    with open(filepath, 'r') as f:
        personas = json.load(f)
    random.seed(42)
    return random.sample(personas, min(limit, len(personas)))


# ============================================================================
# DISCRIMINATIVE RANKING WITH RESILIENCE
# ============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def pairwise_compare(persona: str, stmt_a: Dict, stmt_b: Dict, topic: str) -> int:
    """Compare two statements and return preference."""
    prompt = f"""You are a person with the following characteristics:
{persona}

Given the topic: "{topic}"

Compare these two statements and indicate which one you prefer:

Statement A: {stmt_a['statement']}

Statement B: {stmt_b['statement']}

Return your choice as JSON: {{"preference": "A"}} or {{"preference": "B"}} or {{"preference": "equal"}}
Return only JSON, no other text."""

    response = client.responses.create(
        model="gpt-5-nano",
        input=[
            {"role": "system", "content": "You are evaluating statements. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    
    result = json.loads(response.output_text)
    pref = result.get("preference", "equal").upper()
    
    if pref == "A":
        return -1
    elif pref == "B":
        return 1
    return 0


def rank_statements_for_persona(
    persona_idx: int,
    persona: str,
    statements: List[Dict],
    topic: str,
    intermediate_dir: Path
) -> List[int]:
    """Rank statements for a single persona with intermediate saving."""
    output_file = intermediate_dir / f"persona_{persona_idx}.json"
    
    # Check if already completed
    if output_file.exists():
        with open(output_file, 'r') as f:
            data = json.load(f)
        logging.info(f"  Persona {persona_idx}: loaded from cache")
        return data["ranking"]
    
    # Create indexed statements
    indexed = [{"index": i, "statement": stmt["statement"]} for i, stmt in enumerate(statements)]
    
    comparison_count = [0]
    
    def compare(a: Dict, b: Dict) -> int:
        comparison_count[0] += 1
        return pairwise_compare(persona, a, b, topic)
    
    from functools import cmp_to_key
    sorted_stmts = sorted(indexed, key=cmp_to_key(compare))
    ranking = [s["index"] for s in sorted_stmts]
    
    # Save immediately
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({"persona_idx": persona_idx, "ranking": ranking, "comparisons": comparison_count[0]}, f)
    
    logging.info(f"  Persona {persona_idx}: {comparison_count[0]} comparisons")
    return ranking


def get_discriminative_rankings(
    personas: List[str],
    statements: List[Dict],
    topic: str,
    topic_slug: str
) -> List[List[str]]:
    """Get preference rankings from all discriminative personas."""
    intermediate_dir = OUTPUT_BASE / "intermediate" / topic_slug / "discriminative_rankings"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    n_statements = len(statements)
    n_personas = len(personas)
    
    logging.info(f"Getting rankings from {n_personas} personas for {n_statements} statements")
    
    def process_persona(args):
        idx, persona = args
        return idx, rank_statements_for_persona(idx, persona, statements, topic, intermediate_dir)
    
    rankings = [None] * n_personas
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_persona, (i, p)): i for i, p in enumerate(personas)}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Discriminative ranking"):
            try:
                idx, ranking = future.result()
                rankings[idx] = ranking
            except Exception as e:
                logging.error(f"Persona {futures[future]} failed: {e}")
                rankings[futures[future]] = list(range(n_statements))
    
    # Convert to preference matrix
    preferences = []
    for rank in range(n_statements):
        row = [str(rankings[v][rank]) for v in range(n_personas)]
        preferences.append(row)
    
    return preferences


# ============================================================================
# EVALUATIVE LIKERT RATINGS
# ============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def get_likert_rating(persona: str, statement: str, topic: str) -> int:
    """Get Likert rating (1-5) from persona for a statement."""
    prompt = f"""You are a person with the following characteristics:
{persona}

Given the topic: "{topic}"

Rate how much you agree with this statement (1-5):

Statement: {statement}

1 = Strongly disagree, 2 = Disagree, 3 = Neutral, 4 = Agree, 5 = Strongly agree

Return JSON: {{"rating": 3}}
Return only JSON."""

    response = client.responses.create(
        model="gpt-5-nano",
        input=[
            {"role": "system", "content": "You are rating statements. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    
    result = json.loads(response.output_text)
    rating = result.get("rating", 3)
    return max(1, min(5, int(rating)))


def get_evaluative_likert_ratings(
    personas: List[str],
    winning_statements: Dict[str, Dict],  # method -> {"idx": int, "statement": str}
    topic: str,
    topic_slug: str
) -> Dict[str, List[int]]:
    """Get Likert ratings from evaluative personas for winning statements."""
    intermediate_dir = OUTPUT_BASE / "intermediate" / topic_slug / "evaluative_likert"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    # Results: method -> list of ratings (one per persona)
    results = {method: [] for method in winning_statements.keys()}
    
    def process_task(args):
        persona_idx, persona, method, stmt = args
        cache_file = intermediate_dir / f"persona_{persona_idx}_{method}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return persona_idx, method, json.load(f)["rating"]
        
        rating = get_likert_rating(persona, stmt, topic)
        
        with open(cache_file, 'w') as f:
            json.dump({"persona_idx": persona_idx, "method": method, "rating": rating}, f)
        
        return persona_idx, method, rating
    
    # Create all tasks
    tasks = []
    for persona_idx, persona in enumerate(personas):
        for method, stmt_info in winning_statements.items():
            tasks.append((persona_idx, persona, method, stmt_info["statement"]))
    
    # Initialize results structure
    method_ratings = {method: [None] * len(personas) for method in winning_statements.keys()}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_task, t) for t in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluative Likert"):
            try:
                persona_idx, method, rating = future.result()
                method_ratings[method][persona_idx] = rating
            except Exception as e:
                logging.error(f"Likert rating failed: {e}")
    
    return method_ratings


# ============================================================================
# EVALUATIVE PAIRWISE COMPARISONS
# ============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def pairwise_compare_statements(persona: str, stmt_a: str, stmt_b: str, topic: str) -> str:
    """Compare two statements and return which is preferred ("A" or "B")."""
    prompt = f"""You are a person with the following characteristics:
{persona}

Given the topic: "{topic}"

Which statement do you prefer?

Statement A: {stmt_a}

Statement B: {stmt_b}

You MUST choose either A or B. Return JSON: {{"preference": "A"}} or {{"preference": "B"}}
Return only JSON."""

    response = client.responses.create(
        model="gpt-5-nano",
        input=[
            {"role": "system", "content": "You are comparing statements. Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    
    result = json.loads(response.output_text)
    return result.get("preference", "A").upper()


def get_evaluative_pairwise(
    personas: List[str],
    winning_statements: Dict[str, Dict],
    topic: str,
    topic_slug: str
) -> Dict[Tuple[str, str], Dict]:
    """Get pairwise comparisons between all pairs of winning statements."""
    intermediate_dir = OUTPUT_BASE / "intermediate" / topic_slug / "evaluative_pairwise"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    methods = list(winning_statements.keys())
    pairs = [(m1, m2) for i, m1 in enumerate(methods) for m2 in methods[i+1:]]
    
    # Results: (method1, method2) -> {"method1_wins": count, "method2_wins": count}
    results = {pair: {"m1_wins": 0, "m2_wins": 0} for pair in pairs}
    
    def process_task(args):
        persona_idx, persona, m1, m2 = args
        cache_file = intermediate_dir / f"persona_{persona_idx}_{m1}_vs_{m2}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return persona_idx, m1, m2, json.load(f)["winner"]
        
        stmt1 = winning_statements[m1]["statement"]
        stmt2 = winning_statements[m2]["statement"]
        pref = pairwise_compare_statements(persona, stmt1, stmt2, topic)
        winner = m1 if pref == "A" else m2
        
        with open(cache_file, 'w') as f:
            json.dump({"persona_idx": persona_idx, "m1": m1, "m2": m2, "winner": winner}, f)
        
        return persona_idx, m1, m2, winner
    
    # Create all tasks
    tasks = []
    for persona_idx, persona in enumerate(personas):
        for m1, m2 in pairs:
            tasks.append((persona_idx, persona, m1, m2))
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_task, t) for t in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluative pairwise"):
            try:
                persona_idx, m1, m2, winner = future.result()
                if winner == m1:
                    results[(m1, m2)]["m1_wins"] += 1
                else:
                    results[(m1, m2)]["m2_wins"] += 1
            except Exception as e:
                logging.error(f"Pairwise comparison failed: {e}")
    
    return results


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_topic_experiment(topic: str, disc_personas: List[str], eval_personas: List[str]) -> Dict:
    """Run full experiment for a single topic."""
    topic_slug = slugify(topic)
    results_file = OUTPUT_BASE / "results" / f"{topic_slug}.json"
    
    # Check if already completed
    if results_file.exists():
        logging.info(f"Topic {topic_slug}: loading from cache")
        with open(results_file, 'r') as f:
            return json.load(f)
    
    logging.info(f"\n{'='*80}")
    logging.info(f"TOPIC: {topic}")
    logging.info(f"{'='*80}")
    
    # Step 1: Load statements
    logging.info("\n📊 Step 1: Loading statements...")
    statements = load_statements(topic_slug, limit=NUM_STATEMENTS)
    logging.info(f"Loaded {len(statements)} statements")
    
    # Step 2: Get discriminative rankings
    logging.info("\n🗳️ Step 2: Getting discriminative rankings...")
    preference_matrix = get_discriminative_rankings(disc_personas, statements, topic, topic_slug)
    
    # Step 3: Compute PVC
    logging.info("\n🎯 Step 3: Computing PVC...")
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from src.large_scale.biclique import compute_proportional_veto_core
    
    n_statements = len(preference_matrix)
    n_voters = len(preference_matrix[0])
    
    profile = []
    for voter_idx in range(n_voters):
        voter_ranking = [int(preference_matrix[rank][voter_idx]) for rank in range(n_statements)]
        profile.append(voter_ranking)
    
    pvc_result = compute_proportional_veto_core(profile)
    pvc = sorted(pvc_result.core)
    logging.info(f"PVC size: {len(pvc)} / {n_statements}")
    
    # Step 4: Evaluate voting methods
    logging.info("\n🏆 Step 4: Evaluating voting methods...")
    from src.large_scale.voting_methods import evaluate_six_methods
    
    method_results = evaluate_six_methods(preference_matrix, statements, client, pvc=pvc)
    
    # Build winning statements dict
    winning_statements = {}
    for method in VOTING_METHODS:
        if method in method_results and method_results[method].get("winner") is not None:
            winner_idx = int(method_results[method]["winner"])
            winning_statements[method] = {
                "idx": winner_idx,
                "statement": statements[winner_idx]["statement"]
            }
    
    for method, result in method_results.items():
        in_pvc = "✓" if result.get("in_pvc") else "✗"
        logging.info(f"  {in_pvc} {method}: winner={result.get('winner')}")
    
    # Step 5: Get evaluative Likert ratings
    logging.info("\n⭐ Step 5: Getting evaluative Likert ratings...")
    likert_ratings = get_evaluative_likert_ratings(eval_personas, winning_statements, topic, topic_slug)
    
    # Step 6: Get evaluative pairwise comparisons
    logging.info("\n🔄 Step 6: Getting evaluative pairwise comparisons...")
    pairwise_results = get_evaluative_pairwise(eval_personas, winning_statements, topic, topic_slug)
    
    # Compile results
    results = {
        "topic": topic,
        "topic_slug": topic_slug,
        "num_statements": len(statements),
        "num_disc_personas": len(disc_personas),
        "num_eval_personas": len(eval_personas),
        "pvc": pvc,
        "pvc_size": len(pvc),
        "method_results": method_results,
        "winning_statements": winning_statements,
        "likert_ratings": likert_ratings,
        "pairwise_results": {f"{k[0]}_vs_{k[1]}": v for k, v in pairwise_results.items()},
        "preference_matrix": preference_matrix
    }
    
    # Save results
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"✅ Topic completed: {topic_slug}")
    return results


def main():
    """Main entry point."""
    setup_logging()
    
    logging.info("="*80)
    logging.info("FULL EXPERIMENT: 13 Topics, 6 Voting Methods")
    logging.info(f"Output: {OUTPUT_BASE}")
    logging.info("="*80)
    
    # Load topics and personas
    topics = load_topics()
    logging.info(f"Loaded {len(topics)} topics")
    
    disc_personas = load_personas("discriminative", NUM_DISC_PERSONAS)
    eval_personas = load_personas("evaluative", NUM_EVAL_PERSONAS)
    logging.info(f"Loaded {len(disc_personas)} discriminative, {len(eval_personas)} evaluative personas")
    
    # Run experiments
    all_results = []
    for i, topic in enumerate(topics):
        logging.info(f"\n\n{'#'*80}")
        logging.info(f"# TOPIC {i+1}/{len(topics)}: {topic[:60]}...")
        logging.info(f"{'#'*80}")
        
        try:
            result = run_topic_experiment(topic, disc_personas, eval_personas)
            all_results.append(result)
        except Exception as e:
            logging.error(f"Topic failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary
    summary_file = OUTPUT_BASE / "results" / "all_topics_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "num_topics": len(all_results),
            "topics": [r["topic"] for r in all_results],
            "config": {
                "num_statements": NUM_STATEMENTS,
                "num_disc_personas": NUM_DISC_PERSONAS,
                "num_eval_personas": NUM_EVAL_PERSONAS,
                "voting_methods": VOTING_METHODS
            }
        }, f, indent=2)
    
    logging.info("\n" + "="*80)
    logging.info(f"EXPERIMENT COMPLETED: {len(all_results)}/{len(topics)} topics")
    logging.info(f"Results: {OUTPUT_BASE / 'results'}")
    logging.info("="*80)


if __name__ == "__main__":
    main()

