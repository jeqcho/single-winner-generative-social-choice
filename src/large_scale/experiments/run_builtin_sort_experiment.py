"""
Run experiment with built-in sort for 100 statements and 25 discriminative personas.

This script:
- Loads first 100 statements for "What should guide laws concerning abortion?"
- Samples 25 discriminative personas
- Runs pairwise ranking for each persona using Python's built-in sort
- Applies voting methods: Plurality, Borda, IRV, ChatGPT, ChatGPT+Rankings
- Computes PVC using biclique algorithm
- Outputs a CSV with: Method | Winner Statement | In PVC (Yes/No)
"""

import csv
import json
import os
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import openai
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI()

# Setup logging
def setup_logging():
    """Setup timestamped logging directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"logs/{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "builtin_sort_experiment.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_dir

def load_statements(limit: int = 100) -> List[Dict]:
    """Load first N statements for abortion topic."""
    statements_file = Path("data/large_scale/prod/statements/what-should-guide-laws-concerning-abortion.json")
    with open(statements_file, 'r') as f:
        data = json.load(f)
    
    statements = data[:limit]
    logging.info(f"Loaded {len(statements)} statements from {statements_file}")
    return statements

def load_and_sample_personas(seed: int = 42, num_personas: int = 25) -> List[str]:
    """Load discriminative personas and sample a subset."""
    personas_file = Path("data/personas/prod/discriminative.json")
    with open(personas_file, 'r') as f:
        personas = json.load(f)
    
    random.seed(seed)
    sampled = random.sample(personas, min(num_personas, len(personas)))
    logging.info(f"Sampled {len(sampled)} personas with seed={seed}")
    return sampled


# Import pairwise ranking module (uses built-in sort now)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.large_scale.pairwise_ranking import rank_statements_pairwise
from src.large_scale.biclique import compute_proportional_veto_core


def get_preference_rankings_parallel(
    personas: List[str],
    statements: List[Dict],
    topic: str,
    max_workers: int = 25
) -> List[List[str]]:
    """
    Get preference rankings from all personas in parallel.
    
    Returns:
        Preference matrix where preferences[rank][voter] is the alternative at rank 'rank' for voter 'voter'
    """
    n_statements = len(statements)
    n_personas = len(personas)
    
    logging.info(f"Getting preference rankings from {n_personas} personas for {n_statements} statements")
    
    def process_persona(args):
        """Process a single persona and return (index, ranking)."""
        idx, persona = args
        logging.info(f"Processing persona {idx+1}/{n_personas}")
        ranking = rank_statements_pairwise(persona, statements, topic, client)
        return idx, ranking
    
    rankings = [None] * n_personas
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_persona, (i, persona)): i 
            for i, persona in enumerate(personas)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Ranking personas", unit="persona"):
            try:
                idx, ranking = future.result()
                rankings[idx] = ranking
            except Exception as e:
                logging.error(f"Error processing persona {futures[future]}: {e}")
                # Create a default ranking (sequential) if failed
                rankings[futures[future]] = list(range(n_statements))
    
    # Convert to preference matrix format: preferences[rank][voter]
    preferences = []
    for rank in range(n_statements):
        rank_row = []
        for voter in range(n_personas):
            statement_idx = rankings[voter][rank]
            rank_row.append(str(statement_idx))
        preferences.append(rank_row)
    
    return preferences


def evaluate_voting_methods(
    preference_matrix: List[List[str]],
    statements: List[Dict],
    personas: List[str],
    pvc: List
) -> Dict:
    """
    Evaluate voting methods: Plurality, Borda, IRV, ChatGPT, ChatGPT+Rankings.
    """
    from votekit import RankProfile, RankBallot
    from votekit.elections import Plurality, Borda, IRV
    
    m = len(preference_matrix)  # number of alternatives
    n = len(preference_matrix[0]) if preference_matrix else 0  # number of voters
    
    # Create candidate names with prefix
    candidates = [f"c{i}" for i in range(m)]
    
    # Convert to VoteKit format
    ballots = []
    for voter in range(n):
        ranking = []
        for rank in range(m):
            alt_idx = int(preference_matrix[rank][voter])
            ranking.append(frozenset([f"c{alt_idx}"]))
        ballots.append(RankBallot(ranking=tuple(ranking)))
    
    profile = RankProfile(ballots=ballots, candidates=candidates)
    
    results = {}
    pvc_set = set(pvc) if pvc else set()
    
    def extract_winner_idx(winner):
        """Convert winner from 'c{idx}' format back to int."""
        if winner and isinstance(winner, str) and winner.startswith('c'):
            return int(winner[1:])
        return None
    
    # Plurality
    try:
        plurality_election = Plurality(profile, m=1, tiebreak="random")
        elected = plurality_election.get_elected()
        winner_idx = extract_winner_idx(list(elected[0])[0] if elected and elected[0] else None)
        results["Plurality"] = {
            "winner_idx": winner_idx,
            "winner_statement": statements[winner_idx]["statement"][:100] + "..." if winner_idx is not None else None,
            "in_pvc": winner_idx in pvc_set if winner_idx is not None else False
        }
    except Exception as e:
        results["Plurality"] = {"winner_idx": None, "winner_statement": None, "in_pvc": False, "error": str(e)}
    
    # Borda
    try:
        borda_election = Borda(profile, m=1, tiebreak="random")
        elected = borda_election.get_elected()
        winner_idx = extract_winner_idx(list(elected[0])[0] if elected and elected[0] else None)
        results["Borda"] = {
            "winner_idx": winner_idx,
            "winner_statement": statements[winner_idx]["statement"][:100] + "..." if winner_idx is not None else None,
            "in_pvc": winner_idx in pvc_set if winner_idx is not None else False
        }
    except Exception as e:
        results["Borda"] = {"winner_idx": None, "winner_statement": None, "in_pvc": False, "error": str(e)}
    
    # IRV
    try:
        irv_election = IRV(profile, tiebreak="random")
        elected = irv_election.get_elected()
        winner_idx = extract_winner_idx(list(elected[0])[0] if elected and elected[0] else None)
        results["IRV"] = {
            "winner_idx": winner_idx,
            "winner_statement": statements[winner_idx]["statement"][:100] + "..." if winner_idx is not None else None,
            "in_pvc": winner_idx in pvc_set if winner_idx is not None else False
        }
    except Exception as e:
        results["IRV"] = {"winner_idx": None, "winner_statement": None, "in_pvc": False, "error": str(e)}
    
    # ChatGPT baseline
    chatgpt_result = chatgpt_select_baseline(statements)
    winner_idx = chatgpt_result.get("winner_idx")
    results["ChatGPT"] = {
        "winner_idx": winner_idx,
        "winner_statement": statements[winner_idx]["statement"][:100] + "..." if winner_idx is not None else None,
        "in_pvc": winner_idx in pvc_set if winner_idx is not None else False
    }
    
    # ChatGPT + Rankings
    chatgpt_rankings_result = chatgpt_select_with_rankings(statements, preference_matrix)
    winner_idx = chatgpt_rankings_result.get("winner_idx")
    results["ChatGPT+Rankings"] = {
        "winner_idx": winner_idx,
        "winner_statement": statements[winner_idx]["statement"][:100] + "..." if winner_idx is not None else None,
        "in_pvc": winner_idx in pvc_set if winner_idx is not None else False
    }
    
    return results


def chatgpt_select_baseline(statements: List[Dict]) -> Dict:
    """ChatGPT baseline selection (just sees statements)."""
    statements_text = "\n\n".join([
        f"Statement {i}: {statement['statement']}"
        for i, statement in enumerate(statements)
    ])
    
    prompt = f"""Here are {len(statements)} statements from a discussion:

{statements_text}

Which statement do you think would be the best choice as a consensus/bridging statement? 
Consider which one best represents a reasonable middle ground that could satisfy diverse perspectives.

Return your choice as a JSON object with this format:
{{"selected_statement_index": 0}}

Where the value is the index (0-{len(statements)-1}) of the statement you select.
Return only the JSON, no additional text."""

    try:
        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": "You are a helpful assistant that selects consensus statements. Return ONLY valid JSON, no other text."},
                {"role": "user", "content": prompt}
            ]
        )
        result = json.loads(response.output_text)
        return {"winner_idx": result.get("selected_statement_index")}
    except Exception as e:
        logging.error(f"ChatGPT baseline error: {e}")
        return {"winner_idx": None, "error": str(e)}


def chatgpt_select_with_rankings(statements: List[Dict], preference_matrix: List[List[str]]) -> Dict:
    """ChatGPT selection with preference rankings."""
    statements_text = "\n\n".join([
        f"Statement {i}: {statement['statement']}"
        for i, statement in enumerate(statements)
    ])
    
    n_voters = len(preference_matrix[0]) if preference_matrix else 0
    rankings_summary = []
    for voter in range(min(n_voters, 10)):
        ranking = [preference_matrix[rank][voter] for rank in range(len(preference_matrix))]
        rankings_summary.append(f"Voter {voter+1}: {' > '.join(ranking[:10])}... (most to least preferred)")
    
    if n_voters > 10:
        rankings_summary.append(f"... and {n_voters - 10} more voters")
    
    rankings_text = "\n".join(rankings_summary)
    
    prompt = f"""Here are {len(statements)} statements from a discussion:

{statements_text}

Here are preference rankings from {n_voters} voters:

{rankings_text}

Based on both the statements and the preference rankings, which statement would be the best choice as a consensus/bridging statement?

Return your choice as a JSON object with this format:
{{"selected_statement_index": 0}}

Where the value is the index (0-{len(statements)-1}) of the statement you select.
Return only the JSON, no additional text."""

    try:
        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": "You are a helpful assistant that selects consensus statements. Return ONLY valid JSON, no other text."},
                {"role": "user", "content": prompt}
            ]
        )
        result = json.loads(response.output_text)
        return {"winner_idx": result.get("selected_statement_index")}
    except Exception as e:
        logging.error(f"ChatGPT+Rankings error: {e}")
        return {"winner_idx": None, "error": str(e)}


def generate_csv_table(results: Dict, pvc: List, output_file: str):
    """Generate CSV table with results."""
    rows = [["Method", "Winner Index", "Winner Statement (truncated)", "In PVC"]]
    
    for method, result in results.items():
        winner_idx = result.get("winner_idx", "N/A")
        winner_statement = result.get("winner_statement", "N/A")
        in_pvc = "Yes" if result.get("in_pvc", False) else "No"
        rows.append([method, str(winner_idx), winner_statement or "N/A", in_pvc])
    
    # Add PVC info row
    rows.append([])
    rows.append(["PVC Size", str(len(pvc)), f"Indices: {pvc[:20]}{'...' if len(pvc) > 20 else ''}", ""])
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    logging.info(f"CSV table written to {output_file}")


def run_experiment(num_statements: int = 100, num_personas: int = 25, seed: int = 42, max_workers: int = 25):
    """Main function to run the experiment."""
    log_dir = setup_logging()
    logging.info("=" * 80)
    logging.info("STARTING BUILT-IN SORT EXPERIMENT")
    logging.info(f"Statements: {num_statements}, Personas: {num_personas}, Seed: {seed}")
    logging.info("=" * 80)
    
    topic = "What should guide laws concerning abortion?"
    
    # Step 1: Load data
    logging.info("\n📊 Step 1: Loading data...")
    statements = load_statements(limit=num_statements)
    personas = load_and_sample_personas(seed=seed, num_personas=num_personas)
    
    # Step 2: Get preference rankings (parallelized)
    logging.info("\n🗳️ Step 2: Getting preference rankings...")
    preference_matrix = get_preference_rankings_parallel(personas, statements, topic, max_workers=max_workers)
    logging.info(f"Got preference matrix: {len(preference_matrix)} ranks x {len(preference_matrix[0])} voters")
    
    # Step 3: Compute PVC
    logging.info("\n🎯 Step 3: Computing PVC...")
    # Convert preference_matrix[rank][voter] -> profile[voter][rank]
    n_statements = len(preference_matrix)
    n_voters = len(preference_matrix[0]) if preference_matrix else 0
    
    profile = []
    for voter_idx in range(n_voters):
        voter_ranking = []
        for rank in range(n_statements):
            statement_idx = int(preference_matrix[rank][voter_idx])
            voter_ranking.append(statement_idx)
        profile.append(voter_ranking)
    
    pvc_result = compute_proportional_veto_core(profile)
    pvc = sorted(pvc_result.core)
    logging.info(f"PVC size: {len(pvc)} / {n_statements}")
    logging.info(f"PVC indices: {pvc}")
    logging.info(f"PVC parameters: r={pvc_result.r}, t={pvc_result.t}, alpha={pvc_result.alpha}")
    
    # Step 4: Evaluate voting methods
    logging.info("\n🏆 Step 4: Evaluating voting methods...")
    results = evaluate_voting_methods(preference_matrix, statements, personas, pvc)
    
    for method, result in results.items():
        in_pvc = "✓" if result.get("in_pvc", False) else "✗"
        winner_idx = result.get("winner_idx", "N/A")
        logging.info(f"  {in_pvc} {method}: winner={winner_idx}, in_pvc={result.get('in_pvc', False)}")
    
    # Step 5: Generate CSV
    logging.info("\n📝 Step 5: Generating CSV table...")
    output_dir = Path("data/large_scale/prod")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = output_dir / f"builtin_sort_experiment_{timestamp}.csv"
    generate_csv_table(results, pvc, str(csv_file))
    
    # Also save full results as JSON
    json_file = output_dir / f"builtin_sort_experiment_{timestamp}.json"
    full_results = {
        "topic": topic,
        "num_statements": num_statements,
        "num_personas": num_personas,
        "pvc": pvc,
        "pvc_size": len(pvc),
        "method_results": results,
        "preference_matrix": preference_matrix
    }
    with open(json_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    logging.info(f"Full results saved to {json_file}")
    
    logging.info("=" * 80)
    logging.info("EXPERIMENT COMPLETED")
    logging.info(f"CSV output: {csv_file}")
    logging.info(f"JSON output: {json_file}")
    logging.info(f"Logs: {log_dir}")
    logging.info("=" * 80)
    
    return results, pvc


if __name__ == "__main__":
    run_experiment(num_statements=100, num_personas=25, seed=42, max_workers=25)

