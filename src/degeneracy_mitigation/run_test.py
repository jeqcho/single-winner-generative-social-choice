"""
Main CLI entry point for degeneracy mitigation tests.

Run either Approach A (iterative ranking) or Approach B (scoring) 
across reasoning effort levels.
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from .config import (
    MODEL,
    REASONING_EFFORTS,
    TEST_TOPIC,
    TEST_REP,
    N_VOTERS,
    N_STATEMENTS,
    MAX_WORKERS,
    OUTPUT_DIR,
    PERSONAS_PATH,
    SAMPLED_CONTEXT_DIR,
    SAMPLED_STATEMENTS_DIR,
    TOPIC_SLUGS,
    TOPIC_QUESTIONS,
    HASH_SEED,
    api_timer,
)
from .iterative_ranking import rank_voter
from .iterative_ranking_star import rank_voter as rank_voter_star
from .scoring_ranking import score_voter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_personas(personas_path: Path) -> list[str]:
    """Load personas from JSON file."""
    with open(personas_path, 'r') as f:
        data = json.load(f)
    
    # Handle both list format and dict format
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Assume values are persona strings
        return list(data.values())
    else:
        raise ValueError(f"Unexpected persona file format: {type(data)}")


def load_statements_for_rep(topic: str, rep: int) -> list[dict]:
    """
    Load statements for a specific topic and rep.
    
    Uses the sampled context statements (Alt1) for the given rep.
    """
    # Context files use short topic names (abortion, electoral)
    context_path = SAMPLED_CONTEXT_DIR / topic / f"rep{rep}.json"
    if not context_path.exists():
        raise FileNotFoundError(f"Context file not found: {context_path}")
    
    with open(context_path, 'r') as f:
        context_data = json.load(f)
    
    # Load pre-generated Alt1 statements (also use short topic names)
    alt1_path = SAMPLED_STATEMENTS_DIR / "persona_no_context" / f"{topic}.json"
    if not alt1_path.exists():
        raise FileNotFoundError(f"Alt1 statements not found: {alt1_path}")
    
    with open(alt1_path, 'r') as f:
        all_statements_data = json.load(f)
    
    # all_statements_data has format: {"statements": {"persona_id": "text", ...}}
    all_statements = all_statements_data.get('statements', all_statements_data)
    
    # Get the context persona IDs
    context_persona_ids = context_data.get('context_persona_ids', [])
    if len(context_persona_ids) < N_STATEMENTS:
        logger.warning(f"Only {len(context_persona_ids)} context IDs, expected {N_STATEMENTS}")
    
    # Build statement list using the context persona IDs
    statements = []
    for persona_id in context_persona_ids[:N_STATEMENTS]:
        persona_id_str = str(persona_id)
        if persona_id_str in all_statements:
            stmt_text = all_statements[persona_id_str]
            statements.append({'statement': stmt_text, 'persona_id': persona_id_str})
        else:
            logger.warning(f"Persona ID {persona_id_str} not found in statements")
    
    if len(statements) < N_STATEMENTS:
        logger.warning(f"Only loaded {len(statements)} statements, expected {N_STATEMENTS}")
    
    return statements


def load_voters_for_rep(topic: str, rep: int, personas: list[str]) -> list[str]:
    """
    Load or sample voters for a specific topic and rep.
    
    For this test, we just sample uniformly from all personas.
    """
    import random
    
    # Use deterministic sampling based on rep
    rng = random.Random(42 + rep)
    
    if len(personas) < N_VOTERS:
        logger.warning(f"Only {len(personas)} personas available, using all")
        return personas
    
    # Sample N_VOTERS uniformly
    indices = rng.sample(range(len(personas)), N_VOTERS)
    return [personas[i] for i in indices]


def run_approach_a(
    client: OpenAI,
    voters: list[str],
    statements: list[dict],
    topic_question: str,
    reasoning_effort: str,
    output_dir: Path,
    max_workers: int = MAX_WORKERS
) -> dict:
    """
    Run Approach A (iterative ranking) for all voters.
    
    Args:
        client: OpenAI client
        voters: List of persona strings
        statements: List of statement dicts
        topic_question: The topic question
        reasoning_effort: Reasoning effort level
        output_dir: Directory to save results
        max_workers: Maximum parallel workers
    
    Returns:
        Statistics dictionary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running Approach A (iterative ranking) with {reasoning_effort} reasoning")
    logger.info(f"  {len(voters)} voters × {len(statements)} statements")
    
    results = []
    voter_retries = {}
    
    def process_voter(voter_idx: int) -> dict:
        return rank_voter(
            client=client,
            voter_idx=voter_idx,
            persona=voters[voter_idx],
            statements=statements,
            topic=topic_question,
            reasoning_effort=reasoning_effort,
            hash_seed=HASH_SEED,
        )
    
    # Run in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_voter, i): i 
            for i in range(len(voters))
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc=f"Ranking ({reasoning_effort})"):
            voter_idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                voter_retries[voter_idx] = result['total_retries']
            except Exception as e:
                logger.error(f"Voter {voter_idx} failed: {e}")
                results.append({
                    'voter_idx': voter_idx,
                    'ranking': [],
                    'total_retries': 0,
                    'all_valid': False,
                    'error': str(e),
                })
                voter_retries[voter_idx] = -1  # Error marker
    
    # Sort results by voter index
    results.sort(key=lambda r: r['voter_idx'])
    
    # Extract rankings
    rankings = [r['ranking'] for r in results]
    
    # Compute statistics
    valid_count = sum(1 for r in results if r.get('all_valid', False))
    total_retries = sum(r.get('total_retries', 0) for r in results)
    voters_with_retries = sum(1 for r in results if r.get('total_retries', 0) > 0)
    
    # Retry distribution
    retry_dist = {}
    for retries in voter_retries.values():
        retry_dist[retries] = retry_dist.get(retries, 0) + 1
    
    stats = {
        'approach': 'A',
        'reasoning_effort': reasoning_effort,
        'n_voters': len(voters),
        'n_statements': len(statements),
        'valid_count': valid_count,
        'invalid_count': len(voters) - valid_count,
        'total_retries': total_retries,
        'voters_with_retries': voters_with_retries,
        'retry_distribution': retry_dist,
        'api_stats': api_timer.get_stats(),
    }
    
    # Save results
    with open(output_dir / 'rankings.json', 'w') as f:
        json.dump(rankings, f, indent=2)
    
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save detailed round logs
    round_logs_dir = output_dir / 'round_logs'
    round_logs_dir.mkdir(exist_ok=True)
    
    with open(round_logs_dir / 'voter_retries.json', 'w') as f:
        json.dump(voter_retries, f, indent=2)
    
    with open(round_logs_dir / 'full_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"  Valid: {valid_count}/{len(voters)}")
    logger.info(f"  Total retries: {total_retries}")
    
    return stats


def run_approach_a_star(
    client: OpenAI,
    voters: list[str],
    statements: list[dict],
    topic_question: str,
    reasoning_effort: str,
    output_dir: Path,
    max_workers: int = MAX_WORKERS
) -> dict:
    """
    Run Approach A* (iterative ranking with "least preferred first" for bottom-K).
    
    Variant of Approach A where bottom-K is requested with "least preferred first"
    instead of "least preferred last". The hypothesis is that outputting the worst
    statement first is cognitively easier than "reserving space" for it at the end.
    
    Args:
        client: OpenAI client
        voters: List of persona strings
        statements: List of statement dicts
        topic_question: The topic question
        reasoning_effort: Reasoning effort level
        output_dir: Directory to save results
        max_workers: Maximum parallel workers
    
    Returns:
        Statistics dictionary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running Approach A* (iterative ranking, bottom-K reversed) with {reasoning_effort} reasoning")
    logger.info(f"  {len(voters)} voters × {len(statements)} statements")
    
    results = []
    voter_retries = {}
    
    def process_voter(voter_idx: int) -> dict:
        return rank_voter_star(
            client=client,
            voter_idx=voter_idx,
            persona=voters[voter_idx],
            statements=statements,
            topic=topic_question,
            reasoning_effort=reasoning_effort,
            hash_seed=HASH_SEED,
        )
    
    # Run in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_voter, i): i 
            for i in range(len(voters))
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc=f"Ranking A* ({reasoning_effort})"):
            voter_idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                voter_retries[voter_idx] = result['total_retries']
            except Exception as e:
                logger.error(f"Voter {voter_idx} failed: {e}")
                results.append({
                    'voter_idx': voter_idx,
                    'ranking': [],
                    'total_retries': 0,
                    'all_valid': False,
                    'error': str(e),
                })
                voter_retries[voter_idx] = -1  # Error marker
    
    # Sort results by voter index
    results.sort(key=lambda r: r['voter_idx'])
    
    # Extract rankings
    rankings = [r['ranking'] for r in results]
    
    # Compute statistics
    valid_count = sum(1 for r in results if r.get('all_valid', False))
    total_retries = sum(r.get('total_retries', 0) for r in results)
    voters_with_retries = sum(1 for r in results if r.get('total_retries', 0) > 0)
    
    # Retry distribution
    retry_dist = {}
    for retries in voter_retries.values():
        retry_dist[retries] = retry_dist.get(retries, 0) + 1
    
    stats = {
        'approach': 'A*',
        'reasoning_effort': reasoning_effort,
        'n_voters': len(voters),
        'n_statements': len(statements),
        'valid_count': valid_count,
        'invalid_count': len(voters) - valid_count,
        'total_retries': total_retries,
        'voters_with_retries': voters_with_retries,
        'retry_distribution': retry_dist,
        'api_stats': api_timer.get_stats(),
    }
    
    # Save results
    with open(output_dir / 'rankings.json', 'w') as f:
        json.dump(rankings, f, indent=2)
    
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save detailed round logs
    round_logs_dir = output_dir / 'round_logs'
    round_logs_dir.mkdir(exist_ok=True)
    
    with open(round_logs_dir / 'voter_retries.json', 'w') as f:
        json.dump(voter_retries, f, indent=2)
    
    with open(round_logs_dir / 'full_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"  Valid: {valid_count}/{len(voters)}")
    logger.info(f"  Total retries: {total_retries}")
    
    return stats


def run_approach_b(
    client: OpenAI,
    voters: list[str],
    statements: list[dict],
    topic_question: str,
    reasoning_effort: str,
    output_dir: Path,
    max_workers: int = MAX_WORKERS
) -> dict:
    """
    Run Approach B (scoring) for all voters.
    
    Args:
        client: OpenAI client
        voters: List of persona strings
        statements: List of statement dicts
        topic_question: The topic question
        reasoning_effort: Reasoning effort level
        output_dir: Directory to save results
        max_workers: Maximum parallel workers
    
    Returns:
        Statistics dictionary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running Approach B (scoring) with {reasoning_effort} reasoning")
    logger.info(f"  {len(voters)} voters × {len(statements)} statements")
    
    results = []
    
    def process_voter(voter_idx: int) -> dict:
        return score_voter(
            client=client,
            voter_idx=voter_idx,
            persona=voters[voter_idx],
            statements=statements,
            topic=topic_question,
            reasoning_effort=reasoning_effort,
            hash_seed=HASH_SEED,
        )
    
    # Run in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_voter, i): i 
            for i in range(len(voters))
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc=f"Scoring ({reasoning_effort})"):
            voter_idx = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Voter {voter_idx} failed: {e}")
                results.append({
                    'voter_idx': voter_idx,
                    'scores': {},
                    'ranking': [],
                    'dedup_rounds': 0,
                    'has_unresolved_duplicates': True,
                    'error': str(e),
                })
    
    # Sort results by voter index
    results.sort(key=lambda r: r['voter_idx'])
    
    # Extract rankings and scores
    rankings = [r['ranking'] for r in results]
    scores = [r['scores'] for r in results]
    
    # Compute statistics
    valid_count = sum(1 for r in results if not r.get('has_unresolved_duplicates', True))
    total_dedup_rounds = sum(r.get('dedup_rounds', 0) for r in results)
    voters_needing_dedup = sum(1 for r in results if r.get('dedup_rounds', 0) > 0)
    unresolved_count = sum(1 for r in results if r.get('has_unresolved_duplicates', False))
    
    stats = {
        'approach': 'B',
        'reasoning_effort': reasoning_effort,
        'n_voters': len(voters),
        'n_statements': len(statements),
        'valid_count': valid_count,
        'unresolved_duplicates_count': unresolved_count,
        'total_dedup_rounds': total_dedup_rounds,
        'voters_needing_dedup': voters_needing_dedup,
        'api_stats': api_timer.get_stats(),
    }
    
    # Save results
    with open(output_dir / 'rankings.json', 'w') as f:
        json.dump(rankings, f, indent=2)
    
    with open(output_dir / 'scores.json', 'w') as f:
        json.dump(scores, f, indent=2)
    
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    with open(output_dir / 'full_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"  Valid (no unresolved dups): {valid_count}/{len(voters)}")
    logger.info(f"  Voters needing dedup: {voters_needing_dedup}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Run degeneracy mitigation tests"
    )
    parser.add_argument(
        '--approach', '-a',
        choices=['ranking', 'ranking_star', 'scoring', 'both', 'all'],
        default='both',
        help='Which approach to test: ranking (A), ranking_star (A*), scoring (B), both (A+B), all (A+A*+B)'
    )
    parser.add_argument(
        '--reasoning-effort', '-r',
        choices=['minimal', 'low', 'medium', 'all'],
        default='all',
        help='Reasoning effort level (default: all)'
    )
    parser.add_argument(
        '--topic', '-t',
        default=TEST_TOPIC,
        help=f'Topic short name (default: {TEST_TOPIC})'
    )
    parser.add_argument(
        '--rep',
        type=int,
        default=TEST_REP,
        help=f'Rep number (default: {TEST_REP})'
    )
    parser.add_argument(
        '--max-workers', '-w',
        type=int,
        default=MAX_WORKERS,
        help=f'Maximum parallel workers (default: {MAX_WORKERS})'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=OUTPUT_DIR,
        help=f'Output directory (default: {OUTPUT_DIR})'
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize OpenAI client
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    # Load data
    logger.info("Loading data...")
    
    personas = load_personas(PERSONAS_PATH)
    logger.info(f"  Loaded {len(personas)} personas")
    
    statements = load_statements_for_rep(args.topic, args.rep)
    logger.info(f"  Loaded {len(statements)} statements for {args.topic} rep{args.rep}")
    
    voters = load_voters_for_rep(args.topic, args.rep, personas)
    logger.info(f"  Selected {len(voters)} voters")
    
    topic_question = TOPIC_QUESTIONS.get(args.topic, args.topic)
    
    # Determine which reasoning efforts to test
    if args.reasoning_effort == 'all':
        efforts = REASONING_EFFORTS
    else:
        efforts = [args.reasoning_effort]
    
    # Determine which approaches to run
    run_ranking = args.approach in ['ranking', 'both', 'all']
    run_ranking_star = args.approach in ['ranking_star', 'all']
    run_scoring = args.approach in ['scoring', 'both', 'all']
    
    # Run tests
    all_stats = []
    
    for effort in efforts:
        api_timer.reset()  # Reset timer for each condition
        
        if run_ranking:
            output_dir = args.output_dir / 'approach_a' / effort
            stats = run_approach_a(
                client=client,
                voters=voters,
                statements=statements,
                topic_question=topic_question,
                reasoning_effort=effort,
                output_dir=output_dir,
                max_workers=args.max_workers,
            )
            all_stats.append(stats)
        
        if run_ranking_star:
            api_timer.reset()
            output_dir = args.output_dir / 'approach_a_star' / effort
            stats = run_approach_a_star(
                client=client,
                voters=voters,
                statements=statements,
                topic_question=topic_question,
                reasoning_effort=effort,
                output_dir=output_dir,
                max_workers=args.max_workers,
            )
            all_stats.append(stats)
        
        if run_scoring:
            api_timer.reset()
            output_dir = args.output_dir / 'approach_b' / effort
            stats = run_approach_b(
                client=client,
                voters=voters,
                statements=statements,
                topic_question=topic_question,
                reasoning_effort=effort,
                output_dir=output_dir,
                max_workers=args.max_workers,
            )
            all_stats.append(stats)
    
    # Save summary
    summary_path = args.output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'topic': args.topic,
            'rep': args.rep,
            'n_voters': len(voters),
            'n_statements': len(statements),
            'stats': all_stats,
        }, f, indent=2)
    
    logger.info(f"\nSummary saved to {summary_path}")
    logger.info("Done!")


if __name__ == '__main__':
    main()
