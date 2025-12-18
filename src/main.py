"""
Main orchestration script for social choice experiment pipeline.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

from src.persona_generation import generate_personas
from src.generative_queries import generate_statements
from src.generate_summaries import generate_summaries
from src.discriminative_queries import get_preference_rankings
from src.compute_pvc import compute_pvc
from src.evaluate_methods import evaluate_methods
from src.bridging_evaluation import evaluate_bridging


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text[:50]  # Limit length


def load_topics(filepath: str = "data/topics.txt") -> List[str]:
    """Load topics from file."""
    with open(filepath, 'r') as f:
        topics = [line.strip() for line in f if line.strip()]
    return topics


def run_experiment(
    topic: str,
    openai_client: OpenAI,
    skip_summaries: bool = False,
    output_dir: str = "data/results"
) -> Dict:
    """
    Run a single experiment for a topic.
    
    Args:
        topic: The topic/question to run the experiment on
        openai_client: OpenAI client instance
        skip_summaries: Whether to skip summary generation
        output_dir: Directory to save results
    
    Returns:
        Dictionary with all experiment results
    """
    print(f"\n{'='*80}")
    print(f"Running experiment for topic: {topic}")
    print(f"{'='*80}\n")
    
    # Step 1: Generate 30 personas (split into 3 groups of 10)
    print("Step 1: Generating 30 personas...")
    all_personas = generate_personas(n=30, openai_client=openai_client)
    
    persona_generators = all_personas[0:10]
    persona_rankers = all_personas[10:20]
    persona_evaluators = all_personas[20:30]
    
    print(f"  Generated {len(all_personas)} personas")
    print(f"  - Generators: {len(persona_generators)}")
    print(f"  - Rankers: {len(persona_rankers)}")
    print(f"  - Evaluators: {len(persona_evaluators)}")
    
    # Step 2: Generate 10 statements
    print("\nStep 2: Generating 10 statements...")
    statements = generate_statements(topic, persona_generators, openai_client)
    print(f"  Generated {len(statements)} statements")
    
    # Step 3: (Optional) Generate 10 summaries
    summaries = None
    if not skip_summaries:
        print("\nStep 3: Generating 10 summaries...")
        summaries = generate_summaries(statements, persona_generators, openai_client)
        print(f"  Generated {len(summaries)} summaries")
    else:
        print("\nStep 3: Skipping summary generation (using statements as summaries)")
        # Use statements as summaries for ranking
        summaries = [
            {"persona": stmt["persona"], "summary": stmt["statement"]}
            for stmt in statements
        ]
    
    # Step 4: Get preference rankings
    print("\nStep 4: Getting preference rankings...")
    # Create alternative labels (indices as strings)
    alternatives = [str(i) for i in range(len(summaries))]
    
    preference_matrix = get_preference_rankings(summaries, persona_rankers, openai_client)
    print(f"  Got rankings from {len(persona_rankers)} personas")
    
    # Step 5: Compute PVC
    print("\nStep 5: Computing PVC...")
    pvc_result = compute_pvc(preference_matrix, alternatives)
    print(f"  PVC result: {pvc_result}")
    
    # Step 6: Evaluate other methods
    print("\nStep 6: Evaluating other methods...")
    method_results = evaluate_methods(preference_matrix, summaries, openai_client)
    
    # Check if winners are in PVC
    pvc_set = set(pvc_result)
    for method_name, method_result in method_results.items():
        winner = method_result.get("winner")
        if winner is not None:
            method_result["in_pvc"] = winner in pvc_set
            print(f"  {method_name}: winner={winner}, in_pvc={method_result['in_pvc']}")
        else:
            print(f"  {method_name}: no winner (error: {method_result.get('error', 'unknown')})")
    
    # Step 7: Evaluate bridging quality
    print("\nStep 7: Evaluating bridging quality...")
    bridging_eval = evaluate_bridging(
        pvc_result, summaries, statements, persona_evaluators, openai_client
    )
    print(f"  Got evaluations from {len(persona_evaluators)} personas")
    
    # Compile results
    results = {
        "topic": topic,
        "personas": {
            "generators": persona_generators,
            "rankers": persona_rankers,
            "evaluators": persona_evaluators
        },
        "statements": statements,
        "summaries": summaries,
        "preference_rankings": preference_matrix,
        "pvc": pvc_result,
        "method_results": method_results,
        "bridging_evaluation": bridging_eval
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    topic_slug = slugify(topic)
    output_path = os.path.join(output_dir, f"{topic_slug}.json")
    
    # Check if file exists and skip if it does
    if os.path.exists(output_path):
        print(f"\nSkipping {topic_slug}.json (already exists)")
        return None
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run social choice experiment pipeline")
    parser.add_argument(
        "--skip-summaries",
        action="store_true",
        help="Skip summary generation step"
    )
    parser.add_argument(
        "--topic-index",
        type=int,
        help="Run only topic at index N (0-indexed)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Output directory for results (default: data/results)"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=api_key)
    
    # Load topics
    topics = load_topics("data/topics.txt")
    print(f"Loaded {len(topics)} topics")
    
    # Filter topics if index specified
    if args.topic_index is not None:
        if args.topic_index < 0 or args.topic_index >= len(topics):
            raise ValueError(f"Topic index {args.topic_index} out of range (0-{len(topics)-1})")
        topics = [topics[args.topic_index]]
        print(f"Running only topic at index {args.topic_index}")
    
    # Run experiments
    all_results = []
    for i, topic in enumerate(topics):
        try:
            result = run_experiment(
                topic,
                openai_client,
                skip_summaries=args.skip_summaries,
                output_dir=args.output_dir
            )
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR processing topic {i} ({topic}): {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"Completed {len(all_results)}/{len(topics)} experiments")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

