"""
Production experiment runner for the sampling experiment.

Runs the experiment on ALL topics.

Usage:
    uv run python -m src.sampling_experiment.run_production
    uv run python -m src.sampling_experiment.run_production --topics trust littering
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

from .config import (
    OUTPUT_DIR,
    ALL_TOPICS,
    N_REPS,
    K_VALUES,
    P_VALUES,
    TOPIC_SHORT_NAMES,
)
from .run_experiment import run_experiment, setup_logging
from .visualizer import plot_summary_across_topics


def main():
    parser = argparse.ArgumentParser(description="Run sampling experiment (all topics)")
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="Specific topics to run (use short names from TOPIC_SHORT_NAMES)"
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=N_REPS,
        help="Number of replications"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory"
    )
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip visualization generation"
    )
    
    args = parser.parse_args()
    
    # Determine which topics to run
    if args.topics:
        # Map short names back to full slugs
        short_to_full = {v: k for k, v in TOPIC_SHORT_NAMES.items()}
        topics = []
        for t in args.topics:
            if t in short_to_full:
                topics.append(short_to_full[t])
            elif t in ALL_TOPICS:
                topics.append(t)
            else:
                print(f"Unknown topic: {t}")
                print(f"Valid short names: {list(TOPIC_SHORT_NAMES.values())}")
                sys.exit(1)
    else:
        topics = ALL_TOPICS
    
    # Setup
    setup_logging(args.output_dir, test_mode=False)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting production run at {datetime.now().isoformat()}")
    logger.info(f"Topics: {len(topics)}")
    logger.info(f"Reps per topic: {args.reps}")
    logger.info(f"K values: {K_VALUES}")
    logger.info(f"P values: {P_VALUES}")
    logger.info(f"Total samples per topic: {len(K_VALUES) * len(P_VALUES) * args.reps}")
    
    # Create OpenAI client
    openai_client = OpenAI(timeout=120.0)
    
    # Run experiment on each topic
    for idx, topic_slug in enumerate(topics):
        logger.info(f"\n{'#' * 80}")
        logger.info(f"TOPIC {idx + 1}/{len(topics)}: {TOPIC_SHORT_NAMES.get(topic_slug, topic_slug)}")
        logger.info(f"{'#' * 80}")
        
        try:
            run_experiment(topic_slug, args.output_dir, openai_client, args.reps)
        except Exception as e:
            logger.error(f"Topic {topic_slug} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Generate summary visualizations across all topics
    if not args.skip_visualizations:
        logger.info("\n" + "=" * 80)
        logger.info("Generating summary visualizations across all topics...")
        try:
            plot_summary_across_topics(args.output_dir, topics, args.reps)
        except Exception as e:
            logger.error(f"Summary visualization failed: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("PRODUCTION RUN COMPLETE")
    logger.info(f"Finished at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
