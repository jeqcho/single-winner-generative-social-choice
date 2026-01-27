"""
Unified pipeline entry point for the sample-alt-voters experiment.

Usage:
    # Run full pipeline (default behavior, skips completed work):
    uv run python -m src.sample_alt_voters

    # Force re-run everything:
    uv run python -m src.sample_alt_voters --force

    # Run individual stages:
    uv run python -m src.sample_alt_voters --stage generate-statements
    uv run python -m src.sample_alt_voters --stage run-experiment
    uv run python -m src.sample_alt_voters --stage fix-epsilons
    uv run python -m src.sample_alt_voters --stage run-triple-star
    uv run python -m src.sample_alt_voters --stage visualize
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Base command for running Python modules
PYTHON_CMD = [sys.executable, "-m"]


def run_stage(stage_name: str, force: bool = False) -> bool:
    """
    Run a single pipeline stage.
    
    Returns True if successful, False otherwise.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"STAGE: {stage_name}")
    logger.info(f"{'='*60}\n")
    
    try:
        if stage_name == "generate-statements":
            # Generate Alt1 and Alt4 statements for all topics
            cmd = PYTHON_CMD + ["src.sample_alt_voters.generate_statements", "--all"]
            if force:
                cmd.append("--force")
            result = subprocess.run(cmd, check=True)
            
        elif stage_name == "run-experiment":
            # Run experiment for both voter distributions
            for voter_dist in ["uniform", "clustered"]:
                cmd = PYTHON_CMD + [
                    "src.sample_alt_voters.run_experiment",
                    "--voter-dist", voter_dist,
                    "--all-topics",
                    "--all-alts"
                ]
                if force:
                    cmd.append("--force")
                result = subprocess.run(cmd, check=True)
                
        elif stage_name == "fix-epsilons":
            # Fix epsilon values for GPT* and GPT** methods
            cmd = PYTHON_CMD + ["src.sample_alt_voters.fix_star_epsilons"]
            # Note: fix_star_epsilons already has skip logic (checks if epsilon is null)
            result = subprocess.run(cmd, check=True)
            
        elif stage_name == "run-triple-star":
            # Run GPT*** method
            cmd = PYTHON_CMD + ["src.sample_alt_voters.run_triple_star"]
            if force:
                cmd.append("--force")
            result = subprocess.run(cmd, check=True)
            
        elif stage_name == "visualize":
            # Generate visualization plots (always regenerates)
            cmd = PYTHON_CMD + ["src.sample_alt_voters.visualizer", "--all"]
            result = subprocess.run(cmd, check=True)
            
        else:
            logger.error(f"Unknown stage: {stage_name}")
            return False
            
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Stage '{stage_name}' failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"Stage '{stage_name}' failed: {e}")
        return False


def run_full_pipeline(force: bool = False) -> bool:
    """
    Run all pipeline stages in order.
    
    Returns True if all stages succeed, False otherwise.
    """
    stages = [
        "generate-statements",
        "run-experiment",
        "fix-epsilons",
        "run-triple-star",
        "visualize",
    ]
    
    logger.info("="*60)
    logger.info("RUNNING FULL PIPELINE")
    logger.info(f"Stages: {' -> '.join(stages)}")
    logger.info(f"Force: {force}")
    logger.info("="*60)
    
    for stage in stages:
        success = run_stage(stage, force=force)
        if not success:
            logger.error(f"\nPipeline failed at stage: {stage}")
            return False
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline for the sample-alt-voters experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline (default):
    uv run python -m src.sample_alt_voters

    # Force re-run everything:
    uv run python -m src.sample_alt_voters --force

    # Run specific stage:
    uv run python -m src.sample_alt_voters --stage run-experiment
        """
    )
    
    parser.add_argument(
        "--stage",
        choices=[
            "generate-statements",
            "run-experiment",
            "fix-epsilons",
            "run-triple-star",
            "visualize"
        ],
        help="Run a specific stage only (default: run all stages)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if outputs exist"
    )
    
    args = parser.parse_args()
    
    if args.stage:
        # Run single stage
        success = run_stage(args.stage, force=args.force)
    else:
        # Run full pipeline
        success = run_full_pipeline(force=args.force)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
