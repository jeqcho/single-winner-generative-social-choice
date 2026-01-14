"""
Migration script to reorganize existing experiment data into sample subdirectories.

Moves existing k{k}_p{p}/sample_info.json and results.json into k{k}_p{p}/sample0/

Usage:
    uv run python -m src.sampling_experiment.migrate_to_samples
    uv run python -m src.sampling_experiment.migrate_to_samples --dry-run
"""

import argparse
import shutil
from pathlib import Path

from .config import (
    OUTPUT_DIR,
    K_VALUES,
    P_VALUES,
    N_REPS,
    TEST_TOPIC,
)


def migrate_topic(topic_slug: str, output_dir: Path, dry_run: bool = False) -> int:
    """
    Migrate all reps for a topic to the new sample directory structure.
    
    Returns:
        Number of directories migrated
    """
    topic_dir = output_dir / "data" / topic_slug
    if not topic_dir.exists():
        print(f"Topic directory not found: {topic_dir}")
        return 0
    
    migrated = 0
    
    for rep_idx in range(N_REPS):
        rep_dir = topic_dir / f"rep{rep_idx}"
        if not rep_dir.exists():
            continue
        
        for k in K_VALUES:
            for p in P_VALUES:
                kp_dir = rep_dir / f"k{k}_p{p}"
                if not kp_dir.exists():
                    continue
                
                # Check if already migrated (sample0 exists)
                sample0_dir = kp_dir / "sample0"
                if sample0_dir.exists():
                    print(f"  Already migrated: {kp_dir}")
                    continue
                
                # Check if there are files to migrate
                sample_info = kp_dir / "sample_info.json"
                results = kp_dir / "results.json"
                
                if not sample_info.exists() and not results.exists():
                    print(f"  No files to migrate: {kp_dir}")
                    continue
                
                print(f"  Migrating: {kp_dir}")
                
                if not dry_run:
                    # Create sample0 directory
                    sample0_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Move files
                    if sample_info.exists():
                        shutil.move(str(sample_info), str(sample0_dir / "sample_info.json"))
                    if results.exists():
                        shutil.move(str(results), str(sample0_dir / "results.json"))
                
                migrated += 1
    
    return migrated


def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing experiment data to sample subdirectories"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=TEST_TOPIC,
        help="Topic slug to migrate (default: public trust topic)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN - no changes will be made\n")
    
    print(f"Migrating topic: {args.topic}")
    print(f"Output dir: {args.output_dir}")
    print()
    
    migrated = migrate_topic(args.topic, args.output_dir, args.dry_run)
    
    print()
    print(f"Total directories migrated: {migrated}")
    
    if args.dry_run:
        print("\nThis was a dry run. Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
