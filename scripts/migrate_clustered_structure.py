#!/usr/bin/env python3
"""
Migration script to restructure clustered voter data.

This script performs the following operations:
1. Move clustered persona_no_context data to new path structure:
   - {topic}/clustered/persona_no_context/rep0_progressive_liberal/ 
     -> {topic}/clustered/progressive_liberal/persona_no_context/rep0/
   - {topic}/clustered/persona_no_context/rep1_conservative_traditional/
     -> {topic}/clustered/conservative_traditional/persona_no_context/rep0/
2. Delete other 3 alt_dists from clustered (no_persona_context, no_persona_no_context, persona_context)
3. Delete mini_rep4/ from ALL data (uniform + clustered)
4. Delete chatgpt_triple_star.json at rep level (72 files)
5. Delete new_random entries from all results.json files (buggy old algorithm)

Usage:
    uv run python scripts/migrate_clustered_structure.py
    uv run python scripts/migrate_clustered_structure.py --dry-run
"""

import argparse
import json
import shutil
from pathlib import Path


# Topics to process
TOPICS = ["abortion", "healthcare", "electoral", "policing", "trust", "environment"]

# Alt distributions to delete from clustered
ALT_DISTS_TO_DELETE = ["no_persona_context", "no_persona_no_context", "persona_context"]

# Base data directory
DATA_DIR = Path(__file__).parent.parent / "outputs" / "sample_alt_voters" / "data"


def move_clustered_data(dry_run: bool = False) -> dict:
    """
    Move clustered persona_no_context data to new structure.
    
    Old: {topic}/clustered/persona_no_context/rep{id}_{cluster}/
    New: {topic}/clustered/{cluster}/persona_no_context/rep0/
    
    Note: Both rep0_progressive_liberal and rep1_conservative_traditional
    become rep0 in their respective cluster directories.
    """
    stats = {"moved": 0, "skipped": 0, "errors": 0}
    
    moves = [
        ("rep0_progressive_liberal", "progressive_liberal"),
        ("rep1_conservative_traditional", "conservative_traditional"),
    ]
    
    for topic in TOPICS:
        old_alt_dir = DATA_DIR / topic / "clustered" / "persona_no_context"
        
        if not old_alt_dir.exists():
            print(f"  Skipping {topic}: no clustered/persona_no_context directory")
            stats["skipped"] += 1
            continue
        
        for old_rep_name, cluster_name in moves:
            old_rep_dir = old_alt_dir / old_rep_name
            new_rep_dir = DATA_DIR / topic / "clustered" / cluster_name / "persona_no_context" / "rep0"
            
            if not old_rep_dir.exists():
                print(f"  Skipping {topic}/{old_rep_name}: source directory not found")
                stats["skipped"] += 1
                continue
            
            if new_rep_dir.exists():
                print(f"  Skipping {topic}/{old_rep_name}: destination already exists")
                stats["skipped"] += 1
                continue
            
            print(f"  Moving: {old_rep_dir.relative_to(DATA_DIR)}")
            print(f"      -> {new_rep_dir.relative_to(DATA_DIR)}")
            
            if not dry_run:
                new_rep_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_rep_dir), str(new_rep_dir))
            
            stats["moved"] += 1
    
    return stats


def delete_clustered_alt_dists(dry_run: bool = False) -> dict:
    """Delete non-persona_no_context alt_dists from clustered data."""
    stats = {"deleted": 0, "not_found": 0}
    
    for topic in TOPICS:
        clustered_dir = DATA_DIR / topic / "clustered"
        
        if not clustered_dir.exists():
            continue
        
        for alt_dist in ALT_DISTS_TO_DELETE:
            alt_dir = clustered_dir / alt_dist
            
            if alt_dir.exists():
                print(f"  Deleting: {alt_dir.relative_to(DATA_DIR)}")
                if not dry_run:
                    shutil.rmtree(alt_dir)
                stats["deleted"] += 1
            else:
                stats["not_found"] += 1
        
        # Also delete the old persona_no_context directory if empty after moves
        old_persona_dir = clustered_dir / "persona_no_context"
        if old_persona_dir.exists():
            # Check if it's empty or only has empty subdirs
            remaining = list(old_persona_dir.iterdir())
            if not remaining:
                print(f"  Deleting empty: {old_persona_dir.relative_to(DATA_DIR)}")
                if not dry_run:
                    old_persona_dir.rmdir()
    
    return stats


def delete_mini_rep4(dry_run: bool = False) -> dict:
    """Delete mini_rep4 directories from ALL data (uniform + clustered)."""
    stats = {"deleted": 0}
    
    for mini_rep4_dir in DATA_DIR.rglob("mini_rep4"):
        if mini_rep4_dir.is_dir():
            print(f"  Deleting: {mini_rep4_dir.relative_to(DATA_DIR)}")
            if not dry_run:
                shutil.rmtree(mini_rep4_dir)
            stats["deleted"] += 1
    
    return stats


def delete_triple_star_files(dry_run: bool = False) -> dict:
    """Delete chatgpt_triple_star.json files at rep level."""
    stats = {"deleted": 0}
    
    for triple_star_file in DATA_DIR.rglob("chatgpt_triple_star.json"):
        print(f"  Deleting: {triple_star_file.relative_to(DATA_DIR)}")
        if not dry_run:
            triple_star_file.unlink()
        stats["deleted"] += 1
    
    return stats


def delete_new_random_entries(dry_run: bool = False) -> dict:
    """Delete new_random entries from all results.json files."""
    stats = {"modified": 0, "entries_removed": 0, "no_entry": 0}
    
    for results_file in DATA_DIR.rglob("mini_rep*/results.json"):
        try:
            with open(results_file) as f:
                data = json.load(f)
            
            results = data.get("results", {})
            
            if "new_random" in results:
                print(f"  Removing new_random from: {results_file.relative_to(DATA_DIR)}")
                if not dry_run:
                    del results["new_random"]
                    with open(results_file, "w") as f:
                        json.dump(data, f, indent=2)
                stats["entries_removed"] += 1
                stats["modified"] += 1
            else:
                stats["no_entry"] += 1
        except Exception as e:
            print(f"  Error processing {results_file}: {e}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Migrate clustered data to new structure"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    args = parser.parse_args()
    
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN - No changes will be made")
        print("=" * 60)
    
    print("\n" + "=" * 60)
    print("Step 1: Moving clustered persona_no_context data")
    print("=" * 60)
    move_stats = move_clustered_data(dry_run=args.dry_run)
    print(f"  Moved: {move_stats['moved']}, Skipped: {move_stats['skipped']}")
    
    print("\n" + "=" * 60)
    print("Step 2: Deleting other alt_dists from clustered")
    print("=" * 60)
    alt_stats = delete_clustered_alt_dists(dry_run=args.dry_run)
    print(f"  Deleted: {alt_stats['deleted']}, Not found: {alt_stats['not_found']}")
    
    print("\n" + "=" * 60)
    print("Step 3: Deleting mini_rep4 directories")
    print("=" * 60)
    mini_rep4_stats = delete_mini_rep4(dry_run=args.dry_run)
    print(f"  Deleted: {mini_rep4_stats['deleted']}")
    
    print("\n" + "=" * 60)
    print("Step 4: Deleting chatgpt_triple_star.json files")
    print("=" * 60)
    triple_star_stats = delete_triple_star_files(dry_run=args.dry_run)
    print(f"  Deleted: {triple_star_stats['deleted']}")
    
    print("\n" + "=" * 60)
    print("Step 5: Deleting new_random entries from results.json")
    print("=" * 60)
    new_random_stats = delete_new_random_entries(dry_run=args.dry_run)
    print(f"  Modified: {new_random_stats['modified']}, Entries removed: {new_random_stats['entries_removed']}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Clustered data moved: {move_stats['moved']}")
    print(f"  Alt_dists deleted: {alt_stats['deleted']}")
    print(f"  mini_rep4 directories deleted: {mini_rep4_stats['deleted']}")
    print(f"  chatgpt_triple_star.json files deleted: {triple_star_stats['deleted']}")
    print(f"  new_random entries removed: {new_random_stats['entries_removed']}")
    
    if args.dry_run:
        print("\n(This was a dry run - no changes were made)")


if __name__ == "__main__":
    main()
