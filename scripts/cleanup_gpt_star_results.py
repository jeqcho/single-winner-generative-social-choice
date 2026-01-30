#!/usr/bin/env python3
"""
Cleanup script to remove GPT** and GPT*** results from all output files.

This removes:
1. From results.json: chatgpt_double_star, chatgpt_double_star_rankings, chatgpt_double_star_personas
2. Delete all chatgpt_triple_star.json files

Run before re-running with updated model configuration.
"""

import json
from pathlib import Path


DOUBLE_STAR_KEYS = [
    "chatgpt_double_star",
    "chatgpt_double_star_rankings",
    "chatgpt_double_star_personas",
]


def cleanup_all_gpt_star_results(base_dir: Path) -> dict:
    """Remove GPT** keys from results.json and delete GPT*** files.
    
    Args:
        base_dir: Base directory to search
        
    Returns:
        Dict with cleanup statistics
    """
    stats = {
        "results_files_modified": 0,
        "double_star_keys_removed": 0,
        "triple_star_files_deleted": 0,
    }
    
    # 1. Remove double star keys from results.json files
    for results_file in base_dir.rglob("mini_rep*/results.json"):
        try:
            with open(results_file) as f:
                data = json.load(f)
            
            results = data.get("results", {})
            modified = False
            
            for key in DOUBLE_STAR_KEYS:
                if key in results:
                    del results[key]
                    stats["double_star_keys_removed"] += 1
                    modified = True
            
            if modified:
                with open(results_file, "w") as f:
                    json.dump(data, f, indent=2)
                stats["results_files_modified"] += 1
        except Exception as e:
            print(f"Error processing {results_file}: {e}")
    
    # 2. Delete all chatgpt_triple_star.json files
    for triple_star_file in base_dir.rglob("chatgpt_triple_star.json"):
        try:
            triple_star_file.unlink()
            stats["triple_star_files_deleted"] += 1
        except Exception as e:
            print(f"Error deleting {triple_star_file}: {e}")
    
    return stats


def main():
    """Main entry point."""
    base_dir = Path("outputs/sample_alt_voters/data")
    
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return
    
    print(f"Cleaning up GPT** and GPT*** results in {base_dir}...")
    print("This will remove:")
    print("  - chatgpt_double_star, chatgpt_double_star_rankings, chatgpt_double_star_personas from results.json")
    print("  - All chatgpt_triple_star.json files")
    print()
    
    stats = cleanup_all_gpt_star_results(base_dir)
    
    print("Cleanup complete:")
    print(f"  Results files modified: {stats['results_files_modified']}")
    print(f"  Double star keys removed: {stats['double_star_keys_removed']}")
    print(f"  Triple star files deleted: {stats['triple_star_files_deleted']}")


if __name__ == "__main__":
    main()
