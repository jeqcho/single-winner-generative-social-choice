"""One-off script to remove double star data from results.json files.

The GPT** (double star) rankings had a bug making them inconsistent with other methods.
This script removes the following keys from all results.json files:
- chatgpt_double_star
- chatgpt_double_star_rankings
- chatgpt_double_star_personas
"""

import json
from pathlib import Path


DOUBLE_STAR_KEYS = [
    "chatgpt_double_star",
    "chatgpt_double_star_rankings",
    "chatgpt_double_star_personas",
]


def cleanup_double_star_data(base_dir: Path) -> tuple[int, int]:
    """Remove double star keys from all results.json files.
    
    Args:
        base_dir: Base directory to search for results.json files
        
    Returns:
        Tuple of (files_modified, keys_removed)
    """
    files_modified = 0
    keys_removed = 0
    
    # Find all results.json files in mini_rep directories
    for results_file in base_dir.rglob("mini_rep*/results.json"):
        with open(results_file) as f:
            data = json.load(f)
        
        results = data.get("results", {})
        modified = False
        
        for key in DOUBLE_STAR_KEYS:
            if key in results:
                del results[key]
                keys_removed += 1
                modified = True
        
        if modified:
            with open(results_file, "w") as f:
                json.dump(data, f, indent=2)
            files_modified += 1
    
    return files_modified, keys_removed


def main():
    """Main entry point."""
    base_dir = Path("outputs/sample_alt_voters/data")
    
    if not base_dir.exists():
        print(f"Error: Directory not found: {base_dir}")
        return
    
    print(f"Searching for results.json files in {base_dir}...")
    files_modified, keys_removed = cleanup_double_star_data(base_dir)
    
    print(f"Cleanup complete:")
    print(f"  Files modified: {files_modified}")
    print(f"  Keys removed: {keys_removed}")


if __name__ == "__main__":
    main()
