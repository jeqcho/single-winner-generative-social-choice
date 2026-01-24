"""
Generate human-readable voter files for sense-checking rankings.

Creates a text file for each voter in each condition showing:
1. The voter's persona
2. Statements ordered from most to least preferred
"""

import json
import random
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "degeneracy_mitigation"

PERSONAS_PATH = DATA_DIR / "personas" / "prod" / "adult.json"
STATEMENTS_PATH = DATA_DIR / "sample-alt-voters" / "sampled-statements" / "persona_no_context" / "abortion.json"
CONTEXT_PATH = DATA_DIR / "sample-alt-voters" / "sampled-context" / "abortion" / "rep0.json"

# Constants
N_VOTERS = 100
N_STATEMENTS = 100
VOTER_SEED = 42  # Same seed used in run_test.py


def load_personas() -> list[str]:
    """Load all personas from the adult personas file."""
    with open(PERSONAS_PATH, 'r') as f:
        return json.load(f)


def load_statements() -> tuple[list[str], list[str]]:
    """
    Load statements and their source persona IDs.
    
    Returns:
        Tuple of (statement_texts, statement_persona_ids)
        Both lists are indexed 0-99 matching the ranking indices.
    """
    with open(STATEMENTS_PATH, 'r') as f:
        stmt_data = json.load(f)
    
    with open(CONTEXT_PATH, 'r') as f:
        context_data = json.load(f)
    
    statements_dict = stmt_data['statements']
    context_persona_ids = context_data['context_persona_ids'][:N_STATEMENTS]
    
    statement_texts = []
    statement_persona_ids = []
    
    for persona_id in context_persona_ids:
        pid_str = str(persona_id)
        statement_texts.append(statements_dict[pid_str])
        statement_persona_ids.append(pid_str)
    
    return statement_texts, statement_persona_ids


def sample_voters(all_personas: list[str]) -> list[str]:
    """Sample the same 100 voters used in the experiment."""
    rng = random.Random(VOTER_SEED)
    indices = rng.sample(range(len(all_personas)), N_VOTERS)
    return [all_personas[i] for i in indices]


def load_rankings(condition_dir: Path) -> list[list[int]]:
    """Load rankings for a condition."""
    rankings_path = condition_dir / "rankings.json"
    with open(rankings_path, 'r') as f:
        return json.load(f)


def generate_voter_file(
    voter_idx: int,
    persona: str,
    ranking: list[int],
    statement_texts: list[str],
    statement_persona_ids: list[str],
    approach: str,
    effort: str,
    output_path: Path
) -> None:
    """Generate a single voter file."""
    
    approach_name = "Iterative Ranking" if approach == "a" else "Scoring"
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"VOTER {voter_idx} - Approach {approach.upper()} ({effort} reasoning) - {approach_name}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("PERSONA:")
    lines.append("-" * 40)
    lines.append(persona)
    lines.append("")
    lines.append("=" * 80)
    lines.append("STATEMENTS (Most Preferred → Least Preferred)")
    lines.append("=" * 80)
    lines.append("")
    
    for rank, stmt_idx in enumerate(ranking, start=1):
        if 0 <= stmt_idx < len(statement_texts):
            stmt_text = statement_texts[stmt_idx]
            stmt_persona_id = statement_persona_ids[stmt_idx]
            
            lines.append(f"[RANK {rank}] Statement Index: {stmt_idx} (from persona {stmt_persona_id})")
            lines.append("-" * 40)
            lines.append(stmt_text)
            lines.append("")
        else:
            lines.append(f"[RANK {rank}] Statement Index: {stmt_idx} - INVALID INDEX")
            lines.append("")
    
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))


def generate_all_voter_files() -> None:
    """Generate voter files for all conditions."""
    
    print("Loading data...")
    all_personas = load_personas()
    statement_texts, statement_persona_ids = load_statements()
    voters = sample_voters(all_personas)
    
    print(f"Loaded {len(all_personas)} personas")
    print(f"Loaded {len(statement_texts)} statements")
    print(f"Sampled {len(voters)} voters")
    print()
    
    conditions = [
        ("a", "minimal"),
        ("a", "low"),
        ("a", "medium"),
        ("b", "minimal"),
        ("b", "low"),
        ("b", "medium"),
    ]
    
    for approach, effort in conditions:
        condition_dir = OUTPUT_DIR / f"approach_{approach}" / effort
        
        if not condition_dir.exists():
            print(f"Skipping {approach}-{effort}: directory not found")
            continue
        
        rankings_path = condition_dir / "rankings.json"
        if not rankings_path.exists():
            print(f"Skipping {approach}-{effort}: rankings.json not found")
            continue
        
        print(f"Processing {approach.upper()}-{effort}...")
        
        rankings = load_rankings(condition_dir)
        
        # Create voter_files directory
        voter_files_dir = condition_dir / "voter_files"
        voter_files_dir.mkdir(exist_ok=True)
        
        # Generate file for each voter
        for voter_idx in range(min(len(rankings), len(voters))):
            ranking = rankings[voter_idx]
            persona = voters[voter_idx]
            
            output_path = voter_files_dir / f"voter_{voter_idx:02d}.txt"
            
            generate_voter_file(
                voter_idx=voter_idx,
                persona=persona,
                ranking=ranking,
                statement_texts=statement_texts,
                statement_persona_ids=statement_persona_ids,
                approach=approach,
                effort=effort,
                output_path=output_path
            )
        
        print(f"  Generated {len(rankings)} voter files in {voter_files_dir}")
    
    print()
    print("Done!")


if __name__ == "__main__":
    generate_all_voter_files()
