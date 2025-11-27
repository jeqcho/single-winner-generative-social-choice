"""Extract statements involved in preference cycles."""
import json
from pathlib import Path


def load_statements(json_path):
    """Load statements from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_cycle_statements(statements_data, cycle_ids):
    """Extract statements for given IDs."""
    result = []
    for stmt_id in cycle_ids:
        if stmt_id < len(statements_data):
            result.append({
                'id': stmt_id,
                'statement': statements_data[stmt_id]['statement']
            })
    return result


def main():
    """Extract statements for all cycles."""
    statements_path = Path("data/large_scale/prod/statements/what-should-guide-laws-concerning-abortion.json")
    personas_path = Path("data/personas/prod/discriminative.json")

    # Load all statements
    print("Loading statements...")
    statements = load_statements(statements_path)
    print(f"Loaded {len(statements)} statements")

    # Load all personas
    print("Loading personas...")
    personas = load_statements(personas_path)
    print(f"Loaded {len(personas)} personas")

    # Define cycles found (persona_id -> cycle)
    cycles = {
        'pairwise_100_abortion_persona_1.csv': {'persona_id': 1, 'cycle': [13, 7, 90]},
        'pairwise_100_abortion_persona_40.csv': {'persona_id': 40, 'cycle': [0, 51, 12]},
        'pairwise_100_abortion_persona_7.csv': {'persona_id': 7, 'cycle': [0, 1, 22]}
    }

    # Extract statements for each cycle
    results = {}
    for filename, cycle_info in cycles.items():
        print(f"\nExtracting statements for {filename}")
        results[filename] = {
            'persona': personas[cycle_info['persona_id']],
            'cycle': extract_cycle_statements(statements, cycle_info['cycle'])
        }

    # Write markdown report
    output_path = Path("reports/cycle_analysis/cycle_analysis.md")
    with open(output_path, 'w') as f:
        f.write("# Preference Cycle Analysis\n\n")
        f.write("Analysis of preference cycles found in abortion pairwise preference data.\n\n")
        f.write("## Summary\n\n")
        f.write("All three persona files contain preference cycles, meaning a Directed Acyclic Graph (DAG) cannot be formed. ")
        f.write("This indicates fundamental inconsistencies in the pairwise preferences.\n\n")

        for filename, result_data in results.items():
            f.write(f"## {filename}\n\n")

            # Write persona information
            f.write(f"### Persona\n\n")
            f.write(f"```\n{result_data['persona']}\n```\n\n")

            # Write cycle path
            cycle_data = result_data['cycle']
            cycle_path = " → ".join(str(s['id']) for s in cycle_data)
            cycle_path += f" → {cycle_data[0]['id']}"
            f.write(f"**Cycle:** {cycle_path}\n\n")

            # Write each statement
            for i, stmt_data in enumerate(cycle_data):
                f.write(f"### Statement {stmt_data['id']}\n\n")
                f.write(f"{stmt_data['statement']}\n\n")

            # Write preference explanation
            f.write("### Cycle Explanation\n\n")
            for i in range(len(cycle_data)):
                next_i = (i + 1) % len(cycle_data)
                f.write(f"- Statement {cycle_data[i]['id']} is preferred to Statement {cycle_data[next_i]['id']}\n")
            f.write("\n")
            f.write("---\n\n")

    print(f"\n✓ Markdown report written to {output_path}")


if __name__ == "__main__":
    main()
