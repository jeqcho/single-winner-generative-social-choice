"""Check if pairwise preference graphs form DAGs or contain cycles."""
import csv
from collections import defaultdict, deque
from pathlib import Path


def read_preferences(csv_path):
    """Read pairwise preferences from CSV file."""
    edges = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            i = int(row['statement_id_i'])
            j = int(row['statement_id_j'])
            preferred = int(row['preferred_statement_id'])

            # Edge direction: preferred -> not preferred
            if preferred == i:
                edges.append((i, j))
            elif preferred == j:
                edges.append((j, i))
            # If neither, it's a tie - we can skip or handle as needed

    return edges


def build_graph(edges):
    """Build adjacency list from edges."""
    graph = defaultdict(list)
    nodes = set()

    for u, v in edges:
        graph[u].append(v)
        nodes.add(u)
        nodes.add(v)

    return graph, nodes


def find_cycle_dfs(graph, nodes):
    """
    Find a cycle in the graph using DFS.
    Returns the cycle if found, None otherwise.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in nodes}
    parent = {node: None for node in nodes}

    def dfs(node, path):
        color[node] = GRAY
        path.append(node)

        for neighbor in graph[node]:
            if color[neighbor] == GRAY:
                # Found a cycle - extract it
                cycle_start_idx = path.index(neighbor)
                cycle = path[cycle_start_idx:] + [neighbor]
                return cycle
            elif color[neighbor] == WHITE:
                result = dfs(neighbor, path[:])
                if result:
                    return result

        color[node] = BLACK
        return None

    for node in nodes:
        if color[node] == WHITE:
            result = dfs(node, [])
            if result:
                return result

    return None


def find_shortest_cycle_bfs(graph, nodes):
    """
    Find the shortest cycle using BFS from each node.
    This is more expensive but finds the shortest cycle.
    """
    shortest_cycle = None
    min_length = float('inf')

    for start_node in nodes:
        # BFS to find shortest path back to start_node
        queue = deque([(start_node, [start_node])])
        visited = {start_node}

        while queue:
            node, path = queue.popleft()

            for neighbor in graph[node]:
                if neighbor == start_node and len(path) > 1:
                    # Found a cycle back to start
                    cycle = path + [start_node]
                    if len(cycle) < min_length:
                        min_length = len(cycle)
                        shortest_cycle = cycle
                    break
                elif neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

    return shortest_cycle


def analyze_file(csv_path, find_shortest=True):
    """Analyze a CSV file for cycles."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {csv_path.name}")
    print(f"{'='*60}")

    edges = read_preferences(csv_path)
    graph, nodes = build_graph(edges)

    print(f"Number of nodes: {len(nodes)}")
    print(f"Number of edges: {len(edges)}")

    # First, quick check for any cycle
    cycle = find_cycle_dfs(graph, nodes)

    if cycle is None:
        print("✓ Result: DAG can be formed (no cycles detected)")
        return True
    else:
        print("✗ Result: DAG cannot be formed (cycles detected)")

        if find_shortest:
            print("\nSearching for shortest cycle...")
            shortest = find_shortest_cycle_bfs(graph, nodes)
            if shortest and len(shortest) < len(cycle):
                cycle = shortest

        print(f"\nCycle found (length {len(cycle) - 1}):")
        print(" -> ".join(map(str, cycle)))

        # Show the preference relationships in the cycle
        print("\nCycle explanation:")
        for i in range(len(cycle) - 1):
            print(f"  Statement {cycle[i]} is preferred to Statement {cycle[i+1]}")

        return False


def main():
    """Main function to analyze all pairwise preference files."""
    data_dir = Path("data/large_scale/prod")
    files = sorted(data_dir.glob("pairwise_100_abortion_persona_*.csv"))

    print(f"Found {len(files)} files to analyze")

    results = {}
    for csv_file in files:
        is_dag = analyze_file(csv_file, find_shortest=True)
        results[csv_file.name] = is_dag

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for filename, is_dag in results.items():
        status = "DAG ✓" if is_dag else "Has cycles ✗"
        print(f"{filename}: {status}")


if __name__ == "__main__":
    main()
