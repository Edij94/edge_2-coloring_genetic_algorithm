"""Run both crossover variants of genetic_algorithm_01 on a folder of graphs.

Each file in GRAPH_FOLDER must contain one graph in graph6 format.
"""

from pathlib import Path

import networkx as nx

from genetic_algorithm_naive import genetic_algorithm_naive
from genetic_algorithm_01 import genetic_algorithm_01

GRAPH_FOLDER = Path("graphs/high_vertex_orbits")


def load_graph6_strings(folder: Path) -> list[str]:
    """Read every file in `folder` and return its stripped graph6 text."""
    graphs = []
    for file in folder.iterdir():
        with file.open("r", encoding="utf-8") as f:
            graphs.append(f.read().strip())
    return graphs


def main() -> None:
    graph_strings = load_graph6_strings(GRAPH_FOLDER)
    print(graph_strings)

    results_uniform = []
    results_edge_cut = []

    for graph6 in graph_strings:
        graph = nx.from_graph6_bytes(bytes(graph6, "utf-8"))
        print(len(graph.edges()), len(graph.nodes()))  # edges, nodes
        results_uniform.append(genetic_algorithm_01(graph, crossover='uniform'))
        results_edge_cut.append(genetic_algorithm_01(graph, crossover='edge_cut'))


if __name__ == "__main__":
    main()
