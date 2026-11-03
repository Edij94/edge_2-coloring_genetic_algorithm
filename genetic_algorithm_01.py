"""Genetic algorithm for 2-coloring the edges of a graph (variant 01).

Each genome is a list of 0/1 values, one per edge of the graph: the value is
the "color" of that edge. The fitness of a genome is the total number of
connected components of the two single-colored subgraphs (all 0-edges and
all 1-edges). A higher fitness is better.

Two crossover strategies are available:
    * 'uniform'   - every edge is inherited from a randomly chosen parent.
    * 'edge_cut'  - the graph is split by a minimum edge cut; one side is
                    taken from parent 1, the other from parent 2, and the
                    cut edges are chosen randomly.
"""

import csv
import math
from itertools import accumulate
from random import choice, choices, random, sample

import networkx as nx


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def genetic_algorithm_01(graph: nx.Graph, pop_size: int = 100, generations: int = 300,
                         elite_rate: float = 0.1, mutation_rate: float = 0.02,
                         crossover: str = 'uniform') -> list[int]:
    """Run the genetic algorithm and return the best genome found.

    Args:
        graph: the graph whose edges are to be colored.
        pop_size: number of genomes in the population.
        generations: number of generations to simulate.
        elite_rate: fraction of the population copied unchanged to the next
            generation (at least one genome is always kept).
        mutation_rate: probability that a single gene is flipped.
        crossover: 'uniform' or 'edge_cut'.

    The best (fitness, genome) of every generation is written to the file
    "log". The final fitness and its ratio to a matching-based reference
    value are printed.
    """
    edges = list(graph.edges())
    n_edges = len(edges)
    n_elite = max(1, math.ceil(pop_size * elite_rate))

    # Initial population: random 0/1 genomes.
    population = [generate_genome(n_edges) for _ in range(pop_size)]

    # Choose the crossover function.
    match crossover:
        case 'uniform':
            crossing = uniform_crossing
        case 'edge_cut':
            crossing = edge_cut_crossover
        case _:
            raise ValueError(f"Unknown crossover: {crossover!r}")

    # Evolve the population, logging the best individual of each generation.
    with open("log", "w", encoding="utf-8") as log:
        writer = csv.writer(log, lineterminator="\n")
        for _ in range(generations):
            population, best_of_generation = new_gen(
                graph, edges, population, n_elite, mutation_rate, crossing)
            writer.writerow([best_of_generation])

    # Pick the final winner and compare it with a reference value.
    best_genome = max(population, key=lambda genome: fitness(graph, edges, genome))
    best_fitness = fitness(graph, edges, best_genome)
    matching = len(nx.max_weight_matching(graph, maxcardinality=True)) + 1
    print(best_fitness, f"{round(best_fitness / matching, 2) * 100}%")
    return best_genome


def new_gen(graph, edges, population, n_elite, mutation_rate, crossing):
    """Create the next generation.

    Returns:
        (new_population, best) where `best` is the (fitness, genome) pair of
        the best individual of the *current* population.
    """
    grades = [fitness(graph, edges, genome) for genome in population]

    # Selection weights: shifted so the worst genome has weight 1, then
    # squared to favour better genomes more strongly.
    weights = [(grade - min(grades) + 1) ** 2 for grade in grades]
    cum_weights = list(accumulate(weights))

    ranked = sorted(zip(grades, population), key=lambda t: t[0], reverse=True)
    best = ranked[0]
    elite = [genome for (_, genome) in ranked[:n_elite]]

    # Offspring fill the population; the elite survive unchanged.
    new_population = crossing(graph, edges, population, n_elite, mutation_rate, cum_weights)
    new_population.extend(elite)
    return new_population, best


# ---------------------------------------------------------------------------
# Crossover operators
# ---------------------------------------------------------------------------

def uniform_crossing(graph, edges, population, n_elite, mutation_rate, cum_weights):
    """Uniform crossover: each gene comes from either parent with 50% chance."""
    offspring = []
    for _ in range(len(population) - n_elite):
        parent1, parent2 = select_pair(cum_weights, population)
        child = [parent1[i] if random() < 0.5 else parent2[i]
                 for i in range(len(parent1))]
        offspring.append(mutate(child, mutation_rate))
    return offspring


def edge_cut_crossover(graph, edges, population, n_elite, mutation_rate, cum_weights):
    """Edge-cut crossover.

    A random minimum edge cut splits the graph in two parts. Edges inside
    part 1 are inherited from parent 1, edges inside part 2 from parent 2,
    and the cut edges from a random parent. The same cut is used for every
    child of this generation.
    """
    offspring = []
    part1_idx, part2_idx, cut_idx = get_crossover_indices(graph, edges)
    for _ in range(len(population) - n_elite):
        parent1, parent2 = select_pair(cum_weights, population)
        child = [0] * len(edges)
        for i in part1_idx:
            child[i] = parent1[i]
        for i in part2_idx:
            child[i] = parent2[i]
        for i in cut_idx:
            child[i] = choice([parent1[i], parent2[i]])
        offspring.append(mutate(child, mutation_rate))
    return offspring


def get_crossover_indices(graph, edges):
    """Split the edge indices according to a random minimum edge cut.

    Two distinct random nodes s and t are picked, and a minimum s-t edge cut
    is computed. Removing the cut leaves s's component ("part 1") separated
    from the rest.

    Returns:
        (part1_indices, part2_indices, cut_indices) - sets of indices into
        `edges`: edges inside part 1, edges inside the rest, and cut edges.
    """
    s, t = sample(list(graph.nodes()), 2)
    cut = nx.minimum_edge_cut(graph, s, t)

    graph_without_cut = graph.copy()
    graph_without_cut.remove_edges_from(cut)
    part1 = nx.node_connected_component(graph_without_cut, s)

    cut_edges = {frozenset(edge) for edge in cut}
    part1_indices, part2_indices, cut_indices = set(), set(), set()

    for i, (u, v) in enumerate(edges):
        if frozenset((u, v)) in cut_edges:
            cut_indices.add(i)
        elif u in part1:
            part1_indices.add(i)
        else:
            part2_indices.add(i)

    return part1_indices, part2_indices, cut_indices


# ---------------------------------------------------------------------------
# Genetic operators and helpers
# ---------------------------------------------------------------------------

def mutate(genome, mutation_rate):
    """Flip each gene (0 <-> 1) independently with probability `mutation_rate`."""
    return [g ^ 1 if random() < mutation_rate else g for g in genome]


def select_pair(cum_weights, population):
    """Pick two parents (with replacement) using cumulative selection weights."""
    return list(choices(population, cum_weights=cum_weights, k=2))


def fitness(graph: nx.Graph, edges, genome: list[int]) -> float:
    """Return the number of connected components of the 0-colored subgraph
    plus that of the 1-colored subgraph (higher is better)."""
    zero_edges = []
    one_edges = []

    for edge, color in zip(edges, genome):
        if color == 0:
            zero_edges.append(edge)
        else:
            one_edges.append(edge)

    zero_graph = graph.edge_subgraph(zero_edges)
    one_graph = graph.edge_subgraph(one_edges)

    return nx.number_connected_components(zero_graph) + nx.number_connected_components(one_graph)


def generate_genome(length: int) -> list[int]:
    """Create a random 0/1 genome of the given length."""
    return list(choices([0, 1], k=length))
