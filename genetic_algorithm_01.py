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
import numpy as np
from networkx.algorithms.bridges import bridges


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
    n_edges = len(graph.edges())
    e = [(min(u,v),max(u,v)) for (u, v) in graph.edges()]
    edges = dict(zip(e, range(n_edges)))
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
            population, fitnesses, grades, ones, zeroes = new_gen(
                graph, edges, population, n_elite, mutation_rate, crossing)
            np_colors = np.array([f[0] for f in fitnesses])
            np_grades = np.array(grades)
            np_ones = np.array(ones)
            np_zeroes = np.array(zeroes)
            writer.writerow([np.max(np_colors),np.max(np_grades), np.mean(np_colors), np.mean(np_grades), np.mean(np_ones), np.mean(np_zeroes)])

    # Pick the final winner and compare it with a reference value.
    best_genome = max(population, key=lambda genome: fitness(graph, edges, genome))
    best_fitness = list(fitness(graph, edges, best_genome))
    best_fitness[1] = best_fitness[1]*5
    matching = len(nx.max_weight_matching(graph, maxcardinality=True)) + 1
    print(f"Počet farieb: {best_fitness[0]}, Počet 'dobrych' mostov: {best_fitness[1]}, V porovnaní z matching {round(best_fitness[0] / matching, 2) * 100}%")
    return best_genome


def new_gen(graph, edges, population, n_elite, mutation_rate, crossing):
    """Create the next generation.

    Returns:
        (new_population, best) where `best` is the (fitness, genome) pair of
        the best individual of the *current* population.
    """
    p = [normalize(genome) for genome in population]
    fitnesses = [fitness(graph,edges,genome) for genome in p]
    grades = [sum(f) for f in fitnesses]

    ## 01 rate for logs
    ones = [sum(genome)/len(genome) for genome in p]
    zeroes = [(len(genome) - sum(genome))/len(genome) for genome in p]

    # Selection weights: shifted so the worst genome has weight 1, then
    # squared to favour better genomes more strongly.
    weights = [(grade - min(grades) + 1) ** 2 for grade in grades]
    cum_weights = list(accumulate(weights))

    ranked = sorted(zip(grades, population), key=lambda t: t[0], reverse=True)
    elite = [genome for (_, genome) in ranked[:n_elite]]

    # Offspring fill the population; the elite survive unchanged.
    new_population = crossing(graph, edges, population, n_elite, mutation_rate, cum_weights)
    new_population.extend(elite)
    return new_population, fitnesses, grades, ones, zeroes


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


def edge_cut_crossover(graph, edges: dict[tuple[int,int], int], population, n_elite, mutation_rate, cum_weights):
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

    for (u, v), i in edges.items():
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

def normalize(genome):
    np_genome = np.array(genome)
    flipped = [g ^ 1 for g in genome]
    np_flipped = np.array(flipped)
    if np.count_nonzero(np_flipped) > np.count_nonzero(np_genome):
        return genome
    return flipped
def select_pair(cum_weights, population):
    """Pick two parents (with replacement) using cumulative selection weights."""
    return list(choices(population, cum_weights=cum_weights, k=2))


def fitness(graph: nx.Graph, edges, genome: list[int]):
    zero_components, zero_bridges = dfs_components_bridges(graph,1, edges, genome)
    one_components, one_bridges = dfs_components_bridges(graph,0, edges, genome)
    return (one_components + zero_components, 0.2*(zero_bridges + one_bridges))

def dfs_components_bridges(graph: nx.Graph, color, edges, genome: list[int]):
    adj = graph._adj
    n = len(graph.nodes())
    tin = [-1]*n
    low = [-1]*n
    subtree = [1]*n
    timer = 0
    bridges = []
    components = 0

    for start in graph.nodes():
        if tin[start] != -1:
            continue

        for neighbor in adj[start]:
            if genome[edges[(min(start, neighbor), max(start, neighbor))]] != color:
                components += 1
                break



        stack = [(start, -1, iter(adj[start]))]

        tin[start] = low[start] = timer
        timer += 1

        while stack:
            v, parent, neighbors = stack[-1]

            try:
                to = next(neighbors)

                if to == parent:
                    continue

                if genome[edges[(min(v,to),max(v,to))]] == color:
                    continue

                if tin[to] != -1:
                    low[v] = min(low[v], tin[to])

                else:
                    tin[to] = low[to] = timer
                    timer += 1

                    stack.append((to, v, iter(adj[to])))

            except StopIteration:
                stack.pop()

                if parent != -1:
                    subtree[parent] += subtree[v]

                    low[parent] = min(low[parent], low[v])

                    if low[v] > tin[parent] and subtree[v] > 1:
                        bridges.append((parent, v))

    return components, len(bridges)

def generate_genome(length: int) -> list[int]:
    """Create a random 0/1 genome of the given length."""
    return list(choices([0, 1], k=length))
