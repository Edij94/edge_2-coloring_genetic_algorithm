"""Naive genetic algorithm for coloring the edges of a graph.

A genome is a list with one color (an integer) per edge. At most
min(#vertices, #edges) different colors are used.

Fitness is a tuple (violations, number_of_colors_used):
    * violations  - minus the number of vertices that see more than 2 colors
                    on their incident edges (0 is best, more negative is worse);
    * colors used - number of distinct colors in the genome.
Tuples compare lexicographically, so removing violations has priority.

The algorithm has two modes, chosen from the best genome's violations:
    * mode 0 (violations present): high mutation that pushes genes toward the
      most common color, to reduce the number of colors at a vertex.
    * mode 1 (no violations): low mutation with random colors, and fresh random
      genomes are added to keep diversity.
"""

import csv
import math
from collections import Counter
from random import choice, choices, random

import networkx as nx


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def genetic_algorithm_naive(graph: nx.Graph, pop_size: int = 100, generations: int = 300,
                            normalization_rate: float = 0.8) -> list[int]:
    """Run the algorithm and return the best genome found.

    Args:
        graph: the graph whose edges are to be colored.
        pop_size: number of genomes in the population.
        generations: number of generations to simulate.
        normalization_rate: probability that an elite genome is normalized
            (colors relabeled 0, 1, 2, ...) before crossover.

    The best (fitness, genome) of every generation is written to the file
    "log". The final fitness and its ratio to a matching-based reference
    value are printed.
    """
    n_elite = math.ceil(pop_size * 0.1)
    n_edges = len(graph.edges())
    n_vertices = len(graph.nodes())
    max_colors = min(n_vertices, n_edges)
    matching = len(nx.max_weight_matching(graph, maxcardinality=True)) + 1

    population = [generate_genome(n_edges, max_colors) for _ in range(pop_size)]

    with open("log", "w", encoding="utf-8") as log:
        writer = csv.writer(log, lineterminator="\n")
        for _ in range(generations):
            # Rank the population from best to worst.
            population = sorted(population, key=lambda genome: fitness(graph, genome), reverse=True)
            best = fitness(graph, population[0])
            writer.writerow((best, population[0]))
            elite = population[0:n_elite]

            # Pick the mode: fix violations first, then minimize colors.
            has_violations = best[0] < 0
            mutation_rate = 0.075 if has_violations else 0.02
            mode = 0 if has_violations else 1

            population = crossing(elite, pop_size, n_elite, n_edges, n_vertices,
                                  mutation_rate, mode, normalization_rate)
            population.extend(elite)

    result = max(population, key=lambda genome: fitness(graph, genome))
    grade = fitness(graph, result)
    print(grade, f"{round(grade[1] / matching, 2) * 100}%")
    return result


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def crossing(generation: list[list[int]], pop_size, n_elite, n_edges, n_vertices,
             mutation_rate, mode, normalization_rate):
    """Build `pop_size - n_elite` children from the elite genomes.

    In mode 1 a few brand-new random genomes are added to the parent pool
    (a quarter of the elite count) to keep the population diverse.
    """
    parent_pool = [normalize(genome) if random() < normalization_rate else genome
                   for genome in generation]
    if mode == 1:
        new_genomes = [generate_genome(n_edges, min(n_vertices, n_edges))
                       for _ in range(math.ceil(len(generation) / 4))]
        parent_pool.extend(new_genomes)

    population = []
    for _ in range(pop_size - n_elite):
        parent1 = choice(parent_pool)
        parent2 = choice(parent_pool)
        # Uniform crossover: each gene from a random parent.
        child = [parent1[i] if random() < 0.5 else parent2[i]
                 for i in range(len(parent1))]
        child = mutate(child, mutation_rate, min(n_vertices, n_edges), mode)
        population.append(child)
    return population


def mutate(genome, mutation_rate, number_of_colors, mode):
    """Mutate genes in place with probability `mutation_rate` each.

    mode 0: set the gene to the genome's most common color.
    mode 1: set the gene to a random color.
    """
    most_common_color = Counter(genome).most_common(1)[0][0]
    for i in range(len(genome)):
        if random() < mutation_rate:
            if mode == 0:
                genome[i] = most_common_color
            else:
                genome[i] = choice(range(number_of_colors))
    return genome


def normalize(genome):
    """Relabel colors as 0, 1, 2, ... so equivalent colorings look the same."""
    colors = set(genome)
    mapping = {color: i for i, color in enumerate(colors)}
    return [mapping[color] for color in genome]


def generate_genome(length: int, n_colors: int) -> list[int]:
    """Create a random genome of `length` genes with colors in range(n_colors)."""
    return list(choices(range(n_colors), k=length))


def fitness(graph, genome) -> tuple[int, int]:
    """Return (violations, number_of_colors_used) for the given coloring.

    `violations` is minus the number of vertices with more than 2 distinct
    colors on their incident edges.
    """
    # Map every edge to its color, in both directions (u, v) and (v, u).
    colors = dict(zip(graph.edges(), genome))
    for edge in graph.edges():
        colors[(edge[1], edge[0])] = colors[edge]

    violations = 0
    for node in graph.nodes():
        node_colors = {colors[edge] for edge in graph.edges(node)}
        if len(node_colors) > 2:
            violations -= 1

    return violations, len(set(genome))
