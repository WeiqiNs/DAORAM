"""Graph data structures and algorithms for ORAM constructions."""

import random

# A graph as an adjacency list: {vertex: neighbors}.
Graph = dict[int, list[int]]


def split_set(input_list: list, split_num: int) -> list[list]:
    n = len(input_list)
    chunk_size = n // split_num
    remainder = n % split_num

    parts = []
    start = 0

    for i in range(split_num):
        end = start + chunk_size + (1 if i < remainder else 0)
        parts.append(input_list[start:end])
        start = end

    return parts


def fixed_undirected_graph_gen(num_ver: int, num_neigh: int) -> Graph:
    """Deterministically build a ``num_neigh``-regular undirected graph on vertices 1..``num_ver``."""
    if not (0 < num_neigh < num_ver) or num_ver % 2 != 0:
        raise ValueError("Need 0 < number of neighbors < number of vertices and number of vertices even.")

    result: Graph = {v: [] for v in range(1, num_ver + 1)}

    # Connect each vertex to its num_neigh/2 nearest neighbors on each side of a ring.
    for i in range(1, num_ver + 1):
        for j in range(1, num_neigh // 2 + 1):
            u = ((i - 1 + j) % num_ver) + 1
            result[i].append(u)
            result[u].append(i)

    return result


def random_undirected_graph_gen(num_ver: int, num_neigh: int, seed: int | None = None) -> Graph:
    """Build a random ``num_neigh``-regular undirected graph on vertices 1..``num_ver`` (``seed`` for
    reproducibility)."""
    if not (0 < num_neigh < num_ver) or num_ver % 2 != 0:
        raise ValueError("Need 0 < number of neighbors < number of vertices and number of vertices even.")

    rng = random.Random(seed)

    num_matchings = num_ver - 1

    # Generate all perfect matchings via round-robin 1-factorization.
    factorization = []
    for r in range(num_matchings):
        matching = [(num_ver, r + 1)]
        for i in range(1, (num_ver - 1) // 2 + 1):
            a = ((r + i) % (num_ver - 1)) + 1
            b = ((r - i) % (num_ver - 1)) + 1
            if a > b:
                a, b = b, a
            matching.append((a, b))
        factorization.append(matching)

    chosen_rounds = rng.sample(range(num_matchings), num_neigh)

    # Randomly permute vertex labels to avoid structural bias.
    perm = list(range(1, num_ver + 1))
    rng.shuffle(perm)
    relabel = {old: i + 1 for i, old in enumerate(perm)}

    result: Graph = {v: [] for v in range(1, num_ver + 1)}

    for r in chosen_rounds:
        for a, b in factorization[r]:
            u, v = relabel[a], relabel[b]
            result[u].append(v)
            result[v].append(u)

    return result


def sparsify_graph(graph: Graph, cur_neigh: int, max_neigh: int) -> Graph:
    """Bound every vertex's degree to ``max_neigh`` by spreading its neighbors across new intermediate
    vertices (the original degree is ``cur_neigh``)."""
    split_num = cur_neigh // max_neigh + 1

    starting_vertex = len(graph) + 1

    for vertex in range(1, len(graph) + 1):
        intermediate_vertices = [starting_vertex + i for i in range(split_num)]

        neighbors = split_set(input_list=sorted(graph[vertex]), split_num=split_num)

        for i in range(split_num):
            graph[starting_vertex] = neighbors[i]
            starting_vertex += 1

        graph[vertex] = intermediate_vertices

    return graph
