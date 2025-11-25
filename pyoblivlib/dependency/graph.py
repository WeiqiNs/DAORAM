import random
from typing import Dict, List

# Set the desired graph types.
Graph = Dict[int, List[int]]


def split_set(input_list: list, split_num: int) -> List[list]:
    """Split a list evenly to desired number of chunks.

    :param input_list: The input list to split.
    :param split_num: The number of desired chunks to get.
    :return: A list of list.
    """
    # Total number of elements.
    n = len(input_list)
    # Base size of each portion.
    chunk_size = n // split_num
    # Extra elements that need to be distributed.
    remainder = n % split_num

    parts = []
    start = 0

    for i in range(split_num):
        # The first `remainder` parts get one extra element each.
        end = start + chunk_size + (1 if i < remainder else 0)
        # Slice the input list for this portion.
        parts.append(input_list[start:end])
        # Update the start index for the next portion.
        start = end

    return parts


def fixed_undirected_graph_gen(num_ver: int, num_neigh: int) -> Graph:
    """Deterministically generate simple undirected graph on vertices 1 to num_ver.

    Requires 0 < num_neigh < num_ver and num_ver * num_neigh  to be even.
    :param num_ver: The desired number of vertex to generate.
    :param num_neigh: The number of neighbors each vertex should have.
    :return: a dictionary {vertex: neighbors}.
    """
    # Check for necessary condition before generating.
    if not (0 < num_neigh < num_ver) or num_ver % 2 != 0:
        raise ValueError("Need 0 < number of neighbors < number of vertices and number of vertices even.")

    # Generate a dictionary mapping each vertex to a set.
    result: Graph = {v: [] for v in range(1, num_ver + 1)}

    # Correctly sample the vertices.
    for i in range(1, num_ver + 1):
        for j in range(1, num_neigh // 2 + 1):
            u = ((i - 1 + j) % num_ver) + 1
            w = ((i - 1 - j) % num_ver) + 1
            result[i].append(u)
            result[u].append(i)
            result[i].append(w)
            result[w].append(i)

    # Return the result.
    return result


def random_undirected_graph_gen(num_ver: int, num_neigh: int, seed: int | None = None) -> Graph:
    """Randomly generate simple undirected graph on vertices 1 to num_ver.

    :param num_ver: The desired number of vertex to generate.
    :param num_neigh: The number of neighbors each vertex should have.
    :param seed: The random seed to use to derive the same "random" graph.
    :return: a dictionary {vertex: neighbors}.
    """
    # Check for necessary condition before generating.
    if not (0 < num_neigh < num_ver) or num_ver % 2 != 0:
        raise ValueError("Need 0 < number of neighbors < number of vertices and number of vertices even.")

    # Use the desired seed if provided.
    if seed is not None:
        random.seed(seed)

    # There can be num_ver - 1 perfect matchings.
    num_matchings = num_ver - 1

    # Generate all perfect matchings via round-robin 1-factorization.
    factorization = []
    for r in range(num_matchings):
        # Pair (n, r + 1) and then pair up remaining vertices symmetrically.
        matching = [(num_ver, r + 1)]
        for i in range(1, (num_ver - 1) // 2 + 1):
            a = ((r + i) % (num_ver - 1)) + 1
            b = ((r - i) % (num_ver - 1)) + 1
            if a > b:
                a, b = b, a
            matching.append((a, b))
        factorization.append(matching)

    # Pick k random distinct matchings.
    chosen_rounds = random.sample(range(num_matchings), num_neigh)

    # Randomly permute vertex labels to avoid structural bias.
    perm = list(range(1, num_ver + 1))
    random.shuffle(perm)
    relabel = {old: i + 1 for i, old in enumerate(perm)}

    # Build adjacency list from selected matchings.
    result: Graph = {v: [] for v in range(1, num_ver + 1)}

    for r in chosen_rounds:
        for a, b in factorization[r]:
            # Unpack the vertices represented by the edge.
            u, v = relabel[a], relabel[b]
            result[u].append(v)
            result[v].append(u)

    return result


def sparsify_graph(graph: Graph, cur_neigh: int, max_neigh: int) -> Graph:
    """Sparsify graph by inserting intermediate nodes into the graph.

    :param graph: A dictionary representing a graph {vertex: neighbors}.
    :param cur_neigh: The current number of neighbors each vertex has.
    :param max_neigh: The maximum number of neighbors each vertex could have.
    :return: A dictionary representing a graph {vertex: neighbors}.
    """
    # Compute the number of intermediate vertices each vertex should have.
    split_num = cur_neigh // max_neigh + 1

    # Set the starting vertex.
    starting_vertex = len(graph) + 1

    # For each vertex in the graph.
    for vertex in range(1, len(graph) + 1):
        # Compute the intermediate vertices.
        intermediate_vertices = [starting_vertex + i for i in range(split_num)]

        # Split neighbors.
        neighbors = split_set(input_list=sorted(graph[vertex]), split_num=split_num)

        # Insert intermediate vertices.
        for i in range(split_num):
            graph[starting_vertex] = neighbors[i]
            starting_vertex += 1

        # Update the original vertex.
        graph[vertex] = intermediate_vertices

    return graph
