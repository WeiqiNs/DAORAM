from oblivlib.dependency.types import Data, PathData


def compute_level(num_data: int) -> int:
    return (num_data - 1).bit_length() + 1


def tree_size(level: int) -> int:
    return (1 << level) - 1


def leaf_index(leaf: int, level: int) -> int:
    return leaf + (1 << (level - 1)) - 1


def parent(index: int) -> int:
    return (index - 1) // 2


def path_to_root(index: int) -> list[int]:
    path = []
    while index >= 0:
        path.append(index)
        index = parent(index)
    return path


def path_indices(leaves: list[int], level: int) -> list[int]:
    nodes: set[int] = set()
    for leaf in leaves:
        nodes.update(path_to_root(leaf_index(leaf, level)))
    return sorted(nodes)


def leaf_lca(leaf_a: int, leaf_b: int, level: int) -> int:
    offset = 1 << (level - 1)
    a = leaf_a + offset
    b = leaf_b + offset
    return (a >> (a ^ b).bit_length()) - 1


def empty_path(level: int, leaves: list[int]) -> PathData:
    return {index: [] for index in path_indices(leaves, level)}


def fill_data_to_path(data: Data, path: PathData, *, leaves: list[int], level: int, bucket_size: int) -> bool:
    data_leaf = data.require_leaf()
    index = max(leaf_lca(data_leaf, leaf, level) for leaf in leaves)

    while index >= 0:
        if len(path[index]) < bucket_size:
            path[index].append(data)
            return True
        index = parent(index)

    return False
