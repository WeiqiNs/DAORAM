import pytest

from oblivlib.dependency import Data
from oblivlib.dependency.heap_index import (
    compute_level,
    empty_path,
    fill_data_to_path,
    leaf_lca,
    parent,
    path_to_root,
    union_of_paths,
)
from oblivlib.dependency.types import Block


def _as_data(block: Block) -> Data:
    assert isinstance(block, Data)
    return block


@pytest.mark.parametrize("p", list(range(0, 32)))
def test_compute_level_exact_at_powers_of_two(p):
    n = pow(2, p)
    assert pow(2, compute_level(n) - 1) == n
    assert compute_level(n + 1) == compute_level(n) + 1


def test_parent():
    assert [parent(index) for index in (1, 2, 3, 4, 1023, 1024)] == [0, 0, 1, 1, 511, 511]


def test_path_to_root():
    assert path_to_root(0) == [0]
    assert path_to_root(15) == [15, 7, 3, 1, 0]
    assert path_to_root(1023) == [1023, 511, 255, 127, 63, 31, 15, 7, 3, 1, 0]
    assert path_to_root(99)[1:] == path_to_root(100)[1:]


def test_union_of_paths_dedup_deepest_first():
    assert union_of_paths([15, 16]) == [16, 15, 7, 3, 1, 0]
    assert union_of_paths([1023, 1025]) == [1025, 1023, 512, 511, 255, 127, 63, 31, 15, 7, 3, 1, 0]
    assert union_of_paths([]) == []


def test_leaf_lca_known_values():
    level = 11
    assert leaf_lca(0, 512, level) == 0
    assert leaf_lca(0, 511, level) == 1
    assert leaf_lca(10, 10, level) == 1023 + 10
    assert leaf_lca(3, 700, level) == leaf_lca(700, 3, level)

    level = 6
    first_leaf = (1 << (level - 1)) - 1
    for leaf_a in range(1 << (level - 1)):
        for leaf_b in range(1 << (level - 1)):
            a, b = leaf_a + first_leaf, leaf_b + first_leaf
            while a != b:
                a, b = (a - 1) // 2, (b - 1) // 2
            assert leaf_lca(leaf_a, leaf_b, level) == a


def test_empty_path():
    path = empty_path(level=11, leaves=[0, 2])
    assert sorted(path, reverse=True) == [1025, 1023, 512, 511, 255, 127, 63, 31, 15, 7, 3, 1, 0]
    assert all(bucket == [] for bucket in path.values())


def test_fill_data_to_path_places_at_deepest_legal_bucket():
    path = empty_path(level=11, leaves=[0, 2])
    for data in (
        Data(key=0, leaf=0, value="Path0"),
        Data(key=0, leaf=0, value="Up"),
        Data(key=1, leaf=2, value="Path2"),
        Data(key=2, leaf=10, value="Common"),
    ):
        assert fill_data_to_path(data, path, leaves=[0, 2], level=11, bucket_size=1)
    assert _as_data(path[1023][0]).value == "Path0"
    assert _as_data(path[1025][0]).value == "Path2"
    assert _as_data(path[511][0]).value == "Up"
    assert _as_data(path[63][0]).value == "Common"


def test_fill_data_to_path_returns_false_when_full():
    path = empty_path(level=4, leaves=[0])
    for i in range(4):
        assert fill_data_to_path(Data(key=i, leaf=0, value=i), path, leaves=[0], level=4, bucket_size=1)
    assert not fill_data_to_path(Data(key=99, leaf=0, value=99), path, leaves=[0], level=4, bucket_size=1)
