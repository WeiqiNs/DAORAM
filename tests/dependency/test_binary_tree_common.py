"""Shared index-math tests for BinaryTree and FlexibleBinaryTree.

These two classes implement construction (level/size/start_leaf) and the static path helpers
(`get_parent_index`, `get_path_indices`, `get_mul_path_indices`, `get_mul_path_dict`) with identical
signatures and identical results, so one parametrized suite covers both via the `binary_tree_cls` fixture
(see `conftest.py`). Each test merges the assertions that previously lived in
`test_binary_tree.py` and `test_flexible_binary_tree.py`. Scheme-specific behavior (cross-index math,
leaf paths, fill helpers, storage round-trips, and FlexibleBinaryTree's scaling) stays per-file.
"""

import pytest


class TestTreeIndexCommon:
    @pytest.mark.parametrize(
        ("num_data", "level", "size", "start_leaf"),
        [
            (1, 1, 1, 0),
            (2, 2, 3, 1),
            (3, 3, 7, 3),
            (7, 4, 15, 7),
            (8, 4, 15, 7),
            (pow(2, 10), 11, 2047, 1023),
            (pow(2, 10) + 1, 12, 4095, 2047),
        ],
    )
    def test_init(self, binary_tree_cls, num_data, level, size, start_leaf):
        tree = binary_tree_cls(num_data=num_data, bucket_size=4)
        assert tree.level == level
        assert tree.size == size
        assert tree.start_leaf == start_leaf
        assert pow(2, level - 1) >= num_data
        assert pow(2, level - 2) < num_data or num_data <= 1

    def test_get_path_indices(self, binary_tree_cls):
        assert binary_tree_cls.get_path_indices(index=0) == [0]
        assert binary_tree_cls.get_path_indices(index=15) == [15, 7, 3, 1, 0]
        assert binary_tree_cls.get_path_indices(index=99)[1:] == binary_tree_cls.get_path_indices(index=100)[1:]

    def test_get_mul_path_indices_dedup_sorted(self, binary_tree_cls):
        assert binary_tree_cls.get_mul_path_indices(indices=[15, 16]) == [16, 15, 7, 3, 1, 0]

    def test_get_mul_path_dict(self, binary_tree_cls):
        path = binary_tree_cls.get_mul_path_dict(level=11, indices=[0, 2])
        assert sorted(path.keys(), reverse=True) == [1025, 1023, 512, 511, 255, 127, 63, 31, 15, 7, 3, 1, 0]
        assert all(bucket == [] for bucket in path.values())
