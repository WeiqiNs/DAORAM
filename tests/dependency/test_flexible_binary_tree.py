"""Characterization tests for the revived FlexibleBinaryTree.

The module had no prior working spec (it could not even be constructed), so these tests pin the
behavior the revived code exhibits -- the scale_up/scale_down assertions in particular CHARACTERIZE
current behavior rather than enforce an external contract. Leaf labels are (leaf, level) tuples whose
storage index is 2 ** (level - 1) - 1 + leaf.
"""

from typing import Any, cast

import pytest

from oblivlib.dependency.flexible_binary_tree import FlexibleBinaryTree
from oblivlib.dependency.helper import Data
from oblivlib.dependency.types import Block, Buckets


def _fdata(key: Any = None, leaf: Any = None, value: Any = None) -> Data:
    """Build flexible-tree Data whose leaf is a (leaf, level) tuple; Data.leaf is typed int|None,
    so cast at this boundary the same way the source does (see fill_data_to_storage_leaf)."""
    return Data(key=key, leaf=cast("int | None", leaf), value=value)


def _as_data(block: Block) -> Data:
    """Narrow a read Block to Data; these plaintext trees only ever store Data."""
    assert isinstance(block, Data)
    return block


class TestFlexibleBinaryTree:
    def test_fill_buckets_with_dummy_data(self):
        buckets: Buckets = [[], [_fdata(key=1, leaf=(0, 4), value="x")]]
        FlexibleBinaryTree.fill_buckets_with_dummy_data(buckets, bucket_size=3)
        assert all(len(b) == 3 for b in buckets)
        assert buckets[0] == [Data(), Data(), Data()]

    def test_get_cross_index(self):
        assert FlexibleBinaryTree.get_cross_index(leaf_a=1023, leaf_b=1024) == 511
        assert FlexibleBinaryTree.get_cross_index(leaf_a=1023, leaf_b=1023) == 1023
        assert FlexibleBinaryTree.get_cross_index(leaf_a=7, leaf_b=12) == FlexibleBinaryTree.get_cross_index(
            leaf_a=12, leaf_b=7
        )

    def test_get_cross_index_level(self):
        assert FlexibleBinaryTree.get_cross_index_level(leaf_a=1023, leaf_b=1024) == 9
        assert FlexibleBinaryTree.get_cross_index_level(leaf_a=1023, leaf_b=1023) == 10
        assert FlexibleBinaryTree.get_cross_index_level(leaf_a=0, leaf_b=0) == 0

    def test_fill_data_to_path_single(self):
        level = 4
        leaf = (0, 4)
        path = [[] for _ in range(level)]
        assert FlexibleBinaryTree.fill_data_to_path(
            _fdata(key=1, leaf=(0, 4), value="same"), path, leaf, bucket_size=1, level=level
        )
        assert [i for i, b in enumerate(path) if b] == [3]
        assert FlexibleBinaryTree.fill_data_to_path(
            _fdata(key=2, leaf=(1, 4), value="sib"), path, leaf, bucket_size=1, level=level
        )
        assert [i for i, b in enumerate(path) if b] == [2, 3]

    def test_fill_data_to_path_returns_false_when_full(self):
        level = 4
        leaf = (0, 4)
        path = [[] for _ in range(level)]
        for i in range(level):
            assert FlexibleBinaryTree.fill_data_to_path(
                _fdata(key=i, leaf=(0, 4), value=i), path, leaf, bucket_size=1, level=level
            )
        assert not FlexibleBinaryTree.fill_data_to_path(
            _fdata(key=99, leaf=(0, 4), value=99), path, leaf, bucket_size=1, level=level
        )

    def test_fill_data_to_path_dict(self):
        path = {i: [] for i in [7, 3, 1, 0]}
        assert FlexibleBinaryTree.fill_data_to_path_dict(
            _fdata(key=1, leaf=(0, 4), value="x"), path, (0, 4), bucket_size=1
        )
        assert [k for k, v in path.items() if v] == [7]

    def test_fill_data_to_mul_path(self):
        path = {i: [] for i in [7, 8, 3, 4, 1, 0]}
        assert FlexibleBinaryTree.fill_data_to_mul_path(
            Data(key=1, leaf=7, value="m"), path, leaves=[7, 8], bucket_size=1
        )
        assert [k for k, v in path.items() if v] == [7]

    def test_adjust_to_same_level(self):
        assert FlexibleBinaryTree.adjust_to_same_level(1023, 3, 11, 3) == (3, 3)
        assert FlexibleBinaryTree.adjust_to_same_level(3, 1023, 3, 11) == (3, 3)

    def test_adjust_to_lowest_level(self):
        assert FlexibleBinaryTree.adjust_to_lowest_level(0, old_level=1, level=11) == 1023
        assert FlexibleBinaryTree.adjust_to_lowest_level(1023, old_level=11, level=11) == 1023
        assert FlexibleBinaryTree.adjust_to_lowest_level(7, old_level=4, level=3) == 3

    def test_get_actual_leaf_index(self):
        assert FlexibleBinaryTree.get_actual_leaf_index((0, 11)) == 1023
        assert FlexibleBinaryTree.get_actual_leaf_index((2, 11)) == 1025
        assert FlexibleBinaryTree.get_actual_leaf_index((0, 1)) == 0
        assert FlexibleBinaryTree.get_actual_leaf_index((1, 4)) == 8

    def test_get_actual_leaf_index_rejects_non_tuple(self):
        not_a_tuple: Any = 5
        with pytest.raises(TypeError):
            FlexibleBinaryTree.get_actual_leaf_index(not_a_tuple)

    def test_scale_up_changes_level_and_preserves_data(self):
        tree = FlexibleBinaryTree(num_data=8, bucket_size=4)
        before = (tree.level, tree.size, tree.start_leaf)
        assert before == (4, 15, 7)
        tree.fill_data_to_storage_leaf(_fdata(key=1, leaf=(0, 4), value="A"))

        tree.scale_up()
        assert (tree.level, tree.size, tree.start_leaf) == (5, 31, 15)
        assert _as_data(tree.storage[7][0]).value == "A"
        retrieved = [d for bucket in tree.read_path((0, 4)) for d in map(_as_data, bucket) if d.value == "A"]
        assert retrieved and retrieved[0].key == 1

    def test_scale_down_changes_level_and_preserves_data(self):
        tree = FlexibleBinaryTree(num_data=8, bucket_size=4)
        tree.fill_data_to_storage_leaf(_fdata(key=1, leaf=(0, 4), value="A"))
        assert _as_data(tree.storage[7][0]).value == "A"

        assert tree.scale_down() is True
        assert (tree.level, tree.size, tree.start_leaf) == (3, 7, 3)
        assert _as_data(tree.storage[3][0]).value == "A"
        retrieved = [d for bucket in tree.read_path((0, 4)) for d in map(_as_data, bucket) if d.value == "A"]
        assert retrieved and retrieved[0].key == 1

    def test_scale_down_refuses_when_data_would_not_fit(self):
        tree = FlexibleBinaryTree(num_data=8, bucket_size=1)
        for i in range(4):
            assert tree.fill_data_to_storage_leaf(_fdata(key=i, leaf=(0, 4), value=i))
        assert tree.scale_down() is False
        assert (tree.level, tree.size) == (4, 15)

    def test_scale_down_boundary_is_feasible(self):
        tree = FlexibleBinaryTree(num_data=8, bucket_size=1)
        for i in range(3):
            assert tree.fill_data_to_storage_leaf(_fdata(key=i, leaf=(0, 4), value=i))
        assert tree.scale_down() is True
        assert tree.level == 3

    def test_scale_up_then_scale_down_round_trips(self):
        tree = FlexibleBinaryTree(num_data=8, bucket_size=4)
        tree.fill_data_to_storage_leaf(_fdata(key=1, leaf=(0, 4), value="A"))
        original = (tree.level, tree.size, tree.start_leaf)
        tree.scale_up()
        assert tree.scale_down() is True
        assert (tree.level, tree.size, tree.start_leaf) == original
        retrieved = [d for bucket in tree.read_path((0, 4)) for d in map(_as_data, bucket) if d.value == "A"]
        assert retrieved and retrieved[0].key == 1

    def test_fill_data_to_storage_leaf(self):
        tree = FlexibleBinaryTree(num_data=pow(2, 10), bucket_size=4)
        for i in range(10):
            assert tree.fill_data_to_storage_leaf(_fdata(key=i, leaf=(0, 11), value=i))
        path = tree.read_path((0, 11))
        assert _as_data(path[0][0]).key == 0
        assert _as_data(path[1][0]).key == 4
        assert _as_data(path[2][1]).key == 9

    def test_get_leaf_path(self):
        tree = FlexibleBinaryTree(num_data=pow(2, 10), bucket_size=4)
        assert tree.get_leaf_path(leaf=(0, 11)) == [1023, 511, 255, 127, 63, 31, 15, 7, 3, 1, 0]
        assert tree.get_leaf_path(leaf=(0, 1)) == tree.get_leaf_path(leaf=(0, 11))

    def test_get_mul_leaf_path(self):
        tree = FlexibleBinaryTree(num_data=pow(2, 10), bucket_size=4)
        assert tree.get_mul_leaf_path(leaves=[(0, 11), (1, 11)]) == [1024, 1023, 511, 255, 127, 63, 31, 15, 7, 3, 1, 0]

    def test_get_leaf_block(self):
        tree = FlexibleBinaryTree(num_data=pow(2, 10), bucket_size=4)
        assert tree.get_leaf_block(leaf=(0, 11), index=0) == 0
        assert tree.get_leaf_block(leaf=(0, 11), index=1) == 1
        assert tree.get_leaf_block(leaf=(0, 11), index=tree.level - 1) == 1023

    def test_read_write_path_round_trip(self):
        tree = FlexibleBinaryTree(num_data=8, bucket_size=2)
        leaf = (0, 4)
        path = tree.read_path(leaf)
        assert len(path) == tree.level
        new_path: Buckets = [[_fdata(key=i, leaf=leaf, value=f"v{i}")] for i in range(len(path))]
        tree.write_path(leaf, new_path)
        reread = tree.read_path(leaf)
        assert [_as_data(b[0]).value for b in reread] == ["v0", "v1", "v2", "v3"]

    def test_read_path_rejects_bad_leaf(self):
        tree = FlexibleBinaryTree(num_data=8, bucket_size=2)
        bad_leaf: Any = 5
        with pytest.raises(TypeError):
            tree.read_path(bad_leaf)

    def test_write_path_wrong_length_raises(self):
        tree = FlexibleBinaryTree(num_data=8, bucket_size=2)
        with pytest.raises(ValueError):
            tree.write_path([(0, 4), (1, 4)], data=[[]])

    def test_read_write_block_round_trip(self):
        tree = FlexibleBinaryTree(num_data=8, bucket_size=2)
        leaf = (0, 4)
        new_path: Buckets = [[_fdata(key=i, leaf=leaf, value=f"v{i}")] for i in range(tree.level)]
        tree.write_path(leaf, new_path)
        assert _as_data(tree.read_block(leaf, bucket_id=0, block_id=0)).value == "v3"
        tree.write_block(leaf, bucket_id=0, block_id=0, data=_fdata(key=42, leaf=leaf, value="blk"))
        assert _as_data(tree.read_block(leaf, bucket_id=0, block_id=0)).key == 42
        assert _as_data(tree.read_block(leaf, bucket_id=0, block_id=0)).value == "blk"
