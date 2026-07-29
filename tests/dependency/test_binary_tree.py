import pytest

from oblivlib.dependency import BinaryTree, Data
from oblivlib.dependency.types import Block, BlockKey, BucketKey


def _as_data(block: Block) -> Data:
    """Narrow a read Block to Data; these plaintext trees only ever store Data."""
    assert isinstance(block, Data)
    return block


class TestBinaryTree:
    @pytest.mark.parametrize("p", list(range(0, 32)))
    def test_level_is_exact_at_powers_of_two(self, p):
        # Call compute_level directly: constructing BinaryTree(num_data=2**p) would eagerly allocate
        # a 2**(p+1)-bucket storage list (hundreds of GB at p=31) just to read one integer.
        n = pow(2, p)
        assert pow(2, BinaryTree.compute_level(num_data=n) - 1) == n

    def test_get_cross_index(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        assert tree.get_cross_index(leaf_a=0, leaf_b=512, level=tree.level) == 0
        assert tree.get_cross_index(leaf_a=0, leaf_b=511, level=tree.level) == 1
        assert tree.get_cross_index(leaf_a=10, leaf_b=10, level=tree.level) == tree.start_leaf + 10
        assert tree.get_cross_index(leaf_a=3, leaf_b=700, level=tree.level) == tree.get_cross_index(
            leaf_a=700, leaf_b=3, level=tree.level
        )

    def test_get_cross_index_level(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        assert tree.get_cross_index_level(leaf_a=0, leaf_b=512, level=tree.level) == 0
        assert tree.get_cross_index_level(leaf_a=0, leaf_b=511, level=tree.level) == 1
        assert tree.get_cross_index_level(leaf_a=10, leaf_b=10, level=tree.level) == 10

    def test_get_cross_index_level_is_exact_at_deep_levels(self):
        assert BinaryTree.get_cross_index_level(leaf_a=0, leaf_b=0, level=50) == 49
        for level in (40, 50, 53):
            idx = BinaryTree.get_cross_index(leaf_a=0, leaf_b=0, level=level)
            assert BinaryTree.get_cross_index_level(0, 0, level) == idx.bit_length()

    def test_fill_data_to_path_places_at_deepest_legal_bucket(self):
        path = BinaryTree.get_mul_path_dict(level=11, indices=[0, 2])
        assert BinaryTree.fill_data_to_path(
            data=Data(key=0, leaf=0, value="Path0"), path=path, leaves=[0, 2], level=11, bucket_size=1
        )
        assert BinaryTree.fill_data_to_path(
            data=Data(key=0, leaf=0, value="Up"), path=path, leaves=[0, 2], level=11, bucket_size=1
        )
        assert BinaryTree.fill_data_to_path(
            data=Data(key=1, leaf=2, value="Path2"), path=path, leaves=[0, 2], level=11, bucket_size=1
        )
        assert BinaryTree.fill_data_to_path(
            data=Data(key=2, leaf=10, value="Common"), path=path, leaves=[0, 2], level=11, bucket_size=1
        )
        assert _as_data(path[1023][0]).value == "Path0"
        assert _as_data(path[1025][0]).value == "Path2"
        assert _as_data(path[511][0]).value == "Up"
        assert _as_data(path[63][0]).value == "Common"

    def test_fill_data_to_path_returns_false_when_full(self):
        path = BinaryTree.get_mul_path_dict(level=4, indices=[0])
        for i in range(4):
            assert BinaryTree.fill_data_to_path(
                data=Data(key=i, leaf=0, value=i), path=path, leaves=[0], level=4, bucket_size=1
            )
        assert not BinaryTree.fill_data_to_path(
            data=Data(key=99, leaf=0, value=99), path=path, leaves=[0], level=4, bucket_size=1
        )

    def test_fill_data_to_storage_leaf(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        for i in range(10):
            tree.fill_data_to_storage_leaf(data=Data(key=i, leaf=0, value=i))
        path = tree.extract_path(0, tree.read_path([0]))
        assert _as_data(path[0][0]).key == 0
        assert _as_data(path[1][0]).key == 4
        assert _as_data(path[2][1]).key == 9

    def test_get_leaf_path(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        assert tree.get_leaf_path(leaf=0) == [1023, 511, 255, 127, 63, 31, 15, 7, 3, 1, 0]
        assert tree.get_leaf_path(leaf=100)[1:] == tree.get_leaf_path(leaf=101)[1:]

    def test_get_mul_leaf_path(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        assert tree.get_mul_leaf_path(leaves=[0, 1]) == [1024, 1023, 511, 255, 127, 63, 31, 15, 7, 3, 1, 0]
        assert tree.get_mul_leaf_path(leaves=[0, 2]) == [1025, 1023, 512, 511, 255, 127, 63, 31, 15, 7, 3, 1, 0]

    def test_get_leaf_block(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        assert tree.get_leaf_block(leaf=0, index=0) == 0
        assert tree.get_leaf_block(leaf=0, index=1) == 1
        assert tree.get_leaf_block(leaf=0, index=tree.level - 1) == 1023

    def test_read_write_path_round_trip(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        for i in range(4):
            tree.fill_data_to_storage_leaf(data=Data(key=i, leaf=0, value=i))

        path_data = tree.read_path([0])
        assert isinstance(path_data, dict)
        path = tree.extract_path(0, path_data)
        assert len(path) == tree.level

        leaf_index = tree.get_leaf_block(leaf=0, index=0)
        path_data[leaf_index] = [Data(key=42, leaf=0, value="written")]
        tree.write_path(path_data)
        reread = tree.read_path([0])
        assert _as_data(reread[leaf_index][0]).key == 42
        assert _as_data(reread[leaf_index][0]).value == "written"

    def test_extract_path_missing_raises(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        with pytest.raises(KeyError):
            tree.extract_path(1023, tree.read_path([0]))

    def test_read_write_bucket_round_trip(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        tree.fill_data_to_storage_leaf(data=Data(key=7, leaf=0, value="bucket"))
        leaf_bucket = tree.level - 1
        bucket_data = tree.read_bucket([BucketKey(0, leaf_bucket), BucketKey(0, 0)])
        assert _as_data(bucket_data[BucketKey(0, leaf_bucket)][0]).key == 7
        assert bucket_data[BucketKey(0, 0)] == []

        bucket_data[BucketKey(0, 0)] = [Data(key=8, leaf=0, value="new")]
        tree.write_bucket(bucket_data)
        assert _as_data(tree.read_bucket([BucketKey(0, 0)])[BucketKey(0, 0)][0]).key == 8

    def test_read_write_block_round_trip(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        tree.fill_data_to_storage_leaf(data=Data(key=0, leaf=0, value="orig"))
        key = BlockKey(0, tree.level - 1, 0)
        block_data = tree.read_block([key])
        assert _as_data(block_data[key]).value == "orig"

        block_data[key] = Data(key=99, leaf=0, value="new")
        tree.write_block(block_data)
        assert _as_data(tree.read_block([key])[key]).key == 99
        assert _as_data(tree.read_block([key])[key]).value == "new"
