from oblivlib.dependency import BinaryTree, Data
from oblivlib.dependency.codec import DefaultCodec
from oblivlib.dependency.types import Block

_CODEC = DefaultCodec(block_size=128)


def _as_data(block: Block) -> Data:
    """Narrow a read Block to Data; these plaintext trees only ever store Data."""
    assert isinstance(block, Data)
    return block


class TestBinaryTree:
    def test_fill_data_to_storage_leaf(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4, codec=_CODEC)
        for i in range(10):
            tree.fill_data_to_storage_leaf(data=Data(key=i, leaf=0, value=i))
        path = tree.read_path([0])
        assert [_as_data(block).key for block in path[1023]] == [0, 1, 2, 3]
        assert [_as_data(block).key for block in path[511]] == [4, 5, 6, 7]
        assert [_as_data(block).key for block in path[255]] == [8, 9]

    def test_read_write_path_round_trip(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4, codec=_CODEC)
        for i in range(4):
            tree.fill_data_to_storage_leaf(data=Data(key=i, leaf=0, value=i))

        path_data = tree.read_path([0])
        assert sorted(path_data, reverse=True) == [1023, 511, 255, 127, 63, 31, 15, 7, 3, 1, 0]

        leaf_bucket = 1023
        path_data[leaf_bucket] = [Data(key=42, leaf=0, value="written")]
        tree.write_path(path_data)
        reread = tree.read_path([0])
        assert _as_data(reread[leaf_bucket][0]).key == 42
        assert _as_data(reread[leaf_bucket][0]).value == "written"
