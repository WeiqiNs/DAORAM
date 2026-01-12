from daoram.dependency import BinaryTree, Data


class TestBinaryTree:
    def test_init(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        assert tree.level == 11
        assert tree.size == 2047
        assert tree.start_leaf == 1023

    def test_get_parent_index(self):
        assert BinaryTree.get_parent_index(index=0) < 0
        assert BinaryTree.get_parent_index(index=16) == 7
        assert BinaryTree.get_parent_index(index=15) == BinaryTree.get_parent_index(index=16)

    def test_get_path_indices(self):
        assert BinaryTree.get_path_indices(index=0) == [0]
        assert BinaryTree.get_path_indices(index=15) == [15, 7, 3, 1, 0]
        assert BinaryTree.get_path_indices(index=99)[1:] == BinaryTree.get_path_indices(index=100)[1:]

    def test_get_mul_path_dict(self):
        result = sorted(list(BinaryTree.get_mul_path_dict(level=11, indices=[0, 2]).keys()), reverse=True)
        assert result == [1025, 1023, 512, 511, 255, 127, 63, 31, 15, 7, 3, 1, 0]

    def test_fill_data_to_path(self):
        path = BinaryTree.get_mul_path_dict(level=11, indices=[0, 2])
        BinaryTree.fill_data_to_path(
            data=Data(key=0, leaf=0, value="Path0"), path=path, leaves=[0, 2], level=11, bucket_size=1)
        BinaryTree.fill_data_to_path(
            data=Data(key=0, leaf=0, value="Up"), path=path, leaves=[0, 2], level=11, bucket_size=1)
        BinaryTree.fill_data_to_path(
            data=Data(key=1, leaf=2, value="Path2"), path=path, leaves=[0, 2], level=11, bucket_size=1)
        BinaryTree.fill_data_to_path(
            data=Data(key=2, leaf=10, value="Common"), path=path, leaves=[0, 2], level=11, bucket_size=1)
        assert path[1023][0].value == "Path0"
        assert path[1025][0].value == "Path2"
        assert path[511][0].value == "Up"
        assert path[63][0].value == "Common"

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

    def test_get_cross_index(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        assert tree.get_cross_index(leaf_one=0, leaf_two=512, level=tree.level) == 0
        assert tree.get_cross_index(leaf_one=0, leaf_two=511, level=tree.level) == 1
        assert tree.get_cross_index(leaf_one=10, leaf_two=10, level=tree.level) == tree.start_leaf + 10

    def test_get_cross_index_level(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        assert tree.get_cross_index_level(leaf_one=0, leaf_two=512, level=tree.level) == 0
        assert tree.get_cross_index_level(leaf_one=0, leaf_two=511, level=tree.level) == 1
        assert tree.get_cross_index_level(leaf_one=10, leaf_two=10, level=tree.level) == 10

    def test_fill_data_to_storage_leaf(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        for i in range(10):
            tree.fill_data_to_storage_leaf(data=Data(key=i, leaf=0, value=i))
        # read_path now returns dict, extract path for leaf 0
        path_data = tree.read_path([0])
        path = tree.extract_path(0, path_data)
        assert path[0][0].key == 0
        assert path[1][0].key == 4
        assert path[2][1].key == 9

    def test_read_write_path(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        # Fill some data
        for i in range(4):
            tree.fill_data_to_storage_leaf(data=Data(key=i, leaf=0, value=i))

        # Read path as dict
        path_data = tree.read_path([0])
        assert isinstance(path_data, dict)

        # Extract and verify
        path = tree.extract_path(0, path_data)
        assert len(path) == tree.level

        # Modify and write back
        tree.write_path(path_data)

    def test_read_write_bucket(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        # Fill some data
        tree.fill_data_to_storage_leaf(data=Data(key=0, leaf=0, value="test"))

        # Read bucket
        bucket_data = tree.read_bucket([(0, 0), (0, 1)])
        assert isinstance(bucket_data, dict)
        assert (0, 0) in bucket_data

        # Write bucket back
        tree.write_bucket(bucket_data)

    def test_read_write_block(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        # Fill some data
        tree.fill_data_to_storage_leaf(data=Data(key=0, leaf=0, value="test"))

        # Read block
        block_data = tree.read_block([(0, tree.level - 1, 0)])
        assert isinstance(block_data, dict)
        assert (0, tree.level - 1, 0) in block_data

        # Write block
        tree.write_block(block_data)
