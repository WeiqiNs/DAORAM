import random

from pyoblivlib.dependency import AVLTree, BinaryTree, BPlusTree, BPlusTreeNode, Data


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

    def test_fill_data_to_mul_path(self):
        path = BinaryTree.get_mul_path_dict(level=11, indices=[0, 2])
        BinaryTree.fill_data_to_mul_path(
            data=Data(key=0, leaf=0, value="Path0"), path=path, leaves=[0, 2], level=11, bucket_size=1)
        BinaryTree.fill_data_to_mul_path(
            data=Data(key=0, leaf=0, value="Up"), path=path, leaves=[0, 2], level=11, bucket_size=1)
        BinaryTree.fill_data_to_mul_path(
            data=Data(key=1, leaf=2, value="Path2"), path=path, leaves=[0, 2], level=11, bucket_size=1)
        BinaryTree.fill_data_to_mul_path(
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

    def test_fill_data_to_leaf(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        for i in range(10):
            tree.fill_data_to_storage_leaf(data=Data(key=i, leaf=0, value=i))
        assert tree.read_path(0)[0][0].key == 0
        assert tree.read_path(0)[1][0].key == 4
        assert tree.read_path(0)[2][1].key == 9

    def test_fill_data_to_path(self):
        tree = BinaryTree(num_data=pow(2, 10), bucket_size=4)
        path = [[] for _ in range(tree.level)]
        tree.fill_data_to_path(data=Data(key=0, leaf=0, value=0), path=path, leaf=1, level=tree.level, bucket_size=4)
        tree.fill_data_to_path(data=Data(key=0, leaf=0, value=0), path=path, leaf=0, level=tree.level, bucket_size=4)
        tree.fill_data_to_path(data=Data(key=1, leaf=0, value=1), path=path, leaf=7, level=tree.level, bucket_size=4)

        # Access the second last bucket.
        assert path[-2][0].key == 0
        assert path[-4][0].key == 1


class TestAVLTree:
    def test_insert(self):
        # Create an avl tree object.
        avl_tree = AVLTree(leaf_range=1000)

        # Set the beginning root to None and create a list of kv pairs.
        root = None
        kv_pairs = [(i, i) for i in range(1000)]

        # Perform insertion.
        for kv_pair in kv_pairs:
            root = avl_tree.insert(root=root, kv_pair=kv_pair)

        # Test left children.
        assert root.key == 511
        assert root.left_node.key == 255
        assert root.left_node.left_node.key == 127
        assert root.left_node.right_node.key == 383

        # Test right children.
        assert root.right_node.key == 767
        assert root.right_node.left_node.key == 639
        assert root.right_node.right_node.key == 895

    def test_recursive_insert(self):
        # Create an avl tree object.
        avl_tree = AVLTree(leaf_range=1000)

        # Set the beginning root to None and create a list of kv pairs.
        root = None
        root_recursive = None
        kv_pairs = [(i, i) for i in range(1000)]

        # Perform insertion.
        for kv_pair in kv_pairs:
            root = avl_tree.insert(root=root, kv_pair=kv_pair)
            root_recursive = avl_tree.recursive_insert(root=root_recursive, kv_pair=kv_pair)

        # Test left children.
        assert root.key == root_recursive.key
        assert root.left_node.key == root_recursive.left_node.key
        assert root.left_node.left_node.key == root_recursive.left_node.left_node.key
        assert root.left_node.right_node.key == root_recursive.left_node.right_node.key

        # Test right children.
        assert root.right_node.key == root_recursive.right_node.key
        assert root.right_node.left_node.key == root_recursive.right_node.left_node.key
        assert root.right_node.right_node.key == root_recursive.right_node.right_node.key

    def test_str_insert(self):
        # Create an avl tree object.
        avl_tree = AVLTree(leaf_range=10)

        # Set the beginning root to None and create a list of kv pairs.
        root = None
        kv_pairs = [(f"{i}", f"{i}") for i in range(1000)]

        # Perform insertion.
        for kv_pair in kv_pairs:
            root = avl_tree.recursive_insert(root=root, kv_pair=kv_pair)

        # Test left children.
        assert root.key == "60"
        assert root.left_node.key == "35"
        assert root.left_node.left_node.key == "22"
        assert root.left_node.right_node.key == "5"

        # Test right children.
        assert root.right_node.key == "72"
        assert root.right_node.left_node.key == "66"
        assert root.right_node.right_node.key == "87"

    def test_str_recursive_insert(self):
        # Create an avl tree object.
        avl_tree = AVLTree(leaf_range=1000)

        # Set the beginning root to None and create a list of kv pairs.
        root = None
        root_recursive = None
        kv_pairs = [(f"{i}", f"{i}") for i in range(1000)]

        # Perform insertion.
        for kv_pair in kv_pairs:
            root = avl_tree.insert(root=root, kv_pair=kv_pair)
            root_recursive = avl_tree.recursive_insert(root=root_recursive, kv_pair=kv_pair)

        # Test left children.
        assert root.key == root_recursive.key
        assert root.left_node.key == root_recursive.left_node.key
        assert root.left_node.left_node.key == root_recursive.left_node.left_node.key
        assert root.left_node.right_node.key == root_recursive.left_node.right_node.key

        # Test right children.
        assert root.right_node.key == root_recursive.right_node.key
        assert root.right_node.left_node.key == root_recursive.right_node.left_node.key
        assert root.right_node.right_node.key == root_recursive.right_node.right_node.key

    def test_search(self):
        # Create an avl tree object.
        avl_tree = AVLTree(leaf_range=1000)

        # Set the beginning root to None.
        root = None

        # Generate some random values.
        random_values = set([random.randint(0, 10000) for _ in range(1000)])

        # Perform insertion.
        for i in random_values:
            root = avl_tree.insert(root=root, kv_pair=(i, i))

        # Perform search.
        for i in random_values:
            assert i == avl_tree.search(key=i, root=root)

    def test_get_list(self):
        # Create an avl tree object.
        avl_tree = AVLTree(leaf_range=10)

        # Set the beginning root to None and create a list of kv pairs.
        root = None
        kv_pairs = [(i, i) for i in range(10)]

        # Perform insertion.
        for kv_pair in kv_pairs:
            root = avl_tree.insert(root=root, kv_pair=kv_pair)

        # Convert the avl tree nodes to a list.
        data_list = avl_tree.get_data_list(root=root)

        # The root node should be 3.
        assert data_list[0].key == 3
        assert data_list[0].value.r_key == 7
        assert data_list[0].value.l_key == 1
        assert data_list[0].value.r_height == 3
        assert data_list[0].value.l_height == 2

        # The second node added should be the right node. (As they are more recent in stack.)
        assert data_list[1].key == 7
        assert data_list[1].value.r_key == 8
        assert data_list[1].value.l_key == 5
        assert data_list[1].value.r_height == 2
        assert data_list[1].value.l_height == 2


class TestBPlusTree:
    def test_insert(self):
        # Create a b+ tree object.
        bplus_tree = BPlusTree(order=3, leaf_range=1000)

        # Set an empty b+ tree node.
        root = BPlusTreeNode()

        # Insert some values in order.
        for i in range(21):
            root = bplus_tree.insert(root=root, kv_pair=(i, i))

        # Test left children.
        assert root.keys == [8]
        assert root.values[0].keys == [4]
        assert root.values[0].values[0].keys == [2]
        assert root.values[0].values[1].keys == [6]
        assert root.values[0].values[0].values[0].keys == [1]
        assert root.values[0].values[0].values[1].keys == [3]
        assert root.values[0].values[1].values[0].values[1].values == [5]
        assert root.values[0].values[1].values[0].values[0].is_leaf is True

        # Test right children.
        assert root.values[1].keys == [12, 16]
        assert root.values[1].values[0].keys == [10]
        assert root.values[1].values[1].keys == [14]
        assert root.values[1].values[2].keys == [18]
        assert root.values[1].values[2].values[1].keys == [19]
        assert root.values[1].values[2].values[1].values[1].keys == [19, 20]
        assert root.values[1].values[2].values[1].values[1].is_leaf is True

    def test_rand_insert(self):
        # Create a b+ tree object.
        bplus_tree = BPlusTree(order=4, leaf_range=1000)

        # Set an empty b+ tree node.
        root = BPlusTreeNode()

        # Create a list to hold some values not in order.
        random_values = [0, 10, 9, 2, 3, 8, 7, 4, 5, 6]

        # Insert some values in order.
        for i in random_values:
            root = bplus_tree.insert(root=root, kv_pair=(i, i))

        # Test left children.
        assert root.keys == [7]
        assert root.values[0].keys == [3, 5]
        assert root.values[0].values[1].keys == [3, 4]
        assert root.values[0].values[2].keys == [5, 6]
        assert root.values[0].values[0].is_leaf is True

        # Test right children.
        assert root.values[1].keys == [9]
        assert root.values[1].values[0].keys == [7, 8]
        assert root.values[1].values[1].keys == [9, 10]

    def test_str_insert(self):
        # Create a b+ tree object.
        bplus_tree = BPlusTree(order=5, leaf_range=1000)

        # Set an empty b+ tree node.
        root = BPlusTreeNode()

        # Insert some values in order.
        for i in range(21):
            root = bplus_tree.insert(root=root, kv_pair=(f"{i}", f"{i}"))

        # Test left children.
        assert root.keys == ["2"]
        assert root.values[0].keys == ["10", "12", "14", "16"]
        assert root.values[0].values[0].keys == ["0", "1"]
        assert root.values[0].values[4].keys == ["16", "17", "18", "19"]
        assert root.values[0].values[2].is_leaf is True

        # Test right children.
        assert root.values[1].keys == ["4", "6"]
        assert root.values[1].values[0].keys == ["2", "20", "3"]
        assert root.values[1].values[2].keys == ["6", "7", "8", "9"]
        assert root.values[1].values[1].is_leaf is True

    def test_search(self):
        # Create an avl tree object.
        bplus_tree = BPlusTree(order=10, leaf_range=1000)

        # Set the beginning root to None.
        root = BPlusTreeNode()

        # Generate some random values.
        random_values = set([random.randint(0, 10000) for _ in range(1000)])

        # Perform insertion.
        for i in random_values:
            root = bplus_tree.insert(root=root, kv_pair=(i, i))

        # Perform search.
        for i in random_values:
            assert i == bplus_tree.search(key=i, root=root)

    def test_post_order(self):
        # Create a b+ tree object.
        bplus_tree = BPlusTree(order=4, leaf_range=1000)

        # Set an empty b+ tree node.
        root = BPlusTreeNode()

        # Insert some values in order.
        for i in range(21):
            root = bplus_tree.insert(root=root, kv_pair=(i, i))

        # Convert the B+ tree nodes to a list.
        data_list = bplus_tree.get_data_list(root=root)

        # The data list should contain 14 values.
        assert len(data_list) == 14

        # The root should have keys 6, 12 and has three values.
        assert data_list[0].value.keys == [6, 12]
        assert len(data_list[0].value.values) == 3

        # The right-most child of the root should have the following.
        assert data_list[1].key == 3
        assert data_list[1].value.keys == [14, 16, 18]
        assert len(data_list[1].value.values) == 4
