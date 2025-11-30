import random

from daoram.dependency import BPlusTree, BPlusTreeNode, KVPair


class TestBPlusTree:
    def test_insert(self):
        # Create a b+ tree object.
        bplus_tree = BPlusTree(order=3, leaf_range=1000)

        # Set an empty b+ tree node.
        root = BPlusTreeNode()

        # Insert some values in order.
        for i in range(21):
            root = bplus_tree.insert(root=root, kv_pair=KVPair(key=i, value=i))

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
            root = bplus_tree.insert(root=root, kv_pair=KVPair(key=i, value=i))

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
            root = bplus_tree.insert(root=root, kv_pair=KVPair(key=f"{i}", value=f"{i}"))

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
            root = bplus_tree.insert(root=root, kv_pair=KVPair(key=i, value=i))

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
            root = bplus_tree.insert(root=root, kv_pair=KVPair(key=i, value=i))

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
