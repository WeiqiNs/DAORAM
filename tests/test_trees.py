import random

from daoram.dependency import AVLTree, BinaryTree, BPlusTree, BPlusTreeNode, Data


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

    # def test_delete_leaf_node(self):
    #     """Test deleting a leaf node (no children)."""
    #     avl_tree = AVLTree(leaf_range=1000)
    #     root = None

    #     # Build a small tree.
    #     for v in [10, 5, 15, 3, 7]:
    #         root = avl_tree.insert(root, (v, v))

    #     # Delete a leaf node.
    #     root = avl_tree.delete(root, 3)
    #     assert avl_tree.search(3, root) is None
    #     assert avl_tree.search(5, root) == 5
    #     assert avl_tree.search(7, root) == 7

    # def test_delete_node_with_one_child(self):
    #     """Test deleting a node with one child."""
    #     avl_tree = AVLTree(leaf_range=1000)
    #     root = None

    #     # Build a tree where node 15 has only one child.
    #     for v in [10, 5, 15, 20]:
    #         root = avl_tree.insert(root, (v, v))

    #     # Delete node with one child.
    #     root = avl_tree.delete(root, 15)
    #     assert avl_tree.search(15, root) is None
    #     assert avl_tree.search(20, root) == 20
    #     assert avl_tree.search(10, root) == 10

    # def test_delete_node_with_two_children(self):
    #     """Test deleting a node with two children."""
    #     avl_tree = AVLTree(leaf_range=1000)
    #     root = None

    #     # Build a tree.
    #     for v in [10, 5, 15, 3, 7, 12, 20]:
    #         root = avl_tree.insert(root, (v, v))

    #     # Delete node with two children (root).
    #     root = avl_tree.delete(root, 10)
    #     assert avl_tree.search(10, root) is None

    #     # All other nodes should still exist.
    #     for v in [5, 15, 3, 7, 12, 20]:
    #         assert avl_tree.search(v, root) == v

    # def test_delete_root_only(self):
    #     """Test deleting when there's only a root node."""
    #     avl_tree = AVLTree(leaf_range=1000)
    #     root = avl_tree.insert(None, (5, 5))

    #     root = avl_tree.delete(root, 5)
    #     assert root is None

    # def test_delete_nonexistent_key(self):
    #     """Test deleting a key that doesn't exist."""
    #     avl_tree = AVLTree(leaf_range=1000)
    #     root = None

    #     for v in [10, 5, 15]:
    #         root = avl_tree.insert(root, (v, v))

    #     # Deleting nonexistent key should return original tree.
    #     original_root_key = root.key
    #     root = avl_tree.delete(root, 100)
    #     assert root.key == original_root_key

    # def test_delete_maintains_balance(self):
    #     """Test that tree remains balanced after deletions."""
    #     avl_tree = AVLTree(leaf_range=1000)
    #     root = None

    #     # Insert values that create a balanced tree.
    #     values = list(range(100))
    #     for v in values:
    #         root = avl_tree.insert(root, (v, v))

    #     # Delete half the values.
    #     for v in values[::2]:
    #         root = avl_tree.delete(root, v)

    #     # Verify remaining values are searchable.
    #     for v in values[1::2]:
    #         assert avl_tree.search(v, root) == v

    #     # Verify deleted values are gone.
    #     for v in values[::2]:
    #         assert avl_tree.search(v, root) is None

    # def test_delete_random_order(self):
    #     """Test deleting in random order."""
    #     avl_tree = AVLTree(leaf_range=1000)
    #     root = None

    #     # Insert 100 values.
    #     values = list(range(100))
    #     for v in values:
    #         root = avl_tree.insert(root, (v, v))

    #     # Delete in shuffled order.
    #     random.seed(42)
    #     delete_order = values.copy()
    #     random.shuffle(delete_order)

    #     remaining = set(values)
    #     for v in delete_order:
    #         root = avl_tree.delete(root, v)
    #         remaining.remove(v)

    #         # Verify remaining values still exist.
    #         for r in remaining:
    #             assert avl_tree.search(r, root) == r

    #     assert root is None

    # def test_delete_all_sequential(self):
    #     """Test deleting all values sequentially."""
    #     avl_tree = AVLTree(leaf_range=1000)
    #     root = None

    #     values = list(range(50))
    #     for v in values:
    #         root = avl_tree.insert(root, (v, v))

    #     for v in values:
    #         root = avl_tree.delete(root, v)

    #     assert root is None


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

    def test_delete_single_element(self):
        """Test deleting from a tree with a single element."""
        bplus_tree = BPlusTree(order=3, leaf_range=1000)
        root = BPlusTreeNode()

        root = bplus_tree.insert(root, (1, 1))
        root = bplus_tree.delete(root, 1)

        assert root is None

    def test_delete_from_leaf(self):
        """Test basic deletion from a leaf node without underflow."""
        bplus_tree = BPlusTree(order=4, leaf_range=1000)
        root = BPlusTreeNode()

        # Insert values.
        for i in range(10):
            root = bplus_tree.insert(root, (i, i))

        # Delete a value.
        root = bplus_tree.delete(root, 5)

        # Verify deletion.
        try:
            bplus_tree.search(5, root)
            assert False, "Key 5 should not be found"
        except KeyError:
            pass

        # Verify other values still exist.
        for i in [0, 1, 2, 3, 4, 6, 7, 8, 9]:
            assert bplus_tree.search(i, root) == i

    def test_delete_nonexistent_key(self):
        """Test deleting a key that doesn't exist raises KeyError."""
        bplus_tree = BPlusTree(order=4, leaf_range=1000)
        root = BPlusTreeNode()

        for i in range(5):
            root = bplus_tree.insert(root, (i, i))

        try:
            bplus_tree.delete(root, 100)
            assert False, "Should raise KeyError"
        except KeyError:
            pass

    def test_delete_with_redistribution_left(self):
        """Test deletion that triggers redistribution from left sibling."""
        bplus_tree = BPlusTree(order=4, leaf_range=1000)
        root = BPlusTreeNode()

        # Insert values to create a specific tree structure.
        for i in range(12):
            root = bplus_tree.insert(root, (i, i))

        # Delete values to trigger redistribution.
        root = bplus_tree.delete(root, 6)
        root = bplus_tree.delete(root, 7)

        # Verify remaining values.
        for i in [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]:
            assert bplus_tree.search(i, root) == i

    def test_delete_with_redistribution_right(self):
        """Test deletion that triggers redistribution from right sibling."""
        bplus_tree = BPlusTree(order=4, leaf_range=1000)
        root = BPlusTreeNode()

        for i in range(12):
            root = bplus_tree.insert(root, (i, i))

        # Delete from left side to trigger redistribution from right.
        root = bplus_tree.delete(root, 0)
        root = bplus_tree.delete(root, 1)

        # Verify remaining values.
        for i in range(2, 12):
            assert bplus_tree.search(i, root) == i

    def test_delete_with_merge(self):
        """Test deletion that triggers node merging."""
        bplus_tree = BPlusTree(order=3, leaf_range=1000)
        root = BPlusTreeNode()

        for i in range(10):
            root = bplus_tree.insert(root, (i, i))

        # Delete multiple values to trigger merging.
        for i in range(5):
            root = bplus_tree.delete(root, i)

        # Verify remaining values.
        for i in range(5, 10):
            assert bplus_tree.search(i, root) == i

    def test_delete_causes_root_shrink(self):
        """Test deletion that causes the root to shrink."""
        bplus_tree = BPlusTree(order=3, leaf_range=1000)
        root = BPlusTreeNode()

        # Build a tree with multiple levels.
        for i in range(10):
            root = bplus_tree.insert(root, (i, i))

        # Delete enough values to cause root shrinking.
        for i in range(8):
            root = bplus_tree.delete(root, i)

        # Verify remaining values.
        assert bplus_tree.search(8, root) == 8
        assert bplus_tree.search(9, root) == 9

    def test_delete_all_sequential(self):
        """Test deleting all values sequentially."""
        bplus_tree = BPlusTree(order=4, leaf_range=1000)
        root = BPlusTreeNode()

        values = list(range(20))
        for v in values:
            root = bplus_tree.insert(root, (v, v))

        for v in values:
            root = bplus_tree.delete(root, v)

        assert root is None

    def test_delete_all_reverse(self):
        """Test deleting all values in reverse order."""
        bplus_tree = BPlusTree(order=4, leaf_range=1000)
        root = BPlusTreeNode()

        values = list(range(20))
        for v in values:
            root = bplus_tree.insert(root, (v, v))

        for v in reversed(values):
            root = bplus_tree.delete(root, v)

        assert root is None

    def test_delete_random_order(self):
        """Test deleting values in random order."""
        bplus_tree = BPlusTree(order=5, leaf_range=1000)
        root = BPlusTreeNode()

        values = list(range(50))
        for v in values:
            root = bplus_tree.insert(root, (v, v))

        random.seed(42)
        delete_order = values.copy()
        random.shuffle(delete_order)

        remaining = set(values)
        for v in delete_order:
            root = bplus_tree.delete(root, v)
            remaining.remove(v)

            # Verify remaining values still exist.
            if root:
                for r in remaining:
                    assert bplus_tree.search(r, root) == r

        assert root is None

    def test_delete_various_orders(self):
        """Test delete with various tree orders."""
        for order in [3, 4, 5, 6, 10]:
            bplus_tree = BPlusTree(order=order, leaf_range=1000)
            root = BPlusTreeNode()

            values = list(range(100))
            for v in values:
                root = bplus_tree.insert(root, (v, v))

            random.seed(order)
            random.shuffle(values)

            for v in values:
                root = bplus_tree.delete(root, v)

            assert root is None, f"Tree with order {order} should be empty"

    def test_delete_first_key_updates_parent(self):
        """Test that deleting the first key in a leaf updates parent keys."""
        bplus_tree = BPlusTree(order=4, leaf_range=1000)
        root = BPlusTreeNode()

        for i in range(15):
            root = bplus_tree.insert(root, (i, i))

        # Delete the first key of a leaf.
        root = bplus_tree.delete(root, 6)

        # Verify tree is still valid.
        for i in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]:
            assert bplus_tree.search(i, root) == i
