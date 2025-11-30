import random

from daoram.dependency import AVLTree, KVPair


class TestAVLTree:
    def test_insert(self):
        # Create an avl tree object.
        avl_tree = AVLTree(leaf_range=1000)

        # Set the beginning root to None and create a list of kv pairs.
        root = None
        kv_pairs = [KVPair(key=i, value=i) for i in range(1000)]

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
        kv_pairs = [KVPair(key=i, value=i) for i in range(1000)]

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
        kv_pairs = [KVPair(key=f"{i}", value=f"{i}") for i in range(1000)]

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
        kv_pairs = [KVPair(key=f"{i}", value=f"{i}") for i in range(1000)]

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
            root = avl_tree.insert(root=root, kv_pair=KVPair(key=i, value=i))

        # Perform search.
        for i in random_values:
            assert i == avl_tree.search(key=i, root=root)

    def test_get_list(self):
        # Create an avl tree object.
        avl_tree = AVLTree(leaf_range=10)

        # Set the beginning root to None and create a list of kv pairs.
        root = None
        kv_pairs = [KVPair(key=i, value=i) for i in range(10)]

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

