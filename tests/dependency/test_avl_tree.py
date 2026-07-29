"""AVL-specific structural tests. Generic insert/search/delete behavior lives in
`test_search_tree_common.py`; here we pin the AVL invariants and cross-check the recursive and
non-recursive implementations against each other (the standard the oblivious AVL map mirrors)."""

import random

import pytest

from oblivlib.dependency import AVLTree, KVPair


def _invariants(node, low=None, high=None) -> int:
    """Assert BST order, AVL balance (|height diff| <= 1) and height bookkeeping; return the height."""
    if node is None:
        return 0
    assert low is None or node.key > low
    assert high is None or node.key < high
    left = _invariants(node.left_node, low, node.key)
    right = _invariants(node.right_node, node.key, high)
    assert abs(left - right) <= 1
    assert node.height == 1 + max(left, right)
    return node.height


def _shape(node):
    """Full (key, value, height, left, right) shape, so two trees can be compared exactly."""
    if node is None:
        return None
    return node.key, node.value, node.height, _shape(node.left_node), _shape(node.right_node)


class TestAVLTree:
    def test_multi_search_matches_search(self):
        keys = list(range(300))
        tree = AVLTree(leaf_range=1000)
        root = None
        for k in keys:
            root = tree.insert(root=root, kv_pair=KVPair(key=k, value=k * 2))

        query = keys + [-1, 500]
        results = tree.multi_search(keys=query, root=root)
        for k in query:
            assert results[k] == AVLTree.search(key=k, root=root)

    def test_recursive_insert_matches_iterative(self):
        for keys in (list(range(500)), [f"{i}" for i in range(500)]):
            tree = AVLTree(leaf_range=1000)
            root_it = root_rec = None
            for k in keys:
                root_it = tree.insert(root=root_it, kv_pair=KVPair(key=k, value=k))
                root_rec = tree.recursive_insert(root=root_rec, kv_pair=KVPair(key=k, value=k))
            _invariants(root_it)
            assert _shape(root_it) == _shape(root_rec)

    def test_insert_into_local_rejects_unfetched_node(self):
        # The guard enforces the closure that lets the oblivious port skip re-fetching in phase 2.
        tree = AVLTree(leaf_range=100)
        root = None
        for k in range(10):
            root = tree.insert(root=root, kv_pair=KVPair(key=k, value=k))
        with pytest.raises(ValueError):
            tree._insert_into_local(root=root, local=set(), kv_pair=KVPair(key=100, value=100))

    def test_multi_insert_matches_sequential(self):
        keys = random.Random(2).sample(range(100000), 500)
        tree = AVLTree(leaf_range=200000)
        root_batched = root_single = None
        for k in keys:
            root_single = tree.insert(root=root_single, kv_pair=KVPair(key=k, value=k))
        root_batched = tree.multi_insert(root=root_batched, kv_pairs=[KVPair(key=k, value=k) for k in keys])

        _invariants(root_batched)
        assert _shape(root_batched) == _shape(root_single)

    def test_multi_insert_into_existing_matches_sequential(self):
        # From an empty root the closure check is trivially satisfied (every node is phase-2-created);
        # only a pre-populated tree makes phase 2's rotations prove they stay within the fetched paths.
        rng = random.Random(11)
        base = rng.sample(range(0, 100000, 2), 1000)
        batch = rng.sample(range(1, 100000, 2), 500)
        tree = AVLTree(leaf_range=200000)
        root_batched = root_single = None
        for k in base:
            root_batched = tree.insert(root=root_batched, kv_pair=KVPair(key=k, value=k))
            root_single = tree.insert(root=root_single, kv_pair=KVPair(key=k, value=k))
        root_batched = tree.multi_insert(root=root_batched, kv_pairs=[KVPair(key=k, value=k) for k in batch])
        for k in batch:
            root_single = tree.insert(root=root_single, kv_pair=KVPair(key=k, value=k))

        _invariants(root_batched)
        assert _shape(root_batched) == _shape(root_single)

    def test_recursive_delete_matches_iterative(self):
        tree = AVLTree(leaf_range=1000)
        root_it = root_rec = None
        keys = list(range(300))
        for k in keys:
            root_it = tree.insert(root=root_it, kv_pair=KVPair(key=k, value=k))
            root_rec = tree.insert(root=root_rec, kv_pair=KVPair(key=k, value=k))

        random.Random(0).shuffle(keys)
        for k in keys:
            root_it = tree.delete(root=root_it, key=k)
            root_rec = tree.recursive_delete(root=root_rec, key=k)
            _invariants(root_it)
            assert _shape(root_it) == _shape(root_rec)
        assert root_it is None and root_rec is None

    def test_get_data_list_layout(self):
        tree = AVLTree(leaf_range=10)
        root = None
        for i in range(10):
            root = tree.insert(root=root, kv_pair=KVPair(key=i, value=i))
        assert root is not None

        data_list = tree.get_data_list(root=root)
        assert len(data_list) == 10
        assert root.left_node is not None and root.right_node is not None
        assert data_list[0].key == root.key
        assert data_list[0].value.l_key == root.left_node.key
        assert data_list[0].value.r_key == root.right_node.key
