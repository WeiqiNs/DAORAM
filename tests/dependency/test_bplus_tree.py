"""B+ tree-specific structural tests. Generic insert/search/delete behavior lives in
`test_search_tree_common.py`; here we pin the B+ invariants and cross-check the recursive and
non-recursive implementations against each other (the standard the oblivious B+ map mirrors)."""

import random

import pytest

from oblivlib.dependency import BPlusTree, BPlusTreeNode, KVPair


def _invariants(node) -> None:
    """Assert keys stay sorted, internal nodes hold one more child than keys, and all leaves are level."""

    def check(n, depth):
        assert n.keys == sorted(n.keys)
        if n.is_leaf:
            return {depth}
        assert len(n.values) == len(n.keys) + 1
        return set().union(*(check(child, depth + 1) for child in n.values))

    assert len(check(node, 0)) == 1


def _shape(node):
    """Full structural shape, so two trees can be compared exactly: for a leaf its keys and stored
    values, for an internal node its separator keys and the shapes of its children."""
    if node is None:
        return None
    if node.is_leaf:
        return True, list(node.keys), list(node.values)
    return False, list(node.keys), [_shape(child) for child in node.values]


class TestBPlusTree:
    def test_multi_search_matches_search(self):
        tree = BPlusTree(order=4, leaf_range=1000)
        root = BPlusTreeNode()
        present = list(range(300))
        for k in present:
            root = tree.insert(root=root, kv_pair=KVPair(key=k, value=k * 2))

        results = tree.multi_search(keys=present + [-1, 500], root=root)
        for k in present:
            assert results[k] == tree.search(key=k, root=root)
        assert results[-1] is None and results[500] is None

    def test_recursive_insert_matches_iterative(self):
        for order in (3, 4, 5):
            tree = BPlusTree(order=order, leaf_range=1000)
            root_it, root_rec = BPlusTreeNode(), BPlusTreeNode()
            for k in range(200):
                root_it = tree.insert(root=root_it, kv_pair=KVPair(key=k, value=k))
                root_rec = tree.recursive_insert(root=root_rec, kv_pair=KVPair(key=k, value=k))
            _invariants(root_it)
            assert _shape(root_it) == _shape(root_rec)

    def test_insert_into_local_rejects_unfetched_node(self):
        # The guard enforces the closure that lets the oblivious port skip re-fetching in phase 2.
        tree = BPlusTree(order=4, leaf_range=1000)
        root = BPlusTreeNode()
        for k in range(20):
            root = tree.insert(root=root, kv_pair=KVPair(key=k, value=k))
        with pytest.raises(ValueError):
            tree._insert_into_local(root=root, local=set(), kv_pair=KVPair(key=100, value=100))

    def test_multi_insert_matches_sequential(self):
        for order in (3, 4, 5):
            keys = random.Random(order).sample(range(100000), 400)
            tree = BPlusTree(order=order, leaf_range=200000)
            root_single = BPlusTreeNode()
            for k in keys:
                root_single = tree.insert(root=root_single, kv_pair=KVPair(key=k, value=k))
            root_batched = tree.multi_insert(root=BPlusTreeNode(), kv_pairs=[KVPair(key=k, value=k) for k in keys])

            _invariants(root_batched)
            assert _shape(root_batched) == _shape(root_single)

    def test_multi_insert_into_existing_matches_sequential(self):
        # From an empty root the closure check is trivially satisfied (every node is phase-2-created);
        # only a pre-populated tree makes phase 2's splits prove they stay within the fetched closure.
        for order in (3, 4, 5):
            rng = random.Random(order + 10)
            base = rng.sample(range(0, 100000, 2), 800)
            batch = rng.sample(range(1, 100000, 2), 400)
            tree = BPlusTree(order=order, leaf_range=200000)
            root_batched, root_single = BPlusTreeNode(), BPlusTreeNode()
            for k in base:
                root_batched = tree.insert(root=root_batched, kv_pair=KVPair(key=k, value=k))
                root_single = tree.insert(root=root_single, kv_pair=KVPair(key=k, value=k))
            root_batched = tree.multi_insert(root=root_batched, kv_pairs=[KVPair(key=k, value=k) for k in batch])
            for k in batch:
                root_single = tree.insert(root=root_single, kv_pair=KVPair(key=k, value=k))

            _invariants(root_batched)
            assert _shape(root_batched) == _shape(root_single)

    def test_recursive_delete_matches_iterative(self):
        for order in (3, 4, 5):
            tree = BPlusTree(order=order, leaf_range=1000)
            root_it, root_rec = BPlusTreeNode(), BPlusTreeNode()
            keys = list(range(200))
            for k in keys:
                root_it = tree.insert(root=root_it, kv_pair=KVPair(key=k, value=k))
                root_rec = tree.insert(root=root_rec, kv_pair=KVPair(key=k, value=k))

            random.Random(order).shuffle(keys)
            for k in keys:
                root_it = tree.delete(root=root_it, key=k)
                root_rec = tree.recursive_delete(root=root_rec, key=k)
                if root_it is not None:
                    _invariants(root_it)
                assert _shape(root_it) == _shape(root_rec)
            assert root_it is None and root_rec is None

    def test_get_data_list_layout(self):
        tree = BPlusTree(order=4, leaf_range=1000)
        root = BPlusTreeNode()
        for i in range(21):
            root = tree.insert(root=root, kv_pair=KVPair(key=i, value=i))

        data_list = tree.get_data_list(root=root)
        assert data_list[0].value.keys == root.keys
        assert len(data_list[0].value.values) == len(root.values)
