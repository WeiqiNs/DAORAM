"""Behavioral tests shared by AVLTree and BPlusTree, parametrized via the `make_search_tree` factory.

These assert the insert/search/delete contract holds regardless of internal structure; the
structure-specific assertions live in `test_avl_tree.py` and `test_bplus_tree.py`.
"""

import random

import pytest


def _assert_absent(tree, key) -> None:
    """Assert a missing key follows the tree's contract: KeyError for B+ tree, None for AVL."""
    if tree.missing_raises:
        with pytest.raises(KeyError):
            tree.search(key)
    else:
        assert tree.search(key) is None


class TestSearchTreeBehavior:
    def test_int_key_round_trip(self, make_search_tree):
        tree = make_search_tree(leaf_range=10000)
        for k in range(500):
            tree.insert(k, k * 10)
        for k in range(500):
            assert tree.search(k) == k * 10

    def test_string_key_round_trip(self, make_search_tree):
        tree = make_search_tree(leaf_range=10000)
        keys = [f"key-{i}" for i in range(200)]
        for k in keys:
            tree.insert(k, f"val-{k}")
        for k in keys:
            assert tree.search(k) == f"val-{k}"

    def test_random_insertion_order_round_trip(self, make_search_tree):
        tree = make_search_tree(leaf_range=200000)
        keys = random.Random(1234).sample(range(100000), 500)
        for k in keys:
            tree.insert(k, k + 1)
        for k in keys:
            assert tree.search(k) == k + 1

    def test_single_element(self, make_search_tree):
        tree = make_search_tree(leaf_range=10)
        tree.insert(42, "only")
        assert tree.search(42) == "only"
        _assert_absent(tree, 7)

    def test_missing_key(self, make_search_tree):
        tree = make_search_tree(leaf_range=1000)
        for k in range(50):
            tree.insert(k, k)
        _assert_absent(tree, 999)
        _assert_absent(tree, -1)

    def test_delete_then_absent_others_present(self, make_search_tree):
        tree = make_search_tree(leaf_range=10000)
        for k in range(20):
            tree.insert(k, k)
        tree.delete(10)
        _assert_absent(tree, 10)
        for k in (i for i in range(20) if i != 10):
            assert tree.search(k) == k

    def test_delete_all_empties_tree(self, make_search_tree):
        tree = make_search_tree(leaf_range=10000)
        keys = list(range(30))
        for k in keys:
            tree.insert(k, k)
        random.Random(7).shuffle(keys)
        for k in keys:
            tree.delete(k)
        assert tree.is_empty

    def test_delete_stress_random(self, make_search_tree):
        tree = make_search_tree(leaf_range=200000)
        keys = list(range(200))
        random.Random(99).shuffle(keys)
        for k in keys:
            tree.insert(k, k)
        to_delete, remaining = keys[:100], set(keys[100:])
        for k in to_delete:
            tree.delete(k)
        for k in remaining:
            assert tree.search(k) == k
        for k in to_delete:
            _assert_absent(tree, k)


class TestSearchTreeBatched:
    """The batched (multi-key) descent that the oblivious port will mirror: one level-synchronized
    descent serves the whole key set rather than one descent per key. These pin the result -- it
    matches the single-key path for every key, present or absent."""

    def test_multi_ops_handle_empty_inputs(self, make_search_tree):
        tree = make_search_tree(leaf_range=1000)
        assert tree.multi_search([1, 2, 3]) == {1: None, 2: None, 3: None}
        tree.multi_insert([])  # no-op: must not raise.
        assert tree.multi_search([]) == {}

        tree.multi_insert([(5, 50), (6, 60)])
        assert tree.multi_search([5, 6, 7]) == {5: 50, 6: 60, 7: None}

    def test_multi_insert_matches_repeated_single(self, make_search_tree):
        pairs = [(k, k * 10) for k in random.Random(3).sample(range(100000), 500)]
        batched = make_search_tree(leaf_range=200000)
        single = make_search_tree(leaf_range=200000)
        batched.multi_insert(pairs)
        for k, v in pairs:
            single.insert(k, v)

        assert batched.height == single.height
        for k, v in pairs:
            assert batched.search(k) == v

    def test_multi_search_matches_single(self, make_search_tree):
        tree = make_search_tree(leaf_range=200000)
        present = random.Random(5).sample(range(100000), 400)
        for k in present:
            tree.insert(k, k + 1)

        query = present + [100001, 100002, 100003]
        results = tree.multi_search(query)
        for k in present:
            assert results[k] == k + 1
        for k in (100001, 100002, 100003):
            assert results[k] is None
