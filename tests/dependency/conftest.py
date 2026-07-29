"""Fixtures shared by the dependency tests.

The AVL and B+ trees expose the same insert/search/delete contract, so a small adapter lets one
parametrized behavioral suite (`test_search_tree_common.py`) cover both. Scheme-specific structural
assertions stay in `test_avl_tree.py` / `test_bplus_tree.py`.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, override

import pytest

from oblivlib.dependency import AVLTree, BPlusTree, BPlusTreeNode, KVPair
from oblivlib.dependency.binary_tree import BinaryTree
from oblivlib.dependency.flexible_binary_tree import FlexibleBinaryTree


@pytest.fixture(params=[BinaryTree, FlexibleBinaryTree], ids=["binary", "flexible"])
def binary_tree_cls(request):
    """Return a binary-tree class (BinaryTree or FlexibleBinaryTree) for the shared index-math suite.

    Both classes share an identical constructor signature and identical implementations of the
    level/size index math and the static path helpers exercised in `test_binary_tree_common.py`.
    """
    return request.param


class _SearchTreeAdapter(ABC):
    """Uniform stateful wrapper over a search tree, holding the tree and its current root."""

    # True if search() raises KeyError on a missing key; False if it returns None instead.
    missing_raises: ClassVar[bool]

    @abstractmethod
    def insert(self, key, value) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, key):
        raise NotImplementedError

    @abstractmethod
    def delete(self, key) -> None:
        raise NotImplementedError

    @abstractmethod
    def multi_insert(self, pairs) -> None:
        """Insert (key, value) pairs in one batched descent."""
        raise NotImplementedError

    @abstractmethod
    def multi_search(self, keys) -> dict:
        """Look up keys in one batched descent; return {key: value or None}."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_empty(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def height(self) -> int:
        """Number of node levels on a root-to-leaf path (0 when empty); the single-op descent cost."""
        raise NotImplementedError


class _AVLAdapter(_SearchTreeAdapter):
    missing_raises = False

    def __init__(self, leaf_range: int):
        self._tree = AVLTree(leaf_range=leaf_range)
        self._root = None

    @override
    def insert(self, key, value) -> None:
        self._root = self._tree.insert(root=self._root, kv_pair=KVPair(key=key, value=value))

    @override
    def search(self, key):
        return self._tree.search(key=key, root=self._root)

    @override
    def delete(self, key) -> None:
        self._root = self._tree.delete(root=self._root, key=key)

    @override
    def multi_insert(self, pairs) -> None:
        kv_pairs = [KVPair(key=key, value=value) for key, value in pairs]
        self._root = self._tree.multi_insert(root=self._root, kv_pairs=kv_pairs)

    @override
    def multi_search(self, keys) -> dict:
        return self._tree.multi_search(keys=keys, root=self._root)

    @property
    @override
    def is_empty(self) -> bool:
        return self._root is None

    @property
    @override
    def height(self) -> int:
        return self._root.height if self._root is not None else 0


class _BPlusAdapter(_SearchTreeAdapter):
    missing_raises = True

    def __init__(self, leaf_range: int, order: int = 4):
        self._tree = BPlusTree(order=order, leaf_range=leaf_range)
        self._root: BPlusTreeNode | None = BPlusTreeNode()

    @override
    def insert(self, key, value) -> None:
        if self._root is None:
            self._root = BPlusTreeNode()
        self._root = self._tree.insert(root=self._root, kv_pair=KVPair(key=key, value=value))

    @override
    def search(self, key):
        if self._root is None:
            raise KeyError(key)
        return self._tree.search(key=key, root=self._root)

    @override
    def delete(self, key) -> None:
        self._root = self._tree.delete(root=self._root, key=key)

    @override
    def multi_insert(self, pairs) -> None:
        if self._root is None:
            self._root = BPlusTreeNode()
        kv_pairs = [KVPair(key=key, value=value) for key, value in pairs]
        self._root = self._tree.multi_insert(root=self._root, kv_pairs=kv_pairs)

    @override
    def multi_search(self, keys) -> dict:
        if self._root is None:
            return dict.fromkeys(keys)
        return self._tree.multi_search(keys=keys, root=self._root)

    @property
    @override
    def is_empty(self) -> bool:
        return self._root is None

    @property
    @override
    def height(self) -> int:
        node, levels = self._root, 0
        while node is not None:
            levels += 1
            if node.is_leaf:
                break
            node = node.values[0]
        return levels


@pytest.fixture(params=["avl", "bplus"])
def make_search_tree(request):
    """Return a factory `make_search_tree(leaf_range=...)` producing an AVL or B+ tree adapter."""
    builders = {"avl": _AVLAdapter, "bplus": _BPlusAdapter}
    return builders[request.param]
