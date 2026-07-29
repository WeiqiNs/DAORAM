"""Defines the AVL tree; note that inserting repeated keys may cause unexpected behavior."""

from __future__ import annotations

import pickle
import secrets
from dataclasses import astuple, dataclass
from typing import Any, Self

from oblivlib.dependency.helper import Data
from oblivlib.dependency.types import KVPair


@dataclass
class AVLData:
    """The payload stored in an ORAM block for one AVL node: its value plus each child's (key, leaf,
    height), so the tree can be traversed obliviously. All fields default to None for a dummy block."""

    value: Any = None
    r_key: Any = None
    r_leaf: int | None = None
    r_height: int = 0
    l_key: Any = None
    l_leaf: int | None = None
    l_height: int = 0

    @classmethod
    def from_pickle(cls, data: bytes) -> Self:
        return cls(*pickle.loads(data))

    def dump(self) -> bytes:
        return pickle.dumps(astuple(self))


class AVLTreeNode:
    def __init__(self, kv_pair: KVPair):
        self.key: Any = kv_pair.key
        # leaf is the random ORAM path this node is stored on (assigned later).
        self.leaf: int | None = None
        self.value: Any = kv_pair.value
        self.height: int = 1
        self.left_node: AVLTreeNode | None = None
        self.right_node: AVLTreeNode | None = None


class AVLTree:
    """AVL tree used to initialize the OMAP storage from a list of key-value pairs."""

    def __init__(self, leaf_range: int):
        self._leaf_range = leaf_range

    def _get_new_leaf(self) -> int:
        return secrets.randbelow(self._leaf_range)

    @staticmethod
    def _get_height(node: AVLTreeNode | None) -> int:
        return node.height if node is not None else 0

    @staticmethod
    def _get_balance(node: AVLTreeNode | None) -> int:
        return AVLTree._get_height(node.left_node) - AVLTree._get_height(node.right_node) if node is not None else 0

    def _update_height(self, node: AVLTreeNode) -> None:
        node.height = 1 + max(self._get_height(node.left_node), self._get_height(node.right_node))

    def _rotate_left(self, in_node: AVLTreeNode) -> AVLTreeNode:
        """Left-rotate at in_node and return the new subtree root (its former right child)."""
        assert in_node.right_node is not None
        p_node = in_node.right_node
        tmp_node = p_node.left_node

        p_node.left_node = in_node
        in_node.right_node = tmp_node

        self._update_height(in_node)
        self._update_height(p_node)

        return p_node

    def _rotate_right(self, in_node: AVLTreeNode) -> AVLTreeNode:
        """Right-rotate at in_node and return the new subtree root (its former left child)."""
        assert in_node.left_node is not None
        p_node = in_node.left_node
        tmp_node = p_node.right_node

        p_node.right_node = in_node
        in_node.left_node = tmp_node

        self._update_height(in_node)
        self._update_height(p_node)

        return p_node

    def _balance(self, node: AVLTreeNode) -> AVLTreeNode:
        """Re-balance a node if it is unbalanced, returning the new subtree root."""
        self._update_height(node)
        balance = self._get_balance(node)

        if balance > 1:
            assert node.left_node is not None
            if self._get_balance(node.left_node) < 0:  # left-right case
                node.left_node = self._rotate_left(node.left_node)
            return self._rotate_right(node)

        if balance < -1:
            assert node.right_node is not None
            if self._get_balance(node.right_node) > 0:  # right-left case
                node.right_node = self._rotate_right(node.right_node)
            return self._rotate_left(node)

        return node

    @staticmethod
    def search(key: Any, root: AVLTreeNode | None) -> Any:
        """Return the value stored under key, or None if it is not present."""
        while root:
            if key < root.key:
                root = root.left_node
            elif key > root.key:
                root = root.right_node
            else:
                return root.value

        return None

    @staticmethod
    def multi_search(keys: list[Any], root: AVLTreeNode | None) -> dict[Any, Any]:
        """Look up many keys in a single level-synchronized descent; returns {key: value or None}.

        All cursors start at the root and advance one level per round; within a round we hold exactly
        the nodes a batched ODS read would fetch for that layer (one per still-active key). A cursor
        stops when it matches its key or falls off the tree -- one descent serving the whole key set,
        not one per key. Absent keys map to None.
        """
        results: dict[Any, Any] = dict.fromkeys(keys)
        if root is None:
            return results

        cursors = [(key, root) for key in keys]
        while cursors:
            next_cursors = []
            for key, node in cursors:
                if key == node.key:
                    results[key] = node.value
                elif key < node.key:
                    if node.left_node is not None:
                        next_cursors.append((key, node.left_node))
                elif node.right_node is not None:
                    next_cursors.append((key, node.right_node))
            cursors = next_cursors
        return results

    def insert(self, root: AVLTreeNode | None, kv_pair: KVPair) -> AVLTreeNode:
        """Insert kv_pair into the tree rooted at root and return the updated root."""
        if not root:
            return AVLTreeNode(kv_pair)

        stack = []
        node = root

        while node:
            stack.append(node)
            if kv_pair.key < node.key:
                if not node.left_node:
                    node.left_node = AVLTreeNode(kv_pair)
                    stack.append(node.left_node)
                    break
                node = node.left_node
            else:
                if not node.right_node:
                    node.right_node = AVLTreeNode(kv_pair)
                    stack.append(node.right_node)
                    break
                node = node.right_node

        while stack:
            node = stack.pop()
            balanced_node = self._balance(node)

            if stack:
                parent = stack[-1]
                if parent.left_node == node:
                    parent.left_node = balanced_node
                else:
                    parent.right_node = balanced_node
            else:
                return balanced_node

        # The loop above always returns once the root is reached.
        raise ValueError("The node was not successfully inserted.")

    def recursive_insert(self, root: AVLTreeNode | None, kv_pair: KVPair) -> AVLTreeNode:
        """Recursive variant of insert; kept to validate the non-recursive implementation."""
        if root is None:
            return AVLTreeNode(kv_pair)
        elif kv_pair.key < root.key:
            root.left_node = self.recursive_insert(root=root.left_node, kv_pair=kv_pair)
        else:
            root.right_node = self.recursive_insert(root=root.right_node, kv_pair=kv_pair)

        root.height = 1 + max(self._get_height(node=root.left_node), self._get_height(node=root.right_node))
        balance = self._get_balance(node=root)

        if balance > 1:
            assert root.left_node is not None
            if self._get_balance(node=root.left_node) < 0:  # left-right case
                root.left_node = self._rotate_left(root.left_node)
            return self._rotate_right(in_node=root)

        if balance < -1:
            assert root.right_node is not None
            if self._get_balance(root.right_node) > 0:  # right-left case
                root.right_node = self._rotate_right(root.right_node)
            return self._rotate_left(in_node=root)

        return root

    @staticmethod
    def _collect_insert_paths(root: AVLTreeNode | None, keys: list[Any]) -> set[AVLTreeNode]:
        """Batched insertion descent: fetch every key's root-to-insertion path into one local partial tree.

        All cursors start at the root and advance one level per round (left on a smaller key, right
        otherwise), each round fetching the whole layer at once -- a single descent serving the whole
        key set, not one per key. Returns ``local``, the set of fetched nodes (the partial tree phase 2
        mutates).
        """
        local: set[AVLTreeNode] = set()
        if root is None:
            return local

        cursors = [(key, root) for key in keys]
        while cursors:
            next_cursors = []
            for key, node in cursors:
                local.add(node)
                # Mirror insert's routing: smaller keys go left, equal-or-larger go right.
                child = node.left_node if key < node.key else node.right_node
                if child is not None:
                    next_cursors.append((key, child))
            cursors = next_cursors
        return local

    def _insert_into_local(self, root: AVLTreeNode | None, local: set[AVLTreeNode], kv_pair: KVPair) -> AVLTreeNode:
        """Insert one pair operating only on the local partial tree ``local`` (multi_insert's engine).

        Same shape as the single-key ``insert`` -- descend, link, rebalance bottom-up via ``_balance``
        -- but every node stepped onto must already be in ``local`` (else the batched paths were
        incomplete and we raise), and each newly created node joins ``local``. Rebalancing needs no
        guard: rotations only re-point nodes already on the descended path.
        """
        if root is None:
            new_node = AVLTreeNode(kv_pair)
            local.add(new_node)
            return new_node

        stack = []
        node = root
        while node:
            if node not in local:
                raise ValueError("multi_insert stepped onto an unfetched node; the batched paths were incomplete.")
            stack.append(node)
            if kv_pair.key < node.key:
                if not node.left_node:
                    node.left_node = AVLTreeNode(kv_pair)
                    local.add(node.left_node)
                    stack.append(node.left_node)
                    break
                node = node.left_node
            else:
                if not node.right_node:
                    node.right_node = AVLTreeNode(kv_pair)
                    local.add(node.right_node)
                    stack.append(node.right_node)
                    break
                node = node.right_node

        while stack:
            node = stack.pop()
            balanced_node = self._balance(node)
            if stack:
                parent = stack[-1]
                if parent.left_node == node:
                    parent.left_node = balanced_node
                else:
                    parent.right_node = balanced_node
            else:
                return balanced_node

        raise ValueError("The node was not successfully inserted.")

    def multi_insert(self, root: AVLTreeNode | None, kv_pairs: list[KVPair]) -> AVLTreeNode | None:
        """Insert many pairs against a single fetched partial tree; returns the new root.

        Phase 1 (``_collect_insert_paths``) is the only storage access: one batched descent over all
        keys (not one per key) gathering the union of their insertion paths into ``local``. Phase 2
        replays single-key inserts against ``local`` only, raising if it ever needs an unfetched node.
        Inserts rebalance per key (AVL can't be batch-rebalanced in one pass), so the result equals
        sequential single inserts -- the shape the oblivious port mirrors.
        """
        local = self._collect_insert_paths(root=root, keys=[kv_pair.key for kv_pair in kv_pairs])
        for kv_pair in kv_pairs:
            root = self._insert_into_local(root=root, local=local, kv_pair=kv_pair)
        return root

    def delete(self, root: AVLTreeNode | None, key: Any) -> AVLTreeNode | None:
        """Delete the node with the given key (non-recursive); returns the new root, or None if empty.

        This serves as a template for the oblivious version: it tracks the visited path in a
        ``local`` list with clear phases, and for the two-children case uses the in-order successor
        (go right, then keep going left) or predecessor depending on subtree heights.
        """
        if not root:
            return None

        local = []
        current = root

        while current is not None:
            local.append(current)
            if current.key == key:
                break
            current = current.left_node if key < current.key else current.right_node

        if current is None:
            return root

        node = local[-1]
        node_index = len(local) - 1

        if node.left_node is None and node.right_node is None:
            if len(local) == 1:
                return None
            parent = local[node_index - 1]
            if parent.left_node == node:
                parent.left_node = None
            else:
                parent.right_node = None
            local.pop()

        elif node.left_node is None or node.right_node is None:
            child = node.left_node if node.left_node is not None else node.right_node
            if len(local) == 1:
                return child
            parent = local[node_index - 1]
            if parent.left_node == node:
                parent.left_node = child
            else:
                parent.right_node = child
            local.pop()

        else:
            # Use predecessor (left then all right) if left is taller, else successor (right then all left).
            use_predecessor = self._get_height(node.left_node) > self._get_height(node.right_node)

            current = node.left_node if use_predecessor else node.right_node
            local.append(current)

            next_node = current.right_node if use_predecessor else current.left_node
            while next_node is not None:
                current = next_node
                local.append(current)
                next_node = current.right_node if use_predecessor else current.left_node

            replacement_node = local[-1]
            replacement_index = len(local) - 1

            node.key = replacement_node.key
            node.value = replacement_node.value

            # The replacement has at most one child, on the side opposite the traversal direction.
            child = replacement_node.left_node if use_predecessor else replacement_node.right_node

            parent = local[replacement_index - 1]
            if parent.left_node == replacement_node:
                parent.left_node = child
            else:
                parent.right_node = child

            local.pop()

        for i in range(len(local) - 1, -1, -1):
            curr = local[i]
            balanced = self._balance(curr)

            if i > 0:
                parent = local[i - 1]
                if parent.left_node == curr:
                    parent.left_node = balanced
                else:
                    parent.right_node = balanced
            else:
                root = balanced

        return root

    def recursive_delete(self, root: AVLTreeNode | None, key: Any) -> AVLTreeNode | None:
        """Recursive variant of delete; kept to validate the non-recursive implementation.

        It makes the same two-children choice as ``delete`` (replace with the in-order predecessor when
        the left subtree is taller, otherwise the successor), so both produce structurally identical trees.
        """
        if root is None:
            return None

        if key < root.key:
            root.left_node = self.recursive_delete(root=root.left_node, key=key)
        elif key > root.key:
            root.right_node = self.recursive_delete(root=root.right_node, key=key)

        else:
            if root.left_node is None:
                return root.right_node
            if root.right_node is None:
                return root.left_node

            if self._get_height(root.left_node) > self._get_height(root.right_node):
                predecessor = root.left_node
                while predecessor.right_node is not None:
                    predecessor = predecessor.right_node
                root.key, root.value = predecessor.key, predecessor.value
                root.left_node = self.recursive_delete(root=root.left_node, key=predecessor.key)
            else:
                successor = root.right_node
                while successor.left_node is not None:
                    successor = successor.left_node
                root.key, root.value = successor.key, successor.value
                root.right_node = self.recursive_delete(root=root.right_node, key=successor.key)

        return self._balance(root)

    def get_data_list(self, root: AVLTreeNode, encryption: bool = False) -> list[Data]:
        """Expand the tree rooted at root into a list of Data blocks, sampling a leaf per node.

        With encryption enabled, each block's value is the pickled AVLData rather than the object.
        """
        root.leaf = self._get_new_leaf()
        stack = [root]

        result = []

        while stack:
            node = stack.pop()
            avl_data = AVLData(value=node.value)

            if node.left_node:
                node.left_node.leaf = self._get_new_leaf()
                avl_data.l_key = node.left_node.key
                avl_data.l_leaf = node.left_node.leaf
                avl_data.l_height = node.left_node.height
                stack.append(node.left_node)

            if node.right_node:
                node.right_node.leaf = self._get_new_leaf()
                avl_data.r_key = node.right_node.key
                avl_data.r_leaf = node.right_node.leaf
                avl_data.r_height = node.right_node.height
                stack.append(node.right_node)

            if encryption:
                result.append(Data(key=node.key, leaf=node.leaf, value=avl_data.dump()))
            else:
                result.append(Data(key=node.key, leaf=node.leaf, value=avl_data))

        return result
