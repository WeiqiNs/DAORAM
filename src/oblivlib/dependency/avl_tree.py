from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any

from oblivlib.dependency.types import Data, FieldTuplePickle, KVPair


@dataclass
class AVLData(FieldTuplePickle):
    value: Any = None
    r_key: Any = None
    r_leaf: int | None = None
    r_height: int = 0
    l_key: Any = None
    l_leaf: int | None = None
    l_height: int = 0


class AVLTreeNode:
    def __init__(self, kv_pair: KVPair):
        self.key: Any = kv_pair.key
        self.leaf: int | None = None
        self.value: Any = kv_pair.value
        self.height: int = 1
        self.left_node: AVLTreeNode | None = None
        self.right_node: AVLTreeNode | None = None


class AVLTree:
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
        assert in_node.right_node is not None
        p_node = in_node.right_node
        tmp_node = p_node.left_node

        p_node.left_node = in_node
        in_node.right_node = tmp_node

        self._update_height(in_node)
        self._update_height(p_node)

        return p_node

    def _rotate_right(self, in_node: AVLTreeNode) -> AVLTreeNode:
        assert in_node.left_node is not None
        p_node = in_node.left_node
        tmp_node = p_node.right_node

        p_node.right_node = in_node
        in_node.left_node = tmp_node

        self._update_height(in_node)
        self._update_height(p_node)

        return p_node

    def _balance(self, node: AVLTreeNode) -> AVLTreeNode:
        self._update_height(node)
        balance = self._get_balance(node)

        if balance > 1:
            assert node.left_node is not None
            if self._get_balance(node.left_node) < 0:
                node.left_node = self._rotate_left(node.left_node)
            return self._rotate_right(node)

        if balance < -1:
            assert node.right_node is not None
            if self._get_balance(node.right_node) > 0:
                node.right_node = self._rotate_right(node.right_node)
            return self._rotate_left(node)

        return node

    @staticmethod
    def search(key: Any, root: AVLTreeNode | None) -> Any:
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

        raise ValueError("The node was not successfully inserted.")

    def recursive_insert(self, root: AVLTreeNode | None, kv_pair: KVPair) -> AVLTreeNode:
        if root is None:
            return AVLTreeNode(kv_pair)
        elif kv_pair.key < root.key:
            root.left_node = self.recursive_insert(root=root.left_node, kv_pair=kv_pair)
        else:
            root.right_node = self.recursive_insert(root=root.right_node, kv_pair=kv_pair)

        return self._balance(root)

    @staticmethod
    def _collect_insert_paths(root: AVLTreeNode | None, keys: list[Any]) -> set[AVLTreeNode]:
        local: set[AVLTreeNode] = set()
        if root is None:
            return local

        cursors = [(key, root) for key in keys]
        while cursors:
            next_cursors = []
            for key, node in cursors:
                local.add(node)
                child = node.left_node if key < node.key else node.right_node
                if child is not None:
                    next_cursors.append((key, child))
            cursors = next_cursors
        return local

    def _insert_into_local(self, root: AVLTreeNode | None, local: set[AVLTreeNode], kv_pair: KVPair) -> AVLTreeNode:
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
        local = self._collect_insert_paths(root=root, keys=[kv_pair.key for kv_pair in kv_pairs])
        for kv_pair in kv_pairs:
            root = self._insert_into_local(root=root, local=local, kv_pair=kv_pair)
        return root

    def delete(self, root: AVLTreeNode | None, key: Any) -> AVLTreeNode | None:
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

    def get_data_list(self, root: AVLTreeNode) -> list[Data]:
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

            result.append(Data(key=node.key, leaf=node.leaf, value=avl_data))

        return result
