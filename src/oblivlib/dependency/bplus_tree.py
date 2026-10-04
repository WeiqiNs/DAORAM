from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

from oblivlib.dependency.types import Data, FieldTuplePickle, KVPair


@dataclass
class BPlusData(FieldTuplePickle):
    keys: list[Any] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)


class BPlusTreeNode:
    def __init__(self):
        self.id: int | None = None
        self.leaf: int | None = None
        self.keys: list = []
        self.values: list = []
        self.is_leaf: bool = True

    def add_kv_pair(self, kv_pair: KVPair):
        key, value = kv_pair.key, kv_pair.value

        if self.keys:
            for index, each_key in enumerate(self.keys):
                if key < each_key:
                    self.keys = self.keys[:index] + [key] + self.keys[index:]
                    self.values = self.values[:index] + [value] + self.values[index:]
                    return

        self.keys.append(key)
        self.values.append(value)


class BPlusTree:
    def __init__(self, order: int, leaf_range: int):
        self._order = order
        self._mid = order // 2
        self._min_keys = (order - 1) // 2
        self._leaf_range = leaf_range

    def _get_new_leaf(self) -> int:
        return secrets.randbelow(self._leaf_range)

    @staticmethod
    def _child_index(node: BPlusTreeNode, key: Any) -> int:
        for index, each_key in enumerate(node.keys):
            if key == each_key:
                return index + 1
            if key < each_key:
                return index
        return len(node.keys)

    @staticmethod
    def _find_leaf(key: Any, root: BPlusTreeNode) -> BPlusTreeNode:
        cur_node = root
        while not cur_node.is_leaf:
            cur_node = cur_node.values[BPlusTree._child_index(node=cur_node, key=key)]
        return cur_node

    @staticmethod
    def _find_leaf_path(key: Any, root: BPlusTreeNode) -> list[BPlusTreeNode]:
        result = [root]
        cur_node = root
        while not cur_node.is_leaf:
            cur_node = cur_node.values[BPlusTree._child_index(node=cur_node, key=key)]
            result.append(cur_node)
        return result

    def search(self, key: Any, root: BPlusTreeNode) -> Any:
        leaf = self._find_leaf(root=root, key=key)

        for index, each_key in enumerate(leaf.keys):
            if key == each_key:
                return leaf.values[index]

        raise KeyError(f"The key {key} is not found.")

    def multi_search(self, keys: list[Any], root: BPlusTreeNode) -> dict[Any, Any]:
        if not keys:
            return {}

        cursors = [(key, root) for key in keys]
        while not cursors[0][1].is_leaf:
            cursors = [(key, node.values[self._child_index(node=node, key=key)]) for key, node in cursors]

        results: dict[Any, Any] = {}
        for key, leaf in cursors:
            index = next((i for i, each_key in enumerate(leaf.keys) if each_key == key), None)
            results[key] = leaf.values[index] if index is not None else None
        return results

    def _split_node(self, node: BPlusTreeNode) -> BPlusTreeNode:
        right_node = BPlusTreeNode()

        if node.is_leaf:
            right_node.keys = node.keys[self._mid :]
            right_node.values = node.values[self._mid :]
            node.keys = node.keys[: self._mid]
            node.values = node.values[: self._mid]

        else:
            right_node.is_leaf = False
            right_node.keys = node.keys[self._mid + 1 :]
            right_node.values = node.values[self._mid + 1 :]
            node.keys = node.keys[: self._mid]
            node.values = node.values[: self._mid + 1]

        return right_node

    def _insert_in_parent(self, child_node: BPlusTreeNode, parent_node: BPlusTreeNode) -> None:
        insert_key = child_node.keys[self._mid]
        right_node = self._split_node(node=child_node)

        for index, each_key in enumerate(parent_node.keys):
            if insert_key < each_key:
                parent_node.keys = parent_node.keys[:index] + [insert_key] + parent_node.keys[index:]
                parent_node.values = parent_node.values[: index + 1] + [right_node] + parent_node.values[index + 1 :]
                return
            elif index + 1 == len(parent_node.keys):
                parent_node.keys.append(insert_key)
                parent_node.values.append(right_node)
                return

    def _create_parent(self, child_node: BPlusTreeNode) -> BPlusTreeNode:
        insert_key = child_node.keys[self._mid]
        right_node = self._split_node(node=child_node)

        parent_node = BPlusTreeNode()
        parent_node.is_leaf = False
        parent_node.keys.append(insert_key)
        parent_node.values = [child_node, right_node]

        return parent_node

    def insert(self, root: BPlusTreeNode, kv_pair: KVPair) -> BPlusTreeNode:
        leaves = self._find_leaf_path(root=root, key=kv_pair.key)
        leaves[-1].add_kv_pair(kv_pair=kv_pair)

        index = len(leaves) - 1
        while index >= 0:
            if len(leaves[index].keys) >= self._order:
                if index > 0:
                    self._insert_in_parent(child_node=leaves[index], parent_node=leaves[index - 1])
                    index -= 1
                else:
                    return self._create_parent(child_node=leaves[index])
            else:
                break

        return root

    def recursive_insert(self, root: BPlusTreeNode, kv_pair: KVPair) -> BPlusTreeNode:
        self._recursive_insert(node=root, kv_pair=kv_pair)
        if len(root.keys) >= self._order:
            return self._create_parent(child_node=root)
        return root

    def _recursive_insert(self, node: BPlusTreeNode, kv_pair: KVPair) -> None:
        if node.is_leaf:
            node.add_kv_pair(kv_pair=kv_pair)
            return

        child = node.values[self._child_index(node=node, key=kv_pair.key)]
        self._recursive_insert(node=child, kv_pair=kv_pair)
        if len(child.keys) >= self._order:
            self._insert_in_parent(child_node=child, parent_node=node)

    @staticmethod
    def _collect_insert_paths(root: BPlusTreeNode | None, keys: list[Any]) -> set[BPlusTreeNode]:
        local: set[BPlusTreeNode] = set()
        if root is None or not keys:
            return local

        cursors = [(key, root) for key in keys]
        while True:
            for _, node in cursors:
                local.add(node)
            if cursors[0][1].is_leaf:
                return local
            cursors = [(key, node.values[BPlusTree._child_index(node=node, key=key)]) for key, node in cursors]

    def _split_and_promote_local(self, child: BPlusTreeNode, parent: BPlusTreeNode, local: set[BPlusTreeNode]) -> None:
        insert_key = child.keys[self._mid]
        right = self._split_node(node=child)
        local.add(right)

        position = self._child_index(node=parent, key=insert_key)
        parent.keys.insert(position, insert_key)
        parent.values.insert(position + 1, right)

    def _grow_root_local(self, child: BPlusTreeNode, local: set[BPlusTreeNode]) -> BPlusTreeNode:
        insert_key = child.keys[self._mid]
        right = self._split_node(node=child)
        local.add(right)

        parent = BPlusTreeNode()
        parent.is_leaf = False
        parent.keys = [insert_key]
        parent.values = [child, right]
        local.add(parent)
        return parent

    def _insert_into_local(self, root: BPlusTreeNode, local: set[BPlusTreeNode], kv_pair: KVPair) -> BPlusTreeNode:
        leaves = self._find_leaf_path(root=root, key=kv_pair.key)
        if any(node not in local for node in leaves):
            raise ValueError("multi_insert stepped onto an unfetched node; the batched paths were incomplete.")
        leaves[-1].add_kv_pair(kv_pair=kv_pair)

        index = len(leaves) - 1
        while index >= 0:
            if len(leaves[index].keys) >= self._order:
                if index > 0:
                    self._split_and_promote_local(child=leaves[index], parent=leaves[index - 1], local=local)
                    index -= 1
                else:
                    return self._grow_root_local(child=leaves[index], local=local)
            else:
                break
        return root

    def multi_insert(self, root: BPlusTreeNode, kv_pairs: list[KVPair]) -> BPlusTreeNode:
        local = self._collect_insert_paths(root=root, keys=[kv_pair.key for kv_pair in kv_pairs])
        for kv_pair in kv_pairs:
            root = self._insert_into_local(root=root, local=local, kv_pair=kv_pair)
        return root

    def _fix_underflow(self, parent: BPlusTreeNode, child_index: int) -> None:
        is_left = child_index > 0
        sibling_index = child_index - 1 if is_left else child_index + 1
        node = parent.values[child_index]
        sibling = parent.values[sibling_index]

        if len(sibling.keys) > self._min_keys:
            if is_left:
                if node.is_leaf:
                    node.keys.insert(0, sibling.keys.pop())
                    node.values.insert(0, sibling.values.pop())
                    parent.keys[child_index - 1] = node.keys[0]
                else:
                    node.keys.insert(0, parent.keys[child_index - 1])
                    node.values.insert(0, sibling.values.pop())
                    parent.keys[child_index - 1] = sibling.keys.pop()
            else:
                if node.is_leaf:
                    node.keys.append(sibling.keys.pop(0))
                    node.values.append(sibling.values.pop(0))
                    parent.keys[child_index] = sibling.keys[0]
                else:
                    node.keys.append(parent.keys[child_index])
                    node.values.append(sibling.values.pop(0))
                    parent.keys[child_index] = sibling.keys.pop(0)
            return

        if is_left:
            if not node.is_leaf:
                sibling.keys.append(parent.keys[child_index - 1])
            sibling.keys.extend(node.keys)
            sibling.values.extend(node.values)
            parent.keys.pop(child_index - 1)
            parent.values.pop(child_index)
        else:
            if not node.is_leaf:
                node.keys.append(parent.keys[child_index])
            node.keys.extend(sibling.keys)
            node.values.extend(sibling.values)
            parent.keys.pop(child_index)
            parent.values.pop(child_index + 1)

    def delete(self, root: BPlusTreeNode | None, key: Any) -> BPlusTreeNode | None:
        if root is None:
            return None

        local: list[tuple[BPlusTreeNode, int]] = []
        current = root
        while not current.is_leaf:
            child_index = self._child_index(node=current, key=key)
            local.append((current, child_index))
            current = current.values[child_index]
        leaf = current

        key_index = next((i for i, k in enumerate(leaf.keys) if k == key), None)
        if key_index is None:
            return root
        leaf.keys.pop(key_index)
        leaf.values.pop(key_index)

        if not local:
            return None if not leaf.keys else root

        node = leaf
        for parent, child_index in reversed(local):
            if len(node.keys) >= self._min_keys:
                break
            self._fix_underflow(parent=parent, child_index=child_index)
            node = parent

        if not root.keys:
            return None if root.is_leaf else root.values[0]
        return root

    def recursive_delete(self, root: BPlusTreeNode | None, key: Any) -> BPlusTreeNode | None:
        if root is None:
            return None
        self._recursive_delete(node=root, key=key)
        if not root.keys:
            return None if root.is_leaf else root.values[0]
        return root

    def _recursive_delete(self, node: BPlusTreeNode, key: Any) -> None:
        if node.is_leaf:
            key_index = next((i for i, k in enumerate(node.keys) if k == key), None)
            if key_index is not None:
                node.keys.pop(key_index)
                node.values.pop(key_index)
            return

        child_index = self._child_index(node=node, key=key)
        child = node.values[child_index]
        self._recursive_delete(node=child, key=key)
        if len(child.keys) < self._min_keys:
            self._fix_underflow(parent=node, child_index=child_index)

    def get_data_list(self, root: BPlusTreeNode, block_id: int = 0) -> list[Data]:
        root.id = block_id
        root.leaf = self._get_new_leaf()
        block_id += 1

        stack = [root]
        result = []

        while stack:
            node = stack.pop()

            if not node.is_leaf:
                for child in node.values:
                    child.id = block_id
                    child.leaf = self._get_new_leaf()
                    block_id += 1

                bplus_data = BPlusData(keys=node.keys, values=[(child.id, child.leaf) for child in node.values])
                stack.extend(node.values)

            else:
                bplus_data = BPlusData(keys=node.keys, values=node.values)

            result.append(Data(key=node.id, leaf=node.leaf, value=bplus_data))

        return result
