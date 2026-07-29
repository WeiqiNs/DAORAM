"""Defines the B+ tree; note that the minimum order is 3 and inserting repeated keys may cause unexpected behavior."""

from __future__ import annotations

import pickle
import secrets
from dataclasses import astuple, dataclass, field
from typing import Any, Self

from oblivlib.dependency.helper import Data
from oblivlib.dependency.types import KVPair


@dataclass
class BPlusData:
    """The payload stored in an ORAM block for one B+ tree node: parallel lists of keys and values
    (values are child (id, leaf) pairs for internal nodes, or the actual values for leaves)."""

    keys: list[Any] = field(default_factory=list)
    values: list[Any] = field(default_factory=list)

    @classmethod
    def from_pickle(cls, data: bytes) -> Self:
        return cls(*pickle.loads(data))

    def dump(self) -> bytes:
        return pickle.dumps(astuple(self))


class BPlusTreeNode:
    def __init__(self):
        # values holds the actual values in a leaf, the child nodes in an internal node.
        self.id: int | None = None
        self.leaf: int | None = None
        self.keys: list = []
        self.values: list = []
        self.is_leaf: bool = True

    def add_kv_pair(self, kv_pair: KVPair):
        """Insert a key-value pair into this leaf node, keeping keys sorted ascending."""
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
        """Index of the child to descend into for ``key``: equal-or-larger keys go right, smaller left."""
        for index, each_key in enumerate(node.keys):
            if key == each_key:
                return index + 1
            if key < each_key:
                return index
        # Larger than every separator: under the rightmost child.
        return len(node.keys)

    @staticmethod
    def _find_leaf(key: Any, root: BPlusTreeNode) -> BPlusTreeNode:
        cur_node = root
        while not cur_node.is_leaf:
            cur_node = cur_node.values[BPlusTree._child_index(node=cur_node, key=key)]
        return cur_node

    @staticmethod
    def _find_leaf_path(key: Any, root: BPlusTreeNode) -> list[BPlusTreeNode]:
        """Like ``_find_leaf`` but return the full root-to-leaf path of nodes."""
        result = [root]
        cur_node = root
        while not cur_node.is_leaf:
            cur_node = cur_node.values[BPlusTree._child_index(node=cur_node, key=key)]
            result.append(cur_node)
        return result

    def search(self, key: Any, root: BPlusTreeNode) -> Any:
        """Return the value stored under the key, or raise KeyError if absent."""
        leaf = self._find_leaf(root=root, key=key)

        for index, each_key in enumerate(leaf.keys):
            if key == each_key:
                return leaf.values[index]

        raise KeyError(f"The key {key} is not found.")

    def multi_search(self, keys: list[Any], root: BPlusTreeNode) -> dict[Any, Any]:
        """Look up many keys in a single level-synchronized descent; returns {key: value or None}.

        All cursors descend together, one batched layer-fetch per level; since every leaf sits at the
        same depth, this is a single descent serving the whole key set, not one per key. Absent keys
        map to None (unlike single-key ``search``, which raises, a batch lookup reports per-key misses
        rather than aborting).
        """
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
        """Split a full node about its midpoint, modifying it in place to keep the left half
        and returning a new node holding the right half. Leaves and internal nodes split differently."""
        right_node = BPlusTreeNode()

        if node.is_leaf:
            # A leaf keeps the median key in the new right half; the caller copies it up as the
            # parent separator, so the key is duplicated (not moved out as in an internal split).
            right_node.keys = node.keys[self._mid :]
            right_node.values = node.values[self._mid :]
            node.keys = node.keys[: self._mid]
            node.values = node.values[: self._mid]

        else:
            # An internal split drops the median key (it gets promoted to the parent) and
            # carries the extra child pointer (values has one more entry than keys).
            right_node.is_leaf = False
            right_node.keys = node.keys[self._mid + 1 :]
            right_node.values = node.values[self._mid + 1 :]
            node.keys = node.keys[: self._mid]
            node.values = node.values[: self._mid + 1]

        return right_node

    def _insert_in_parent(self, child_node: BPlusTreeNode, parent_node: BPlusTreeNode) -> None:
        """Split the full child node and insert the promoted median key (and new right node) into the parent."""
        # Read the median before the split mutates the child's keys.
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
        """Split a full root node and return a fresh parent holding the two halves (grows the tree's height)."""
        # The median is promoted before the split mutates the child's keys.
        insert_key = child_node.keys[self._mid]
        right_node = self._split_node(node=child_node)

        parent_node = BPlusTreeNode()
        parent_node.is_leaf = False
        parent_node.keys.append(insert_key)
        parent_node.values = [child_node, right_node]

        return parent_node

    def insert(self, root: BPlusTreeNode, kv_pair: KVPair) -> BPlusTreeNode:
        """Insert a key-value pair into the tree and return the (possibly new) root."""
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
        """Recursive variant of insert; kept to validate the non-recursive implementation."""
        self._recursive_insert(node=root, kv_pair=kv_pair)
        # If the root overflowed, grow a new root above it (this is the only way the tree gets taller).
        if len(root.keys) >= self._order:
            return self._create_parent(child_node=root)
        return root

    def _recursive_insert(self, node: BPlusTreeNode, kv_pair: KVPair) -> None:
        """Insert into the subtree at ``node``, splitting a child into ``node`` if it overflowed."""
        if node.is_leaf:
            node.add_kv_pair(kv_pair=kv_pair)
            return

        child = node.values[self._child_index(node=node, key=kv_pair.key)]
        self._recursive_insert(node=child, kv_pair=kv_pair)
        if len(child.keys) >= self._order:
            self._insert_in_parent(child_node=child, parent_node=node)

    @staticmethod
    def _collect_insert_paths(root: BPlusTreeNode | None, keys: list[Any]) -> set[BPlusTreeNode]:
        """Batched root-to-leaf descent: fetch every key's path into one local partial tree.

        Every cursor descends together, one batched fetch per level; since all B+ leaves sit at the
        same depth, the cursors reach the leaf layer on the same round -- a single descent serving the
        whole key set, not one per key. Returns ``local``, the set of fetched nodes (the partial tree
        phase 2 mutates).
        """
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
        """Split the overflowed ``child`` and promote the median into ``parent`` (multi_insert helper).

        Reuses the pure ``_split_node`` (which returns the new right node) and adds that node to
        ``local`` so a later insert in the batch may descend onto this fresh sibling.
        """
        # The median is promoted before the split mutates the child's keys.
        insert_key = child.keys[self._mid]
        right = self._split_node(node=child)
        local.add(right)

        position = self._child_index(node=parent, key=insert_key)
        parent.keys.insert(position, insert_key)
        parent.values.insert(position + 1, right)

    def _grow_root_local(self, child: BPlusTreeNode, local: set[BPlusTreeNode]) -> BPlusTreeNode:
        """Split the overflowed root ``child`` and return a fresh parent over the two halves (multi_insert
        helper). Both the new right node and the new root join ``local``."""
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
        """Insert one pair operating only on the local partial tree ``local`` (multi_insert's engine).

        Same shape as the single-key ``insert`` -- add to the target leaf, then split overflowed nodes
        bottom-up -- but every node on the descended path must already be in ``local`` (else the batched
        paths were incomplete and we raise), and each node a split creates joins ``local`` via the
        ``_*_local`` helpers.
        """
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
        """Insert many pairs against a single fetched partial tree; returns the new root.

        Phase 1 (``_collect_insert_paths``) is the only storage access: one batched descent over all
        keys (not one per key) gathering the union of their root-to-leaf paths into ``local``. Phase 2
        replays single-key inserts against ``local`` only, raising if it ever needs an unfetched node;
        each split adds its new nodes to ``local`` so later inserts can descend onto them. The result
        equals sequential single inserts -- the shape the oblivious port mirrors.
        """
        local = self._collect_insert_paths(root=root, keys=[kv_pair.key for kv_pair in kv_pairs])
        for kv_pair in kv_pairs:
            root = self._insert_into_local(root=root, local=local, kv_pair=kv_pair)
        return root

    def _fix_underflow(self, parent: BPlusTreeNode, child_index: int) -> None:
        """Repair the underflowed child ``parent.values[child_index]`` against a single sibling --
        the left one when it exists, otherwise the right -- by borrowing a key from it when it can
        spare one, else merging the two. Mutates ``parent`` in place.

        Considering exactly one sibling (rather than the better of the two) is the standard the
        oblivious map follows: it pre-fetches just this sibling, so the work per level is fixed and
        independent of the borrow/merge outcome. ``delete`` and ``recursive_delete`` both route
        through here, so they make identical structural choices and can be cross-checked.
        """
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

        # The sibling is at its minimum too: merge the two, dropping the separator that sat between
        # them. The left node always absorbs the right one (so the surviving node keeps its index).
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
        """Delete a key from the B+ tree (non-recursive); returns the new root, or None if empty.

        Serves as the template for the oblivious version: it records the visited path as
        (node, child_index) pairs in a ``local`` list, removes the key from the leaf, then repairs
        any underflow bottom-up. ``recursive_delete`` mirrors it for cross-verification.
        """
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

        # When the root itself is the leaf, it simply empties out as its last key goes.
        if not local:
            return None if not leaf.keys else root

        node = leaf
        for parent, child_index in reversed(local):
            if len(node.keys) >= self._min_keys:
                break
            self._fix_underflow(parent=parent, child_index=child_index)
            node = parent

        # The root may now be empty: drop it if it was a leaf, else promote its only remaining child.
        if not root.keys:
            return None if root.is_leaf else root.values[0]
        return root

    def recursive_delete(self, root: BPlusTreeNode | None, key: Any) -> BPlusTreeNode | None:
        """Recursive variant of delete; kept to validate the non-recursive implementation."""
        if root is None:
            return None
        self._recursive_delete(node=root, key=key)
        # The root may now be empty: drop it if it was a leaf, else promote its only remaining child.
        if not root.keys:
            return None if root.is_leaf else root.values[0]
        return root

    def _recursive_delete(self, node: BPlusTreeNode, key: Any) -> None:
        """Delete ``key`` from the subtree at ``node``, repairing the descended child if it underflows."""
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

    def get_data_list(self, root: BPlusTreeNode, block_id: int = 0, encryption: bool = False) -> list[Data]:
        """Flatten the B+ tree into ORAM ``Data`` blocks, assigning each node a block id and random leaf.

        ``encryption=True`` stores the value as pickled bytes rather than a live ``BPlusData``.
        """
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
                stack.extend([child for child in node.values])

            else:
                bplus_data = BPlusData(keys=node.keys, values=node.values)

            if encryption:
                result.append(Data(key=node.id, leaf=node.leaf, value=bplus_data.dump()))
            else:
                result.append(Data(key=node.id, leaf=node.leaf, value=bplus_data))

        return result
