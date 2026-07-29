"""OMAP constructed with the AVL tree ODS."""

import copy
import math
import os
from functools import cached_property
from typing import Any, Protocol, cast, override

from oblivlib.dependency import AVLData, AVLTree, AVLTreeNode, Data, Helper, KVPair
from oblivlib.dependency.codec import BlockCodec, NodeCodec
from oblivlib.dependency.config import AvlOmapConfig
from oblivlib.omap.ost_base_omap import ROOT, LocalNodesBase, OstBaseOmap


class AVLNode(Protocol):
    """Type-only view of an AVL block: a ``Data`` whose ``value`` is always an ``AVLData``. Runtime
    objects are plain ``Data``; this exists so the type checker resolves ``node.value.l_key`` etc."""

    key: Any
    leaf: int
    value: AVLData


class LocalNodes(LocalNodesBase[AVLNode]):
    """Nodes retrieved during AVL operations: adds AVL rotation/reparenting helpers to the shared base."""

    def reparent(self, child_key: Any, new_parent_key: Any) -> None:
        """Update a node's parent link (and root_key if it becomes the root)."""
        self.parent_of[child_key] = new_parent_key
        if new_parent_key is None:
            self.root_key = child_key

    def swap_in_path(self, key_a: Any, key_b: Any) -> None:
        """Swap two keys' positions in the path (used after rotation); skipped if either is absent."""
        if key_a in self.path and key_b in self.path:
            idx_a = self.path.index(key_a)
            idx_b = self.path.index(key_b)
            self.path[idx_a], self.path[idx_b] = self.path[idx_b], self.path[idx_a]

    def update_all_leaves(self, get_new_leaf) -> None:
        """Give every node a fresh leaf and fix each parent's stored child-leaf, in a single pass."""
        for key, node in self.nodes.items():
            node.leaf = get_new_leaf()
            parent = self.get_parent(key)
            if parent:
                if parent.value.l_key == key:
                    parent.value.l_leaf = node.leaf
                elif parent.value.r_key == key:
                    parent.value.r_leaf = node.leaf

    def update_child_in_parent(
        self, parent_key: Any, old_child_key: Any, new_key: Any, new_leaf: Any, new_height: int
    ) -> None:
        """Point a parent's child (the side matching old_child_key) at new (key, leaf, height)."""
        parent = cast(AVLNode | None, self.nodes.get(parent_key))
        if parent is None:
            return

        if parent.value.l_key == old_child_key:
            parent.value.l_key = new_key
            parent.value.l_leaf = new_leaf
            parent.value.l_height = new_height
        elif parent.value.r_key == old_child_key:
            parent.value.r_key = new_key
            parent.value.r_leaf = new_leaf
            parent.value.r_height = new_height

    def replace_node_key(self, old_key: Any, new_key: Any) -> None:
        """Rename a node's key across all tracking structures (used in delete with two children)."""
        if old_key not in self.nodes:
            return

        self.nodes[new_key] = self.nodes.pop(old_key)
        self.parent_of[new_key] = self.parent_of.pop(old_key)
        if old_key in self.path:
            self.path[self.path.index(old_key)] = new_key
        if self.root_key == old_key:
            self.root_key = new_key
        # Re-point any children that referenced the old key as their parent.
        for key, parent_key in self.parent_of.items():
            if parent_key == old_key:
                self.parent_of[key] = new_key


class AVLOmap(OstBaseOmap[AvlOmapConfig, LocalNodes]):
    def __init__(self, config: AvlOmapConfig):
        super().__init__(config)

        # Worst-case AVL height for num_data nodes: the classic 1.44*log2(n) bound, floored at 1 so a
        # single-node tree (where 1.44*log2(1)=0) still reports height 1. This drives block sizing and
        # the dummy-eviction counts, which must cover the real worst case.
        self._max_height: int = max(1, math.ceil(1.44 * math.log(self._num_data, 2)))

        # AVL uses a larger block size, so update disk_size for file storage (one blob per bucket).
        if self._filename and self._encryptor:
            self._disk_size = self._encryptor.ciphertext_length(self._bucket_size * self._max_block_size)

    @override
    def _new_local(self) -> LocalNodes:
        return LocalNodes()

    @override
    def update_mul_tree_height(self, num_tree: int) -> None:
        # Per-bucket item count from https://eprint.iacr.org/2021/1280, then the AVL height bound.
        tree_size = math.ceil(math.e ** (Helper.lambert_w(math.e**-1 * (math.log(num_tree, 2) + 128 - 1)).real + 1))
        self._max_height = max(1, math.ceil(1.44 * math.log(tree_size, 2)))

    @cached_property
    @override
    def _max_block_size(self) -> int:
        return len(
            Data(
                key=os.urandom(self._key_size),
                leaf=self._num_data - 1,
                value=AVLData(
                    value=os.urandom(self._data_size),
                    r_key=os.urandom(self._key_size),
                    r_leaf=self._num_data - 1,
                    r_height=self._max_height,
                    l_key=os.urandom(self._key_size),
                    l_leaf=self._num_data - 1,
                    l_height=self._max_height,
                ).dump(),
            ).dump()
        )

    @property
    @override
    def _codec(self) -> BlockCodec:
        """AVL blocks store an AVLData value, packed to a fixed _max_block_size width."""
        return NodeCodec(self._max_block_size, AVLData)

    def _get_avl_data(self, key: Any, value: Any) -> Data:
        return Data(key=key, leaf=self._get_new_leaf(), value=AVLData(value=value))

    @override
    def _build_ods_blocks(self, data: list[KVPair]) -> tuple[list[Data], ROOT]:
        root: AVLTreeNode | None = None
        avl_tree = AVLTree(leaf_range=self._leaf_range)
        for kv_pair in data:
            root = avl_tree.recursive_insert(root=root, kv_pair=kv_pair)

        # The data list is non-empty, so root is set; get_data_list samples its leaf.
        assert root is not None
        blocks = avl_tree.get_data_list(root=root, encryption=self._encryptor is not None)
        assert root.leaf is not None
        return blocks, (root.key, root.leaf)

    def _update_height(self) -> None:
        """Recompute every local node's height bottom-up and update each parent's stored child-height."""
        for node_key in reversed(self._local.path):
            node = self._local.require(node_key)

            l_height = node.value.l_height if node.value.l_key is not None else 0
            r_height = node.value.r_height if node.value.r_key is not None else 0
            new_height = 1 + max(l_height, r_height)

            parent = self._local.get_parent(node_key)
            if parent:
                if parent.value.l_key == node_key:
                    parent.value.l_height = new_height
                else:
                    parent.value.r_height = new_height

    def _rotate_node(self, node_key: Any, rotate_left: bool) -> tuple[Any, int, int]:
        """AVL rotation at ``node_key`` (it becomes the child); returns the new subtree root's
        (key, leaf, height)."""
        node = self._local.get(node_key)
        if node is None:
            raise ValueError(f"Node {node_key} not found in local.")

        # The pivot (the child that becomes the new subtree root) must be loaded.
        pivot_key = node.value.r_key if rotate_left else node.value.l_key
        pivot = self._local.get(pivot_key)
        if pivot is None or pivot.key != pivot_key:
            side = "Right" if rotate_left else "Left"
            raise ValueError(f"{side} node is not loaded when it is supposed to.")

        if rotate_left:
            # Pivot's left subtree becomes node's right subtree; node becomes pivot's left child.
            node.value.r_key = pivot.value.l_key
            node.value.r_leaf = pivot.value.l_leaf
            node.value.r_height = pivot.value.l_height
            pivot.value.l_key = node.key
            pivot.value.l_leaf = node.leaf
            pivot.value.l_height = 1 + max(node.value.l_height, node.value.r_height)
        else:
            # Pivot's right subtree becomes node's left subtree; node becomes pivot's right child.
            node.value.l_key = pivot.value.r_key
            node.value.l_leaf = pivot.value.r_leaf
            node.value.l_height = pivot.value.r_height
            pivot.value.r_key = node.key
            pivot.value.r_leaf = node.leaf
            pivot.value.r_height = 1 + max(node.value.l_height, node.value.r_height)

        # Pivot takes node's position in the tree.
        grandparent_key = self._local.get_parent_key(node_key)
        self._local.reparent(pivot_key, grandparent_key)
        self._local.reparent(node_key, pivot_key)
        self._local.swap_in_path(node_key, pivot_key)

        new_height = 1 + max(pivot.value.l_height, pivot.value.r_height)
        return pivot.key, pivot.leaf, new_height

    def _balance_node(self, node_key: Any, is_delete: bool = False) -> tuple[Any, int, int]:
        """Re-balance ``node_key`` if unbalanced; returns the (key, leaf, height) now at this position.
        On delete the off-path child/grandchild may need loading first (extra reads, see _op_round_bounds)."""
        # The node being balanced is always loaded (callers only pass loaded keys).
        node = self._local.require(node_key)
        balance = node.value.l_height - node.value.r_height

        # Left-heavy.
        if balance > 1:
            child_key = node.value.l_key
            if is_delete and not self._local.get(child_key):
                self._move_node_to_local(key=child_key, leaf=node.value.l_leaf, parent_key=node_key)

            # The unbalanced child is loaded by this point (either already in local or just moved).
            child_node = self._local.require(child_key)
            # Left-right case: first rotate left on the child.
            if child_node.value.l_height - child_node.value.r_height < 0:
                grandchild_key = child_node.value.r_key
                if is_delete and not self._local.get(grandchild_key):
                    self._move_node_to_local(key=grandchild_key, leaf=child_node.value.r_leaf, parent_key=child_key)

                key, leaf, height = self._rotate_node(child_key, rotate_left=True)
                node.value.l_key, node.value.l_leaf, node.value.l_height = key, leaf, height

            # Left-left case: rotate right.
            return self._rotate_node(node_key, rotate_left=False)

        # Right-heavy.
        if balance < -1:
            child_key = node.value.r_key
            if is_delete and not self._local.get(child_key):
                self._move_node_to_local(key=child_key, leaf=node.value.r_leaf, parent_key=node_key)

            # The unbalanced child is loaded by this point (either already in local or just moved).
            child_node = self._local.require(child_key)
            # Right-left case: first rotate right on the child.
            if child_node.value.l_height - child_node.value.r_height > 0:
                grandchild_key = child_node.value.l_key
                if is_delete and not self._local.get(grandchild_key):
                    self._move_node_to_local(key=grandchild_key, leaf=child_node.value.l_leaf, parent_key=child_key)

                key, leaf, height = self._rotate_node(child_key, rotate_left=False)
                node.value.r_key, node.value.r_leaf, node.value.r_height = key, leaf, height

            # Right-right case: rotate left.
            return self._rotate_node(node_key, rotate_left=True)

        return node.key, node.leaf, 1 + max(node.value.l_height, node.value.r_height)

    def _balance_local(self, is_delete: bool = False) -> None:
        """Rebalance the downloaded path bottom-up, re-pointing each parent at the post-rotation child."""
        # Index-based since rotations mutate the path.
        idx = len(self._local.path) - 1

        while idx >= 0:
            node_key = self._local.path[idx]
            node = self._local.get(node_key)
            if node is None:
                idx -= 1
                continue

            original_key = node.key
            key, leaf, height = self._balance_node(node_key=node_key, is_delete=is_delete)

            # The returned key is the node now at this position; update its parent's pointer.
            new_node = self._local.get(key)
            parent = self._local.get_parent(key) if new_node else None

            if parent:
                if parent.value.r_key == original_key or parent.value.r_key == key:
                    parent.value.r_key = key
                    parent.value.r_leaf = leaf
                    parent.value.r_height = height
                elif parent.value.l_key == original_key or parent.value.l_key == key:
                    parent.value.l_key = key
                    parent.value.l_leaf = leaf
                    parent.value.l_height = height
                else:
                    raise ValueError("This node is not connected to its parent.")

            idx -= 1

        root_node = self._local.get_root()
        if root_node:
            self.root = (root_node.key, root_node.leaf)

    def _descend_and_link(self, key: Any, data_block: Data) -> None:
        """Descend from the root to ``key``'s insertion point, link ``data_block`` there, add it to local.
        Pure tree algorithm: the begin/finalize framing belongs to the caller."""
        # Callers handle the empty-tree case before getting here.
        root = self.root
        assert root is not None
        self._move_node_to_local(key=root[0], leaf=root[1], parent_key=None)
        current_key = root[0]

        while True:
            # current_key is always a node just loaded into local.
            node = self._local.require(current_key)

            # Smaller key -> go right; descend into an existing child or link the new node here.
            if node.key < key:
                if node.value.r_key is not None:
                    self._move_node_to_local(key=node.value.r_key, leaf=node.value.r_leaf, parent_key=current_key)
                    current_key = node.value.r_key
                else:
                    node.value.r_key = data_block.key
                    break
            # Otherwise go left.
            else:
                if node.value.l_key is not None:
                    self._move_node_to_local(key=node.value.l_key, leaf=node.value.l_leaf, parent_key=current_key)
                    current_key = node.value.l_key
                else:
                    node.value.l_key = data_block.key
                    break

        self._local.add(node=data_block, parent_key=current_key)

    def _descend_to_key(self, key: Any) -> Any:
        """Descend from the root toward ``key``; return the key where the descent stopped (the match, or
        the last node when ``key`` is absent). Pure tree algorithm."""
        root = self.root
        assert root is not None
        self._move_node_to_local(key=root[0], leaf=root[1], parent_key=None)
        current_key = root[0]

        # The root was just loaded into local.
        node = self._local.require(current_key)
        while node.key != key:
            if node.key < key:
                if node.value.r_key is not None:
                    self._move_node_to_local(key=node.value.r_key, leaf=node.value.r_leaf, parent_key=current_key)
                    current_key = node.value.r_key
                else:
                    break
            else:
                if node.value.l_key is not None:
                    self._move_node_to_local(key=node.value.l_key, leaf=node.value.l_leaf, parent_key=current_key)
                    current_key = node.value.l_key
                else:
                    break
            # Just descended to a loaded key.
            node = self._local.require(current_key)

        return current_key

    def _delete_at_node(self, node_key: Any) -> tuple[Any, bool]:
        """Remove the (already-loaded) node via the three AVL delete cases; returns (deleted_value, done).
        ``done`` is True only for the single-node tree, where the new root is set and local cleared (so the
        caller skips the post-op fixup). Pure tree algorithm."""
        node = self._local.require(node_key)
        deleted_value = node.value.value
        parent_key = self._local.get_parent_key(node_key)

        # Case 1: leaf node (no children).
        if node.value.l_key is None and node.value.r_key is None:
            if len(self._local) == 1:
                self.root = None
                self._local.clear()
                return deleted_value, True

            self._local.update_child_in_parent(parent_key, node_key, None, None, 0)
            self._local.remove(node_key)

        # Case 2: one child.
        elif node.value.l_key is None or node.value.r_key is None:
            has_left = node.value.l_key is not None
            child_key = node.value.l_key if has_left else node.value.r_key
            child_leaf = node.value.l_leaf if has_left else node.value.r_leaf
            child_height = node.value.l_height if has_left else node.value.r_height

            if len(self._local) == 1:
                # The surviving child carries a concrete leaf.
                assert child_leaf is not None
                self.root = (child_key, child_leaf)
                self._local.clear()
                return deleted_value, True

            self._local.update_child_in_parent(parent_key, node_key, child_key, child_leaf, child_height)
            self._local.remove(node_key)

        # Case 3: two children. Replace with the in-order predecessor/successor from the taller subtree --
        # the same height-based choice AVLTree.delete makes (the plaintext standard).
        else:
            use_predecessor = node.value.l_height > node.value.r_height
            original_key = node.key

            # Descend to the replacement node (rightmost of left subtree / leftmost of right subtree).
            traverse_key = node.value.l_key if use_predecessor else node.value.r_key
            traverse_leaf = node.value.l_leaf if use_predecessor else node.value.r_leaf
            self._move_node_to_local(key=traverse_key, leaf=traverse_leaf, parent_key=node_key)
            current = self._local.require(traverse_key)
            parent_of_replacement_key = node_key

            next_key = current.value.r_key if use_predecessor else current.value.l_key
            while next_key is not None:
                next_leaf = current.value.r_leaf if use_predecessor else current.value.l_leaf
                self._move_node_to_local(key=next_key, leaf=next_leaf, parent_key=traverse_key)
                parent_of_replacement_key = traverse_key
                traverse_key = next_key
                current = self._local.require(traverse_key)
                next_key = current.value.r_key if use_predecessor else current.value.l_key

            # Copy the replacement's data into the node being deleted.
            replacement_node = current
            node.key = replacement_node.key
            node.value.value = replacement_node.value.value

            # Splice the replacement out (point its parent at its single child).
            child_key = replacement_node.value.l_key if use_predecessor else replacement_node.value.r_key
            child_leaf = replacement_node.value.l_leaf if use_predecessor else replacement_node.value.r_leaf
            child_height = (
                (replacement_node.value.l_height if use_predecessor else replacement_node.value.r_height)
                if child_key
                else 0
            )
            self._local.update_child_in_parent(
                parent_of_replacement_key, replacement_node.key, child_key, child_leaf, child_height
            )
            self._local.remove(replacement_node.key)

            # Re-point the deleted node's parent at the new key and fix local tracking.
            self._local.update_child_in_parent(parent_key, original_key, node.key, node.leaf, 0)
            self._local.replace_node_key(original_key, node.key)

        return deleted_value, False

    def _post_op_fixup(self, is_delete: bool = False) -> None:
        """Shared post-op fixup: recompute heights, re-leaf every node, then rebalance bottom-up."""
        self._update_height()
        self._local.update_all_leaves(self._get_new_leaf)
        self._balance_local(is_delete=is_delete)

    @override
    def _op_round_bounds(self) -> dict[str, int]:
        """Self-clearing round cost per op (retrieve the nodes, then evict them all back), height h.
        search/insert pull h nodes and need h more rounds to evict them => 2h (+1 margin); delete pulls
        up to 3h (h to locate the node + in-order successor, 2h for a rebalance that can cascade to the
        root, fetching the off-path child + grandchild per level) and needs 3h to evict => 6h. Oblivious
        mode (distinguishable=False) pads all three to their max (see ``_op_budget``);
        ``test_omap_common.py`` enforces both regimes."""
        h = self._max_height
        return {"search": 2 * h + 1, "insert": 2 * h + 1, "delete": 6 * h}

    @override
    def search(self, key: Any, value: Any = None) -> Any:
        """Streaming search: descend one node at a time, re-homing and evicting each before reading the
        next, so ``local`` holds O(1) nodes (not the whole path). Pads to the op budget, so it is
        oblivious when distinguishable=False (fixed read/evict round count) and cheaper/depth-varying
        when distinguishable=True."""
        self._op_rounds = 0
        budget = self._op_budget("search")
        # A dummy op (key is None) or empty tree (e.g. an unused bucket in a composed OMAP) finds nothing.
        if self._short_circuit_read(key=key, num_round=budget):
            return None

        root = self.root
        assert root is not None
        self._move_node_to_local_without_eviction(key=root[0], leaf=root[1], parent_key=None)
        current_key = root[0]

        old_child_path = root[1]
        child_leaf = self._get_new_leaf()
        self.root = (root[0], child_leaf)

        # The root was just loaded into local.
        current = self._local.require(current_key)
        while current.key != key:
            go_right = current.key < key
            next_key = current.value.r_key if go_right else current.value.l_key

            if next_key is None:
                break

            # Re-home the current node onto its fresh leaf and write it back before descending.
            node_to_return = copy.deepcopy(current)
            node_to_return.leaf = child_leaf
            child_leaf = self._get_new_leaf()
            if go_right:
                node_to_return.value.r_leaf = child_leaf
            else:
                node_to_return.value.l_leaf = child_leaf

            self._stash.append(node_to_return)
            self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=[old_child_path]))
            self._client.execute()

            # Descend; a child pointer always carries a real leaf.
            next_leaf = current.value.r_leaf if go_right else current.value.l_leaf
            assert next_leaf is not None
            self._move_node_to_local_without_eviction(key=next_key, leaf=next_leaf, parent_key=current_key)
            old_child_path = next_leaf
            self._local.remove(current_key)
            current_key = next_key
            current = self._local.require(current_key)

        search_value = current.value.value if current.key == key else None
        # Only write on a hit: a miss stops the descent at an unrelated node.
        if value is not None and current.key == key:
            current.value.value = value

        # Write the final node back, then pad the real rounds up to the budget.
        node_to_return = copy.deepcopy(current)
        node_to_return.leaf = child_leaf
        self._stash.append(node_to_return)
        self._local.clear()
        self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=[old_child_path]))
        self._client.execute()

        self._pad_to(budget)
        return search_value

    @override
    def insert(self, key: Any, value: Any = None) -> None:
        """Mirrors plaintext ``AVLTree.insert``: descend to the insertion point, link the new leaf, then
        rebalance bottom-up (at most one rotation, reusing the already-loaded path nodes)."""
        self._op_rounds = 0
        budget = self._op_budget("insert")
        if key is None:
            # A dummy insert must be indistinguishable from a real one.
            self._pad_to(budget)
            return
        data_block = self._get_avl_data(key=key, value=value)

        # Empty tree: the new block is the root.
        if self.root is None:
            self._stash.append(data_block)
            # A fresh block always has a sampled leaf.
            assert data_block.leaf is not None
            self.root = (data_block.key, data_block.leaf)
            # Pad to the same total a populated-tree insert uses (so the first insert blends in).
            self._pad_to(budget)
            return

        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")

        self._descend_and_link(key=key, data_block=data_block)
        self._post_op_fixup()

        self._flush_local_to_stash()
        self._pad_to(budget)

    def delete(self, key: Any) -> Any:
        """Delete ``key`` and return its value (or None if absent)."""
        self._op_rounds = 0
        budget = self._op_budget("delete")
        # A dummy/empty/missing delete removes nothing; pad like a real (found) delete so the three look
        # identical to the server.
        if self._short_circuit_read(key=key, num_round=budget):
            return None

        # Tree is non-empty; descend to the node to delete.
        current_key = self._descend_to_key(key=key)
        node = self._local.require(current_key)

        # Miss: pad to the same total a found delete uses.
        if node.key != key:
            self._flush_local_to_stash()
            self._pad_to(budget)
            return None

        deleted_value, done = self._delete_at_node(node_key=current_key)
        if done:
            # The single-node tree case already set the new root and cleared local; just pad the rest.
            self._pad_to(budget)
            return deleted_value

        self._post_op_fixup(is_delete=True)

        self._flush_local_to_stash()
        self._pad_to(budget)
        return deleted_value
