"""OMAP constructed with the B+ tree ODS."""

import math
import os
from functools import cached_property
from typing import Any, Protocol, cast, override

from oblivlib.dependency import BPlusData, BPlusTree, BPlusTreeNode, Data, Helper, KVPair
from oblivlib.dependency.codec import BlockCodec, NodeCodec
from oblivlib.dependency.config import BPlusOmapConfig
from oblivlib.omap.ost_base_omap import ROOT, LocalNodesBase, OstBaseOmap


class BPlusValue(Protocol):
    """Type-only view of a B+ node's payload (a ``BPlusData``); exists so the type checker resolves
    ``node.value.keys[i]`` etc. through the ``Data.value: Any`` field."""

    keys: list[Any]
    values: list[Any]


class BPlusNode(Protocol):
    """Type-only view of a B+ block: a ``Data`` whose ``value`` is a populated ``BPlusData``. Runtime
    objects are plain ``Data``; the ``LocalNodes`` getters cast their stored ``Data`` to this view."""

    key: Any
    leaf: int
    value: BPlusValue


class LocalNodes(LocalNodesBase[BPlusNode]):
    """Nodes retrieved during B+ operations: adds child-index tracking and leaf helpers to the base."""

    def __init__(self) -> None:
        super().__init__()
        self.child_index_of: dict[Any, int] = {}  # key -> index in its parent's values

    @staticmethod
    def is_leaf_node(node: BPlusNode) -> bool:
        """A node is a leaf when its keys and values lists have equal length."""
        return len(node.value.keys) == len(node.value.values)

    @override
    def add(self, node: Data, parent_key: Any = None, child_index: int | None = None) -> None:
        super().add(node=node, parent_key=parent_key)
        if child_index is not None:
            self.child_index_of[node.key] = child_index

    def get_child_index(self, key: Any) -> int | None:
        return self.child_index_of.get(key)

    def get_leaf(self) -> BPlusNode | None:
        """Get the leaf node (last in path; None when the path is empty)."""
        if not self.path:
            return None
        return cast(BPlusNode | None, self.nodes.get(self.path[-1]))

    def require_leaf(self) -> BPlusNode:
        """Get the leaf node (last in path) known to exist; raise if it is genuinely absent."""
        node = cast(BPlusNode | None, self.nodes.get(self.path[-1])) if self.path else None
        if node is None:
            raise KeyError("The local has no leaf node.")
        return node

    def update_all_leaves(self, get_new_leaf) -> None:
        """Give every node a fresh leaf and update each parent's stored child (key, leaf), in one pass."""
        for key, node in self.nodes.items():
            node.leaf = get_new_leaf()
            parent = self.get_parent(key)
            child_index = self.get_child_index(key)
            if parent is not None and child_index is not None:
                parent.value.values[child_index] = (node.key, node.leaf)

    @override
    def remove(self, key: Any) -> Data | None:
        node = super().remove(key)
        self.child_index_of.pop(key, None)
        return node

    def pop(self, key: Any) -> BPlusNode:
        """Remove and return a node known to be loaded; raise if it is genuinely absent."""
        node = cast(BPlusNode | None, self.remove(key))
        if node is None:
            raise KeyError(f"Node {key} is not loaded in local.")
        return node

    @override
    def clear(self) -> None:
        super().clear()
        self.child_index_of.clear()


class BPlusOmap(OstBaseOmap[BPlusOmapConfig, LocalNodes]):
    def __init__(self, config: BPlusOmapConfig):
        super().__init__(config)

        # Split point and the next block id to hand out.
        self._mid: int = config.order // 2
        self._block_id: int = 0

        # Worst-case B+ height for num_data nodes, floored at 1 so a single-element map (where log gives
        # 0) still has a positive dummy-eviction budget.
        self._max_height: int = max(1, math.ceil(math.log(self._num_data, math.ceil(config.order / 2))))

        # B+ uses a larger block size, so update disk_size for file storage (one blob per bucket).
        if self._filename and self._encryptor:
            self._disk_size = self._encryptor.ciphertext_length(self._bucket_size * self._max_block_size)

    @override
    def _new_local(self) -> LocalNodes:
        return LocalNodes()

    @override
    def update_mul_tree_height(self, num_tree: int) -> None:
        # Per-bucket item count from https://eprint.iacr.org/2021/1280, then the B+ height bound.
        tree_size = math.ceil(math.e ** (Helper.lambert_w(math.e**-1 * (math.log(num_tree, 2) + 128 - 1)).real + 1))
        self._max_height = max(1, math.ceil(math.log(tree_size, math.ceil(self._order / 2))))

    # Scheme-specific construction parameter — read-only view onto the frozen config.
    @property
    def _order(self) -> int:
        return self._config.order

    @cached_property
    @override
    def _max_block_size(self) -> int:
        # Two block types (internal vs leaf); size to the larger.
        return max(
            len(
                Data(  # internal: values are (id, leaf) integer pairs
                    key=self._num_data - 1,
                    leaf=self._num_data - 1,
                    value=BPlusData(
                        keys=[os.urandom(self._key_size) for _ in range(self._order - 1)],
                        values=[(self._num_data - 1, self._num_data - 1) for _ in range(self._order)],
                    ).dump(),
                ).dump()
            ),
            len(
                Data(  # leaf: values are actual values
                    key=self._num_data - 1,
                    leaf=self._num_data - 1,
                    value=BPlusData(
                        keys=[os.urandom(self._key_size) for _ in range(self._order - 1)],
                        values=[os.urandom(self._data_size) for _ in range(self._order - 1)],
                    ).dump(),
                ).dump()
            ),
        )

    @property
    @override
    def _codec(self) -> BlockCodec:
        """B+ blocks store a BPlusData value, packed to a fixed _max_block_size width."""
        return NodeCodec(self._max_block_size, BPlusData)

    def _get_bplus_data(self, keys: Any = None, values: Any = None) -> Data:
        data_block = Data(key=self._block_id, leaf=self._get_new_leaf(), value=BPlusData(keys=keys, values=values))
        self._block_id += 1
        return data_block

    @override
    def _build_ods_blocks(self, data: list[KVPair]) -> tuple[list[Data], ROOT]:
        root = BPlusTreeNode()
        bplus_tree = BPlusTree(order=self._order, leaf_range=self._leaf_range)
        for kv_pair in data:
            root = bplus_tree.insert(root=root, kv_pair=kv_pair)

        # get_data_list assigns each node a block id (starting at _block_id) and a leaf.
        blocks = bplus_tree.get_data_list(root=root, block_id=self._block_id, encryption=self._encryptor is not None)
        self._block_id += len(blocks)
        assert root.id is not None and root.leaf is not None
        return blocks, (root.id, root.leaf)

    def _find_leaf(self, key: Any) -> int:
        """Descend to the leaf holding ``key`` without keeping the visited nodes in local (each is moved
        to the stash and evicted before the next is read); returns the leaf's old path."""
        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")

        # Callers guarantee a non-empty tree.
        root = self.root
        assert root is not None

        self._move_node_to_local_without_eviction(key=root[0], leaf=root[1], parent_key=None, child_index=None)

        # The root was just loaded into local; re-home it onto a fresh leaf.
        node = self._local.require_root()
        old_leaf = node.leaf
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)

        while not self._local.is_leaf_node(node):
            new_leaf = self._get_new_leaf()

            child_index = len(node.value.keys)
            for index, each_key in enumerate(node.value.keys):
                if key == each_key:
                    child_index = index + 1
                    break
                elif key < each_key:
                    child_index = index
                    break

            child_key, child_leaf = node.value.values[child_index]
            # Update the stored child leaf, then stash + evict the current node before the next read.
            node.value.values[child_index] = (child_key, new_leaf)
            self._stash.append(self._local.remove(node.key))
            self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=[old_leaf]))
            self._client.execute()
            self._move_node_to_local_without_eviction(key=child_key, leaf=child_leaf, parent_key=None, child_index=None)

            # The child we just descended to was loaded into local.
            node = self._local.require_root()
            old_leaf = node.leaf
            node.leaf = new_leaf

        return old_leaf

    def _find_leaf_to_local(self, key: Any) -> None:
        """Descend to the leaf holding ``key``, keeping every visited node in local with parent links."""
        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")

        # Callers guarantee a non-empty tree.
        root = self.root
        assert root is not None

        self._move_node_to_local(key=root[0], leaf=root[1], parent_key=None, child_index=None)

        # The root was just loaded into local; re-home it onto a fresh leaf.
        node = self._local.require_root()
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)

        while not self._local.is_leaf_node(node):
            new_leaf = self._get_new_leaf()

            child_index = len(node.value.keys)
            for index, each_key in enumerate(node.value.keys):
                if key == each_key:
                    child_index = index + 1
                    break
                elif key < each_key:
                    child_index = index
                    break

            child_key, child_leaf = node.value.values[child_index]
            self._move_node_to_local(key=child_key, leaf=child_leaf, parent_key=node.key, child_index=child_index)
            node.value.values[child_index] = (child_key, new_leaf)

            # The child is now the last node in the path.
            node = self._local.require_leaf()
            node.leaf = new_leaf

    def _split_node(self, node: BPlusNode) -> tuple[int, int]:
        """Split a full ``node`` in place into a left (the original) and a new right half, depending on
        whether it is a leaf or internal; the new right node is added to the stash. Returns its (key, leaf)."""
        right_node = self._get_bplus_data()

        if self._local.is_leaf_node(node):
            right_node.value.keys = node.value.keys[self._mid :]
            right_node.value.values = node.value.values[self._mid :]
            node.value.keys = node.value.keys[: self._mid]
            node.value.values = node.value.values[: self._mid]
        else:
            right_node.value.keys = node.value.keys[self._mid + 1 :]
            right_node.value.values = node.value.values[self._mid + 1 :]
            node.value.keys = node.value.keys[: self._mid]
            node.value.values = node.value.values[: self._mid + 1]

        self._stash.append(right_node)
        # Both halves are mutated in place, so only the new right node needs returning.
        return right_node.key, right_node.require_leaf()

    def _insert_in_parent(self, child_node: BPlusNode, parent_node: BPlusNode) -> None:
        """Split ``child_node`` and insert the promoted key + new right pointer into ``parent_node``."""
        insert_key = child_node.value.keys[self._mid]
        right_node = self._split_node(node=child_node)

        for index, each_key in enumerate(parent_node.value.keys):
            if insert_key < each_key:
                parent_node.value.keys = parent_node.value.keys[:index] + [insert_key] + parent_node.value.keys[index:]
                parent_node.value.values = (
                    parent_node.value.values[: index + 1] + [right_node] + parent_node.value.values[index + 1 :]
                )
                return
            elif index + 1 == len(parent_node.value.keys):
                parent_node.value.keys.append(insert_key)
                parent_node.value.values.append(right_node)
                return

    def _create_parent(self, child_node: BPlusNode) -> None:
        """Split a rootless full ``child_node`` and create a new root over the two halves."""
        insert_key = child_node.value.keys[self._mid]
        right_node = self._split_node(node=child_node)

        values = [(child_node.key, child_node.leaf), right_node]
        parent_node = self._get_bplus_data(keys=[insert_key], values=values)
        self._stash.append(parent_node)

        # A fresh block always has a sampled leaf.
        assert parent_node.leaf is not None
        self.root = (parent_node.key, parent_node.leaf)

    def _insert_into_loaded_leaf(self, key: Any, value: Any) -> None:
        """Insert ``key``/``value`` into the leaf already loaded in local, then split overflow up the path."""
        leaf = self._local.require_leaf()

        for index, each_key in enumerate(leaf.value.keys):
            if key < each_key:
                leaf.value.keys = leaf.value.keys[:index] + [key] + leaf.value.keys[index:]
                leaf.value.values = leaf.value.values[:index] + [value] + leaf.value.values[index:]
                break
            elif index + 1 == len(leaf.value.keys):
                leaf.value.keys.append(key)
                leaf.value.values.append(value)
                break

        # Splits add nodes to the stash, not new server reads.
        self._perform_insertion()

    def _perform_insertion(self):
        """Split any overflowed node back up the loaded path (leaf to root)."""
        path = self._local.path
        index = len(path) - 1

        while index >= 0:
            node_key = path[index]
            node = self._local.require(node_key)

            if len(node.value.keys) >= self._order:
                # Overflow: push the split into the parent, or grow a new root.
                if index > 0:
                    parent_key = path[index - 1]
                    parent_node = self._local.require(parent_key)
                    self._insert_in_parent(child_node=node, parent_node=parent_node)
                    index -= 1
                else:
                    self._create_parent(child_node=node)
                    break
            else:
                # No overflow above this point: stop.
                break

    def _find_path_with_siblings(
        self, key: Any
    ) -> tuple[dict[int, BPlusNode], list[int], dict[int, BPlusNode], dict[int, int], int]:
        """Find the root-to-leaf path for ``key``, pre-fetching one sibling at every non-root level.

        Pre-fetching is what makes delete oblivious: the path child *and* a sibling are read at each level
        whether or not a later underflow needs the sibling, so the read count depends only on the (public)
        tree height, never on the borrow/merge outcome. Returns (path_nodes by level, child_indices,
        siblings by child-level, sibling_indices by child-level, num path reads performed)."""
        # The base scheme operates on an empty local; each node is popped out to the dicts as it is read.
        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")

        path_nodes: dict[int, BPlusNode] = {}
        child_indices: list[int] = []
        siblings: dict[int, BPlusNode] = {}
        sibling_indices: dict[int, int] = {}
        level = 0
        num_rounds = 0

        # Callers guarantee a non-empty tree.
        root = self.root
        assert root is not None

        self._move_node_to_local(key=root[0], leaf=root[1], parent_key=None, child_index=None)
        num_rounds += 1
        node = self._local.pop(self._local.root_key)
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)
        path_nodes[level] = node

        # Descend to the leaf, reading the path child and one sibling at each level.
        while not (len(node.value.keys) == len(node.value.values)):
            new_leaf = self._get_new_leaf()

            child_index = len(node.value.keys)
            for index, each_key in enumerate(node.value.keys):
                if key < each_key:
                    child_index = index
                    break
                elif key == each_key:
                    child_index = index + 1
                    break
            child_indices.append(child_index)

            # Read the path child, re-homing it onto a fresh leaf.
            child_key, child_leaf = node.value.values[child_index]
            self._move_node_to_local(key=child_key, leaf=child_leaf, parent_key=None, child_index=None)
            num_rounds += 1
            child_node = self._local.pop(self._local.root_key)
            node.value.values[child_index] = (child_key, new_leaf)
            child_node.leaf = new_leaf

            # Read one sibling (prefer left, else right) so an underflow needs no extra round.
            sibling_index: int | None = None
            if child_index > 0:
                sibling_index = child_index - 1
            elif child_index < len(node.value.values) - 1:
                sibling_index = child_index + 1
            if sibling_index is not None:
                sib_key, sib_leaf = node.value.values[sibling_index]
                sib_new_leaf = self._get_new_leaf()
                self._move_node_to_local(key=sib_key, leaf=sib_leaf, parent_key=None, child_index=None)
                num_rounds += 1
                sibling_node = self._local.pop(self._local.root_key)
                sibling_node.leaf = sib_new_leaf
                node.value.values[sibling_index] = (sib_key, sib_new_leaf)
                siblings[level + 1] = sibling_node
                sibling_indices[level + 1] = sibling_index

            level += 1
            path_nodes[level] = child_node
            node = child_node

        return path_nodes, child_indices, siblings, sibling_indices, num_rounds

    def _apply_delete(
        self,
        key: Any,
        path_nodes: dict[int, BPlusNode],
        child_indices: list[int],
        siblings: dict[int, BPlusNode],
        sibling_indices: dict[int, int],
    ) -> Any:
        """Remove ``key`` from the pre-fetched path and rebalance, leaving the survivors in place.

        Operates on the structures from ``_find_path_with_siblings``: removes the key from the leaf,
        resolves underflow bottom-up with the pre-fetched siblings (borrow-or-merge), and collapses the
        root. Merged-away nodes are dropped from ``path_nodes``/``siblings`` so the caller flushes only
        survivors. Returns the deleted value, or None if the key was absent."""
        # Minimum keys a (non-root) node may hold before it underflows (== BPlusTree._min_keys).
        min_keys = (self._order - 1) // 2

        leaf_level = len(path_nodes) - 1
        leaf = path_nodes[leaf_level]

        # Find and remove the key from the leaf.
        key_index = None
        deleted_value = None
        for i, k in enumerate(leaf.value.keys):
            if k == key:
                key_index = i
                deleted_value = leaf.value.values[i]
                break

        if key_index is not None:
            leaf.value.keys.pop(key_index)
            leaf.value.values.pop(key_index)

            if not child_indices:
                # Root is the leaf: drop it if now empty, else it stays as the (still rooted) leaf.
                if len(leaf.value.keys) == 0:
                    self.root = None
                    del path_nodes[leaf_level]
            else:
                # Resolve underflow bottom-up using the pre-fetched siblings.
                node_level = leaf_level
                node = leaf
                for level in range(len(child_indices) - 1, -1, -1):
                    parent = path_nodes[level]
                    child_index = child_indices[level]

                    if len(node.value.keys) >= min_keys:
                        break
                    child_level = level + 1
                    if child_level not in siblings:
                        break

                    sibling = siblings[child_level]
                    sib_index = sibling_indices[child_level]
                    is_left_sibling = sib_index < child_index
                    is_leaf = len(node.value.keys) == len(node.value.values)

                    # Borrow a key from the sibling if it can spare one.
                    if len(sibling.value.keys) > min_keys:
                        if is_left_sibling:
                            if is_leaf:
                                node.value.keys.insert(0, sibling.value.keys.pop())
                                node.value.values.insert(0, sibling.value.values.pop())
                                parent.value.keys[child_index - 1] = node.value.keys[0]
                            else:
                                node.value.keys.insert(0, parent.value.keys[child_index - 1])
                                node.value.values.insert(0, sibling.value.values.pop())
                                parent.value.keys[child_index - 1] = sibling.value.keys.pop()
                        else:
                            if is_leaf:
                                node.value.keys.append(sibling.value.keys.pop(0))
                                node.value.values.append(sibling.value.values.pop(0))
                                parent.value.keys[child_index] = sibling.value.keys[0]
                            else:
                                node.value.keys.append(parent.value.keys[child_index])
                                node.value.values.append(sibling.value.values.pop(0))
                                parent.value.keys[child_index] = sibling.value.keys.pop(0)
                        break

                    # Otherwise merge node and sibling into one.
                    if is_left_sibling:
                        if is_leaf:
                            sibling.value.keys.extend(node.value.keys)
                            sibling.value.values.extend(node.value.values)
                        else:
                            sibling.value.keys.append(parent.value.keys[child_index - 1])
                            sibling.value.keys.extend(node.value.keys)
                            sibling.value.values.extend(node.value.values)
                        parent.value.keys.pop(child_index - 1)
                        parent.value.values.pop(child_index)
                        # node was merged away into its left sibling.
                        del path_nodes[node_level]
                    else:
                        if is_leaf:
                            node.value.keys.extend(sibling.value.keys)
                            node.value.values.extend(sibling.value.values)
                        else:
                            node.value.keys.append(parent.value.keys[child_index])
                            node.value.keys.extend(sibling.value.keys)
                            node.value.values.extend(sibling.value.values)
                        parent.value.keys.pop(child_index)
                        parent.value.values.pop(child_index + 1)
                        # the right sibling was merged away into node.
                        del siblings[child_level]

                    node_level = level
                    node = parent

                # The root may now be empty: drop it, or promote its single remaining child.
                root_node = path_nodes.get(0)
                if root_node is not None and len(root_node.value.keys) == 0:
                    if len(root_node.value.keys) == len(root_node.value.values):
                        self.root = None
                    else:
                        child_key, child_leaf = root_node.value.values[0]
                        self.root = (child_key, child_leaf)
                    del path_nodes[0]

                if self.root is not None and 0 in path_nodes:
                    root_node = path_nodes[0]
                    self.root = (root_node.key, root_node.leaf)

        return deleted_value

    @override
    def _op_round_bounds(self) -> dict[str, int]:
        """Self-clearing round cost per op (retrieve the nodes, then evict them all back), height h.
        search/insert pull h nodes and need h more rounds to evict them => 2h (+1 margin); delete
        pre-fetches the path child + one sibling at each non-root level (~2h retrieval, fixed regardless
        of the borrow/merge outcome -- see _find_path_with_siblings) and needs 2h to evict => 4h.
        Oblivious mode (distinguishable=False) pads all three to their max (see ``_op_budget``);
        ``test_omap_common.py`` enforces both regimes."""
        h = self._max_height
        return {"search": 2 * h + 1, "insert": 2 * h + 1, "delete": 4 * h}

    @override
    def search(self, key: Any, value: Any = None) -> Any:
        """Streaming search: descend to the leaf via _find_leaf (each visited node evicted before the
        next is read, so ``local`` holds O(1) nodes), then pad to the op budget. Oblivious when
        distinguishable=False (fixed read/evict round count); cheaper/depth-varying when True."""
        self._op_rounds = 0
        budget = self._op_budget("search")
        # A dummy op (key is None) or an empty tree finds nothing.
        if self._short_circuit_read(key=key, num_round=budget):
            return None

        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")

        old_leaf = self._find_leaf(key=key)

        # The traversal always lands on a leaf node now in local.
        leaf = self._local.require_leaf()
        search_value = None

        for index, each_key in enumerate(leaf.value.keys):
            if key == each_key:
                search_value = leaf.value.values[index]
                if value is not None:
                    leaf.value.values[index] = value
                break

        # Flush the leaf, run one final eviction, then pad up to the budget.
        self._flush_local_to_stash()
        self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=[old_leaf]))
        self._client.execute()
        self._pad_to(budget)

        return search_value

    @override
    def insert(self, key: Any, value: Any = None) -> None:
        """Mirrors plaintext ``BPlusTree.insert``: descend to the target leaf, add the pair, then split
        any overflowed node back up the path (splits add stash nodes, not new reads)."""
        self._op_rounds = 0
        budget = self._op_budget("insert")
        if key is None:
            # A dummy insert must be indistinguishable from a real one.
            self._pad_to(budget)
            return

        # Empty tree: the new block is the root.
        if self.root is None:
            data_block = self._get_bplus_data(keys=[key], values=[value])
            self._stash.append(data_block)
            # A fresh block always has a sampled leaf.
            assert data_block.leaf is not None
            self.root = (data_block.key, data_block.leaf)
            # Pad to the same total a populated-tree insert uses (so the first insert blends in).
            self._pad_to(budget)
            return

        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")

        self._find_leaf_to_local(key=key)
        self._insert_into_loaded_leaf(key=key, value=value)

        self._flush_local_to_stash()
        self._pad_to(budget)

    def delete(self, key: Any) -> Any:
        """Delete ``key`` and return its value (or None if absent).

        Mirrors plaintext ``BPlusTree.delete`` / ``_fix_underflow`` (single sibling per underflow, prefer
        left else right, borrow-or-merge). Obliviousness comes from pre-fetching that sibling at every
        level, so every path -- hit, miss, borrow, or merge -- does the same reads, padded to the fixed
        budget. So an observer cannot tell a hit from a miss, nor read off the rebalancing."""
        self._op_rounds = 0
        budget = self._op_budget("delete")

        # A dummy/empty/missing delete removes nothing; pad like a real one.
        if self._short_circuit_read(key=key, num_round=budget):
            return None

        path_nodes, child_indices, siblings, sibling_indices, _ = self._find_path_with_siblings(key=key)
        deleted_value = self._apply_delete(
            key=key,
            path_nodes=path_nodes,
            child_indices=child_indices,
            siblings=siblings,
            sibling_indices=sibling_indices,
        )

        # Flush every surviving node to stash, then pad the real rounds up to the delete budget.
        for node in path_nodes.values():
            self._stash.append(node)
        for sib in siblings.values():
            self._stash.append(sib)
        self._pad_to(budget)

        return deleted_value
