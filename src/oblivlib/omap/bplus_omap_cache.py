"""B+ tree OMAP with caching optimization for repeated accesses."""

from typing import Any, override

from oblivlib.dependency.config import BPlusOmapCachedConfig
from oblivlib.omap.bplus_omap import BPlusNode, BPlusOmap


class BPlusOmapCached(BPlusOmap):
    """B+ OMAP with stash caching. Versus ``BPlusOmap``: a cache hit (node already in stash) skips the
    ORAM access; insert keeps the visited path in local, flushed at the start of the next op, so nearby
    repeated accesses reuse cached nodes; fewer interaction rounds -- insert/search are one
    root->leaf descent (h rounds), and delete fetches the path child *and* one sibling together in a
    single batched read-write round per level (see ``_batch_move_nodes_to_local``), so it is h rounds
    (1 root + h-1 batched) instead of the non-cached 2h-1, while bandwidth stays ~2h-1 paths.

    Unlike ``AVLOmapCached.search`` (which keeps the path cached), this ``search`` streams and keeps
    nothing in local. Both cached variants are intentionally NOT per-op oblivious -- caching makes the
    access pattern data-dependent by design."""

    def __init__(self, config: BPlusOmapCachedConfig):
        super().__init__(config)

    @override
    def _move_node_to_local(
        self, key: Any, leaf: int | None, parent_key: Any = None, child_index: int | None = None
    ) -> None:
        # The caching mechanism: serve a node from the stash if a previous op left it there, else fetch.
        stash_idx = self._find_in_stash(key)
        if stash_idx >= 0:
            node = self._stash.pop(stash_idx)
            self._local.add(node=node, parent_key=parent_key, child_index=child_index)
        else:
            super()._move_node_to_local(key=key, leaf=leaf, parent_key=parent_key, child_index=child_index)

    def _batch_move_nodes_to_local(self, nodes_to_fetch: list[tuple[Any, int]]) -> None:
        """Fetch several nodes (e.g. a path child and its sibling) in ONE read-write round.

        Cache hits are served from the stash for free; the misses are read with a single batched
        ``add_read_path`` and written back with a single batched eviction, so any number of nodes costs
        one round. Counts as one real round (``_op_rounds += 1``) only when a server read happens -- a
        full cache hit is free. The fetched nodes are left in local for the caller to ``pop`` by key."""
        keys_to_read: list[Any] = []
        leaves_to_read: list[int] = []

        # Serve cache hits from the stash; collect the misses to read in one batch.
        for node_key, leaf in nodes_to_fetch:
            stash_idx = self._find_in_stash(node_key)
            if stash_idx >= 0:
                self._local.add(node=self._stash.pop(stash_idx))
            else:
                keys_to_read.append(node_key)
                leaves_to_read.append(leaf)

        # Every requested node was cached -- no server round needed.
        if not leaves_to_read:
            return

        # One batched read round for all the misses.
        self._op_rounds += 1
        to_index = len(self._stash)
        self._client.add_read_path(label=self._name, leaves=leaves_to_read)
        result = self._client.execute()
        path = self._decrypt_path_data(path=result.require(self._name))

        keys_to_find = set(keys_to_read)
        for bucket in path.values():
            for data in bucket:
                if data.key in keys_to_find:
                    self._local.add(node=data)
                    keys_to_find.discard(data.key)
                elif data.key is not None:
                    self._stash.append(data)

        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")

        # Any key not on its fetched path must already be in the stash from an earlier op.
        for node_key in list(keys_to_find):
            stash_idx = self._find_in_stash(node_key)
            if 0 <= stash_idx < to_index:
                self._local.add(node=self._stash.pop(stash_idx))
                keys_to_find.discard(node_key)

        if keys_to_find:
            raise KeyError(f"The search key(s) {keys_to_find} are not found.")

        # One batched eviction write-back over the same paths.
        self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=leaves_to_read))
        self._client.execute()

    def _find_path_with_siblings_cached(
        self, key: Any
    ) -> tuple[dict[int, BPlusNode], list[int], dict[int, BPlusNode], dict[int, int]]:
        """Batched-round variant of ``_find_path_with_siblings``: at each non-root level the path child
        and one sibling are read together in a single round, so the descent is h rounds rather than 2h-1.
        Returns the same shape ``_apply_delete`` consumes."""
        self._flush_local_to_stash()

        path_nodes: dict[int, BPlusNode] = {}
        child_indices: list[int] = []
        siblings: dict[int, BPlusNode] = {}
        sibling_indices: dict[int, int] = {}
        level = 0

        root = self.root
        assert root is not None

        # Read the root (one possibly-cached round), re-home it onto a fresh leaf.
        self._move_node_to_local(key=root[0], leaf=root[1], parent_key=None, child_index=None)
        node = self._local.pop(self._local.root_key)
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)
        path_nodes[level] = node

        # Descend to the leaf, batching the path child + one sibling into one round per level.
        while not self._local.is_leaf_node(node):
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
            child_key, child_leaf = node.value.values[child_index]

            # Pick one sibling (prefer left, else right) to pre-fetch in the same round.
            sibling_index: int | None = None
            if child_index > 0:
                sibling_index = child_index - 1
            elif child_index < len(node.value.values) - 1:
                sibling_index = child_index + 1

            nodes_to_fetch: list[tuple[Any, int]] = [(child_key, child_leaf)]
            sib_key = sib_new_leaf = None
            if sibling_index is not None:
                sib_key, sib_leaf = node.value.values[sibling_index]
                sib_new_leaf = self._get_new_leaf()
                nodes_to_fetch.append((sib_key, sib_leaf))

            # One batched round fetches the child (+ sibling) together.
            self._batch_move_nodes_to_local(nodes_to_fetch)

            child_node = self._local.pop(child_key)
            node.value.values[child_index] = (child_key, new_leaf)
            child_node.leaf = new_leaf

            if sibling_index is not None:
                assert sib_new_leaf is not None
                sibling_node = self._local.pop(sib_key)
                sibling_node.leaf = sib_new_leaf
                node.value.values[sibling_index] = (sib_key, sib_new_leaf)
                siblings[level + 1] = sibling_node
                sibling_indices[level + 1] = sibling_index

            level += 1
            path_nodes[level] = child_node
            node = child_node

        return path_nodes, child_indices, siblings, sibling_indices

    @override
    def search(self, key: Any, value: Any = None) -> Any:
        """Flush local first so cache hits land in the stash, then run the base streaming search."""
        self._flush_local_to_stash()
        return super().search(key=key, value=value)

    @override
    def insert(self, key: Any, value: Any = None) -> None:
        if key is None:
            self._perform_dummy_operation(num_round=self._max_height)
            return

        # Empty tree: the new block is the root.
        if self.root is None:
            data_block = self._get_bplus_data(keys=[key], values=[value])
            self._stash.append(data_block)
            # A fresh block always has a sampled leaf.
            assert data_block.leaf is not None
            self.root = (data_block.key, data_block.leaf)
            self._perform_dummy_operation(num_round=self._max_height)
            return

        # Flush cached local first so the base traversal's empty-local guard passes; the overridden
        # _move_node_to_local still serves cache hits while loading the path to the leaf.
        self._flush_local_to_stash()
        self._find_leaf_to_local(key=key)

        num_retrieved_nodes = len(self._local)
        self._insert_into_loaded_leaf(key=key, value=value)

        # Splits make keeping nodes in local complex, so flush, then pad by retrieved count.
        self._flush_local_to_stash()
        self._perform_dummy_operation(num_round=self._max_height - num_retrieved_nodes)

    @override
    def delete(self, key: Any) -> Any:
        """Delete in h interaction rounds: each level's path child and one sibling are fetched together
        in a single batched read-write round (see ``_find_path_with_siblings_cached`` /
        ``_batch_move_nodes_to_local``), so the op is h rounds (1 root + h-1 batched) regardless of the
        borrow/merge outcome -- padded to the max-height budget. Bandwidth is unchanged (~2h-1 paths);
        only the round-trip count drops to h. Cache hits that skip a real round shift onto dummy padding,
        so the total stays a fixed h rounds."""
        self._op_rounds = 0
        budget = self._max_height

        # A dummy/empty/missing delete removes nothing; pad to h rounds.
        if self._short_circuit_read(key=key, num_round=budget):
            return None

        # Descend, batching the path child + one sibling into a single round per level.
        path_nodes, child_indices, siblings, sibling_indices = self._find_path_with_siblings_cached(key=key)

        # Remove the key and rebalance with the pre-fetched siblings (no extra rounds), then flush the
        # survivors to stash -- the cached delete keeps nothing in local.
        deleted_value = self._apply_delete(
            key=key,
            path_nodes=path_nodes,
            child_indices=child_indices,
            siblings=siblings,
            sibling_indices=sibling_indices,
        )
        for node in path_nodes.values():
            self._stash.append(node)
        for sib in siblings.values():
            self._stash.append(sib)

        # Pad the real rounds (root + one per descended level) up to the h-round budget.
        self._pad_to(budget)

        return deleted_value
