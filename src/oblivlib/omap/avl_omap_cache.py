"""AVL OMAP with caching optimization for repeated accesses."""

from typing import Any, override

from oblivlib.dependency.config import AvlOmapCachedConfig
from oblivlib.omap.avl_omap import AVLOmap


class AVLOmapCached(AVLOmap):
    """AVL OMAP with stash caching. Versus ``AVLOmap``: a cache hit (node already in stash) skips the
    ORAM access; insert/search/delete keep the visited path in local, flushed at the start of the next
    op, so nearby repeated accesses reuse cached nodes; padding is a single tree-height traversal.

    Unlike ``BPlusOmapCached.search`` (which streams and keeps nothing), this ``search`` retains the
    path in local for reuse. Both cached variants are intentionally NOT per-op oblivious -- caching
    makes the access pattern data-dependent."""

    def __init__(self, config: AvlOmapCachedConfig):
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

    @override
    def search(self, key: Any, value: Any = None) -> Any:
        # A dummy op (key is None) or an empty tree finds nothing.
        if self._short_circuit_read(key=key, num_round=self._max_height):
            return None

        # Flush cached nodes to stash at start, then run the shared descent.
        self._flush_local_to_stash()
        current_key = self._descend_to_key(key=key)
        node = self._local.require(current_key)

        search_value = node.value.value if node.key == key else None
        if value is not None and node.key == key:
            node.value.value = value

        self._local.update_all_leaves(self._get_new_leaf)
        # After a search the tree is non-empty.
        root_node = self._local.require_root()
        self.root = (root_node.key, root_node.leaf)

        # Dummy ops only -- local stays cached for the next op.
        num_retrieved = len(self._local)
        self._perform_dummy_operation(num_round=self._max_height - num_retrieved)

        return search_value

    @override
    def insert(self, key: Any, value: Any = None) -> None:
        if key is None:
            self._perform_dummy_operation(num_round=self._max_height)
            return

        data_block = self._get_avl_data(key=key, value=value)

        if self.root is None:
            self._stash.append(data_block)
            # A fresh block always has a sampled leaf.
            assert data_block.leaf is not None
            self.root = (data_block.key, data_block.leaf)
            self._perform_dummy_operation(num_round=self._max_height)
            return

        # Flush cached nodes to stash at start, then run the shared insert algorithm.
        self._flush_local_to_stash()
        self._descend_and_link(key=key, data_block=data_block)
        self._post_op_fixup()

        # Dummy ops only -- local stays cached for the next op.
        num_retrieved = len(self._local)
        self._perform_dummy_operation(num_round=self._max_height - num_retrieved)

    @override
    def delete(self, key: Any) -> Any:
        # A dummy/empty/missing/found delete all pad to the same 2h total, so they are indistinguishable.
        if self._short_circuit_read(key=key, num_round=2 * self._max_height):
            return None

        # Flush cached nodes to stash at start, then run the shared descent.
        self._flush_local_to_stash()
        current_key = self._descend_to_key(key=key)
        node = self._local.require(current_key)

        # Miss: re-leaf the cached path, keep it local, and pad by retrieved count.
        if node.key != key:
            self._local.update_all_leaves(self._get_new_leaf)
            root_node = self._local.require_root()
            self.root = (root_node.key, root_node.leaf)
            num_retrieved = len(self._local)
            self._perform_dummy_operation(num_round=2 * self._max_height - num_retrieved)
            return None

        deleted_value, done = self._delete_at_node(node_key=current_key)
        if done:
            # The single-node tree case already set the new root and cleared local.
            self._perform_dummy_operation(num_round=2 * self._max_height)
            return deleted_value

        self._post_op_fixup(is_delete=True)

        # Dummy ops only -- local stays cached for the next op.
        num_retrieved = len(self._local)
        self._perform_dummy_operation(num_round=2 * self._max_height - num_retrieved)

        return deleted_value
