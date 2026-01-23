"""AVL OMAP with caching optimization for repeated accesses."""

from typing import Any

from daoram.dependency import Encryptor, InteractServer
from daoram.omap import AVLOmap


class AVLOmapCached(AVLOmap):
    """
    AVL OMAP with stash caching optimization.

    Key differences from AVLOmap:
    1. Checks stash before fetching from server (cache hit avoids ORAM access)
    2. Nodes stay in local at end of operation, flushed to stash at start of next
    3. Reduced interaction rounds: insert/search use h rounds, delete uses 2h rounds
    """

    def __init__(self,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 client: InteractServer,
                 name: str = "avl_opt",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 encryptor: Encryptor = None):
        super().__init__(
            num_data=num_data,
            key_size=key_size,
            data_size=data_size,
            client=client,
            name=name,
            filename=filename,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            encryptor=encryptor,
        )

    def _move_node_to_local(self, key: Any, leaf: int, parent_key: Any = None) -> None:
        """
        Override to check stash first before fetching from server.

        This is the key caching mechanism - if a node is in stash from a previous
        operation, we use it directly without an ORAM server access.
        """
        stash_idx = self._find_in_stash(key)
        if stash_idx >= 0:
            # Cache hit - use node from stash (no ORAM access)
            node = self._stash.pop(stash_idx)
            self._local.add(node=node, parent_key=parent_key)
        else:
            # Cache miss - fetch from server
            super()._move_node_to_local(key=key, leaf=leaf, parent_key=parent_key)

    def insert(self, key: Any, value: Any = None) -> None:
        """
        Given key-value pair, insert the pair to the tree with caching.

        :param key: The search key of interest.
        :param value: The value to insert.
        """
        if key is None:
            self._perform_dummy_operation(num_round=self._max_height)
            return

        data_block = self._get_avl_data(key=key, value=value)

        if self.root is None:
            self._stash.append(data_block)
            self.root = (data_block.key, data_block.leaf)
            self._perform_dummy_operation(num_round=self._max_height)
            return

        # Flush cached nodes to stash at start
        self._flush_local_to_stash()

        # Get root and traverse (uses overridden _move_node_to_local for cache benefit)
        self._move_node_to_local(key=self.root[0], leaf=self.root[1], parent_key=None)
        current_key = self.root[0]

        while True:
            node = self._local.get(current_key)
            if node.key < key:
                if node.value.r_key is not None:
                    self._move_node_to_local(key=node.value.r_key, leaf=node.value.r_leaf, parent_key=current_key)
                    current_key = node.value.r_key
                else:
                    node.value.r_key = data_block.key
                    break
            else:
                if node.value.l_key is not None:
                    self._move_node_to_local(key=node.value.l_key, leaf=node.value.l_leaf, parent_key=current_key)
                    current_key = node.value.l_key
                else:
                    node.value.l_key = data_block.key
                    break

        self._local.add(node=data_block, parent_key=current_key)
        self._update_height()
        self._local.update_all_leaves(self._get_new_leaf)
        self._balance_local()

        # Dummy ops only - local stays cached for next operation
        num_retrieved = len(self._local)
        self._perform_dummy_operation(num_round=self._max_height - num_retrieved)

    def search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value with caching.

        If the input value is not None, the value corresponding to the search tree will be updated.
        :param key: The search key of interest.
        :param value: The updated value.
        :return: The (old) value corresponding to the search key.
        """
        if key is None:
            self._perform_dummy_operation(num_round=self._max_height)
            return None

        # If the current root is empty, we can't perform search.
        if self.root is None:
            raise ValueError("Cannot search in an empty tree.")

        # Flush cached nodes to stash at start
        self._flush_local_to_stash()

        self._move_node_to_local(key=self.root[0], leaf=self.root[1], parent_key=None)
        current_key = self.root[0]

        node = self._local.get(current_key)
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
            node = self._local.get(current_key)

        search_value = node.value.value if node.key == key else None
        if value is not None and node.key == key:
            node.value.value = value

        self._local.update_all_leaves(self._get_new_leaf)
        root_node = self._local.get_root()
        self.root = (root_node.key, root_node.leaf)

        # Dummy ops only - local stays cached for next operation
        num_retrieved = len(self._local)
        self._perform_dummy_operation(num_round=self._max_height - num_retrieved)

        return search_value

    def fast_search(self, key: Any, value: Any = None) -> Any:
        """Fast search is identical to search in cached version."""
        return self.search(key=key, value=value)

    def delete(self, key: Any) -> Any:
        """Delete with caching: flush at start, keep in local at end."""
        if self.root is None:
            raise ValueError("Cannot delete from an empty tree.")

        # Flush cached nodes to stash at start
        self._flush_local_to_stash()

        self._move_node_to_local(key=self.root[0], leaf=self.root[1], parent_key=None)
        current_key = self.root[0]
        node = self._local.get(current_key)

        # Find node to delete
        while node.key != key:
            if node.key < key:
                if node.value.r_key is not None:
                    self._move_node_to_local(key=node.value.r_key, leaf=node.value.r_leaf, parent_key=current_key)
                    current_key = node.value.r_key
                    node = self._local.get(current_key)
                else:
                    # Key not found
                    self._local.update_all_leaves(self._get_new_leaf)
                    root_node = self._local.get_root()
                    self.root = (root_node.key, root_node.leaf)
                    num_retrieved = len(self._local)
                    self._perform_dummy_operation(num_round=2 * self._max_height - num_retrieved)
                    return None
            else:
                if node.value.l_key is not None:
                    self._move_node_to_local(key=node.value.l_key, leaf=node.value.l_leaf, parent_key=current_key)
                    current_key = node.value.l_key
                    node = self._local.get(current_key)
                else:
                    # Key not found
                    self._local.update_all_leaves(self._get_new_leaf)
                    root_node = self._local.get_root()
                    self.root = (root_node.key, root_node.leaf)
                    num_retrieved = len(self._local)
                    self._perform_dummy_operation(num_round=2 * self._max_height - num_retrieved)
                    return None

        # At this point, node contains the key to delete
        node_key = node.key
        parent_key = self._local.get_parent_key(node_key)

        # Perform the deletion using the helper
        deleted_value, early_returned = self._delete_node_from_local(node, node_key, parent_key)
        if early_returned:
            self._perform_dummy_operation(num_round=2 * self._max_height)
            return deleted_value

        self._update_height()
        self._local.update_all_leaves(self._get_new_leaf)
        self._balance_local(is_delete=True)

        # Dummy ops only - local stays cached for next operation
        num_retrieved = len(self._local)
        self._perform_dummy_operation(num_round=2 * self._max_height - num_retrieved)

        return deleted_value
