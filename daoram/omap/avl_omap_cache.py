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
    3. Reduced dummy operations for insert/search (single tree height traversal)
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
            name=name,
            client=client,
            num_data=num_data,
            key_size=key_size,
            filename=filename,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            encryptor=encryptor
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
        """Insert with caching: flush at start, keep in local at end."""
        if key is None:
            self._perform_dummy_operation(num_round=self._max_height + 1)
            return

        data_block = self._get_avl_data(key=key, value=value)

        if self.root is None:
            self._stash.append(data_block)
            self.root = (data_block.key, data_block.leaf)
            self._perform_dummy_operation(num_round=self._max_height + 1)
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
        self._perform_dummy_operation(num_round=self._max_height + 1 - num_retrieved)

    def search(self, key: Any, value: Any = None) -> Any:
        """Search with caching: flush at start, keep in local at end."""
        if key is None or self.root is None:
            self._perform_dummy_operation(num_round=self._max_height + 1)
            return None

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
        self._perform_dummy_operation(num_round=self._max_height + 1 - num_retrieved)

        return search_value

    def fast_search(self, key: Any, value: Any = None) -> Any:
        """Fast search: flush local first so parent can find nodes in stash."""
        self._flush_local_to_stash()
        return super().fast_search(key=key, value=value)

    def delete(self, key: Any) -> Any:
        """Delete with caching: flush at start, keep in local at end."""
        if self.root is None:
            self._perform_dummy_operation(num_round=self._max_height + 1)
            return None

        # Flush cached nodes to stash at start
        self._flush_local_to_stash()

        # Use parent's delete logic (which now uses our cached _move_node_to_local)
        # But we need to handle the "no flush at end" differently

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
                    self._perform_dummy_operation(num_round=self._max_height + 1 - num_retrieved)
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
                    self._perform_dummy_operation(num_round=self._max_height + 1 - num_retrieved)
                    return None

        deleted_value = node.value.value
        node_key = node.key
        parent_key = self._local.get_parent_key(node_key)

        # Case 1: No children
        if node.value.l_key is None and node.value.r_key is None:
            if len(self._local) == 1:
                self.root = None
                self._local.clear()
                self._perform_dummy_operation(num_round=self._max_height)
                return deleted_value
            self._local.update_child_in_parent(parent_key, node_key, None, None, 0)
            self._local.remove(node_key)

        # Case 2: One child
        elif node.value.l_key is None or node.value.r_key is None:
            has_left = node.value.l_key is not None
            child_key = node.value.l_key if has_left else node.value.r_key
            child_leaf = node.value.l_leaf if has_left else node.value.r_leaf
            child_height = node.value.l_height if has_left else node.value.r_height

            if len(self._local) == 1:
                self.root = (child_key, child_leaf)
                self._local.clear()
                self._perform_dummy_operation(num_round=self._max_height)
                return deleted_value
            self._local.update_child_in_parent(parent_key, node_key, child_key, child_leaf, child_height)
            self._local.remove(node_key)

        # Case 3: Two children
        else:
            use_predecessor = node.value.l_height > node.value.r_height
            original_key = node.key

            traverse_key = node.value.l_key if use_predecessor else node.value.r_key
            traverse_leaf = node.value.l_leaf if use_predecessor else node.value.r_leaf
            self._move_node_to_local(key=traverse_key, leaf=traverse_leaf, parent_key=node_key)
            current = self._local.get(traverse_key)
            parent_of_replacement_key = node_key

            next_key = current.value.r_key if use_predecessor else current.value.l_key
            while next_key is not None:
                next_leaf = current.value.r_leaf if use_predecessor else current.value.l_leaf
                self._move_node_to_local(key=next_key, leaf=next_leaf, parent_key=traverse_key)
                parent_of_replacement_key = traverse_key
                traverse_key = next_key
                current = self._local.get(traverse_key)
                next_key = current.value.r_key if use_predecessor else current.value.l_key

            replacement_node = current
            node.key = replacement_node.key
            node.value.value = replacement_node.value.value

            child_key = replacement_node.value.l_key if use_predecessor else replacement_node.value.r_key
            child_leaf = replacement_node.value.l_leaf if use_predecessor else replacement_node.value.r_leaf
            child_height = (
                replacement_node.value.l_height if use_predecessor else replacement_node.value.r_height
            ) if child_key else 0
            self._local.update_child_in_parent(
                parent_of_replacement_key, replacement_node.key, child_key, child_leaf, child_height
            )
            self._local.remove(replacement_node.key)
            self._local.update_child_in_parent(parent_key, original_key, node.key, node.leaf, 0)
            self._local.replace_node_key(original_key, node.key)

        self._update_height()
        self._local.update_all_leaves(self._get_new_leaf)
        self._balance_local(is_delete=True)

        # Dummy ops only - local stays cached for next operation
        num_retrieved = len(self._local)
        self._perform_dummy_operation(num_round=2 * self._max_height + 1 - num_retrieved)

        return deleted_value
