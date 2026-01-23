"""AVL OMAP with caching optimization for repeated accesses."""

from typing import Any, Dict, List, Set, Tuple

from daoram.dependency import Data, Encryptor, InteractServer
from daoram.omap import AVLOmap


class AVLOmapCached(AVLOmap):
    """
    AVL OMAP with stash caching optimization.

    Key differences from AVLOmap:
    1. Checks stash before fetching from server (cache hit avoids ORAM access)
    2. Nodes stay in local at end of operation, flushed to stash at start of next
    3. Reduced interaction rounds: insert/search use h rounds, delete uses 2h rounds
    # Todo: Do insert and serch use h+1 rounds?
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

    def _find_node_in_stash_no_remove(self, node_key: Any) -> Data:
        """
        Find a node in stash by its key without removing it.

        :param node_key: The key of the node to find.
        :return: The Data node if found, None otherwise.
        """
        for node in self._stash:
            if node.key == node_key:
                return node
        return None

    def batch_search(self, keys: List[Any]) -> Dict[Any, Any]:
        """
        Search for multiple keys in the oblivious AVL tree using level-by-level reading.

        This approach optimizes reads based on the number of nodes at each level:
        - At level l, there are at most 2^l nodes in an AVL tree
        - If searching for k keys and 2^l >= k, read k paths (one per key)
        - If 2^l < k, read all 2^l nodes at that level (more efficient)

        This reduces communication compared to reading k full paths independently.

        :param keys: List of keys to search for.
        :return: Dict mapping each key to its corresponding value (None if not found).
        """
        if not keys:
            return {}

        k = len(keys)
        results: Dict[Any, Any] = {key: None for key in keys}

        # Handle empty tree case
        if self.root is None:
            self._perform_dummy_operation(num_round=self._max_height)
            return results

        # Flush local to stash at start
        self._flush_local_to_stash()

        # Track nodes retrieved during this operation
        local_nodes: Dict[Any, Data] = {}  # key -> Data node
        keys_in_stash: Set[Any] = {node.key for node in self._stash}
        all_leaves_read: List[int] = []

        # For each key i:
        #   current_node[i] = (node_key, leaf) - the node we need to visit next
        #   finished[i] = whether we're done with this key
        current_node: List[Tuple[Any, int]] = [(self.root[0], self.root[1])] * k
        finished: List[bool] = [False] * k

        # Process level by level - read paths but delay eviction until the end
        for level in range(self._max_height):
            # Maximum possible nodes at this level in a balanced tree
            max_nodes_at_level = 2 ** level

            # Collect unique leaves needed for this level
            leaf_to_node_keys: Dict[int, Set[Any]] = {}
            for i in range(k):
                if finished[i]:
                    continue
                node_key, leaf = current_node[i]

                # Check if node is already in local
                if node_key in local_nodes:
                    continue

                # Check if node is in stash - if so, move to local
                stash_idx = self._find_in_stash(node_key)
                if stash_idx >= 0:
                    node = self._stash.pop(stash_idx)
                    local_nodes[node.key] = node
                    keys_in_stash.discard(node.key)
                    continue

                # Need to read from this leaf
                if leaf not in leaf_to_node_keys:
                    leaf_to_node_keys[leaf] = set()
                leaf_to_node_keys[leaf].add(node_key)

            unique_leaves = list(leaf_to_node_keys.keys())
            actual_reads_needed = len(unique_leaves)

            # Determine how many paths to read for obliviousness
            # If max_nodes_at_level < k, we read all possible paths at this level
            # Otherwise, we read k paths (padding with dummies if needed)
            if max_nodes_at_level < k:
                # Read all nodes at this level - dummy padding to max_nodes_at_level
                dummy_count = max(0, max_nodes_at_level - actual_reads_needed)
            else:
                # Read k paths - dummy padding to k
                dummy_count = max(0, k - actual_reads_needed)

            # Perform batch read if needed (WITHOUT eviction - delay until end)
            if unique_leaves or dummy_count > 0:
                # Add dummy leaves for padding
                all_leaves = list(unique_leaves)
                for _ in range(dummy_count):
                    all_leaves.append(self._get_new_leaf())

                all_leaves_read.extend(all_leaves)

                # Read paths from server
                self._client.add_read_path(label=self._name, leaves=all_leaves)
                result = self._client.execute()
                path_data = result.results[self._name]

                # Decrypt the path
                path = self.decrypt_path_data(path=path_data)

                # Collect target node keys we're looking for
                target_node_keys: Set[Any] = set()
                for node_keys in leaf_to_node_keys.values():
                    target_node_keys.update(node_keys)

                # Process path data - move target nodes to local, others to stash
                for bucket in path.values():
                    for data in bucket:
                        if data.key in target_node_keys and data.key not in local_nodes:
                            local_nodes[data.key] = data
                        elif data.key not in keys_in_stash:
                            self._stash.append(data)
                            keys_in_stash.add(data.key)

            # Process each key - determine next node to visit
            for i in range(k):
                if finished[i]:
                    continue

                node_key, _ = current_node[i]

                # Find node in local first, then stash
                node = local_nodes.get(node_key)
                if node is None:
                    node = self._find_node_in_stash_no_remove(node_key)
                if node is None:
                    # Node not found - key doesn't exist
                    finished[i] = True
                    continue

                # Check if this is the target key
                if node.key == keys[i]:
                    results[keys[i]] = node.value.value
                    finished[i] = True

                # Navigate right if key > node.key
                elif keys[i] > node.key:
                    if node.value.r_key is not None:
                        current_node[i] = (node.value.r_key, node.value.r_leaf)
                    else:
                        finished[i] = True  # Key not in tree

                # Navigate left if key < node.key
                else:
                    if node.value.l_key is not None:
                        current_node[i] = (node.value.l_key, node.value.l_leaf)
                    else:
                        finished[i] = True  # Key not in tree

        # Move local nodes back to stash for final processing
        for node in local_nodes.values():
            if self._find_in_stash(node.key) < 0:
                self._stash.append(node)

        # Re-randomize leaves for all nodes in stash and update parent-child pointers
        key_to_node: Dict[Any, Data] = {node.key: node for node in self._stash}
        for node in self._stash:
            node.leaf = self._get_new_leaf()

        # Fix parent-child leaf pointers
        for node in self._stash:
            if node.value.l_key is not None and node.value.l_key in key_to_node:
                node.value.l_leaf = key_to_node[node.value.l_key].leaf
            if node.value.r_key is not None and node.value.r_key in key_to_node:
                node.value.r_leaf = key_to_node[node.value.r_key].leaf

        # Update root pointer
        if self.root is not None:
            root_key = self.root[0]
            if root_key in key_to_node:
                self.root = (root_key, key_to_node[root_key].leaf)

        # Final eviction - evict to all paths we read
        if all_leaves_read:
            unique_leaves_for_eviction = list(set(all_leaves_read))
            evicted = self._evict_stash(leaves=unique_leaves_for_eviction)
            self._client.add_write_path(label=self._name, data=evicted)
            self._client.execute()

        return results
