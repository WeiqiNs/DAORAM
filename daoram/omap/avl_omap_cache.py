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
    # Todo: Do insert and search use h+1 rounds for indistinguishability?
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

        Uses per-level eviction with pre-allocation:
        - Before evicting parent nodes, pre-allocate new leaves for children we will visit
        - Update parent's l_leaf/r_leaf to point to the pre-allocated leaves
        - Save old child leaves for reading in next level
        - Evict parents (they go to intersection of read path and new leaf path)

        :param keys: List of keys to search for.
        :return: Dict mapping each key to its corresponding value (None if not found).
        """
        if not keys:
            return {}

        k = len(keys)
        results: Dict[Any, Any] = {key: None for key in keys}

        # Handle empty tree case
        if self.root is None:
            # Perform level-based dummy accesses to match batch search access pattern
            for level in range(self._max_height):
                max_nodes_at_level = 2 ** level
                required = min(max_nodes_at_level, k)
                if required <= 0:
                    continue

                leaves: List[int] = []
                while len(leaves) < required:
                    leaf = self._get_new_leaf()
                    if leaf not in leaves:
                        leaves.append(leaf)

                self._client.add_read_path(label=self._name, leaves=leaves)
                result = self._client.execute()
                path_data = result.results[self._name]

                # Decrypt the path and add all real data to stash
                path = self.decrypt_path_data(path=path_data)
                for bucket in path.values():
                    for data in bucket:
                        self._stash.append(data)

                if len(self._stash) > self._stash_size:
                    raise OverflowError(
                        f"Stash overflow in {self._name}: size {len(self._stash)} exceeds max {self._stash_size}.")

                evicted = self._evict_stash(leaves=leaves)
                self._client.add_write_path(label=self._name, data=evicted)
                self._client.execute()

            return results

        # Flush local to stash at start
        self._flush_local_to_stash()

        # Track keys we've seen in stash to avoid duplicates
        keys_in_stash: Set[Any] = {node.key for node in self._stash}

        # Save root's old leaf for reading, pre-allocate new leaf
        root_key = self.root[0]
        root_old_leaf = self.root[1]

        # For each key i:
        #   current_node[i] = (node_key, leaf_to_read) - use old leaf for reading
        #   finished[i] = whether we're done with this key
        current_node: List[Tuple[Any, int]] = [(root_key, root_old_leaf)] * k
        finished: List[bool] = [False] * k

        # Pre-allocated leaves: child_key -> new_leaf
        pre_allocated_leaves: Dict[Any, int] = {}

        # Old leaves for reading: child_key -> old_leaf (the leaf where the node actually is)
        old_leaves_for_reading: Dict[Any, int] = {}

        # Process level by level with per-level eviction
        for level in range(self._max_height):
            # Maximum possible nodes at this level in a balanced tree
            max_nodes_at_level = 2 ** level

            # Nodes retrieved this level
            level_local_nodes: Dict[Any, Data] = {}

            # Collect unique leaves needed for this level
            leaf_to_node_keys: Dict[int, Set[Any]] = {}
            for i in range(k):
                if finished[i]:
                    continue
                node_key, leaf = current_node[i]

                # Check if node is already retrieved this level
                if node_key in level_local_nodes:
                    continue

                # Check if node is in stash - if so, move to level_local
                stash_idx = self._find_in_stash(node_key)
                if stash_idx >= 0:
                    node = self._stash.pop(stash_idx)
                    level_local_nodes[node.key] = node
                    keys_in_stash.discard(node.key)
                    continue

                # Need to read from this leaf
                if leaf not in leaf_to_node_keys:
                    leaf_to_node_keys[leaf] = set()
                leaf_to_node_keys[leaf].add(node_key)

            unique_leaves = list(leaf_to_node_keys.keys())
            actual_reads_needed = len(unique_leaves)

            # Determine how many paths to read for obliviousness
            if max_nodes_at_level < k:
                dummy_count = max(0, max_nodes_at_level - actual_reads_needed)
            else:
                dummy_count = max(0, k - actual_reads_needed)

            # Perform batch read if needed
            leaves_this_level = []
            if unique_leaves or dummy_count > 0:
                # Add dummy leaves for padding
                all_leaves = list(unique_leaves)
                tmp = 0
                while tmp < dummy_count:
                    tmp_path = self._get_new_leaf()
                    if tmp_path not in all_leaves:
                        all_leaves.append(tmp_path)
                        tmp += 1

                leaves_this_level = all_leaves

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

                # Process path data - move target nodes to level_local, others to stash
                for bucket in path.values():
                    for data in bucket:
                        if data.key in level_local_nodes:
                            continue

                        stash_idx = self._find_in_stash(data.key)

                        if data.key in target_node_keys:
                            if stash_idx >= 0:
                                self._stash.pop(stash_idx)
                                keys_in_stash.discard(data.key)
                            level_local_nodes[data.key] = data
                        else:
                            if stash_idx >= 0:
                                # Refresh existing stash copy with latest data
                                self._stash[stash_idx] = data
                            else:
                                self._stash.append(data)
                                keys_in_stash.add(data.key)

            # Determine which child nodes will be visited next level
            children_to_visit: Set[Any] = set()
            for i in range(k):
                if finished[i]:
                    continue

                node_key, _ = current_node[i]
                node = level_local_nodes.get(node_key)
                if node is None:
                    stash_idx = self._find_in_stash(node_key)
                    if stash_idx >= 0:
                        node = self._stash.pop(stash_idx)
                        level_local_nodes[node.key] = node
                        keys_in_stash.discard(node.key)
                if node is None:
                    finished[i] = True
                    continue

                # Check if this is the target key
                if node.key == keys[i]:
                    results[keys[i]] = node.value.value
                    finished[i] = True
                    continue

                # Determine which child to visit and save its OLD leaf for reading
                if keys[i] > node.key:
                    if node.value.r_key is not None:
                        child_key = node.value.r_key
                        children_to_visit.add(child_key)
                        # Save the OLD leaf (where the child actually is) for reading
                        if child_key not in old_leaves_for_reading:
                            old_leaves_for_reading[child_key] = node.value.r_leaf
                    else:
                        finished[i] = True
                else:
                    if node.value.l_key is not None:
                        child_key = node.value.l_key
                        children_to_visit.add(child_key)
                        # Save the OLD leaf (where the child actually is) for reading
                        if child_key not in old_leaves_for_reading:
                            old_leaves_for_reading[child_key] = node.value.l_leaf
                    else:
                        finished[i] = True

            # Pre-allocate new leaves for children we'll visit
            for child_key in children_to_visit:
                if child_key not in pre_allocated_leaves:
                    pre_allocated_leaves[child_key] = self._get_new_leaf()

            # Update current_node for next level - use OLD leaf for reading!
            for i in range(k):
                if finished[i]:
                    continue

                node_key, _ = current_node[i]
                node = level_local_nodes.get(node_key)
                if node is None:
                    stash_idx = self._find_in_stash(node_key)
                    if stash_idx >= 0:
                        node = self._stash.pop(stash_idx)
                        level_local_nodes[node.key] = node
                        keys_in_stash.discard(node.key)
                if node is None:
                    finished[i] = True
                    continue

                if keys[i] > node.key:
                    if node.value.r_key is not None:
                        child_key = node.value.r_key
                        # Use the saved OLD leaf for reading
                        child_leaf = old_leaves_for_reading.get(child_key, node.value.r_leaf)
                        current_node[i] = (child_key, child_leaf)
                else:
                    if node.value.l_key is not None:
                        child_key = node.value.l_key
                        # Use the saved OLD leaf for reading
                        child_leaf = old_leaves_for_reading.get(child_key, node.value.l_leaf)
                        current_node[i] = (child_key, child_leaf)

            # Update parent pointers to use pre-allocated leaves (AFTER saving old leaves)
            for node in level_local_nodes.values():
                if node.value.l_key is not None and node.value.l_key in pre_allocated_leaves:
                    node.value.l_leaf = pre_allocated_leaves[node.value.l_key]
                if node.value.r_key is not None and node.value.r_key in pre_allocated_leaves:
                    node.value.r_leaf = pre_allocated_leaves[node.value.r_key]

            # Assign new leaves for nodes at this level
            for node in level_local_nodes.values():
                if node.key in pre_allocated_leaves:
                    # Use pre-allocated leaf from parent
                    node.leaf = pre_allocated_leaves[node.key]
                else:
                    # Root or node without pre-allocation - assign new leaf
                    node.leaf = self._get_new_leaf()

            # Move level_local nodes to stash for eviction
            for node in level_local_nodes.values():
                if self._find_in_stash(node.key) < 0:
                    self._stash.append(node)
                    keys_in_stash.add(node.key)

            # Update root pointer
            if self.root is not None and root_key in level_local_nodes:
                self.root = (root_key, level_local_nodes[root_key].leaf)

            # Evict this level using the paths we read
            if leaves_this_level:
                evicted = self._evict_stash(leaves=leaves_this_level)
                self._client.add_write_path(label=self._name, data=evicted)
                self._client.execute()

        return results
