"""AVL OMAP with caching optimization for repeated accesses."""

from typing import Any, Dict, List, Set, Tuple

from daoram.dependency import Data, Encryptor, InteractServer
from daoram.omap import AVLOmap
from daoram.oram import MulPathOram


class AVLOmapCached(AVLOmap):
    """
    AVL OMAP with stash caching optimization and internal meta ORAM.

    Key differences from AVLOmap:
    1. Checks stash before fetching from server (cache hit avoids ORAM access)
    2. Nodes stay in local at end of operation, flushed to stash at start of next
    3. Reduced interaction rounds: insert/search use h rounds, delete uses 2h rounds
    4. Internal meta ORAM for delayed duplication (Graph ORAM -> PosMap updates)
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
                 encryptor: Encryptor = None,
                 enable_meta: bool = False):
        """
        Initialize AVLOmapCached with optional internal meta ORAM.
        
        :param enable_meta: If True, create an internal meta ORAM for delayed duplication.
                           This is used for Graph ORAM -> PosMap updates.
        """
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
        
        # Internal meta ORAM for delayed duplication
        self._enable_meta = enable_meta
        self._meta: MulPathOram = None
        if enable_meta:
            self._meta = MulPathOram(
                num_data=num_data,
                data_size=data_size,
                client=client,
                name=f"{name}_meta",
                filename=None,  # Meta ORAM uses memory storage
                bucket_size=bucket_size,
                stash_scale=stash_scale,
                encryptor=encryptor,
            )

    def init_server_storage(self, data=None) -> None:
        """
        Initialize the server storage for AVLOmapCached and its internal meta ORAM.
        
        :param data: A list of key-value pairs for the AVL OMAP.
        """
        # Initialize parent AVL OMAP storage.
        super().init_server_storage(data=data)
        
        # Initialize internal meta ORAM storage if enabled.
        if self._enable_meta and self._meta is not None:
            self._meta.init_server_storage()

    def add_meta_duplications(self, duplications: List[Data]) -> None:
        """
        Add duplications to the internal meta ORAM stash.
        
        Duplications are used for delayed updates (Graph ORAM -> PosMap).
        Format: Data(key=vertex_key, leaf=pos_leaf, value=new_graph_leaf)
        
        New duplications are added to the FRONT of stash (highest priority).
        
        :param duplications: List of Data objects representing delayed updates.
        """
        if not self._enable_meta or self._meta is None:
            return
        # Prepend new duplications (highest priority)
        self._meta.stash = duplications + self._meta.stash
    
    def meta_de_duplication(self) -> None:
        """
        Remove duplicate entries from meta ORAM stash.
        
        For each key, keeps only the FIRST dup encountered (highest priority).
        Priority: stash order (front = highest, from process_read_result).
        """
        if not self._enable_meta or self._meta is None:
            return
        seen_keys = set()
        temp_stash = []
        for data in self._meta.stash:
            if data.key not in seen_keys:
                seen_keys.add(data.key)
                temp_stash.append(data)
        self._meta.stash = temp_stash

    def _flush_local_to_stash(self) -> None:
        """
        Override to apply meta duplications to stash nodes before flush.
        
        This ensures that any pending dups in _meta.stash are applied to their
        corresponding nodes in _stash before those nodes are potentially written
        back to storage.
        """
        # First, apply meta duplications to nodes in _stash
        if self._enable_meta and self._meta is not None:
            temp_meta_stash = []
            for dup in self._meta.stash:
                applied = False
                for node in self._stash:
                    if node.key == dup.key:
                        node.value.value = dup.value
                        applied = True
                        break
                if not applied:
                    # Dup target not in stash, keep for later
                    temp_meta_stash.append(dup)
            self._meta.stash = temp_meta_stash
        
        # Then do the normal flush
        super()._flush_local_to_stash()

    def _move_node_to_local(self, key: Any, leaf: int, parent_key: Any = None) -> None:
        """
        Move a node to local cache. Always performs server access for obliviousness.
        
        For obliviousness: whether the node is in stash (cache hit) or not (cache miss),
        we ALWAYS read/write the same paths to the server. This ensures the server
        cannot distinguish between cache hits and misses.
        """
        # Check if node is in stash first
        stash_idx = self._find_in_stash(key)
        cache_hit = stash_idx >= 0
        
        # ALWAYS read from server (for obliviousness)
        # Read AVL ORAM path
        self._client.add_read_path(label=self._name, leaves=[leaf])
        # Also read meta ORAM path to get evicted dups
        if self._enable_meta and self._meta is not None:
            self._client.add_read_path(label=self._meta._name, leaves=[leaf])
        
        result = self._client.execute()
        path_data = result.results[self._name]
        
        # Process AVL ORAM path
        found_in_path = False
        to_index = len(self._stash)
        path = self.decrypt_path_data(path=path_data)
        for bucket in path.values():
            for data in bucket:
                if data.key == key:
                    if cache_hit:
                        # Node already in stash, add path version to stash (will be deduped later)
                        self._stash.append(data)
                    else:
                        # Node not in stash, add to local
                        self._local.add(node=data, parent_key=parent_key)
                        found_in_path = True
                else:
                    self._stash.append(data)
        
        if len(self._stash) > self._stash_size:
            raise OverflowError(f"Stash overflow in {self._name}")
        
        # Process meta ORAM - add dups from storage to _meta.stash
        if self._enable_meta and self._meta is not None and self._meta._name in result.results:
            meta_path_data = result.results[self._meta._name]
            meta_path = self._meta.decrypt_path_data(path=meta_path_data)
            for bucket in meta_path.values():
                for data in bucket:
                    if data.key is not None:
                        self._meta.stash.append(data)
        
        # Now handle where the node actually comes from
        if cache_hit:
            # Use node from stash (original position)
            node = self._stash.pop(stash_idx)
            self._local.add(node=node, parent_key=parent_key)
        elif not found_in_path:
            # Not in path, check stash (before path data was added)
            stash_idx = self._find_in_stash(key)
            if 0 <= stash_idx < to_index:
                self._local.add(node=self._stash[stash_idx], parent_key=parent_key)
                del self._stash[stash_idx]
            else:
                raise KeyError(f"The search key {key} is not found.")
        
        # Apply pending meta dup for this node
        self._apply_meta_dup_to_local_node(key)
        
        # ALWAYS write back (for obliviousness)
        # Evict and write AVL ORAM
        evicted_path = self._evict_stash(leaves=[leaf])
        self._client.add_write_path(label=self._name, data=evicted_path)
        # Evict and write meta ORAM
        if self._enable_meta and self._meta is not None:
            meta_evicted = self._meta._evict_stash(leaves=[leaf])
            self._client.add_write_path(label=self._meta._name, data=meta_evicted)
        
        self._client.execute()
    
    def _apply_meta_dup_to_local_node(self, key: Any) -> None:
        """Apply pending meta dup for a node in _local from _meta.stash."""
        if not self._enable_meta or self._meta is None:
            return
        node = self._local.get(key)
        if node is None:
            return
        temp_stash = []
        for dup in self._meta.stash:
            if dup.key == key:
                node.value.value = dup.value
            else:
                temp_stash.append(dup)
        self._meta.stash = temp_stash
    
    def _sync_meta_dup_leaves(self) -> None:
        """
        Sync _meta.stash dup.leaf with the corresponding node's new leaf in _local.
        
        This must be called after update_all_leaves() to ensure that any remaining
        dups in _meta.stash have their leaf updated to match the node's new leaf.
        Otherwise, the dup would be written to the old path and become unreachable.
        """
        if not self._enable_meta or self._meta is None:
            return
        
        for dup in self._meta.stash:
            node = self._local.get(dup.key)
            if node is not None:
                # Update dup.leaf to match node's new leaf
                dup.leaf = node.leaf

    def insert(self, key: Any, value: Any = None, return_pos_leaf: bool = False) -> Any:
        """
        Given key-value pair, insert the pair to the tree with caching.

        :param key: The search key of interest.
        :param value: The value to insert.
        :param return_pos_leaf: If True, return the node's leaf in the underlying ORAM.
        :return: None, or pos_leaf if return_pos_leaf is True.
        """
        if key is None:
            self._perform_dummy_operation(num_round=self._max_height)
            return None if return_pos_leaf else None

        data_block = self._get_avl_data(key=key, value=value)

        if self.root is None:
            self._stash.append(data_block)
            self.root = (data_block.key, data_block.leaf)
            self._perform_dummy_operation(num_round=self._max_height)
            return data_block.leaf if return_pos_leaf else None

        # Flush cached nodes to stash at start
        self._flush_local_to_stash()

        # Get root and traverse (uses overridden _move_node_to_local for cache benefit)
        self._move_node_to_local(key=self.root[0], leaf=self.root[1], parent_key=None)
        current_key = self.root[0]
        target_node = None  # Track the node for pos_leaf

        while True:
            node = self._local.get(current_key)
            if node.key == key:
                # Key already exists - update value instead of inserting duplicate.
                node.value.value = value
                target_node = node
                # Update height and leaves, then balance (same as normal insert).
                self._update_height()
                self._local.update_all_leaves(self._get_new_leaf)
                self._sync_meta_dup_leaves()  # Sync dup.leaf with node's new leaf
                self._balance_local()
                num_retrieved = len(self._local)
                self._perform_dummy_operation(num_round=self._max_height - num_retrieved)
                return target_node.leaf if return_pos_leaf else None
            elif node.key < key:
                if node.value.r_key is not None:
                    self._move_node_to_local(key=node.value.r_key, leaf=node.value.r_leaf, parent_key=current_key)
                    current_key = node.value.r_key
                else:
                    node.value.r_key = data_block.key
                    target_node = data_block
                    break
            else:
                if node.value.l_key is not None:
                    self._move_node_to_local(key=node.value.l_key, leaf=node.value.l_leaf, parent_key=current_key)
                    current_key = node.value.l_key
                else:
                    node.value.l_key = data_block.key
                    target_node = data_block
                    break

        self._local.add(node=data_block, parent_key=current_key)
        self._update_height()
        self._local.update_all_leaves(self._get_new_leaf)
        self._sync_meta_dup_leaves()  # Sync dup.leaf with node's new leaf
        self._balance_local()

        # Dummy ops only - local stays cached for next operation
        num_retrieved = len(self._local)
        self._perform_dummy_operation(num_round=self._max_height - num_retrieved)
        
        return target_node.leaf if return_pos_leaf else None

    def search(self, key: Any, value: Any = None, return_pos_leaf: bool = False) -> Any:
        """
        Given a search key, return its corresponding value with caching.

        If the input value is not None, the value corresponding to the search tree will be updated.
        :param key: The search key of interest.
        :param value: The updated value.
        :param return_pos_leaf: If True, also return the node's leaf in the underlying ORAM.
        :return: The (old) value, or (value, pos_leaf) if return_pos_leaf is True.
        """
        if key is None:
            self._perform_dummy_operation(num_round=self._max_height)
            return (None, None) if return_pos_leaf else None

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

        found = node.key == key
        search_value = node.value.value if found else None
        if value is not None and found:
            node.value.value = value

        self._local.update_all_leaves(self._get_new_leaf)
        self._sync_meta_dup_leaves()  # Sync dup.leaf with node's new leaf
        
        # Get the pos_leaf after leaf reassignment
        pos_leaf = node.leaf if found else None
        
        root_node = self._local.get_root()
        self.root = (root_node.key, root_node.leaf)

        # Dummy ops only - local stays cached for next operation
        num_retrieved = len(self._local)
        self._perform_dummy_operation(num_round=self._max_height - num_retrieved)

        if return_pos_leaf:
            return (search_value, pos_leaf)
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
                    self._sync_meta_dup_leaves()  # Sync dup.leaf with node's new leaf
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
                    self._sync_meta_dup_leaves()  # Sync dup.leaf with node's new leaf
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
        self._sync_meta_dup_leaves()  # Sync dup.leaf with node's new leaf
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

    def batch_search(self, keys: List[Any], return_pos_leaf: bool = False,
                     return_visited_nodes: bool = False) -> Dict[Any, Any]:
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
        :param return_pos_leaf: If True, return (value, pos_leaf) tuples.
        :param return_visited_nodes: If True, also return all visited nodes' info.
        :return: Dict mapping each key to its corresponding value (None if not found).
                 If return_pos_leaf is True, returns {key: (value, pos_leaf)}.
                 If return_visited_nodes is True, returns:
                   (results, visited_nodes_map, total_paths_read)
                 where visited_nodes_map is {node_key: (new_pos_leaf, graph_leaf)}
                 containing both the new pos_leaf and the stored graph_leaf for all visited nodes.
        """
        if not keys:
            return ({}, {}, 0) if return_visited_nodes else {}

        k = len(keys)
        results: Dict[Any, Any] = {key: None for key in keys}
        # Track pos_leaf for each key if needed
        pos_leaf_map: Dict[Any, int] = {} if return_pos_leaf else None
        # Track all visited nodes and their new pos_leaf if needed
        visited_nodes_map: Dict[Any, int] = {} if return_visited_nodes else None
        # Track total paths read for obliviousness calculation
        total_paths_read: int = 0 if return_visited_nodes else None

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

            return (results, {}, 0) if return_visited_nodes else results

        # Flush local to stash at start
        self._flush_local_to_stash()
        
        # Keep pending meta duplications in stash - they will be applied during search
        # when we encounter nodes with matching keys (no need to write then re-read)

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
                
                # Track total paths read for obliviousness
                if total_paths_read is not None:
                    total_paths_read += len(all_leaves)

                # Read paths from server (and meta ORAM if enabled)
                self._client.add_read_path(label=self._name, leaves=all_leaves)
                if self._enable_meta and self._meta is not None:
                    self._meta.queue_read(leaves=all_leaves)
                result = self._client.execute()
                path_data = result.results[self._name]

                # Process meta ORAM results if enabled
                if self._enable_meta and self._meta is not None:
                    self._meta.process_read_result(result)
                    # De-duplicate: keep only the first (highest priority) dup for each key
                    self.meta_de_duplication()

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
                
                # Apply meta duplications to update AVL node values (graph_leaf)
                # Meta duplication format: Data(key=vertex_key, leaf=pos_leaf, value=new_graph_leaf)
                # After de-duplication, each key has only one dup (the highest priority one)
                if self._enable_meta and self._meta is not None:
                    meta_temp_stash = []
                    for meta_data in self._meta.stash:
                        applied = False
                        # Check if the node is in level_local_nodes (just retrieved)
                        if meta_data.key in level_local_nodes:
                            node = level_local_nodes[meta_data.key]
                            node.value.value = meta_data.value
                            applied = True
                        else:
                            # Check if the node is in stash (AVL stash, not meta stash)
                            stash_idx = self._find_in_stash(meta_data.key)
                            if stash_idx >= 0:
                                self._stash[stash_idx].value.value = meta_data.value
                                applied = True
                        
                        if not applied:
                            # Node not found yet, keep the duplication for later
                            meta_temp_stash.append(meta_data)
                    self._meta.stash = meta_temp_stash

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
                    # Track the node for pos_leaf capture after leaf reassignment
                    if pos_leaf_map is not None:
                        pos_leaf_map[keys[i]] = node  # Store the node object temporarily
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
                
                # CRITICAL: Update pending meta duplications to use the new leaf
                # Otherwise dup will be written to old path and won't be found later
                if self._enable_meta and self._meta is not None:
                    for dup in self._meta.stash:
                        if dup.key == node.key:
                            dup.leaf = node.leaf
                
                # Track visited nodes' new pos_leaf AND their stored graph_leaf
                # node.key: vertex key in PosMap
                # node.leaf: new pos_leaf (position in PosMap ORAM)
                # node.value.value: graph_leaf (position in Graph ORAM, stored in AVL node)
                if visited_nodes_map is not None:
                    graph_leaf = node.value.value if node.value else None
                    visited_nodes_map[node.key] = (node.leaf, graph_leaf)

            # Move level_local nodes to stash for eviction
            for node in level_local_nodes.values():
                if self._find_in_stash(node.key) < 0:
                    self._stash.append(node)
                    keys_in_stash.add(node.key)

            # Update root pointer
            if self.root is not None and root_key in level_local_nodes:
                self.root = (root_key, level_local_nodes[root_key].leaf)

            # Capture pos_leaf for found keys AFTER leaf reassignment
            if pos_leaf_map is not None:
                for key in list(pos_leaf_map.keys()):
                    node = pos_leaf_map[key]
                    if isinstance(node, Data):
                        pos_leaf_map[key] = node.leaf  # Replace node with its new leaf

            # Evict this level using the paths we read
            if leaves_this_level:
                evicted = self._evict_stash(leaves=leaves_this_level)
                self._client.add_write_path(label=self._name, data=evicted)
                # Also write meta ORAM if enabled - use same paths as read
                if self._enable_meta and self._meta is not None:
                    self._meta.queue_write(leaves=leaves_this_level)
                self._client.execute()

        # Return results with pos_leaf if requested
        if return_visited_nodes:
            if return_pos_leaf:
                final_results = {key: (results[key], pos_leaf_map.get(key)) for key in keys}
            else:
                final_results = results
            # Return (results, visited_nodes_map, total_paths_read)
            return (final_results, visited_nodes_map, total_paths_read)
        
        if return_pos_leaf:
            return {key: (results[key], pos_leaf_map.get(key)) for key in keys}
        return results
