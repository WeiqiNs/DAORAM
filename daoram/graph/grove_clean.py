    def delete(self, key: Any) -> None:
        """
        Delete a vertex from the graph using duplication-based neighbor notification.
        
        Instead of downloading all neighbors, we send duplications to notify them
        that this vertex has been deleted (using new_graph_leaf = -1 as deletion marker).

        :param key: The key of the vertex to delete.
        """
        # Step 1: Delete from PosMap ORAM (search with value=None performs lazy deletion)
        # This also returns the vertex's graph_leaf
        vertex_graph_leaf = self._pos_omap.delete(key=key)
        if vertex_graph_leaf is None:
            # Vertex not found in PosMap
            return
        
        # Step 2: Get RL paths for graph_meta (to notify neighbors via duplication)
        # We need max_degree paths for obliviousness
        graph_meta_rl_paths = self.get_rl_leaf(count=self._max_deg, for_pos_meta=False)
        
        # Step 3: Queue reads for Graph ORAM and Graph Meta
        self._graph_oram.queue_read(leaves=[vertex_graph_leaf])
        self._graph_meta.queue_read(leaves=graph_meta_rl_paths + [vertex_graph_leaf])
        
        # Execute reads
        result = self._client.execute()
        self._graph_oram.process_read_result(result)
        self._graph_meta.process_read_result(result)
        
        # De-duplication to ensure we only apply the newest updates
        self.graph_meta_de_duplication()
        
        # Step 4: Find the vertex in stash and get its neighbors
        vertex_data = None
        vertex_idx = None
        for i, data in enumerate(self._graph_oram.stash):
            if data.key == key:
                vertex_data = data
                vertex_idx = i
                break
        
        if vertex_data is None:
            # Vertex not found in Graph ORAM, write back and return
            self._graph_oram.queue_write()
            self._graph_meta.queue_write(leaves=graph_meta_rl_paths + [vertex_graph_leaf])
            
            # Also queue write for PosMap meta ORAM
            if self._pos_omap._enable_meta and self._pos_omap._meta is not None:
                self._pos_omap._meta.queue_write(leaves=[vertex_graph_leaf])
                
            self._client.execute()
            return
        
        # Get neighbor info: {neighbor_key: graph_leaf}
        neighbor_adjacency = vertex_data.value[1]
        neighbor_keys = list(neighbor_adjacency.keys())

        # Step 5: Apply duplications to TARGET vertices in stash, and discard dups
        # targeting the deleted vertex.
        target_keys = {key}
        retrieved_indices = {data.key: idx for idx, data in enumerate(self._graph_oram.stash)}
        temp_stash = []
        for dup in self._graph_meta.stash:
            if dup.key == key:
                # This dup targets the deleted vertex, discard it
                continue
            
            if dup.key in target_keys and dup.key in retrieved_indices:
                idx = retrieved_indices[dup.key]
                vertex_value = self._graph_oram.stash[idx].value
                
                if isinstance(dup.value, tuple) and len(dup.value) == 2:
                    # Type 1: Neighbor update
                    adjacency_dict = vertex_value[1]
                    source_key, new_graph_leaf_val = dup.value
                    if new_graph_leaf_val < 0:
                        if source_key in adjacency_dict:
                            del adjacency_dict[source_key]
                    else:
                        adjacency_dict[source_key] = new_graph_leaf_val
                elif not isinstance(dup.value, tuple):
                    # Type 2: PosMap update
                    vertex_data_val = vertex_value[0]
                    adjacency_dict_val = vertex_value[1]
                    self._graph_oram.stash[idx].value = (vertex_data_val, adjacency_dict_val, dup.value)
            else:
                # Keep dups for non-target vertices or vertices NOT in stash
                temp_stash.append(dup)
        self._graph_meta.stash = temp_stash
        
        # Step 6: Create deletion duplications for all neighbors
        # Use new_graph_leaf = -1 to indicate deletion
        DELETION_MARKER = -1
        deletion_dups = []
        for i, neighbor_key in enumerate(neighbor_keys):
            neighbor_graph_leaf = neighbor_adjacency[neighbor_key]
            if isinstance(neighbor_graph_leaf, tuple):
                neighbor_graph_leaf = neighbor_graph_leaf[0]
            
            # Duplication format: Data(key=neighbor_key, leaf=neighbor_graph_leaf, value=(key, DELETION_MARKER))
            dup = Data(key=neighbor_key, leaf=neighbor_graph_leaf, value=(key, DELETION_MARKER))
            deletion_dups.append(dup)
        
        # Pad with dummy dups for obliviousness (total = max_degree)
        dummy_leaf = secrets.randbelow(self._leaf_range)
        for _ in range(self._max_deg - len(deletion_dups)):
            dummy_dup = Data(key=None, leaf=dummy_leaf, value=(None, DELETION_MARKER))
            deletion_dups.append(dummy_dup)
        
        # Add deletion dups to graph_meta stash (prepend for highest priority)
        self._graph_meta.stash = deletion_dups + self._graph_meta.stash
        
        # Step 7: Remove the deleted vertex from graph_oram stash
        del self._graph_oram.stash[vertex_idx]
        
        # Step 8: Write back
        self._graph_oram.queue_write()
        self._graph_meta.queue_write(leaves=graph_meta_rl_paths + [vertex_graph_leaf])
        
        # CRITICAL FIX: Also queue write for PosMap meta ORAM to ensure updates 
        # reach the server! We use the same leaves as Graph ORAM for obliviousness.
        if self._pos_omap._enable_meta and self._pos_omap._meta is not None:
            self._pos_omap._meta.queue_write(leaves=[vertex_graph_leaf])

        self._client.execute()

    def neighbor(self, keys: List[Any]) -> dict:
        """
        Perform a neighbor lookup query with proper duplication handling.
        
        For a single center vertex:
        1. Lookup the center vertex to get its neighbor list (adjacency_dict has graph_leaf)
        2. Download all neighbors in one round using graph_leaf from adjacency_dict
        3. Update center's new path in all neighbors
        4. Send duplications to neighbors' neighbors about path changes
        5. Send duplications to PosMap for all path updates (pos_leaf from downloaded neighbors)

        :param keys: List of vertex keys to find neighbors for.
        :return: Dict mapping neighbor_key to (vertex_data, adjacency_dict).
        """
        if not keys:
            return {}
        
        # For simplicity, handle one center vertex at a time
        center_key = keys[0]
        
        # Step 1: Lookup the center vertex (also get visited_nodes_map)
        # We need the new_graph_leaf of the center vertex to update neighbors
        key_graph_leaf_dict, center_visited_nodes, total_posmap_paths = self._pos_omap.batch_search(
            keys=[center_key], return_visited_nodes=True
        )
        if center_key not in key_graph_leaf_dict or key_graph_leaf_dict[center_key] is None:
            return {}
        
        pos_meta_extra_paths = max(0, total_posmap_paths - 1)
        center_lookup_result = self.lookup_without_omap(
            {center_key: key_graph_leaf_dict[center_key]}, 
            center_visited_nodes, 
            pos_meta_extra_paths
        )
        
        if center_key not in center_lookup_result:
            return {}
        
        # lookup_without_omap returns (vertex_data, adjacency_dict, new_graph_leaf)
        center_data, center_adjacency, center_new_graph_leaf = center_lookup_result[center_key]
        
        # adjacency_dict format: {neighbor_key: neighbor_graph_leaf}
        neighbor_keys = list(center_adjacency.keys())
        K = len(neighbor_keys)
        
        if K == 0:
            return {}

        # Use graph_leaf from center_adjacency (should be up-to-date after lookup's dup processing)
        neighbor_graph_leaves = []
        for nk in neighbor_keys:
            gl = center_adjacency[nk]
            if isinstance(gl, tuple):
                gl = gl[0]
            neighbor_graph_leaves.append(gl)
        
        # Pad with dummy leaves for obliviousness (total = max_degree)
        dummy_count = max(0, self._max_deg - K)
        dummy_leaves = [secrets.randbelow(self._leaf_range) for _ in range(dummy_count)]
        all_graph_leaves = neighbor_graph_leaves + dummy_leaves

        # RL paths for meta ORAMs
        graph_meta_rl_count = self._max_deg * self._max_deg
        graph_meta_rl_paths = (
            self.get_rl_leaf(count=graph_meta_rl_count, for_pos_meta=False)
            + all_graph_leaves
        )
        
        # CRITICAL FIX: Before reading Graph ORAM, create dups for ALL visited AVL nodes
        # from the lookup step. These dups notify vertices about their pos_leaf changes.
        avl_pos_update_dups = []
        for vertex_key, (new_pos_leaf, vertex_graph_leaf) in center_visited_nodes.items():
            if vertex_graph_leaf is not None:
                avl_pos_update_dups.append(
                    Data(key=vertex_key, leaf=vertex_graph_leaf, value=new_pos_leaf)
                )
        self._graph_meta.stash = avl_pos_update_dups + self._graph_meta.stash

        # Step 3: Queue reads for Graph ORAM and Graph Meta ORAM
        self._graph_oram.queue_read(leaves=all_graph_leaves)
        self._graph_meta.queue_read(leaves=graph_meta_rl_paths)
        
        result = self._client.execute()
        
        self._graph_oram.process_read_result(result)
        self._graph_meta.process_read_result(result)
        
        # De-duplication for graph_meta
        self.graph_meta_de_duplication()
        
        # Step 4: Apply duplications and process neighbors
        downloaded_neighbors = {}
        for data in self._graph_oram.stash:
            if data.key in neighbor_keys:
                downloaded_neighbors[data.key] = data
        
        # Apply graph_meta duplications to TARGET vertices in stash.
        # Targets in neighbor query are: the neighbors AND the center node.
        all_targets = set(neighbor_keys) | {center_key}
        retrieved_indices = {data.key: idx for idx, data in enumerate(self._graph_oram.stash)}
        temp_stash = []
        for dup in self._graph_meta.stash:
            if dup.key in all_targets and dup.key in retrieved_indices:
                idx = retrieved_indices[dup.key]
                target_data_obj = self._graph_oram.stash[idx]
                vertex_value = target_data_obj.value
                if isinstance(dup.value, tuple) and len(dup.value) == 2:
                    # Type 1: Neighbor update
                    adjacency_dict = vertex_value[1]
                    source_key, new_graph_leaf = dup.value
                    if new_graph_leaf < 0:
                        if source_key in adjacency_dict:
                            del adjacency_dict[source_key]
                    else:
                        adjacency_dict[source_key] = new_graph_leaf
                else:
                    # Type 2: PosMap update
                    vertex_data_val = vertex_value[0]
                    adjacency_dict_val = vertex_value[1]
                    target_data_obj.value = (vertex_data_val, adjacency_dict_val, dup.value)
            else:
                temp_stash.append(dup)
        self._graph_meta.stash = temp_stash
        
        # Prepare result and duplications
        graph_meta_duplications = []
        pos_meta_duplications = []
        neighbor_result = {}
        
        # Pre-assign all new paths for nodes in this batch to allow local synchronization
        new_paths_in_batch = {center_key: center_new_graph_leaf}
        for nk in neighbor_keys:
            if nk in downloaded_neighbors:
                new_paths_in_batch[nk] = secrets.randbelow(self._leaf_range)

        # Update center node in stash if it's there (it might be a passenger)
        if center_key in retrieved_indices:
            center_node_in_stash = self._graph_oram.stash[retrieved_indices[center_key]]
            center_node_val = center_node_in_stash.value
            center_adj_in_stash = center_node_val[1]
            for nk, n_new_leaf in new_paths_in_batch.items():
                if nk != center_key and nk in center_adj_in_stash:
                    center_adj_in_stash[nk] = n_new_leaf
            center_node_in_stash.leaf = center_new_graph_leaf
            center_node_in_stash.value = (center_node_val[0], center_adj_in_stash, center_node_val[2] if len(center_node_val) > 2 else None)
            center_adjacency = center_adj_in_stash

        for neighbor_key, neighbor_data in downloaded_neighbors.items():
            neighbor_new_graph_leaf = new_paths_in_batch[neighbor_key]
            
            if len(neighbor_data.value) == 3:
                neighbor_vertex_data, neighbor_adjacency, neighbor_pos_leaf = neighbor_data.value
            else:
                neighbor_vertex_data, neighbor_adjacency = neighbor_data.value
                neighbor_pos_leaf = None

            if center_key in neighbor_adjacency:
                neighbor_adjacency[center_key] = center_new_graph_leaf
            
            neighbor_data.leaf = neighbor_new_graph_leaf
            
            for other_key, other_graph_leaf in neighbor_adjacency.items():
                if other_key != center_key:
                    if isinstance(other_graph_leaf, tuple):
                        other_graph_leaf = other_graph_leaf[0]
                    if other_graph_leaf is None or other_graph_leaf < 0:
                        continue
                    
                    target_leaf = new_paths_in_batch.get(other_key, other_graph_leaf)
                    if other_key in new_paths_in_batch:
                        neighbor_adjacency[other_key] = target_leaf

                    dup = Data(key=other_key, leaf=target_leaf, 
                              value=(neighbor_key, neighbor_new_graph_leaf))
                    graph_meta_duplications.append(dup)
            
            dup_to_center = Data(key=center_key, leaf=center_new_graph_leaf,
                                value=(neighbor_key, neighbor_new_graph_leaf))
            graph_meta_duplications.append(dup_to_center)
            
            if neighbor_pos_leaf is not None:
                pos_dup = Data(key=neighbor_key, leaf=neighbor_pos_leaf, value=neighbor_new_graph_leaf)
                pos_meta_duplications.append(pos_dup)
            
            neighbor_data.value = (neighbor_vertex_data, neighbor_adjacency, neighbor_pos_leaf)
            neighbor_result[neighbor_key] = (neighbor_vertex_data, neighbor_adjacency)
        
        dummy_leaf = secrets.randbelow(self._leaf_range)
        while len(graph_meta_duplications) < self._max_deg * self._max_deg:
            graph_meta_duplications.append(Data(key=None, leaf=dummy_leaf, value=(None, 0)))
        
        while len(pos_meta_duplications) < self._max_deg + 1:
            pos_meta_duplications.append(Data(key=None, leaf=dummy_leaf, value=0))
        
        self._graph_meta.stash = graph_meta_duplications + self._graph_meta.stash
        self._pos_omap.add_meta_duplications(pos_meta_duplications)
        
        self.graph_meta_de_duplication()
        self.pos_meta_de_duplication()
        self._pos_omap.meta_de_duplication()
        
        self._graph_oram.queue_write(leaves=all_graph_leaves)
        self._graph_meta.queue_write(leaves=graph_meta_rl_paths)
        
        if self._pos_omap._enable_meta and self._pos_omap._meta is not None:
            self._pos_omap._meta.queue_write(leaves=all_graph_leaves)

        self._client.execute()
        return neighbor_result
