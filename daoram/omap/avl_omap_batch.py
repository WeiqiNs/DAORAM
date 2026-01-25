"""AVL OMAP with batch operations for multiple keys."""

from typing import Any, Dict, List, Set, Tuple

from daoram.dependency import Data, Encryptor, InteractServer
from daoram.omap.avl_omap_cache import AVLOmapCached
from daoram.omap.oblivious_search_tree import KV_LIST


class AVLOmapBatch(AVLOmapCached):
    """
    AVL OMAP with batch operations for multiple keys.

    Key features:
    1. batch_insert: Insert multiple key-value pairs
    2. batch_search: Search/update multiple keys (level-by-level for efficiency)
    3. batch_delete: Delete multiple keys
    4. Leverages stash caching from AVLOmapCached for efficiency

    The batch_search uses a level-by-level approach that reduces communication
    from O(k*h) rounds to O(h) rounds for k keys in a tree of height h.
    """

    def __init__(self,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 client: InteractServer,
                 name: str = "avl_batch",
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

    def _batch_read_paths_with_padding(
        self, leaves: List[int], dummy_count: int, keys_in_stash: Set[Any]
    ) -> List[int]:
        """
        Read specified leaves plus dummy_count random dummy paths.

        Nodes from read paths are added to stash, deduplicating by key.

        :param leaves: List of leaf indices to read.
        :param dummy_count: Number of additional dummy paths to read for obliviousness.
        :param keys_in_stash: Set of keys already in stash (for deduplication).
        :return: List of all leaves read (for tracking).
        """
        all_leaves = list(leaves)
        for _ in range(dummy_count):
            all_leaves.append(self._get_new_leaf())

        # Batch read all paths
        self._client.add_read_path(label=self._name, leaves=all_leaves)
        result = self._client.execute()
        path_data = result.results[self._name]

        # Decrypt and add nodes to stash (deduplicating by key)
        path = self.decrypt_path_data(path=path_data)
        for bucket in path.values():
            for data in bucket:
                if data.key not in keys_in_stash:
                    self._stash.append(data)
                    keys_in_stash.add(data.key)

        return all_leaves

    def _final_eviction(self, all_leaves_read: List[int]) -> None:
        """
        Final eviction after all levels processed.

        Updates all leaves for re-randomization and fixes parent-child pointers,
        then evicts to read paths.

        :param all_leaves_read: List of all leaves that were read during batch search.
        """
        # Build a key-to-node map for efficient lookup
        key_to_node: Dict[Any, Data] = {}
        for node in self._stash:
            key_to_node[node.key] = node

        # Update leaves for all nodes in stash (re-randomization)
        for node in self._stash:
            node.leaf = self._get_new_leaf()

        # Fix parent-child leaf pointers
        # For each node, if its children are in the stash, update the l_leaf/r_leaf pointers
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

        # Evict to all paths we read (deduplicated)
        unique_leaves = list(set(all_leaves_read))
        evicted = self._evict_stash(leaves=unique_leaves)
        self._client.add_write_path(label=self._name, data=evicted)
        self._client.execute()

    def batch_search(self, kv_pairs: KV_LIST) -> List[Any]:
        """
        Search for multiple keys using level-by-level path reading.

        This approach reduces communication from O(k*h) to O(h) rounds
        where k is the number of keys and h is the tree height.

        At each level l, we read exactly min(2^l, k) paths (padding with
        dummy reads if fewer paths are actually needed) for obliviousness.
        After each level, we evict and write back (like normal search).

        :param kv_pairs: List of (key, value) tuples. If value is None, just search.
                        If value is provided, update the key's value.
        :return: List of (old) values corresponding to each key.
        """
        if not kv_pairs:
            return []

        keys = [kv[0] for kv in kv_pairs]
        values = [kv[1] for kv in kv_pairs]
        k = len(keys)

        # Handle empty tree case
        if self.root is None:
            self._perform_dummy_operation(num_round=self._max_height)
            return [None] * k

        # 1. INITIALIZATION
        # Flush local to stash
        self._flush_local_to_stash()

        # Track found nodes in local (similar to normal search)
        local_nodes: Dict[Any, Data] = {}

        # For each key i:
        #   current_node[i] = (node_key, leaf) - the node we need to visit next
        #   finished[i] = whether we're done with this key
        #   results[i] = the found value (or None)
        current_node: List[Tuple[Any, int]] = [(self.root[0], self.root[1])] * k
        finished: List[bool] = [False] * k
        results: List[Any] = [None] * k

        # 2. FOR EACH LEVEL
        for level in range(self._max_height):
            # a. COLLECT UNIQUE LEAVES needed for this level
            # Group active keys by the leaf they need to read
            leaf_to_node_keys: Dict[int, Set[Any]] = {}
            for i in range(k):
                if not finished[i]:
                    node_key, leaf = current_node[i]
                    # Check if node is already in local
                    if node_key in local_nodes:
                        continue  # Already have this node
                    # Check if node is in stash - if so, move to local
                    stash_idx = self._find_in_stash(node_key)
                    if stash_idx >= 0:
                        node = self._stash.pop(stash_idx)
                        local_nodes[node.key] = node
                        continue
                    # Need to read from this leaf
                    if leaf not in leaf_to_node_keys:
                        leaf_to_node_keys[leaf] = set()
                    leaf_to_node_keys[leaf].add(node_key)

            unique_leaves = list(leaf_to_node_keys.keys())

            # b. COMPUTE PADDING for obliviousness
            required = min(2**level, k)
            actual = len(unique_leaves)
            dummy_count = max(0, required - actual)

            # c. BATCH READ paths and EVICT (like normal search pattern)
            if unique_leaves or dummy_count > 0:
                # Add dummy leaves
                all_leaves = list(unique_leaves)
                for _ in range(dummy_count):
                    all_leaves.append(self._get_new_leaf())

                # Read paths
                self._client.add_read_path(label=self._name, leaves=all_leaves)
                result = self._client.execute()
                path_data = result.results[self._name]
                path = self.decrypt_path_data(path=path_data)

                # Track keys of nodes we're looking for
                target_node_keys: Set[Any] = set()
                for node_keys in leaf_to_node_keys.values():
                    target_node_keys.update(node_keys)

                # Process path data - move target nodes to local, others to stash
                for bucket in path.values():
                    for data in bucket:
                        if data.key in target_node_keys and data.key not in local_nodes:
                            local_nodes[data.key] = data
                        elif self._find_in_stash(data.key) < 0:
                            # Not in stash yet, add it
                            self._stash.append(data)

                # Re-randomize leaves for nodes in local
                for node in local_nodes.values():
                    node.leaf = self._get_new_leaf()

                # Evict and write back (like normal search)
                evicted = self._evict_stash(leaves=all_leaves)
                self._client.add_write_path(label=self._name, data=evicted)
                self._client.execute()

            # d. PROCESS EACH KEY - determine next node to visit
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
                    results[i] = node.value.value
                    if values is not None and i < len(values) and values[i] is not None:
                        node.value.value = values[i]
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

        # 3. FINALIZATION
        # Move local nodes back to stash for next operation
        for node in local_nodes.values():
            if self._find_in_stash(node.key) < 0:
                self._stash.append(node)

        # Update parent-child leaf pointers for nodes in stash
        key_to_node: Dict[Any, Data] = {node.key: node for node in self._stash}
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

        return results

    def batch_insert(self, keys: List[Any], values: List[Any] = None) -> None:
        """
        Insert multiple key-value pairs.

        Uses the cached insert method which benefits from stash caching.

        :param keys: List of keys to insert.
        :param values: List of values (if None, uses keys as values).
        """
        if not keys:
            return

        values = values if values is not None else list(keys)

        for key, value in zip(keys, values):
            self.insert(key=key, value=value)

    def batch_delete(self, keys: List[Any]) -> List[Any]:
        """
        Delete multiple keys.

        Uses the cached delete method which benefits from stash caching.

        :param keys: List of keys to delete.
        :return: List of deleted values.
        """
        if not keys:
            return []

        results = []
        for key in keys:
            try:
                result = self.delete(key=key)
                results.append(result)
            except ValueError:
                # Tree became empty or key not found
                results.append(None)

        return results
