"""B+ tree OMAP with caching optimization for repeated accesses."""

from typing import Any, Dict, List

from daoram.dependency import Encryptor, InteractServer
from daoram.dependency.binary_tree import Data
from daoram.omap import BPlusOmap


class BPlusOmapCached(BPlusOmap):
    """
    B+ OMAP with stash caching optimization.

    Key differences from BPlusOmap:
    1. Checks stash before fetching from server (cache hit avoids ORAM access)
    2. Nodes stay in local at end of operation, flushed to stash at start of next
    3. Reduced interaction rounds: insert/search use h rounds, delete uses h rounds (batched)
    """

    def __init__(self,
                 order: int,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 client: InteractServer,
                 name: str = "bplus_opt",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 encryptor: Encryptor = None):
        super().__init__(
            order=order,
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

    def _move_node_to_local(self, key: Any, leaf: int, parent_key: Any = None, child_index: int = None) -> None:
        """
        Override to check stash first before fetching from server.

        This is the key caching mechanism - if a node is in stash from a previous
        operation, we use it directly without an ORAM server access.
        """
        stash_idx = self._find_in_stash(key)
        if stash_idx >= 0:
            # Cache hit - use node from stash (no ORAM access)
            node = self._stash.pop(stash_idx)
            self._local.add(node=node, parent_key=parent_key, child_index=child_index)
        else:
            # Cache miss - fetch from server
            super()._move_node_to_local(key=key, leaf=leaf, parent_key=parent_key, child_index=child_index)

    def _batch_move_nodes_to_local(self, nodes_to_fetch: List[tuple[Any, int]]) -> None:
        """
        Batch fetch multiple nodes in a single read-write round.

        Checks stash first for each node, then batches remaining reads.
        This reduces multiple sequential fetches to a single round.

        :param nodes_to_fetch: List of (key, leaf) tuples to fetch.
        """
        nodes_from_stash = []
        keys_to_read = []
        leaves_to_read = []

        # Check stash first for each node
        for key, leaf in nodes_to_fetch:
            stash_idx = self._find_in_stash(key)
            if stash_idx >= 0:
                nodes_from_stash.append(self._stash.pop(stash_idx))
            else:
                keys_to_read.append(key)
                leaves_to_read.append(leaf)

        # Add nodes found in stash to local
        for node in nodes_from_stash:
            self._local.add(node=node, parent_key=None, child_index=None)

        # If all nodes were in stash, no server access needed
        if not leaves_to_read:
            return

        # Batch read remaining nodes
        self._client.add_read_path(label=self._name, leaves=leaves_to_read)
        result = self._client.execute()
        path_data = result.results[self._name]

        # Decrypt and process
        path = self.decrypt_path_data(path=path_data)

        keys_to_find = set(keys_to_read)
        to_index = len(self._stash)

        for bucket in path.values():
            for data in bucket:
                if data.key in keys_to_find:
                    self._local.add(node=data, parent_key=None, child_index=None)
                    keys_to_find.remove(data.key)
                elif data.key is not None:
                    self._stash.append(data)

        # Check stash overflow
        if len(self._stash) > self._stash_size:
            raise OverflowError(
                f"Stash overflow in {self._name}: size {len(self._stash)} exceeds max {self._stash_size}.")

        # Check stash for any keys not found in paths
        for key in list(keys_to_find):
            stash_idx = self._find_in_stash(key)
            if 0 <= stash_idx < to_index:
                self._local.add(node=self._stash.pop(stash_idx), parent_key=None, child_index=None)
                keys_to_find.remove(key)

        if keys_to_find:
            raise KeyError(f"Keys {keys_to_find} not found in paths or stash.")

        # Batch write back
        evicted = self._evict_stash(leaves=leaves_to_read)
        self._client.add_write_path(label=self._name, data=evicted)
        self._client.execute()

    def _find_leaf_to_local_cached(self, key: Any) -> None:
        """
        Add all nodes we need to visit to local until finding the leaf storing the key.
        Uses caching: flushes at start, checks stash before fetching.

        :param key: Search key of interest.
        """
        # Flush any cached local nodes to stash first
        self._flush_local_to_stash()

        # Get the root node (will check stash first via overridden _move_node_to_local)
        self._move_node_to_local(key=self.root[0], leaf=self.root[1], parent_key=None, child_index=None)

        # Get the node from local
        node = self._local.get_root()

        # Update node leaf and root
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)

        # While we do not reach a leaf
        while not self._local.is_leaf_node(node):
            # Sample a new leaf for updating the current storage
            new_leaf = self._get_new_leaf()

            # Find child index
            child_index = len(node.value.keys)
            for index, each_key in enumerate(node.value.keys):
                if key == each_key:
                    child_index = index + 1
                    break
                elif key < each_key:
                    child_index = index
                    break

            # Get child info
            child_key, child_leaf = node.value.values[child_index]
            # Move the next node to local with parent tracking (checks stash first)
            self._move_node_to_local(key=child_key, leaf=child_leaf, parent_key=node.key, child_index=child_index)
            # Update the current stored value with new leaf
            node.value.values[child_index] = (child_key, new_leaf)

            # Update the node and its leaf
            node = self._local.get_leaf()
            node.leaf = new_leaf

    def insert(self, key: Any, value: Any = None) -> None:
        """
        Given key-value pair, insert the pair to the tree with caching.

        :param key: The search key of interest.
        :param value: The value to insert.
        """
        if key is None:
            self._perform_dummy_operation(num_round=self._max_height)
            return

        # If the current root is empty, we simply set root as this new block
        if self.root is None:
            data_block = self._get_bplus_data(keys=[key], values=[value])
            self._stash.append(data_block)
            self.root = (data_block.key, data_block.leaf)
            self._perform_dummy_operation(num_round=self._max_height)
            return

        # Get all nodes we need to visit until finding the key (with caching)
        self._find_leaf_to_local_cached(key=key)

        # Set the last node in local as leaf
        leaf = self._local.get_leaf()

        # Find the proper place to insert the leaf
        for index, each_key in enumerate(leaf.value.keys):
            if key < each_key:
                leaf.value.keys = leaf.value.keys[:index] + [key] + leaf.value.keys[index:]
                leaf.value.values = leaf.value.values[:index] + [value] + leaf.value.values[index:]
                break
            elif index + 1 == len(leaf.value.keys):
                leaf.value.keys.append(key)
                leaf.value.values.append(value)
                break

        # Save the length of the local
        num_retrieved_nodes = len(self._local)

        # Perform the insertion to handle splits
        self._perform_insertion()

        # Flush local to stash (splits make keeping nodes in local complex)
        self._flush_local_to_stash()

        # Perform dummy operations
        self._perform_dummy_operation(num_round=self._max_height - num_retrieved_nodes)

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

        # Get all nodes we need to visit until finding the key (with caching)
        self._find_leaf_to_local_cached(key=key)

        # Set the last node in local as leaf
        leaf = self._local.get_leaf()
        search_value = None

        # Search the desired key and update its value as needed
        for index, each_key in enumerate(leaf.value.keys):
            if key == each_key:
                search_value = leaf.value.values[index]
                if value is not None:
                    leaf.value.values[index] = value
                break

        # Save the number of retrieved nodes
        num_retrieved_nodes = len(self._local)

        # Flush local to stash
        self._flush_local_to_stash()

        # Perform dummy operations
        self._perform_dummy_operation(num_round=self._max_height - num_retrieved_nodes)

        return search_value

    def fast_search(self, key: Any, value: Any = None) -> Any:
        """Fast search is identical to search in cached version."""
        return self.search(key=key, value=value)

    def _find_path_with_siblings_cached(self, key: Any) -> tuple[
        Dict[int, Data], List[int], Dict[int, Data], Dict[int, int]]:
        """
        Find path from root to leaf, fetching target and sibling in batched rounds.

        At each level (except root), batches fetches for:
        - The target child on the path
        - One sibling (left if available, else right) for underflow handling

        This completes in h rounds (2 executes per level) instead of 2h-1 rounds.

        :param key: Search key of interest.
        :return: Tuple of (path_nodes, child_indices, siblings, sibling_indices).
            - path_nodes: Dict[level, node] - nodes on the path from root to leaf
            - child_indices: List[int] - child index taken at each internal level
            - siblings: Dict[level, node] - pre-fetched sibling at each non-root level
            - sibling_indices: Dict[level, int] - index of sibling in parent's values
        """
        # Flush any cached local nodes to stash first
        self._flush_local_to_stash()

        # Path nodes stored by level: {0: root, 1: child, 2: grandchild, ...}
        path_nodes: Dict[int, Data] = {}
        child_indices: List[int] = []
        siblings: Dict[int, Data] = {}
        sibling_indices: Dict[int, int] = {}
        level = 0

        # Get the root node (checks stash first via overridden _move_node_to_local)
        self._move_node_to_local(key=self.root[0], leaf=self.root[1], parent_key=None, child_index=None)
        node = self._local.remove(self._local.root_key)  # Remove from local, store in our dict

        # Update node leaf and root
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)
        path_nodes[level] = node

        # Traverse down to leaf, fetching target + sibling in ONE batch per level
        while not (len(node.value.keys) == len(node.value.values)):
            # Sample new leaves for the child and sibling
            new_leaf = self._get_new_leaf()

            # Find the child index to descend to
            child_index = len(node.value.keys)
            for index, each_key in enumerate(node.value.keys):
                if key < each_key:
                    child_index = index
                    break
                elif key == each_key:
                    child_index = index + 1
                    break

            # Record the child index
            child_indices.append(child_index)

            # Get child info
            child_key, child_leaf = node.value.values[child_index]

            # Determine sibling: prefer left, fallback to right
            sibling_index = None
            sib_key, sib_leaf, sib_new_leaf = None, None, None
            if child_index > 0:
                sibling_index = child_index - 1
            elif child_index < len(node.value.values) - 1:
                sibling_index = child_index + 1

            if sibling_index is not None:
                sib_key, sib_leaf = node.value.values[sibling_index]
                sib_new_leaf = self._get_new_leaf()

            # Batch fetch target child and sibling in ONE round
            nodes_to_fetch = [(child_key, child_leaf)]
            if sibling_index is not None:
                nodes_to_fetch.append((sib_key, sib_leaf))

            self._batch_move_nodes_to_local(nodes_to_fetch)

            # Extract child node from local
            child_node = self._local.remove(child_key)
            child_node.leaf = new_leaf
            node.value.values[child_index] = (child_key, new_leaf)

            # Extract sibling node from local if fetched
            if sibling_index is not None:
                sibling_node = self._local.remove(sib_key)
                sibling_node.leaf = sib_new_leaf
                node.value.values[sibling_index] = (sib_key, sib_new_leaf)

                # Store sibling at the child's level
                siblings[level + 1] = sibling_node
                sibling_indices[level + 1] = sibling_index

            # Store child in path and move down
            level += 1
            path_nodes[level] = child_node
            node = child_node

        return path_nodes, child_indices, siblings, sibling_indices

    def delete(self, key: Any) -> Any:
        """
        Delete with h-round optimization: batches target + sibling fetches per level.

        At each level, fetches both target node and sibling in ONE batch round,
        completing the entire operation in h rounds regardless of underflow handling.
        """
        # Total accesses for h-level tree: 1 (root) + 1*(h-1) = h
        # Each access is a batch fetch (target + sibling) = 2 executes
        # Pad to max_height for obliviousness
        max_accesses = self._max_height

        if key is None:
            self._perform_dummy_operation(num_round=max_accesses)
            return None

        # If the current root is empty, nothing to delete.
        if self.root is None:
            raise ValueError("Cannot delete from an empty tree.")

        # Compute the minimum number of keys each node should have
        min_keys = (self._order - 1) // 2

        # Find path from root to leaf with pre-fetched siblings (h rounds)
        path_nodes, child_indices, siblings, sibling_indices = self._find_path_with_siblings_cached(key=key)
        leaf_level = len(path_nodes) - 1
        leaf = path_nodes[leaf_level]

        # Find and delete key from leaf
        key_index = None
        deleted_value = None
        for i, k in enumerate(leaf.value.keys):
            if k == key:
                key_index = i
                deleted_value = leaf.value.values[i]
                break

        # Key not found - flush everything and return
        if key_index is None:
            for node in path_nodes.values():
                self._stash.append(node)
            for sib in siblings.values():
                self._stash.append(sib)
            total_nodes = len(path_nodes) + len(siblings)
            self._perform_dummy_operation(num_round=max_accesses - total_nodes)
            return None

        # Delete key and value from leaf
        leaf.value.keys.pop(key_index)
        leaf.value.values.pop(key_index)

        # Handle root leaf case (child_indices is empty when root is the leaf)
        if not child_indices:
            if len(leaf.value.keys) == 0:
                self.root = None
            else:
                self._stash.append(leaf)
            self._perform_dummy_operation(num_round=max_accesses - 1)
            return deleted_value

        # Handle underflow from bottom to top using pre-fetched siblings
        node_level = leaf_level
        node = leaf

        for level in range(len(child_indices) - 1, -1, -1):
            parent = path_nodes[level]
            child_index = child_indices[level]

            # No underflow, done
            if len(node.value.keys) >= min_keys:
                break

            # Get pre-fetched sibling for this level (child's level = level + 1)
            child_level = level + 1
            if child_level not in siblings:
                # No sibling was fetched (shouldn't happen in valid B+ tree)
                break

            sibling = siblings[child_level]
            sib_index = sibling_indices[child_level]
            is_left_sibling = sib_index < child_index

            # Try to borrow from the pre-fetched sibling
            if len(sibling.value.keys) > min_keys:
                is_leaf = len(node.value.keys) == len(node.value.values)
                if is_left_sibling:
                    if is_leaf:
                        # Borrow from left for leaf
                        node.value.keys.insert(0, sibling.value.keys.pop())
                        node.value.values.insert(0, sibling.value.values.pop())
                        parent.value.keys[child_index - 1] = node.value.keys[0]
                    else:
                        # Borrow from left for internal node
                        node.value.keys.insert(0, parent.value.keys[child_index - 1])
                        node.value.values.insert(0, sibling.value.values.pop())
                        parent.value.keys[child_index - 1] = sibling.value.keys.pop()
                else:
                    if is_leaf:
                        # Borrow from right for leaf
                        node.value.keys.append(sibling.value.keys.pop(0))
                        node.value.values.append(sibling.value.values.pop(0))
                        parent.value.keys[child_index] = sibling.value.keys[0]
                    else:
                        # Borrow from right for internal node
                        node.value.keys.append(parent.value.keys[child_index])
                        node.value.values.append(sibling.value.values.pop(0))
                        parent.value.keys[child_index] = sibling.value.keys.pop(0)
                break

            # Cannot borrow, must merge with the pre-fetched sibling
            is_leaf = len(node.value.keys) == len(node.value.values)
            if is_left_sibling:
                # Merge node into left sibling
                if is_leaf:
                    sibling.value.keys.extend(node.value.keys)
                    sibling.value.values.extend(node.value.values)
                else:
                    sibling.value.keys.append(parent.value.keys[child_index - 1])
                    sibling.value.keys.extend(node.value.keys)
                    sibling.value.values.extend(node.value.values)

                # Remove key and child pointer from parent
                parent.value.keys.pop(child_index - 1)
                parent.value.values.pop(child_index)

                # Mark merged node as removed (don't add to stash later)
                del path_nodes[node_level]
            else:
                # Merge right sibling into node
                if is_leaf:
                    node.value.keys.extend(sibling.value.keys)
                    node.value.values.extend(sibling.value.values)
                else:
                    node.value.keys.append(parent.value.keys[child_index])
                    node.value.keys.extend(sibling.value.keys)
                    node.value.values.extend(sibling.value.values)

                # Remove key and child pointer from parent
                parent.value.keys.pop(child_index)
                parent.value.values.pop(child_index + 1)

                # Remove merged sibling from siblings dict (don't add to stash)
                del siblings[child_level]

            # Move up to parent for next iteration
            node_level = level
            node = parent

        # Check if root needs replacement
        root_node = path_nodes.get(0)
        if root_node is not None and len(root_node.value.keys) == 0:
            if len(root_node.value.keys) == len(root_node.value.values):
                self.root = None
            else:
                # Root has no keys but has one child - promote child to root
                child_key, child_leaf = root_node.value.values[0]
                self.root = (child_key, child_leaf)
            # Remove old root from path_nodes
            del path_nodes[0]

        # Update root pointer if root still exists
        if self.root is not None and 0 in path_nodes:
            root_node = path_nodes[0]
            self.root = (root_node.key, root_node.leaf)

        # Flush all remaining path nodes and siblings to stash
        total_nodes = 0
        for node in path_nodes.values():
            self._stash.append(node)
            total_nodes += 1
        for sib in siblings.values():
            self._stash.append(sib)
            total_nodes += 1

        # Perform dummy operations to pad to max_accesses
        self._perform_dummy_operation(num_round=max_accesses - total_nodes)

        return deleted_value
