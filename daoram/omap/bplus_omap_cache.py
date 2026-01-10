from typing import Any, Dict, List, Tuple

from daoram.dependency import Encryptor, InteractServer
from daoram.dependency.binary_tree import Data
from daoram.omap import BPlusOmap


class BPlusOmapCached(BPlusOmap):
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
        """
        Initializes the optimized OMAP based on the B+ tree ODS.

        This version uses stash as a cache - checking stash before fetching from server
        to reduce ORAM accesses when consecutive operations access overlapping paths.

        :param order: The branching order of the B+ tree.
        :param num_data: The number of data points the oram should store.
        :param key_size: The number of bytes the random dummy key should have.
        :param data_size: The number of bytes the random dummy data should have.
        :param client: The instance we use to interact with server.
        :param name: The name of the protocol, this should be unique if multiple schemes are used together.
        :param filename: The filename to save the oram data to.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param encryptor: The encryptor to use for encryption.
        """
        super().__init__(
            name=name,
            order=order,
            client=client,
            filename=filename,
            num_data=num_data,
            key_size=key_size,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            encryptor=encryptor
        )

    def _move_node_to_local_cached(self, key: Any, leaf: int) -> None:
        """
        Move a node to local, checking stash first before fetching from server.

        If the node is in stash, use it directly (no ORAM access needed).
        If not, fetch from server as usual.

        :param key: The key of the node to move.
        :param leaf: The leaf label of the node.
        """
        stash_idx = self._find_in_stash(key)
        if stash_idx >= 0:
            # Found in stash - use it directly (no ORAM access)
            node = self._stash.pop(stash_idx)
            self._local.append(node)
        else:
            # Not in stash - fetch from server
            self._move_node_to_local(key=key, leaf=leaf)

    def _find_leaf_to_local_cached(self, key: Any) -> None:
        """Add all nodes we need to visit to local until finding the leaf storing the key.

        Uses cached version that checks stash before fetching from server.

        :param key: Search key of interest.
        """
        # Move any cached nodes to stash first.
        self._flush_local_to_stash()

        # Get the node information (check stash first).
        self._move_node_to_local_cached(key=self.root[0], leaf=self.root[1])

        # Get the node from local.
        node = self._local[0]

        # Update node leaf and root.
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)

        # While we do not reach a leaf (whose number of children keys and number of children values are the same).
        while len(node.value.keys) != len(node.value.values):
            # Sample a new leaf for updating the current storage.
            new_leaf = self._get_new_leaf()

            for index, each_key in enumerate(node.value.keys):
                # If key equals, it is on the right.
                if key == each_key:
                    # Move the next node to local (check stash first).
                    self._move_node_to_local_cached(
                        key=node.value.values[index + 1][0], leaf=node.value.values[index + 1][1]
                    )
                    # Update the current stored value.
                    node.value.values[index + 1] = (node.value.values[index + 1][0], new_leaf)
                    break
                # If the key is smaller, it is on the left.
                elif key < each_key:
                    # Move the next node to local (check stash first).
                    self._move_node_to_local_cached(
                        key=node.value.values[index][0], leaf=node.value.values[index][1]
                    )
                    # Update the current stored value.
                    node.value.values[index] = (node.value.values[index][0], new_leaf)
                    break
                # If we reached the end, it is on the right.
                elif index + 1 == len(node.value.keys):
                    # Move the next node to local (check stash first).
                    self._move_node_to_local_cached(
                        key=node.value.values[index + 1][0], leaf=node.value.values[index + 1][1]
                    )
                    # Update the current stored value.
                    node.value.values[index + 1] = (node.value.values[index + 1][0], new_leaf)
                    break

            # Update the node and its leaf.
            node = self._local[-1]
            node.leaf = new_leaf

    def _perform_one_insertion(self):
        """Perform a single insertion in local."""
        # We start from the last node.
        index = len(self._local) - 1

        # While the index is larger than 0.
        while index >= 0:
            # We check if the current node has too many values.
            if len(self._local[index].value.keys) >= 2 * self._order:
                raise MemoryError("There is an overflow to the block size.")

            # If so, we perform the insertion.
            if len(self._local[index].value.keys) >= self._order:
                # When insertion is needed, we first locate the parent.
                if index > 0:
                    # Perform the insertion.
                    self._insert_in_parent(child_node=self._local[index], parent_node=self._local[index - 1])
                    index -= 1
                # Or we need a new parent node.
                else:
                    self._create_parent(child_node=self._local[index])
                    break
            else:
                # No more splitting needed.
                break

    def insert(self, key: Any, value: Any = None) -> None:
        """
        Given key-value pair, insert the pair to the tree.

        :param key: The search key of interest.
        :param value: The value to insert.
        """
        if key is None:
            self._perform_dummy_operation(num_round=self._max_height + 1)
            return

        # If the current root is empty, we simply set root as this new block.
        if self.root is None:
            # Create a new bplus data block.
            data_block = self._get_bplus_data(keys=[key], values=[value])
            # Append data block to the stash.
            self._stash.append(data_block)
            self.root = (data_block.key, data_block.leaf)
            # Perform dummy evictions.
            self._perform_dummy_operation(num_round=self._max_height + 1)
            return

        # Get all nodes we need to visit until finding the key (with caching).
        self._find_leaf_to_local_cached(key=key)

        # Set the last node in local as leaf.
        leaf = self._local[-1]

        # Find the proper place to insert the leaf.
        for index, each_key in enumerate(leaf.value.keys):
            if key < each_key:
                leaf.value.keys = leaf.value.keys[:index] + [key] + leaf.value.keys[index:]
                leaf.value.values = leaf.value.values[:index] + [value] + leaf.value.values[index:]
                break
            elif index + 1 == len(leaf.value.keys):
                leaf.value.keys.append(key)
                leaf.value.values.append(value)
                break

        # Save the length of the local.
        num_retrieved_nodes = len(self._local)

        # Perform insertion to handle splits.
        self._perform_one_insertion()

        # Flush local to stash.
        self._flush_local_to_stash()

        # Perform desired number of dummy evictions.
        self._perform_dummy_operation(num_round=self._max_height + 1 - num_retrieved_nodes)

    def search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value.

        If the input value is not None, the value corresponding to the search key will be updated.

        :param key: The search key of interest.
        :param value: The updated value.
        :return: The (old) value corresponding to the search key.
        """
        if key is None:
            self._perform_dummy_operation(num_round=self._max_height + 1)
            return None

        # If the current root is empty, we can't perform search.
        if self.root is None:
            self._perform_dummy_operation(num_round=self._max_height + 1)
            return None

        # Get all nodes we need to visit until finding the key (with caching).
        self._find_leaf_to_local_cached(key=key)

        # Set the last node in local as leaf and set the return search value to None.
        leaf = self._local[-1]
        search_value = None

        # Search the desired key and update its value as needed.
        for index, each_key in enumerate(leaf.value.keys):
            if key == each_key:
                search_value = leaf.value.values[index]
                if value is not None:
                    leaf.value.values[index] = value
                # Terminate the loop after finding the key.
                break

        # Save the number of retrieved nodes.
        num_retrieved_nodes = len(self._local)

        # Flush local to stash and perform dummy evictions.
        self._flush_local_to_stash()
        self._perform_dummy_operation(num_round=self._max_height + 1 - num_retrieved_nodes)

        return search_value

    def fast_search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value.

        The difference here is that fast search will return the node immediately without keeping it in local.
        If the input value is not None, the value corresponding to the search tree will be updated.

        :param key: The search key of interest.
        :param value: The value to update.
        :return: The (old) value corresponding to the search key.
        """
        # Flush any cached local nodes to stash first.
        self._flush_local_to_stash()

        # Use parent's fast_search.
        return super().fast_search(key=key, value=value)

    def _find_path_for_delete_cached(self, key: Any) -> Tuple[Dict[int, Data], List[int]]:
        """
        Find the path from root to leaf for the given key, returning nodes by level.
        Uses cached version that checks stash before fetching from server.

        :param key: Search key of interest.
        :return: Tuple of (path_nodes dict keyed by level, child_indices list).
        """
        # Flush any cached local nodes to stash first.
        self._flush_local_to_stash()

        # Path nodes stored by level: {0: root, 1: child, 2: grandchild, ...}
        path_nodes: Dict[int, Data] = {}
        child_indices: List[int] = []
        level = 0

        # Get the root node (check stash first).
        self._move_node_to_local_cached(key=self.root[0], leaf=self.root[1])
        node = self._local.pop()  # Remove from local, store in our dict

        # Update node leaf and root.
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)
        path_nodes[level] = node

        # Traverse down to leaf.
        while not self._is_leaf_node(node):
            # Sample a new leaf for the child.
            new_leaf = self._get_new_leaf()

            # Find the child index to descend to.
            child_index = len(node.value.keys)
            for index, each_key in enumerate(node.value.keys):
                if key < each_key:
                    child_index = index
                    break
                elif key == each_key:
                    child_index = index + 1
                    break

            # Record the child index.
            child_indices.append(child_index)

            # Move the next node to local (check stash first).
            child_key, child_leaf = node.value.values[child_index]
            self._move_node_to_local_cached(key=child_key, leaf=child_leaf)
            child_node = self._local.pop()  # Remove from local, store in our dict

            # Update the parent's pointer to child with new leaf.
            node.value.values[child_index] = (child_key, new_leaf)

            # Update child's leaf and store in path.
            child_node.leaf = new_leaf
            level += 1
            path_nodes[level] = child_node
            node = child_node

        return path_nodes, child_indices

    def _fetch_sibling_for_delete_cached(self, parent: Data, sibling_index: int) -> Data:
        """
        Fetch a sibling node from storage for delete operation.
        Uses cached version that checks stash before fetching from server.

        :param parent: The parent node containing the sibling.
        :param sibling_index: Index of the sibling in parent's values.
        :return: The sibling node.
        """
        sibling_key, sibling_leaf = parent.value.values[sibling_index]
        new_leaf = self._get_new_leaf()

        # Move sibling to local (check stash first) then pop it out.
        self._move_node_to_local_cached(key=sibling_key, leaf=sibling_leaf)
        sibling = self._local.pop()

        # Update the sibling's leaf.
        sibling.leaf = new_leaf

        # Update parent's pointer to sibling.
        parent.value.values[sibling_index] = (sibling_key, new_leaf)

        return sibling

    def delete(self, key: Any) -> Any:
        """
        Given a search key, delete the corresponding node from the tree.

        Uses cached version that checks stash before fetching from server.

        :param key: The search key of interest.
        :return: The value of the deleted node, or None if not found.
        """
        if key is None:
            self._perform_dummy_operation(num_round=3 * self._max_height)
            return None

        # If the current root is empty, nothing to delete.
        if self.root is None:
            self._perform_dummy_operation(num_round=3 * self._max_height)
            return None

        # Compute the minimum number of keys each node should have.
        min_keys = (self._order - 1) // 2

        # Find path from root to leaf, stored by level (with caching).
        path_nodes, child_indices = self._find_path_for_delete_cached(key=key)
        leaf_level = len(path_nodes) - 1
        leaf = path_nodes[leaf_level]

        # Separately track fetched siblings (not on the original path).
        fetched_siblings: List[Data] = []

        # Find and delete key from leaf.
        key_index = None
        deleted_value = None
        for i, k in enumerate(leaf.value.keys):
            if k == key:
                key_index = i
                deleted_value = leaf.value.values[i]
                break

        # Key not found.
        if key_index is None:
            # Flush all path nodes to stash.
            for node in path_nodes.values():
                self._stash.append(node)
            self._perform_dummy_operation(num_round=3 * self._max_height - len(path_nodes))
            return None

        # Delete key and value from leaf.
        leaf.value.keys.pop(key_index)
        leaf.value.values.pop(key_index)

        # Handle root leaf case (child_indices is empty when root is the leaf).
        if not child_indices:
            if len(leaf.value.keys) == 0:
                self.root = None
            else:
                self._stash.append(leaf)
            self._perform_dummy_operation(num_round=3 * self._max_height - 1)
            return deleted_value

        # Handle underflow from bottom to top.
        node_level = leaf_level
        node = leaf

        for level in range(len(child_indices) - 1, -1, -1):
            parent = path_nodes[level]
            child_index = child_indices[level]

            # No underflow, done.
            if len(node.value.keys) >= min_keys:
                break

            # Get sibling info.
            has_left = child_index > 0
            has_right = child_index < len(parent.value.values) - 1

            left_sib = None
            right_sib = None

            # Try borrow from left sibling.
            if has_left:
                left_sib = self._fetch_sibling_for_delete_cached(parent, child_index - 1)
                fetched_siblings.append(left_sib)
                if len(left_sib.value.keys) > min_keys:
                    if self._is_leaf_node(node):
                        # Borrow from left for leaf.
                        node.value.keys.insert(0, left_sib.value.keys.pop())
                        node.value.values.insert(0, left_sib.value.values.pop())
                        parent.value.keys[child_index - 1] = node.value.keys[0]
                    else:
                        # Borrow from left for internal node.
                        node.value.keys.insert(0, parent.value.keys[child_index - 1])
                        node.value.values.insert(0, left_sib.value.values.pop())
                        parent.value.keys[child_index - 1] = left_sib.value.keys.pop()
                    break

            # Try borrow from right sibling.
            if has_right:
                right_sib = self._fetch_sibling_for_delete_cached(parent, child_index + 1)
                fetched_siblings.append(right_sib)
                if len(right_sib.value.keys) > min_keys:
                    if self._is_leaf_node(node):
                        # Borrow from right for leaf.
                        node.value.keys.append(right_sib.value.keys.pop(0))
                        node.value.values.append(right_sib.value.values.pop(0))
                        parent.value.keys[child_index] = right_sib.value.keys[0]
                    else:
                        # Borrow from right for internal node.
                        node.value.keys.append(parent.value.keys[child_index])
                        node.value.values.append(right_sib.value.values.pop(0))
                        parent.value.keys[child_index] = right_sib.value.keys.pop(0)
                    break

            # Cannot borrow, must merge. Prefer left sibling.
            if has_left and left_sib is not None:
                if self._is_leaf_node(node):
                    # Merge leaf into left sibling.
                    left_sib.value.keys.extend(node.value.keys)
                    left_sib.value.values.extend(node.value.values)
                else:
                    # Merge internal node into left sibling.
                    left_sib.value.keys.append(parent.value.keys[child_index - 1])
                    left_sib.value.keys.extend(node.value.keys)
                    left_sib.value.values.extend(node.value.values)

                # Remove key and child pointer from parent.
                parent.value.keys.pop(child_index - 1)
                parent.value.values.pop(child_index)

                # Mark merged node as removed (don't add to stash later).
                del path_nodes[node_level]

            elif has_right and right_sib is not None:
                if self._is_leaf_node(node):
                    # Merge right sibling into node.
                    node.value.keys.extend(right_sib.value.keys)
                    node.value.values.extend(right_sib.value.values)
                else:
                    # Merge right sibling into node (internal).
                    node.value.keys.append(parent.value.keys[child_index])
                    node.value.keys.extend(right_sib.value.keys)
                    node.value.values.extend(right_sib.value.values)

                # Remove key and child pointer from parent.
                parent.value.keys.pop(child_index)
                parent.value.values.pop(child_index + 1)

                # Remove merged sibling from fetched_siblings (don't add to stash).
                fetched_siblings.remove(right_sib)

            # Move up to parent for next iteration.
            node_level = level
            node = parent

        # Check if root needs replacement.
        root_node = path_nodes.get(0)
        if root_node is not None and len(root_node.value.keys) == 0:
            if self._is_leaf_node(root_node):
                self.root = None
            else:
                # Root has no keys but has one child - promote child to root.
                child_key, child_leaf = root_node.value.values[0]
                self.root = (child_key, child_leaf)
            # Remove old root from path_nodes.
            del path_nodes[0]

        # Update root pointer if root still exists.
        if self.root is not None and 0 in path_nodes:
            root_node = path_nodes[0]
            self.root = (root_node.key, root_node.leaf)

        # Flush all remaining path nodes and fetched siblings to stash.
        total_nodes = 0
        for node in path_nodes.values():
            self._stash.append(node)
            total_nodes += 1
        for sib in fetched_siblings:
            self._stash.append(sib)
            total_nodes += 1

        # Perform dummy operations.
        self._perform_dummy_operation(num_round=3 * self._max_height - total_nodes)

        return deleted_value
