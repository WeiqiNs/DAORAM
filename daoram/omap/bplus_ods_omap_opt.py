from typing import Any

from daoram.dependency import InteractServer
from daoram.omap import BPlusOdsOmap


class BPlusOdsOmapOptimized(BPlusOdsOmap):
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
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True):
        """
        Initializes the OMAP based on the B+ tree ODS.

        :param order: The branching order of the B+ tree.
        :param num_data: The number of data points the oram should store.
        :param key_size: The number of bytes the random dummy key should have.
        :param data_size: The number of bytes the random dummy data should have.
        :param client: The instance we use to interact with server.
        :param name: The name of the protocol, this should be unique if multiple schemes are used together.
        :param filename: The filename to save the oram data to.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param aes_key: The key to use for the AES instance.
        :param num_key_bytes: The number of bytes the aes key should have.
        :param use_encryption: A boolean indicating whether to use encryption.
        """
        # Initialize the parent BaseOmap class.
        super().__init__(
            name=name,
            order=order,
            client=client,
            aes_key=aes_key,
            filename=filename,
            num_data=num_data,
            key_size=key_size,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            num_key_bytes=num_key_bytes,
            use_encryption=use_encryption
        )

    def _move_two_nodes_to_local(self, key1: Any, leaf1: int, key2: Any, leaf2: int) -> None:
        """
        Move two nodes to local storage efficiently by batching reads and writes.

        This method fetches two nodes from ORAM storage with 2 reads and 2 writes,
        but batches them to be more efficient than calling _move_node_to_local twice.
        The nodes are added to local in order: key1 first, then key2.

        :param key1: The key of the first node (child).
        :param leaf1: The leaf path of the first node.
        :param key2: The key of the second node (sibling).
        :param leaf2: The leaf path of the second node.
        """
        node1 = None
        node2 = None

        # Check stash for nodes before reading paths.
        for i in range(len(self._stash) - 1, -1, -1):
            if self._stash[i].key == key1 and node1 is None:
                node1 = self._stash[i]
                del self._stash[i]
            elif self._stash[i].key == key2 and node2 is None:
                node2 = self._stash[i]
                del self._stash[i]
            if node1 is not None and node2 is not None:
                break

        # Read paths for nodes not yet found.
        if node1 is None:
            path1 = self._decrypt_buckets(buckets=self._client.read_query(label=self._name, leaf=leaf1))
            for bucket in path1:
                for data in bucket:
                    if data.key == key1:
                        node1 = data
                    elif data.key == key2 and node2 is None:
                        node2 = data
                    else:
                        self._stash.append(data)
            # Perform eviction for path1.
            self._client.write_query(label=self._name, leaf=leaf1, data=self._evict_stash(leaf=leaf1))

        if node2 is None:
            path2 = self._decrypt_buckets(buckets=self._client.read_query(label=self._name, leaf=leaf2))
            for bucket in path2:
                for data in bucket:
                    if data.key == key2:
                        node2 = data
                    elif data.key == key1 and node1 is None:
                        node1 = data
                    else:
                        self._stash.append(data)
            # Perform eviction for path2.
            self._client.write_query(label=self._name, leaf=leaf2, data=self._evict_stash(leaf=leaf2))

        # Check stash again for any nodes still not found.
        if node1 is None or node2 is None:
            for i in range(len(self._stash) - 1, -1, -1):
                if self._stash[i].key == key1 and node1 is None:
                    node1 = self._stash[i]
                    del self._stash[i]
                elif self._stash[i].key == key2 and node2 is None:
                    node2 = self._stash[i]
                    del self._stash[i]
                if node1 is not None and node2 is not None:
                    break

        # Check if stash overflows.
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")

        # Verify both nodes were found.
        if node1 is None:
            raise KeyError(f"The search key {key1} is not found.")
        if node2 is None:
            raise KeyError(f"The search key {key2} is not found.")

        # Add to local in guaranteed order: key1 first, then key2.
        self._local.append(node1)
        self._local.append(node2)

    def _perform_one_insertion(self):
        """Perform a single insertion in local."""
        # Assume we have not performed any insertion.
        inserted = False
        # We start from the last node.
        index = len(self._local) - 1

        # While we have not inserted and the index is larger than 0.
        while not inserted and index >= 0:
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

                # Set inserted to be True.
                inserted = True

            else:
                # Decrement the index in local.
                index -= 1

    def insert(self, key: Any, value: Any) -> None:
        """
        Given key-value pair, insert the pair to the tree.

        :param key: the search key of interest.
        :param value: the value to insert.
        """
        # If the current root is empty, we simply set root as this new block.
        if self.root is None:
            # Create a new bplus data block.
            data_block = self._get_bplus_data(keys=[key], values=[value])
            # Append data block to the stash.
            self._stash.append(data_block)
            self.root = (data_block.key, data_block.leaf)
            # Perform dummy evictions.
            self._perform_dummy_operation(num_round=self._max_height + 1)
            # Terminates the function.
            return

        # If the current root is not empty, we might have cache.
        self._stash += self._local
        self._local = []

        # Get all nodes we need to visit until finding the key.
        self._find_leaf_to_local(key=key)

        # Set the last node in local as leaf.
        leaf = self._local[-1]

        # Find the proper place to insert the leaf.
        for index, each_key in enumerate(leaf.value.values):
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

        # Assume we have not performed any insertion.
        self._perform_one_insertion()

        # Perform desired number of dummy evictions.
        self._perform_dummy_operation(num_round=self._max_height + 1 - num_retrieved_nodes)

    def search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value.

        The difference here is that fast search will return the node immediately without keeping it in local.
        If the input value is not None, the value corresponding to the search tree will be updated.
        :param key: The search key of interest.
        :param value: The value to update.
        :return: The (old) value corresponding to the search key.
        """
        # If the local cache is not empty, move it to stash and evict while searching.
        if self._local:
            self._perform_insertion()
            self._stash += self._local
            self._local = []

        # Set the value.
        value = super().fast_search(key=key, value=value)

        return value

    def _find_leaf_with_siblings_to_local(self, key: Any) -> int:
        """
        Find the path to the leaf containing the key, fetching siblings at each level.

        At each internal node, we fetch both the child on the path AND one sibling (if it exists)
        so that borrowing/merging can be done without additional server interactions.
        Nodes are stored in _local with siblings interleaved.

        :param key: Search key of interest.
        :return: The number of rounds (levels traversed).
        """
        # Make sure that the local is cleared and is empty at the moment.
        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")

        # Track the number of rounds (each level = 1 round).
        num_rounds = 1

        # Get the root node from oram storage.
        self._move_node_to_local(key=self.root[0], leaf=self.root[1])

        # Get the node from local.
        node = self._local[0]

        # Update node leaf and root.
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)

        # While we do not reach a leaf (whose number of keys equals number of values).
        while len(node.value.keys) != len(node.value.values):
            # Increment round count for this level.
            num_rounds += 1

            # Sample new leaves for the child and sibling.
            new_leaf = self._get_new_leaf()
            sibling_new_leaf = self._get_new_leaf()

            # Find which child to follow and its sibling.
            child_idx = len(node.value.keys)  # Default to rightmost child.
            for index, each_key in enumerate(node.value.keys):
                if key < each_key:
                    child_idx = index
                    break
                elif key == each_key:
                    child_idx = index + 1
                    break

            # Get the child node info.
            child_key, child_leaf = node.value.values[child_idx]

            # Find sibling index (prefer left sibling, fallback to right).
            sibling_idx = -1
            if child_idx > 0:
                sibling_idx = child_idx - 1
            elif child_idx < len(node.value.values) - 1:
                sibling_idx = child_idx + 1

            if sibling_idx >= 0:
                # Fetch both child and sibling using the efficient method.
                sibling_key, sibling_leaf = node.value.values[sibling_idx]
                self._move_two_nodes_to_local(
                    key1=child_key, leaf1=child_leaf,
                    key2=sibling_key, leaf2=sibling_leaf
                )

                # Update stored leaves in parent.
                node.value.values[child_idx] = (child_key, new_leaf)
                node.value.values[sibling_idx] = (sibling_key, sibling_new_leaf)

                # Update the sibling's leaf (sibling is at -1, child is at -2).
                self._local[-1].leaf = sibling_new_leaf

                # Update the child node and its leaf.
                node = self._local[-2]
                node.leaf = new_leaf
            else:
                # No sibling exists, just fetch the child.
                self._move_node_to_local(key=child_key, leaf=child_leaf)
                node.value.values[child_idx] = (child_key, new_leaf)

                # Update the child node and its leaf.
                node = self._local[-1]
                node.leaf = new_leaf

        return num_rounds

    def _handle_underflow_with_sibling(self, node_idx: int, sibling_idx: int, parent_idx: int) -> None:
        """
        Handle underflow at node_idx using the sibling at sibling_idx.

        :param node_idx: Index of the node with underflow in _local.
        :param sibling_idx: Index of the sibling node in _local.
        :param parent_idx: Index of the parent node in _local.
        """
        node = self._local[node_idx]
        sibling = self._local[sibling_idx]
        parent = self._local[parent_idx]
        is_leaf = len(node.value.keys) == len(node.value.values)
        min_keys = self._get_min_keys(is_leaf=is_leaf)

        # Find child indices in parent.
        node_child_idx = -1
        sibling_child_idx = -1
        for i, val in enumerate(parent.value.values):
            if val[0] == node.key:
                node_child_idx = i
            if val[0] == sibling.key:
                sibling_child_idx = i

        if node_child_idx == -1 or sibling_child_idx == -1:
            raise ValueError("Could not find child/sibling in parent node.")

        # Determine if sibling is left or right.
        is_left_sibling = sibling_child_idx < node_child_idx

        # Try to borrow from sibling.
        if len(sibling.value.keys) > min_keys:
            if is_left_sibling:
                if is_leaf:
                    # Borrow last key-value from left sibling.
                    node.value.keys.insert(0, sibling.value.keys.pop())
                    node.value.values.insert(0, sibling.value.values.pop())
                    # Update parent key.
                    parent.value.keys[sibling_child_idx] = node.value.keys[0]
                else:
                    # For internal node: rotate through parent.
                    node.value.keys.insert(0, parent.value.keys[sibling_child_idx])
                    node.value.values.insert(0, sibling.value.values.pop())
                    parent.value.keys[sibling_child_idx] = sibling.value.keys.pop()
            else:
                if is_leaf:
                    # Borrow first key-value from right sibling.
                    node.value.keys.append(sibling.value.keys.pop(0))
                    node.value.values.append(sibling.value.values.pop(0))
                    # Update parent key.
                    parent.value.keys[node_child_idx] = sibling.value.keys[0]
                else:
                    # For internal node: rotate through parent.
                    node.value.keys.append(parent.value.keys[node_child_idx])
                    node.value.values.append(sibling.value.values.pop(0))
                    parent.value.keys[node_child_idx] = sibling.value.keys.pop(0)
            return

        # Must merge.
        if is_left_sibling:
            # Merge node into left sibling.
            if is_leaf:
                sibling.value.keys.extend(node.value.keys)
                sibling.value.values.extend(node.value.values)
            else:
                sibling.value.keys.append(parent.value.keys[sibling_child_idx])
                sibling.value.keys.extend(node.value.keys)
                sibling.value.values.extend(node.value.values)
            # Remove the merged child and corresponding key from parent.
            parent.value.keys.pop(sibling_child_idx)
            parent.value.values.pop(node_child_idx)
            # Mark node as merged (cleared).
            node.value.keys = []
            node.value.values = []
        else:
            # Merge right sibling into node.
            if is_leaf:
                node.value.keys.extend(sibling.value.keys)
                node.value.values.extend(sibling.value.values)
            else:
                node.value.keys.append(parent.value.keys[node_child_idx])
                node.value.keys.extend(sibling.value.keys)
                node.value.values.extend(sibling.value.values)
            # Remove the merged child and corresponding key from parent.
            parent.value.keys.pop(node_child_idx)
            parent.value.values.pop(sibling_child_idx)
            # Mark sibling as merged (cleared).
            sibling.value.keys = []
            sibling.value.values = []

    def delete(self, key: Any) -> Any:
        """
        Given a search key, delete the corresponding key-value pair from the tree.

        This optimized version fetches siblings during traversal so no extra server
        interactions are needed for borrowing/merging.

        :param key: The search key of interest.
        :return: The value of the deleted key, or None if not found.
        """
        # If the local cache is not empty, perform pending insertions first.
        if self._local:
            self._perform_insertion()
            self._stash += self._local
            self._local = []

        # If the current root is empty, we can't perform deletion.
        if self.root is None:
            raise ValueError("It seems the tree is empty and can't perform deletion.")

        # Fetch the path to the leaf with siblings.
        num_rounds = self._find_leaf_with_siblings_to_local(key=key)

        # Find the leaf node (the one that contains the key).
        leaf = None
        leaf_idx = -1
        for i in range(len(self._local) - 1, -1, -1):
            node = self._local[i]
            # Leaf nodes have same number of keys and values.
            if len(node.value.keys) == len(node.value.values):
                # Check if this leaf contains our key.
                if key in node.value.keys:
                    leaf = node
                    leaf_idx = i
                    break

        if leaf is None:
            # Key not found, perform dummy operations and return.
            self._stash += self._local
            self._local = []
            self._perform_dummy_operation(num_round=self._max_height - num_rounds)
            return None

        # Find and remove the key from the leaf.
        key_index = leaf.value.keys.index(key)
        deleted_value = leaf.value.values[key_index]
        deleted_key = leaf.value.keys.pop(key_index)
        leaf.value.values.pop(key_index)

        # Handle the case where we're deleting from root leaf.
        if len(self._local) == 1 or (len(self._local) == 2 and self._local[0].key == self._local[1].key):
            if not leaf.value.keys:
                # Tree is now empty.
                self.root = None
                self._local = []
                self._perform_dummy_operation(num_round=self._max_height - num_rounds)
                return deleted_value
            else:
                # Root leaf still has keys.
                self._stash += self._local
                self._local = []
                self._perform_dummy_operation(num_round=self._max_height - num_rounds)
                return deleted_value

        # Update parent key if the first key of the leaf changed.
        if key_index == 0 and leaf.value.keys:
            new_first_key = leaf.value.keys[0]
            for i in range(len(self._local)):
                node = self._local[i]
                if len(node.value.keys) != len(node.value.values):  # Internal node.
                    for j, k in enumerate(node.value.keys):
                        if k == deleted_key:
                            node.value.keys[j] = new_first_key
                            break

        # Handle underflow if needed.
        min_keys = self._get_min_keys(is_leaf=True)
        if len(leaf.value.keys) < min_keys:
            # Find sibling and parent for this leaf.
            sibling_idx = -1
            parent_idx = -1

            # Search for sibling (a leaf that's not the current leaf).
            for i in range(len(self._local)):
                node = self._local[i]
                if len(node.value.keys) == len(node.value.values) and i != leaf_idx:
                    sibling_idx = i
                    break

            # Search for parent (an internal node).
            for i in range(len(self._local)):
                node = self._local[i]
                if len(node.value.keys) != len(node.value.values):
                    # Check if this is the parent of the leaf.
                    for val in node.value.values:
                        if val[0] == leaf.key:
                            parent_idx = i
                            break
                    if parent_idx >= 0:
                        break

            if sibling_idx >= 0 and parent_idx >= 0:
                self._handle_underflow_with_sibling(leaf_idx, sibling_idx, parent_idx)

                # Check if parent now has underflow (for internal nodes).
                parent = self._local[parent_idx]
                min_keys_internal = self._get_min_keys(is_leaf=False)
                if not parent.value.keys:
                    # Parent is root and has no keys, promote the remaining child.
                    if parent.value.values:
                        self.root = parent.value.values[0]
                    else:
                        self.root = None

        # Move all local nodes to stash and perform dummy operations.
        self._stash += self._local
        self._local = []
        self._perform_dummy_operation(num_round=self._max_height - num_rounds)

        return deleted_value
