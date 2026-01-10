from typing import Any

from daoram.dependency import Encryptor, InteractServer
from daoram.omap import AVLOmap


class AVLOmapOptimized(AVLOmap):
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
        """
        Initializes the optimized OMAP based on the AVL tree ODS.

        This version uses stash as a cache - checking stash before fetching from server
        to reduce ORAM accesses when consecutive operations access overlapping paths.

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
            client=client,
            num_data=num_data,
            key_size=key_size,
            filename=filename,
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

    def _flush_local_to_stash(self) -> None:
        """Move all nodes from local to stash and clear local."""
        self._stash += self._local
        self._local = []

    def insert(self, key: Any, value: Any = None) -> None:
        """
        Given key-value pair, insert the pair to the tree.

        :param key: The search key of interest.
        :param value: The value to insert.
        """
        if key is None:
            self._perform_dummy_operation(num_round=self._max_height + 1)
            return

        # Create a new data block
        data_block = self._get_avl_data(key=key, value=value)

        # If root is empty, set root as this new block
        if self.root is None:
            self._stash.append(data_block)
            self.root = (data_block.key, data_block.leaf)
            self._perform_dummy_operation(num_round=self._max_height + 1)
            return

        # Move any cached nodes to stash
        self._flush_local_to_stash()

        # Get root node (check stash first)
        self._move_node_to_local_cached(key=self.root[0], leaf=self.root[1])

        # Traverse to find insertion point
        while True:
            node = self._local[-1]

            if node.key < key:
                if node.value.r_key is not None:
                    self._move_node_to_local_cached(key=node.value.r_key, leaf=node.value.r_leaf)
                else:
                    node.value.r_key = data_block.key
                    break
            else:
                if node.value.l_key is not None:
                    self._move_node_to_local_cached(key=node.value.l_key, leaf=node.value.l_leaf)
                else:
                    node.value.l_key = data_block.key
                    break

        # Append new node and rebalance
        self._local.append(data_block)
        self._update_height()
        self._update_leaves()
        self._balance_local()

        # Move local to stash and do dummy operations
        num_retrieved = len(self._local)
        self._flush_local_to_stash()
        self._perform_dummy_operation(num_round=self._max_height + 1 - num_retrieved)

    def search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value.

        If input value is not None, the value corresponding to the search key will be updated.

        :param key: The search key of interest.
        :param value: The updated value.
        :return: The (old) value corresponding to the search key.
        """
        if key is None:
            self._perform_dummy_operation(num_round=self._max_height + 1)
            return None

        if self.root is None:
            self._perform_dummy_operation(num_round=self._max_height + 1)
            return None

        # Move any cached nodes to stash
        self._flush_local_to_stash()

        # Get root node (check stash first)
        self._move_node_to_local_cached(key=self.root[0], leaf=self.root[1])

        # Traverse to find key
        node = self._local[-1]
        while node.key != key:
            if node.key < key:
                if node.value.r_key is not None:
                    self._move_node_to_local_cached(key=node.value.r_key, leaf=node.value.r_leaf)
                else:
                    break
            else:
                if node.value.l_key is not None:
                    self._move_node_to_local_cached(key=node.value.l_key, leaf=node.value.l_leaf)
                else:
                    break
            node = self._local[-1]

        # Get value and optionally update
        if node.key == key:
            search_value = node.value.value
            if value is not None:
                node.value.value = value
        else:
            search_value = None

        # Update leaves
        self._update_leaves()

        # Update root
        self.root = (self._local[0].key, self._local[0].leaf)

        # Move local to stash and do dummy operations
        num_retrieved = len(self._local)
        self._flush_local_to_stash()
        self._perform_dummy_operation(num_round=self._max_height + 1 - num_retrieved)

        return search_value

    def delete(self, key: Any) -> Any:
        """
        Given a search key, delete the corresponding node from the tree.

        :param key: The search key of interest.
        :return: The value of the deleted node.
        """
        if self.root is None:
            self._perform_dummy_operation(num_round=self._max_height + 1)
            return None

        # Move any cached nodes to stash
        self._flush_local_to_stash()

        # Get root node (check stash first)
        self._move_node_to_local_cached(key=self.root[0], leaf=self.root[1])
        node = self._local[-1]
        node_index = 0

        # Find the node to delete
        while node.key != key:
            if node.key < key:
                if node.value.r_key is not None:
                    self._move_node_to_local_cached(key=node.value.r_key, leaf=node.value.r_leaf)
                    node = self._local[-1]
                    node_index = len(self._local) - 1
                else:
                    # Key not found
                    self._update_leaves()
                    self.root = (self._local[0].key, self._local[0].leaf)
                    num_retrieved = len(self._local)
                    self._flush_local_to_stash()
                    self._perform_dummy_operation(num_round=self._max_height + 1 - num_retrieved)
                    return None
            else:
                if node.value.l_key is not None:
                    self._move_node_to_local_cached(key=node.value.l_key, leaf=node.value.l_leaf)
                    node = self._local[-1]
                    node_index = len(self._local) - 1
                else:
                    # Key not found
                    self._update_leaves()
                    self.root = (self._local[0].key, self._local[0].leaf)
                    num_retrieved = len(self._local)
                    self._flush_local_to_stash()
                    self._perform_dummy_operation(num_round=self._max_height + 1 - num_retrieved)
                    return None

        # Node found - get deleted value
        deleted_value = node.value.value

        # Case 1: Node has no children (leaf node)
        if node.value.l_key is None and node.value.r_key is None:
            if len(self._local) == 1:
                # Deleting the root
                self.root = None
                self._local = []
                self._perform_dummy_operation(num_round=self._max_height)
                return deleted_value

            # Remove the node from its parent
            parent = self._local[node_index - 1]
            if parent.value.l_key == node.key:
                parent.value.l_key = None
                parent.value.l_leaf = None
                parent.value.l_height = 0
            else:
                parent.value.r_key = None
                parent.value.r_leaf = None
                parent.value.r_height = 0

            # Remove the node from local
            self._local.pop()

        # Case 2: Node has one child
        elif node.value.l_key is None or node.value.r_key is None:
            child_key = node.value.l_key if node.value.l_key is not None else node.value.r_key
            child_leaf = node.value.l_leaf if node.value.l_key is not None else node.value.r_leaf
            child_height = node.value.l_height if node.value.l_key is not None else node.value.r_height

            if len(self._local) == 1:
                # Deleting the root
                self.root = (child_key, child_leaf)
                self._local = []
                self._perform_dummy_operation(num_round=self._max_height)
                return deleted_value

            # Replace the node with its child in the parent
            parent = self._local[node_index - 1]
            if parent.value.l_key == node.key:
                parent.value.l_key = child_key
                parent.value.l_leaf = child_leaf
                parent.value.l_height = child_height
            else:
                parent.value.r_key = child_key
                parent.value.r_leaf = child_leaf
                parent.value.r_height = child_height

            # Remove the node from local
            self._local.pop()

        # Case 3: Node has two children
        else:
            # Use predecessor if left is taller, else successor
            use_predecessor = node.value.l_height > node.value.r_height

            # Save reference to parent of deleted node and original key
            parent_of_node = self._local[node_index - 1] if node_index > 0 else None
            original_key = node.key

            # Go to the taller subtree
            current_key = node.value.l_key if use_predecessor else node.value.r_key
            current_leaf = node.value.l_leaf if use_predecessor else node.value.r_leaf
            self._move_node_to_local_cached(key=current_key, leaf=current_leaf)
            current = self._local[-1]

            # Traverse in opposite direction to find replacement
            next_key = current.value.r_key if use_predecessor else current.value.l_key
            while next_key is not None:
                next_leaf = current.value.r_leaf if use_predecessor else current.value.l_leaf
                self._move_node_to_local_cached(key=next_key, leaf=next_leaf)
                current = self._local[-1]
                next_key = current.value.r_key if use_predecessor else current.value.l_key

            # Replacement node is the last in local
            replacement_node = self._local[-1]
            replacement_index = len(self._local) - 1

            # Copy replacement's data to the node being deleted
            node.key = replacement_node.key
            node.value.value = replacement_node.value.value

            # Get the child to replace with (opposite of traversal direction)
            child_key = replacement_node.value.l_key if use_predecessor else replacement_node.value.r_key
            child_leaf = replacement_node.value.l_leaf if use_predecessor else replacement_node.value.r_leaf
            child_height = (
                replacement_node.value.l_height if use_predecessor else replacement_node.value.r_height
            ) if child_key else 0

            # Update parent's pointer
            parent = self._local[replacement_index - 1]
            if parent.value.l_key == replacement_node.key:
                parent.value.l_key = child_key
                parent.value.l_leaf = child_leaf
                parent.value.l_height = child_height
            else:
                parent.value.r_key = child_key
                parent.value.r_leaf = child_leaf
                parent.value.r_height = child_height

            # Remove replacement node from local
            self._local.pop()

            # Update parent of the deleted node to point to new key
            if parent_of_node is not None:
                if parent_of_node.value.l_key == original_key:
                    parent_of_node.value.l_key = node.key
                else:
                    parent_of_node.value.r_key = node.key

        # Update heights, leaves, and balance
        self._update_height()
        self._update_leaves()
        self._balance_local(is_delete=True)

        # Move local to stash and do dummy operations
        num_retrieved = len(self._local)
        self._flush_local_to_stash()
        self._perform_dummy_operation(num_round=self._max_height + 1 - num_retrieved)

        return deleted_value
