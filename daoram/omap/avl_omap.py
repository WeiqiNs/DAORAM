"""Defines the OMAP constructed with the AVL tree ODS."""
import copy
import math
import os
from functools import cached_property
from typing import Any, List, Tuple

from daoram.dependency import AVLData, AVLTree, BinaryTree, Data, Encryptor, Helper, InteractServer, KVPair, PathData
from daoram.omap.oblivious_search_tree import KV_LIST, ROOT, ObliviousSearchTree


class LocalNodes:
    """
    A container for managing nodes retrieved during AVL tree operations.

    Uses a dictionary with explicit parent tracking instead of a flat list,
    making rotations cleaner and eliminating the need for positional assumptions.
    """

    def __init__(self):
        self.nodes: dict[Any, Data] = {}  # key -> Data node
        self.parent_of: dict[Any, Any] = {}  # key -> parent_key
        self.root_key: Any = None
        self.path: List[Any] = []  # Keys in traversal order (root to current)

    def __len__(self) -> int:
        return len(self.nodes)

    def __bool__(self) -> bool:
        return len(self.nodes) > 0

    def add(self, node: Data, parent_key: Any = None) -> None:
        """Add a node with its parent relationship and track in path."""
        self.nodes[node.key] = node
        self.parent_of[node.key] = parent_key
        self.path.append(node.key)
        if parent_key is None:
            self.root_key = node.key

    def get(self, key: Any) -> Data:
        """Get a node by its key."""
        return self.nodes.get(key)

    def get_parent(self, key: Any) -> Data | None:
        """Get the parent node of a given key."""
        parent_key = self.parent_of.get(key)
        return self.nodes.get(parent_key) if parent_key is not None else None

    def get_parent_key(self, key: Any) -> Any:
        """Get the parent key of a given key."""
        return self.parent_of.get(key)

    def reparent(self, child_key: Any, new_parent_key: Any) -> None:
        """Update the parent relationship for a node."""
        self.parent_of[child_key] = new_parent_key
        # Update root_key if this node becomes the root
        if new_parent_key is None:
            self.root_key = child_key

    def swap_in_path(self, key1: Any, key2: Any) -> None:
        """
        Swap positions of two keys in the path (used after rotation).

        If a key is not in path (e.g., fetched during rotation), swapping is skipped.
        """
        if key1 in self.path and key2 in self.path:
            idx1 = self.path.index(key1)
            idx2 = self.path.index(key2)
            self.path[idx1], self.path[idx2] = self.path[idx2], self.path[idx1]

    def get_root(self) -> Data | None:
        """Get the root node."""
        return self.nodes.get(self.root_key) if self.root_key is not None else None

    def update_all_leaves(self, get_new_leaf) -> None:
        """
        Update leaves for all nodes and fix parent pointers in a single pass.

        :param get_new_leaf: A callable that returns a new random leaf.
        """
        for key, node in self.nodes.items():
            node.leaf = get_new_leaf()
            parent = self.get_parent(key)
            if parent:
                if parent.value.l_key == key:
                    parent.value.l_leaf = node.leaf
                elif parent.value.r_key == key:
                    parent.value.r_leaf = node.leaf

    def remove(self, key: Any) -> Data | None:
        """Remove and return a node by its key."""
        node = self.nodes.pop(key, None)
        if node:
            self.parent_of.pop(key, None)
            if key in self.path:
                self.path.remove(key)
            if self.root_key == key:
                self.root_key = None
        return node

    def update_child_in_parent(self, parent_key: Any, old_child_key: Any,
                               new_key: Any, new_leaf: Any, new_height: int) -> None:
        """
        Update a parent's child pointer to new values.

        :param parent_key: Key of the parent node.
        :param old_child_key: The current child key to identify which side (left or right).
        :param new_key: New child key (can be None to remove child).
        :param new_leaf: New child leaf.
        :param new_height: New child height.
        """
        parent = self.nodes.get(parent_key)
        if parent is None:
            return

        if parent.value.l_key == old_child_key:
            parent.value.l_key = new_key
            parent.value.l_leaf = new_leaf
            parent.value.l_height = new_height
        elif parent.value.r_key == old_child_key:
            parent.value.r_key = new_key
            parent.value.r_leaf = new_leaf
            parent.value.r_height = new_height

    def replace_node_key(self, old_key: Any, new_key: Any) -> None:
        """
        Replace a node's key in all tracking structures.

        Used when a node's key changes (e.g., during delete with two children).

        :param old_key: The current key of the node.
        :param new_key: The new key for the node.
        """
        if old_key not in self.nodes:
            return

        # Update nodes dict
        self.nodes[new_key] = self.nodes.pop(old_key)

        # Update parent_of dict
        self.parent_of[new_key] = self.parent_of.pop(old_key)

        # Update path
        if old_key in self.path:
            idx = self.path.index(old_key)
            self.path[idx] = new_key

        # Update root_key
        if self.root_key == old_key:
            self.root_key = new_key

        # Update any children's parent references
        for key, parent_key in self.parent_of.items():
            if parent_key == old_key:
                self.parent_of[key] = new_key

    def to_list(self) -> List[Data]:
        """Return all nodes as a list (for moving to stash)."""
        return list(self.nodes.values())

    def clear(self) -> None:
        """Clear all stored nodes."""
        self.nodes.clear()
        self.parent_of.clear()
        self.root_key = None
        self.path.clear()


class AVLOmap(ObliviousSearchTree):
    def __init__(self,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 client: InteractServer,
                 name: str = "avl",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 encryptor: Encryptor = None):
        """
        Initializes the OMAP based on the AVL tree ODS.

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
        # Initialize the parent BaseOmap class.
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

        # Compute the maximum height of the AVL tree.
        self._max_height: int = math.ceil(1.44 * math.log(self._num_data, 2))

        # Override _local with LocalNodes for explicit parent tracking.
        self._local: LocalNodes = LocalNodes()

        # AVL uses a larger block size, so update disk_size for file storage.
        if self._filename and self._encryptor:
            self._disk_size = self._encryptor.ciphertext_length(self._max_block_size)

    def update_mul_tree_height(self, num_tree: int) -> None:
        """Suppose the ODS is used to store multiple trees, we update each tree's height.

        :param num_tree: Number of trees to store, which is the same as number of data in upper level oram.
        """
        # First compute how many are in the buckets, according to https://eprint.iacr.org/2021/1280.
        tree_size = math.ceil(
            math.e ** (Helper.lambert_w(math.e ** -1 * (math.log(num_tree, 2) + 128 - 1)).real + 1)
        )

        # Update the height accordingly.
        self._max_height = math.ceil(1.44 * math.log(tree_size, 2))

    @cached_property
    def _max_block_size(self) -> int:
        """Get the number of bytes equal to the size of actual data block stored in the ORAM tree."""
        return len(
            Data(
                key=os.urandom(self._key_size),
                leaf=self._num_data - 1,
                value=AVLData(
                    value=os.urandom(self._data_size),
                    r_key=os.urandom(self._key_size),
                    r_leaf=self._num_data - 1,
                    r_height=self._max_height,
                    l_key=os.urandom(self._key_size),
                    l_leaf=self._num_data - 1,
                    l_height=self._max_height
                ).dump()
            ).dump()
        ) + Helper.LENGTH_HEADER_SIZE

    def _encrypt_path_data(self, path: PathData) -> PathData:
        """
        Encrypt all data in the given PathData dict.

        Note that we first pad data to the desired length and then perform the encryption. This encryption also fills
        the bucket with the desired amount of dummy data.

        :param path: PathData dict mapping storage index to bucket.
        :return: PathData dict with encrypted buckets.
        """

        def _enc_bucket(bucket: List[Data]) -> List[bytes]:
            """Helper function to add dummy data and encrypt a bucket."""
            # First, each data's value is AVLData; we need to dump it.
            for data in bucket:
                data.value = data.value.dump()

            # Perform encryption.
            enc_bucket = [
                self._encryptor.enc(plaintext=Helper.pad_pickle(data=data.dump(), length=self._max_block_size))
                for data in bucket
            ]

            # Compute if dummy block is needed.
            dummy_needed = self._bucket_size - len(bucket)

            # If needed, perform padding.
            if dummy_needed > 0:
                enc_bucket.extend([
                    self._encryptor.enc(plaintext=Helper.pad_pickle(data=Data().dump(), length=self._max_block_size))
                    for _ in range(dummy_needed)
                ])

            return enc_bucket

        # Return the encrypted PathData dict.
        return {idx: _enc_bucket(bucket) for idx, bucket in path.items()} if self._encryptor else path

    def decrypt_path_data(self, path: PathData) -> PathData:
        """
        Decrypt all data in the given PathData dict.

        :param path: PathData dict mapping storage index to encrypted bucket.
        :return: PathData dict with decrypted buckets (dummy blocks filtered out).
        """

        def _dec_bucket(bucket: List[bytes]) -> List[Data]:
            """Helper function to decrypt a bucket and filter out dummy data."""
            # Perform decryption.
            dec_bucket = [
                dec for data in bucket
                if (dec := Data.load_unpad(self._encryptor.dec(ciphertext=data))).key is not None
            ]

            # Load data as AVLData.
            for data in dec_bucket:
                data.value = AVLData.from_pickle(data=data.value)

            return dec_bucket

        # Return the decrypted PathData dict.
        return {idx: _dec_bucket(bucket) for idx, bucket in path.items()} if self._encryptor else path

    def _get_avl_data(self, key: Any, value: Any) -> Data:
        """From the input key and value, create the Data object that should be stored in the AVL tree ORAM."""
        # In the AVLData, only value needs to be filled.
        return Data(key=key, leaf=self._get_new_leaf(), value=AVLData(value=value))

    def _move_node_to_local(self, key: Any, leaf: int, parent_key: Any = None) -> None:
        """
        Retrieve a node and add it to local with parent tracking.

        :param key: Search key of interest.
        :param leaf: Indicate which path the data is stored in the ORAM.
        :param parent_key: The key of the parent node (None if this is the root).
        """
        self._move_node_to_local_without_eviction(key=key, leaf=leaf, parent_key=parent_key)

        # Perform eviction and write the path back.
        self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=[leaf]))
        self._client.execute()

    def _move_node_to_local_without_eviction(self, key: Any, leaf: int, parent_key: Any = None) -> None:
        """
        Retrieve a node and add it to local without eviction.

        :param key: Search key of interest.
        :param leaf: Indicate which path the data is stored in the ORAM.
        :param parent_key: The key of the parent node (None if this is the root).
        """
        found = False
        to_index = len(self._stash)

        # Read the path from the server.
        self._client.add_read_path(label=self._name, leaves=[leaf])
        result = self._client.execute()
        path_data = result.results[self._name]

        # Decrypt the path.
        path = self.decrypt_path_data(path=path_data)

        # Find the desired data in the path.
        for bucket in path.values():
            for data in bucket:
                if data.key == key:
                    self._local.add(node=data, parent_key=parent_key)
                    found = True
                else:
                    self._stash.append(data)

        # Check if stash overflows.
        if len(self._stash) > self._stash_size:
            raise OverflowError(
                f"Stash overflow in {self._name}: size {len(self._stash)} exceeds max {self._stash_size}.")

        # If the desired data is not found in the path, check the stash.
        if not found:
            stash_idx = self._find_in_stash(key)
            if 0 <= stash_idx < to_index:
                self._local.add(node=self._stash[stash_idx], parent_key=parent_key)
                del self._stash[stash_idx]
                return

            raise KeyError(f"The search key {key} is not found.")

    def _flush_local_to_stash(self) -> None:
        """Move all nodes from local to stash and clear local."""
        self._stash += self._local.to_list()
        self._local.clear()

    def _init_ods_storage(self, data: KV_LIST) -> BinaryTree:
        """
        Initialize a binary tree storage to store the AVL tree holding input key-value pairs.

        :param data: A list of key-value pairs.
        :return: The binary tree storage for the input list of key-value pairs.
        """
        # Create the binary tree object.
        tree = BinaryTree(
            filename=self._filename,
            num_data=self._num_data,
            disk_size=self._disk_size,
            bucket_size=self._bucket_size,
            data_size=self._max_block_size,
            encryption=True if self._encryptor else False,
        )

        # Insert all the provided KV pairs to the AVL tree.
        if data:
            # Create an empty root and the avl tree object.
            root = None
            avl_tree = AVLTree(leaf_range=self._leaf_range)

            # Insert the kv pairs to the AVL tree.
            for kv_pair in data:
                # Convert tuple to KVPair if needed.
                if isinstance(kv_pair, tuple):
                    kv_pair = KVPair(key=kv_pair[0], value=kv_pair[1])
                root = avl_tree.recursive_insert(root=root, kv_pair=kv_pair)

            # Get node from avl tree and fill them to the oram storage.
            for avl_data in avl_tree.get_data_list(root=root, encryption=True if self._encryptor else False):
                tree.fill_data_to_storage_leaf(data=avl_data)

            # Store the key and its path in the oram.
            self.root = (root.key, root.leaf)

        # Encryption and fill with dummy data if needed.
        if self._encryptor:
            tree.storage.encrypt(encryptor=self._encryptor)

        return tree

    def _init_mul_tree_ods_storage(self, data_list: List[KV_LIST]) -> Tuple[BinaryTree, List[ROOT]]:
        """
        Initialize a binary tree storage to store multiple AVL trees holding input lists of key-value pairs.

        :param data_list: A list of lists of key-value pairs.
        :return: The binary tree storage for the input list of key-value pairs and a list of AVL tree roots.
        """
        # Create the binary tree object.
        tree = BinaryTree(
            filename=self._filename,
            num_data=self._num_data,
            disk_size=self._disk_size,
            bucket_size=self._bucket_size,
            data_size=self._max_block_size,
            encryption=True if self._encryptor else False,
        )

        # Create a root list for each AVL tree.
        root_list = []

        # Enumerate each key-pair list in the input list.
        for index, data in enumerate(data_list):
            # Insert all data to the AVL tree.
            if data:
                # Create an empty root and the avl tree object.
                root = None
                avl_tree = AVLTree(leaf_range=self._leaf_range)

                # Insert the kv pairs to the AVL tree.
                for kv_pair in data:
                    # Convert tuple to KVPair if needed.
                    if isinstance(kv_pair, tuple):
                        kv_pair = KVPair(key=kv_pair[0], value=kv_pair[1])
                    root = avl_tree.recursive_insert(root=root, kv_pair=kv_pair)

                # Get node from avl tree and fill them to the oram storage.
                for avl_data in avl_tree.get_data_list(root=root, encryption=True if self._encryptor else False):
                    tree.fill_data_to_storage_leaf(data=avl_data)

                # Store the key and its path to the root list.
                root_list.append((root.key, root.leaf))

            else:
                # Otherwise just append None.
                root_list.append(None)

        # Encryption and fill with dummy data if needed.
        if self._encryptor:
            tree.storage.encrypt(encryptor=self._encryptor)

        return tree, root_list

    def _update_height(self) -> None:
        """Traverse the nodes in local (bottom-up) and update their heights accordingly."""
        for node_key in reversed(self._local.path):
            node = self._local.get(node_key)

            # Calculate new height based on children
            l_height = node.value.l_height if node.value.l_key is not None else 0
            r_height = node.value.r_height if node.value.r_key is not None else 0
            new_height = 1 + max(l_height, r_height)

            # Update the parent's record of this node's height
            parent = self._local.get_parent(node_key)
            if parent:
                if parent.value.l_key == node_key:
                    parent.value.l_height = new_height
                else:
                    parent.value.r_height = new_height

    def _rotate_node(self, node_key: Any, rotate_left: bool) -> Tuple[Any, int, int]:
        """
        Perform an AVL rotation at the given node.

        :param node_key: Key of the node to rotate (will become child after rotation).
        :param rotate_left: True for left rotation, False for right rotation.
        :return: Tuple of (new_parent_key, new_parent_leaf, new_parent_height).
        :raises ValueError: If the pivot node is not loaded.
        """
        node = self._local.get(node_key)
        if node is None:
            raise ValueError(f"Node {node_key} not found in local.")

        # Get pivot (the child that will become parent)
        pivot_key = node.value.r_key if rotate_left else node.value.l_key
        pivot = self._local.get(pivot_key)
        if pivot is None or pivot.key != pivot_key:
            side = "Right" if rotate_left else "Left"
            raise ValueError(f"{side} node is not loaded when it is supposed to.")

        if rotate_left:
            # Left rotation: pivot's left subtree becomes node's right subtree
            node.value.r_key = pivot.value.l_key
            node.value.r_leaf = pivot.value.l_leaf
            node.value.r_height = pivot.value.l_height
            # Node becomes pivot's left child
            pivot.value.l_key = node.key
            pivot.value.l_leaf = node.leaf
            pivot.value.l_height = 1 + max(node.value.l_height, node.value.r_height)
        else:
            # Right rotation: pivot's right subtree becomes node's left subtree
            node.value.l_key = pivot.value.r_key
            node.value.l_leaf = pivot.value.r_leaf
            node.value.l_height = pivot.value.r_height
            # Node becomes pivot's right child
            pivot.value.r_key = node.key
            pivot.value.r_leaf = node.leaf
            pivot.value.r_height = 1 + max(node.value.l_height, node.value.r_height)

        # Update parent relationships: pivot takes node's position
        grandparent_key = self._local.get_parent_key(node_key)
        self._local.reparent(pivot_key, grandparent_key)
        self._local.reparent(node_key, pivot_key)

        # Swap positions in path
        self._local.swap_in_path(node_key, pivot_key)

        new_height = 1 + max(pivot.value.l_height, pivot.value.r_height)
        return pivot.key, pivot.leaf, new_height

    def _balance_node(self, node_key: Any, is_delete: bool = False) -> Tuple[Any, int, int]:
        """
        Re-balance a node if it is unbalanced.

        :param node_key: Key of the node to balance.
        :param is_delete: Whether this is during a delete operation.
        :return: The (key, leaf, height) of the node at this position after balancing.
        """
        node = self._local.get(node_key)
        balance = node.value.l_height - node.value.r_height

        # Left heavy subtree rotation.
        if balance > 1:
            child_key = node.value.l_key
            if is_delete and not self._local.get(child_key):
                self._move_node_to_local(key=child_key, leaf=node.value.l_leaf, parent_key=node_key)

            child_node = self._local.get(child_key)
            # Left-right case: first rotate left on child.
            if child_node.value.l_height - child_node.value.r_height < 0:
                grandchild_key = child_node.value.r_key
                if is_delete and not self._local.get(grandchild_key):
                    self._move_node_to_local(key=grandchild_key, leaf=child_node.value.r_leaf, parent_key=child_key)

                key, leaf, height = self._rotate_node(child_key, rotate_left=True)
                node.value.l_key, node.value.l_leaf, node.value.l_height = key, leaf, height

            # Left-left case: rotate right.
            return self._rotate_node(node_key, rotate_left=False)

        # Right heavy subtree rotation.
        if balance < -1:
            child_key = node.value.r_key
            if is_delete and not self._local.get(child_key):
                self._move_node_to_local(key=child_key, leaf=node.value.r_leaf, parent_key=node_key)

            child_node = self._local.get(child_key)
            # Right-left case: first rotate right on child.
            if child_node.value.l_height - child_node.value.r_height > 0:
                grandchild_key = child_node.value.l_key
                if is_delete and not self._local.get(grandchild_key):
                    self._move_node_to_local(key=grandchild_key, leaf=child_node.value.l_leaf, parent_key=child_key)

                key, leaf, height = self._rotate_node(child_key, rotate_left=False)
                node.value.r_key, node.value.r_leaf, node.value.r_height = key, leaf, height

            # Right-right case: rotate left.
            return self._rotate_node(node_key, rotate_left=True)

        return node.key, node.leaf, 1 + max(node.value.l_height, node.value.r_height)

    def _balance_local(self, is_delete: bool = False) -> None:
        """Balance the AVL tree path downloaded to local."""
        # Iterate from the end of the path towards the root (bottom-up)
        # Use index-based iteration since rotations modify the path
        idx = len(self._local.path) - 1

        while idx >= 0:
            node_key = self._local.path[idx]
            node = self._local.get(node_key)
            if node is None:
                idx -= 1
                continue

            original_key = node.key

            # Get balanced information.
            key, leaf, height = self._balance_node(node_key=node_key, is_delete=is_delete)

            # After rotation, the returned key is the node now at this position
            # Update parent's pointer if parent exists
            new_node = self._local.get(key)
            parent = self._local.get_parent(key) if new_node else None

            if parent:
                # If node was the right child, update right child information.
                if parent.value.r_key == original_key or parent.value.r_key == key:
                    parent.value.r_key = key
                    parent.value.r_leaf = leaf
                    parent.value.r_height = height
                # Otherwise, update left child information.
                elif parent.value.l_key == original_key or parent.value.l_key == key:
                    parent.value.l_key = key
                    parent.value.l_leaf = leaf
                    parent.value.l_height = height
                else:
                    raise ValueError("This node is not connected to its parent.")

            idx -= 1

        # Update the root after balance.
        root_node = self._local.get_root()
        if root_node:
            self.root = (root_node.key, root_node.leaf)

    def insert(self, key: Any, value: Any = None) -> None:
        """
        Given key-value pair, insert the pair to the tree.

        :param key: The search key of interest.
        :param value: The value to insert.
        """
        if key is None:
            self._perform_dummy_operation(num_round=2 * self._max_height + 1)
            return
        # Create a new data block that holds the data to insert to tree.
        data_block = self._get_avl_data(key=key, value=value)

        # If the current root is empty, we simply set root as this new block.
        if self.root is None:
            # Add the data to stash and update root.
            self._stash.append(data_block)
            self.root = (data_block.key, data_block.leaf)
            # Perform the desired number of dummy operations.
            self._perform_dummy_operation(num_round=2 * self._max_height + 1)
            return

        # Make sure that the local is cleared and is empty at the moment.
        if self._local:
            raise MemoryError(
                f"Local storage in {self._name} was not emptied before operation (size={len(self._local)}).")

        # Get the root node from oram storage.
        self._move_node_to_local(key=self.root[0], leaf=self.root[1], parent_key=None)
        current_key = self.root[0]

        # Keep adding node to local until we find a place to insert the new node.
        while True:
            node = self._local.get(current_key)

            # If a node key is smaller, we go right and check whether a child is already there.
            if node.key < key:
                # If the child is there, we keep grabbing the next node.
                if node.value.r_key is not None:
                    self._move_node_to_local(key=node.value.r_key, leaf=node.value.r_leaf, parent_key=current_key)
                    current_key = node.value.r_key
                # Else we link the parent with the new node.
                else:
                    node.value.r_key = data_block.key
                    break

            # If the key is not smaller, we go left and check the same as above.
            else:
                if node.value.l_key is not None:
                    self._move_node_to_local(key=node.value.l_key, leaf=node.value.l_leaf, parent_key=current_key)
                    current_key = node.value.l_key
                else:
                    node.value.l_key = data_block.key
                    break

        # Add the newly inserted node to local.
        self._local.add(node=data_block, parent_key=current_key)
        # Update the heights
        self._update_height()
        # Update the leaves
        self._local.update_all_leaves(self._get_new_leaf)
        # Perform balance
        self._balance_local()

        # Save the number of retrieved nodes, move the local nodes to stash and perform dummy evictions.
        num_retrieved_nodes = len(self._local)
        self._stash += self._local.to_list()
        self._local.clear()
        self._perform_dummy_operation(num_round=3 * self._max_height + 1 - num_retrieved_nodes)

    def search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value.

        If the input value is not None, the value corresponding to the search tree will be updated.
        :param key: The search key of interest.
        :param value: The updated value.
        :return: The (old) value corresponding to the search key.
        """
        # If the current root is empty, we can't perform search.
        if self.root is None:
            raise ValueError("Cannot search in an empty tree.")

        # Otherwise get information about node.
        self._move_node_to_local(key=self._root[0], leaf=self._root[1], parent_key=None)
        current_key = self._root[0]

        # Find the desired search key.
        node = self._local.get(current_key)
        while node.key != key:
            # If a node key is smaller, we go right and check whether a child is already there.
            if node.key < key:
                if node.value.r_key is not None:
                    self._move_node_to_local(key=node.value.r_key, leaf=node.value.r_leaf, parent_key=current_key)
                    current_key = node.value.r_key
                else:
                    break
            # If the key is not smaller, we go left and check the same as above.
            else:
                if node.value.l_key is not None:
                    self._move_node_to_local(key=node.value.l_key, leaf=node.value.l_leaf, parent_key=current_key)
                    current_key = node.value.l_key
                else:
                    break
            # Update the node to keep searching.
            node = self._local.get(current_key)

        # Get the desired search value and update the stored value if needed.
        if node.key == key:
            search_value = node.value.value
        else:
            search_value = None
        if value is not None:
            node.value.value = value

        # Update new leaves.
        self._local.update_all_leaves(self._get_new_leaf)

        # Update the root stored.
        root_node = self._local.get_root()
        self.root = (root_node.key, root_node.leaf)

        # Save the number of retrieved nodes, move the local nodes to stash and perform dummy evictions.
        num_retrieved_nodes = len(self._local)
        self._stash += self._local.to_list()
        self._local.clear()
        self._perform_dummy_operation(num_round=3 * self._max_height + 1 - num_retrieved_nodes)

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
        if self.root is None:
            raise ValueError("Cannot search in an empty tree.")

        self._move_node_to_local_without_eviction(key=self.root[0], leaf=self.root[1], parent_key=None)
        current_key = self.root[0]

        old_child_path = self.root[1]
        child_leaf = self._get_new_leaf()
        self.root = (self.root[0], child_leaf)
        num_retrieved_nodes = 1

        current = self._local.get(current_key)
        while current.key != key:
            num_retrieved_nodes += 1
            go_right = current.key < key
            next_key = current.value.r_key if go_right else current.value.l_key

            if next_key is None:
                break

            # Deep copy and update leaf.
            node_to_return = copy.deepcopy(current)
            node_to_return.leaf = child_leaf
            child_leaf = self._get_new_leaf()

            # Update the child leaf pointer.
            if go_right:
                node_to_return.value.r_leaf = child_leaf
            else:
                node_to_return.value.l_leaf = child_leaf

            # Write back to server.
            self._stash.append(node_to_return)
            self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=[old_child_path]))
            self._client.execute()

            # Get next node and update old_child_path.
            next_leaf = current.value.r_leaf if go_right else current.value.l_leaf
            self._move_node_to_local_without_eviction(key=next_key, leaf=next_leaf, parent_key=current_key)
            old_child_path = next_leaf
            self._local.remove(current_key)
            current_key = next_key
            current = self._local.get(current_key)

        # Get result and optionally update value.
        search_value = current.value.value if current.key == key else None
        if value is not None:
            current.value.value = value

        # Write final node back.
        node_to_return = copy.deepcopy(current)
        node_to_return.leaf = child_leaf
        self._stash.append(node_to_return)
        self._local.clear()
        self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=[old_child_path]))
        self._client.execute()

        self._perform_dummy_operation(num_round=self._max_height - num_retrieved_nodes)
        return search_value

    def _delete_node_from_local(self, node: Data, node_key: Any, parent_key: Any) -> Tuple[Any, bool]:
        """
        Remove a node from local storage and update tree structure.

        Handles all three AVL deletion cases:
        - Case 1: No children (leaf node)
        - Case 2: One child
        - Case 3: Two children (finds in-order predecessor/successor)

        :param node: The node to delete.
        :param node_key: The key of the node to delete.
        :param parent_key: The key of the parent node.
        :return: Tuple of (deleted_value, early_returned).
                 early_returned is True if the tree had only the root node.
        """
        deleted_value = node.value.value

        # Case 1: Node has no children (leaf node)
        if node.value.l_key is None and node.value.r_key is None:
            if len(self._local) == 1:
                self.root = None
                self._local.clear()
                return deleted_value, True

            # Remove from parent and local
            self._local.update_child_in_parent(parent_key, node_key, None, None, 0)
            self._local.remove(node_key)

        # Case 2: Node has one child
        elif node.value.l_key is None or node.value.r_key is None:
            has_left = node.value.l_key is not None
            child_key = node.value.l_key if has_left else node.value.r_key
            child_leaf = node.value.l_leaf if has_left else node.value.r_leaf
            child_height = node.value.l_height if has_left else node.value.r_height

            if len(self._local) == 1:
                self.root = (child_key, child_leaf)
                self._local.clear()
                return deleted_value, True

            # Replace in parent and remove from local
            self._local.update_child_in_parent(parent_key, node_key, child_key, child_leaf, child_height)
            self._local.remove(node_key)

        # Case 3: Node has two children
        else:
            use_predecessor = node.value.l_height > node.value.r_height
            original_key = node.key

            # Traverse to find replacement node
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

            # Copy replacement's data to the node being deleted
            replacement_node = current
            node.key = replacement_node.key
            node.value.value = replacement_node.value.value

            # Update replacement's parent to point to replacement's child
            child_key = replacement_node.value.l_key if use_predecessor else replacement_node.value.r_key
            child_leaf = replacement_node.value.l_leaf if use_predecessor else replacement_node.value.r_leaf
            child_height = (
                replacement_node.value.l_height if use_predecessor else replacement_node.value.r_height
            ) if child_key else 0
            self._local.update_child_in_parent(
                parent_of_replacement_key, replacement_node.key, child_key, child_leaf, child_height
            )
            self._local.remove(replacement_node.key)

            # Update parent of deleted node to point to new key, then update LocalNodes tracking
            self._local.update_child_in_parent(parent_key, original_key, node.key, node.leaf, 0)
            self._local.replace_node_key(original_key, node.key)

        return deleted_value, False

    def delete(self, key: Any) -> Any:
        """
        Given a search key, delete the corresponding node from the tree.

        :param key: The search key of interest.
        :return: The value of the deleted node.
        """
        if key is None:
            self._perform_dummy_operation(num_round=2 * self._max_height + 1)
            return None
        # If the current root is empty, we can't perform deletion.
        if self.root is None:
            raise ValueError("Cannot delete from an empty tree.")

        # First, find the node to delete by traversing the tree
        self._move_node_to_local(key=self.root[0], leaf=self.root[1], parent_key=None)
        current_key = self.root[0]
        node = self._local.get(current_key)

        # Find the node to delete
        while node.key != key:
            if node.key < key:
                if node.value.r_key is not None:
                    self._move_node_to_local(key=node.value.r_key, leaf=node.value.r_leaf, parent_key=current_key)
                    current_key = node.value.r_key
                    node = self._local.get(current_key)
                else:
                    # Key not found, return None
                    num_retrieved_nodes = len(self._local)
                    self._stash += self._local.to_list()
                    self._local.clear()
                    self._perform_dummy_operation(num_round=2 * self._max_height + 1 - num_retrieved_nodes)
                    return None

            else:
                if node.value.l_key is not None:
                    self._move_node_to_local(key=node.value.l_key, leaf=node.value.l_leaf, parent_key=current_key)
                    current_key = node.value.l_key
                    node = self._local.get(current_key)
                else:
                    # Key not found, return None
                    num_retrieved_nodes = len(self._local)
                    self._stash += self._local.to_list()
                    self._local.clear()
                    self._perform_dummy_operation(num_round=2 * self._max_height + 1 - num_retrieved_nodes)
                    return None

        # At this point, node contains the key to delete
        node_key = node.key
        parent_key = self._local.get_parent_key(node_key)

        # Perform the deletion using the helper
        deleted_value, early_returned = self._delete_node_from_local(node, node_key, parent_key)
        if early_returned:
            self._perform_dummy_operation(num_round=2 * self._max_height)
            return deleted_value

        # Update heights
        self._update_height()
        # Update leaves
        self._local.update_all_leaves(self._get_new_leaf)
        # Balance the tree
        self._balance_local(True)

        # Save the number of retrieved nodes, move the local nodes to stash and perform dummy evictions.
        num_retrieved_nodes = len(self._local)
        self._stash += self._local.to_list()
        self._local.clear()
        self._perform_dummy_operation(num_round=3 * self._max_height + 1 - num_retrieved_nodes)
        return deleted_value
