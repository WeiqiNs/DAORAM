"""Defines the OMAP constructed with the AVL tree ODS."""
import copy
import math
import os
from functools import cached_property
from typing import Any, List, Tuple

from daoram.dependency import AVLData, AVLTree, BinaryTree, Data, Encryptor, Helper, InteractServer, KVPair, PathData
from daoram.omap.oblivious_search_tree import KV_LIST, ROOT, ObliviousSearchTree


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
            name=name,
            client=client,
            filename=filename,
            num_data=num_data,
            key_size=key_size,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            encryptor=encryptor
        )

        # Compute the maximum height of the AVL tree.
        self._max_height: int = math.ceil(1.44 * math.log(self._num_data, 2))

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

    def _decrypt_path_data(self, path: PathData) -> PathData:
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
        """From the input key and value, create the data should be stored in the AVL tree oram."""
        # In the AVLData, only value needs to be filled.
        return Data(key=key, leaf=self._get_new_leaf(), value=AVLData(value=value))

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
            data_size=self._max_block_size,
            bucket_size=self._bucket_size,
            disk_size=self._disk_size,
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
            data_size=self._max_block_size,
            bucket_size=self._bucket_size,
            disk_size=self._disk_size,
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

    def _update_leaves(self) -> None:
        """Traverse the nodes in local and update their children leaves accordingly."""
        # Iterate backwards from len(local) - 2 down to 0.
        for i in range(len(self._local) - 1, -1, -1):
            # Set the current node as child node and sample a new leaf for it.
            child = self._local[i]
            child.leaf = self._get_new_leaf()

            # If parent exists, we update the parent information.
            if i - 1 >= 0:
                parent = self._local[i - 1]

                # Update the child leaf.
                if parent.value.l_key == child.key:
                    parent.value.l_leaf = child.leaf
                else:
                    parent.value.r_leaf = child.leaf

    def _update_leaves_one(self, index: int) -> None:
        """Update the leaves of the node at the given index and its parent."""
        # Set the current node as child node and sample a new leaf for it.
        child = self._local[index]
        child.leaf = self._get_new_leaf()

        # If parent exists, we update the parent information.
        parent = self._local[index - 1]

        # Update the child leaf.
        if parent.value.l_key == child.key:
            parent.value.l_leaf = child.leaf
        else:
            parent.value.r_leaf = child.leaf

    def _update_height(self) -> None:
        """Traverse the nodes in local and update their heights accordingly."""
        for i in range(len(self._local) - 1, -1, -1):
            node = self._local[i]

            # Calculate new height based on children
            l_height = node.value.l_height if node.value.l_key is not None else 0
            r_height = node.value.r_height if node.value.r_key is not None else 0
            new_height = 1 + max(l_height, r_height)

            # Update the node's height property
            if i > 0:  # Not root
                parent = self._local[i - 1]
                if parent.value.l_key == node.key:
                    parent.value.l_height = new_height
                else:
                    parent.value.r_height = new_height

    def _rotate(self, index: int, in_node: Data, rotate_left: bool, is_delete: bool = False) -> Tuple[Any, int, int]:
        """
        Perform a rotation at the provided input node.

        :param index: The index of the node to rotate in local.
        :param in_node: Some AVLTreeNode to rotate.
        :param rotate_left: True for left rotation, False for right rotation.
        :param is_delete: Whether this rotation is during a delete operation.
        :return: The (key, leaf, height) of the new parent node after rotation.
        """
        # Get the pivot node.
        p_node = self._local[-1] if is_delete else self._local[index + 1]

        if rotate_left:
            # Left rotation: pivot is right child, pivot's left becomes in_node's right.
            if p_node.key != in_node.value.r_key:
                raise ValueError("Right node is not loaded when it is supposed to.")

            # Move pivot's left child to in_node's right position.
            in_node.value.r_key = p_node.value.l_key
            in_node.value.r_leaf = p_node.value.l_leaf
            in_node.value.r_height = p_node.value.l_height

            # Set in_node as pivot's left child.
            p_node.value.l_key = in_node.key
            p_node.value.l_leaf = in_node.leaf
            p_node.value.l_height = 1 + max(in_node.value.l_height, in_node.value.r_height)
        else:
            # Right rotation: pivot is left child, pivot's right becomes in_node's left.
            if p_node.key != in_node.value.l_key:
                raise ValueError("Left node is not loaded when it is supposed to.")

            # Move pivot's right child to in_node's left position.
            in_node.value.l_key = p_node.value.r_key
            in_node.value.l_leaf = p_node.value.r_leaf
            in_node.value.l_height = p_node.value.r_height

            # Set in_node as pivot's right child.
            p_node.value.r_key = in_node.key
            p_node.value.r_leaf = in_node.leaf
            p_node.value.r_height = 1 + max(in_node.value.l_height, in_node.value.r_height)

        # Switch nodes in local.
        if is_delete:
            self._local[-1], self._local[-2] = self._local[-2], self._local[-1]
        else:
            self._local[index], self._local[index + 1] = self._local[index + 1], self._local[index]

        return p_node.key, p_node.leaf, 1 + max(p_node.value.l_height, p_node.value.r_height)

    def _balance_node(self, index: int, node: Data, is_delete: bool = False):
        """Re-balance a node if it is unbalanced."""
        balance = node.value.l_height - node.value.r_height

        # Left heavy subtree rotation.
        if balance > 1:
            # Get left child node.
            if is_delete:
                self._move_node_to_local(key=node.value.l_key, leaf=node.value.l_leaf)
            child_node = self._local[-1] if is_delete else self._local[index + 1]
            if child_node.key != node.value.l_key:
                raise ValueError("Left node is not loaded when it is supposed to.")

            # Left-right case: first rotate left on child.
            if child_node.value.l_height - child_node.value.r_height < 0:
                if is_delete:
                    self._move_node_to_local(key=child_node.value.r_key, leaf=child_node.value.r_leaf)
                key, leaf, height = self._rotate(index + 1, child_node, rotate_left=True, is_delete=is_delete)
                node.value.l_key, node.value.l_leaf, node.value.l_height = key, leaf, height
                if is_delete:
                    self._update_leaves_one(-1)
                    self._stash.append(self._local.pop())

            # Left-left case: rotate right.
            return self._rotate(index, node, rotate_left=False, is_delete=is_delete)

        # Right heavy subtree rotation.
        if balance < -1:
            # Get right child node.
            if is_delete:
                self._move_node_to_local(key=node.value.r_key, leaf=node.value.r_leaf)
            child_node = self._local[-1] if is_delete else self._local[index + 1]
            if child_node.key != node.value.r_key:
                raise ValueError("Right node is not loaded when it is supposed to.")

            # Right-left case: first rotate right on child.
            if child_node.value.l_height - child_node.value.r_height > 0:
                if is_delete:
                    self._move_node_to_local(key=child_node.value.l_key, leaf=child_node.value.l_leaf)
                key, leaf, height = self._rotate(index + 1, child_node, rotate_left=False, is_delete=is_delete)
                node.value.r_key, node.value.r_leaf, node.value.r_height = key, leaf, height
                if is_delete:
                    self._update_leaves_one(-1)
                    self._stash.append(self._local.pop())

            # Right-right case: rotate left.
            return self._rotate(index, node, rotate_left=True, is_delete=is_delete)

        return node.key, node.leaf, 1 + max(node.value.l_height, node.value.r_height)

    def _balance_local(self, is_delete=False):
        """Balance the AVL tree path downloaded to local."""
        # Perform rotation on the nodes stored in local.
        node_index = len(self._local) - 1
        key = None
        leaf = None
        while node_index >= 0:
            # Get the current node of interest.
            node = self._local[node_index]

            # Get balanced information.
            key, leaf, height = self._balance_node(index=node_index, node=node, is_delete=is_delete)

            # If the parent exists, update which node the parent should point to.
            if node_index > 0:
                parent = self._local[node_index - 1]
                # If node was the right child, update right child information.
                if parent.value.r_key == node.key:
                    parent.value.r_key = key
                    parent.value.r_leaf = leaf
                    parent.value.r_height = height
                # Otherwise, update left child information.
                elif parent.value.l_key == node.key:
                    parent.value.l_key = key
                    parent.value.l_leaf = leaf
                    parent.value.l_height = height
                # The between parent and child is broken somehow.
                else:
                    raise ValueError("This node is not connected to a parent.")

            # Decrease the node index.
            node_index -= 1

        # Update the root after balance.
        if is_delete:
            self.root = (key, leaf)
        else:
            self.root = (self._local[0].key, self._local[0].leaf)

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
            raise MemoryError("The local storage was not emptied before this operation.")

        # Get the node information from oram storage.
        self._move_node_to_local(key=self.root[0], leaf=self.root[1])

        # Keep adding node to local until we find a place to insert the new node.
        while True:
            # Save the last node data and its value.
            node = self._local[-1]

            # If a node key is smaller, we go right and check whether a child is already there.
            if node.key < key:
                # If the child is there, we keep grabbing the next node.
                if node.value.r_key is not None:
                    self._move_node_to_local(key=node.value.r_key, leaf=node.value.r_leaf)
                # Else we store link the parent with the new node, other information will be updated during balance.
                else:
                    node.value.r_key = data_block.key
                    break

            # If the key is not smaller, we go left and check the same as above.
            else:
                if node.value.l_key is not None:
                    self._move_node_to_local(key=node.value.l_key, leaf=node.value.l_leaf)
                else:
                    node.value.l_key = data_block.key
                    break

        # Append the newly inserted node to local as well
        self._local.append(data_block)
        # Update the heights
        self._update_height()
        # Update the leaves
        self._update_leaves()
        # Perform balance 
        self._balance_local()

        # Save the number of retrieved nodes, move the local nodes to stash and perform dummy evictions.
        num_retrieved_nodes = len(self._local)
        self._stash += self._local
        self._local = []
        self._perform_dummy_operation(num_round=2 * self._max_height + 1 - num_retrieved_nodes)

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
            raise ValueError(f"It seems the tree is empty and can't perform search.")

        # Otherwise get information about node.
        self._move_node_to_local(key=self._root[0], leaf=self._root[1])

        # Set the node as root.
        node = self._local[-1]

        # Find the desired search key.
        while node.key != key:
            # If a node key is smaller, we go right and check whether a child is already there.
            if node.key < key:
                if node.value.r_key is not None:
                    self._move_node_to_local(key=node.value.r_key, leaf=node.value.r_leaf)
                else:
                    break
            # If the key is not smaller, we go left and check the same as above.
            else:
                if node.value.l_key is not None:
                    self._move_node_to_local(key=node.value.l_key, leaf=node.value.l_leaf)
                else:
                    break
            # Update the node to keep searching.
            node = self._local[-1]

        # Get the desired search value and update the stored value if needed.
        if node.key == key:
            search_value = node.value.value
        else:
            search_value = None
        if value is not None:
            node.value.value = value

        # Update new leaves.
        self._update_leaves()

        # Update the root stored.
        self.root = (self._local[0].key, self._local[0].leaf)

        # Save the number of retrieved nodes, move the local nodes to stash and perform dummy evictions.
        num_retrieved_nodes = len(self._local)
        self._stash += self._local
        self._local = []
        self._perform_dummy_operation(num_round=2 * self._max_height + 1 - num_retrieved_nodes)

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
            return None

        self._move_node_to_local_without_eviction(key=self.root[0], leaf=self.root[1])

        old_child_path = self.root[1]
        child_leaf = self._get_new_leaf()
        self.root = (self.root[0], child_leaf)
        num_retrieved_nodes = 1

        while self._local[0].key != key:
            num_retrieved_nodes += 1
            current = self._local[0]
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
            self._move_node_to_local_without_eviction(key=next_key, leaf=next_leaf)
            old_child_path = next_leaf
            del self._local[0]

        # Get result and optionally update value.
        search_value = self._local[0].value.value if self._local[0].key == key else None
        if value is not None:
            self._local[0].value.value = value

        # Write final node back.
        node_to_return = copy.deepcopy(self._local[0])
        node_to_return.leaf = child_leaf
        self._stash.append(node_to_return)
        self._local = []
        self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=[old_child_path]))
        self._client.execute()

        self._perform_dummy_operation(num_round=self._max_height)
        return search_value

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
            raise ValueError(f"It seems the tree is empty and can't perform deletion.")

        # First, find the node to delete by traversing the tree
        self._move_node_to_local(key=self.root[0], leaf=self.root[1])
        node = self._local[-1]
        # Track the index of the node to delete in local
        node_index = 0

        # Find the node to delete
        while node.key != key:
            if node.key < key:
                if node.value.r_key is not None:
                    self._move_node_to_local(key=node.value.r_key, leaf=node.value.r_leaf)
                    node = self._local[-1]
                    node_index = len(self._local) - 1
                else:
                    # Key not found, return None
                    num_retrieved_nodes = len(self._local)
                    self._stash += self._local
                    self._local = []
                    self._perform_dummy_operation(num_round=2 * self._max_height + 1 - num_retrieved_nodes)
                    return None

            else:
                if node.value.l_key is not None:
                    self._move_node_to_local(key=node.value.l_key, leaf=node.value.l_leaf)
                    node = self._local[-1]
                    node_index = len(self._local) - 1
                else:
                    # Key not found, return None
                    num_retrieved_nodes = len(self._local)
                    self._stash += self._local
                    self._local = []
                    self._perform_dummy_operation(num_round=2 * self._max_height + 1 - num_retrieved_nodes)
                    return None

        # At this point, node contains the key to delete
        deleted_value = node.value.value
        # Case 1: Node has no children (leaf node)
        if node.value.l_key is None and node.value.r_key is None:
            # If deleting the root
            if len(self._local) == 1:
                self.root = None
                self._local = []
                self._perform_dummy_operation(num_round=2 * self._max_height)
                return deleted_value

            if node_index != len(self._local) - 1:
                raise ValueError("node_index is not the last index in local")

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
            # Find the replacement child
            child_key = node.value.l_key if node.value.l_key is not None else node.value.r_key
            child_leaf = node.value.l_leaf if node.value.l_key is not None else node.value.r_leaf
            child_height = node.value.l_height if node.value.l_key is not None else node.value.r_height

            # If deleting the root
            if len(self._local) == 1:
                # update root
                self.root = (child_key, child_leaf)
                self._local = []
                self._perform_dummy_operation(num_round=2 * self._max_height)
                return deleted_value

            else:
                if node_index != len(self._local) - 1:
                    raise ValueError("node_index is not the last index in local")
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

                # Remove the successor node
                self._local.pop()

        # Case 3: Node has two children; choose based on subtree height.
        else:
            # Use predecessor (left then all right) if left is taller, else successor (right then all left).
            use_predecessor = node.value.l_height > node.value.r_height

            # Save reference to parent of deleted node and original key.
            parent_of_node = self._local[node_index - 1] if node_index > 0 else None
            original_key = node.key

            # Go to the taller subtree.
            current_key = node.value.l_key if use_predecessor else node.value.r_key
            current_leaf = node.value.l_leaf if use_predecessor else node.value.r_leaf
            self._move_node_to_local(key=current_key, leaf=current_leaf)
            current = self._local[-1]

            # Traverse in opposite direction to find replacement.
            next_key = current.value.r_key if use_predecessor else current.value.l_key
            while next_key is not None:
                next_leaf = current.value.r_leaf if use_predecessor else current.value.l_leaf
                self._move_node_to_local(key=next_key, leaf=next_leaf)
                current = self._local[-1]
                next_key = current.value.r_key if use_predecessor else current.value.l_key

            # Replacement node is the last in local.
            replacement_node = self._local[-1]
            replacement_index = len(self._local) - 1

            # Copy replacement's data to the node being deleted.
            node.key = replacement_node.key
            node.value.value = replacement_node.value.value

            # Get the child to replace with (opposite of traversal direction).
            child_key = replacement_node.value.l_key if use_predecessor else replacement_node.value.r_key
            child_leaf = replacement_node.value.l_leaf if use_predecessor else replacement_node.value.r_leaf
            child_height = (
                replacement_node.value.l_height if use_predecessor else replacement_node.value.r_height) if child_key else 0

            # Update parent's pointer.
            parent = self._local[replacement_index - 1]
            if parent.value.l_key == replacement_node.key:
                parent.value.l_key = child_key
                parent.value.l_leaf = child_leaf
                parent.value.l_height = child_height
            else:
                parent.value.r_key = child_key
                parent.value.r_leaf = child_leaf
                parent.value.r_height = child_height

            # Remove replacement node from local.
            self._local.pop()

            # Update parent of the deleted node to point to new key.
            if parent_of_node is not None:
                if parent_of_node.value.l_key == original_key:
                    parent_of_node.value.l_key = node.key
                else:
                    parent_of_node.value.r_key = node.key

        # Update heights
        self._update_height()
        # Update leaves
        self._update_leaves()
        # Balance the tree
        self._balance_local(True)

        # Save the number of retrieved nodes, move the local nodes to stash and perform dummy evictions.
        num_retrieved_nodes = len(self._local)
        self._stash += self._local
        self._local = []
        self._perform_dummy_operation(num_round=2 * self._max_height + 1 - num_retrieved_nodes)
        return deleted_value
