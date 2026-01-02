"""Defines the OMAP constructed with the AVL tree ODS."""
import copy
import math
import os
from functools import cached_property
from typing import Any, List, Tuple

from daoram.dependency import AVLData, AVLTree, BinaryTree, Buckets, Data, Helper, InteractServer
from daoram.omap.tree_ods_omap import KV_LIST, ROOT, TreeOdsOmap


class AVLOdsOmap(TreeOdsOmap):
    def __init__(self,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 client: InteractServer,
                 name: str = "avl",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True):
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
        :param aes_key: The key to use for the AES instance.
        :param num_key_bytes: The number of bytes the aes key should have.
        :param use_encryption: A boolean indicating whether to use encryption.
        """
        # Initialize the parent BaseOmap class.
        super().__init__(
            name=name,
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

        # Compute the maximum height of the AVL tree.
        self._max_height: int = math.ceil(1.44 * math.log(self._num_data, 2))

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
        )

    def _encrypt_buckets(self, buckets: List[List[Data]]) -> Buckets:
        """
        Encrypt all data in the given buckets.

        Note that we first pad data to the desired length and then perform the encryption. This encryption also fills
        the bucket with the desired amount of dummy data.
        """

        def _enc_bucket(bucket: List[Data]) -> List[bytes]:
            """Helper function to add dummy data and encrypt a bucket."""
            # First, each data's value is AVLData; we need to dump it.
            for data in bucket:
                data.value = data.value.dump()

            # Perform encryption.
            enc_bucket = [
                self._cipher.enc(plaintext=Helper.pad_pickle(data=data.dump(), length=self._max_block_size))
                for data in bucket
            ]

            # Compute if dummy block is needed.
            dummy_needed = self._bucket_size - len(bucket)

            # If needed, perform padding.
            if dummy_needed > 0:
                enc_bucket.extend([
                    self._cipher.enc(plaintext=Helper.pad_pickle(data=Data().dump(), length=self._max_block_size))
                    for _ in range(dummy_needed)
                ])

            return enc_bucket

        # Return the encrypted list of lists of bytes.
        return [_enc_bucket(bucket=bucket) for bucket in buckets] if self._use_encryption else buckets

    def _decrypt_buckets(self, buckets: Buckets) -> List[List[Data]]:
        """Given encrypted buckets, decrypt all data in it."""

        # First decrypt the data and then load it as AVLData.
        def _dec_bucket(bucket: List[bytes]) -> List[Data]:
            """Helper function to add dummy data and encrypt a bucket."""
            # Perform encryption.
            dec_bucket = [
                dec for data in bucket
                if (dec := Data.from_pickle(
                    Helper.unpad_pickle(data=self._cipher.dec(ciphertext=data)))
                    ).key is not None
            ]

            # Load data.
            for data in dec_bucket:
                data.value = AVLData.from_pickle(data=data.value)

            return dec_bucket

        # Return the decrypted list of lists of Data.
        return [_dec_bucket(bucket=bucket) for bucket in buckets] if self._use_encryption else buckets

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
            enc_key_size=self._num_key_bytes if self._use_encryption else None,
        )

        # Insert all the provided KV pairs to the AVL tree.
        if data:
            # Create an empty root and the avl tree object.
            root = None
            avl_tree = AVLTree(leaf_range=self._leaf_range)

            # Insert the kv pairs to the AVL tree.
            for kv_pair in data:
                root = avl_tree.recursive_insert(root=root, kv_pair=kv_pair)

            # Get node from avl tree and fill them to the oram storage.
            for avl_data in avl_tree.get_data_list(root=root, encryption=self._use_encryption):
                tree.fill_data_to_storage_leaf(data=avl_data)

            # Store the key and its path in the oram.
            self.root = (root.key, root.leaf)

        # Encryption and fill with dummy data if needed.
        if self._use_encryption:
            tree.storage.encrypt(aes=self._cipher)

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
            enc_key_size=self._num_key_bytes if self._use_encryption else None,
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
                    root = avl_tree.recursive_insert(root=root, kv_pair=kv_pair)

                # Get node from avl tree and fill them to the oram storage.
                for avl_data in avl_tree.get_data_list(root=root, encryption=self._use_encryption):
                    tree.fill_data_to_storage_leaf(data=avl_data)

                # Store the key and its path to the root list.
                root_list.append((root.key, root.leaf))

            else:
                # Otherwise just append None.
                root_list.append(None)

        # Encryption and fill with dummy data if needed.
        if self._use_encryption:
            tree.storage.encrypt(aes=self._cipher)

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

    def _update_height(self) ->None:
        """Traverse the nodes in local and update their heights accordingly."""
        for i in range(len(self._local)-1, -1, -1):
            node = self._local[i]

            # Calculate new height based on children
            l_height = node.value.l_height if node.value.l_key is not None else 0
            r_height = node.value.r_height if node.value.r_key is not None else 0
            new_height = 1 + max(l_height, r_height)
                
            # Update the node's height property
            if i > 0:  # Not root
                parent = self._local[i-1]
                if parent.value.l_key == node.key:
                    parent.value.l_height = new_height
                else:
                    parent.value.r_height = new_height
            else:  # Root node
                    pass 
        
    def _rotate_left(self, index: int, in_node: Data, is_delete = False) -> Tuple[Any, int, int]:
        """
        Perform a left rotation at the provided input node.

        :param index: The index of the node to rotate in local.
        :param in_node: Some AVLTreeNode to rotate.
        :return: The (key, leaf, height) of the new parent node after rotation.
        """
        if is_delete:
            # p_node should be sibling node when deleting
            p_node = self._local[-1]
        else:
            # The right child should exist in the next location.
            p_node = self._local[index + 1]

        # Verify that this is the right node.
        if p_node.key != in_node.value.r_key:
            raise ValueError("Right node is not loaded when it is supposed to.")

        # Save the left node information to temp.
        tmp_key = p_node.value.l_key
        tmp_leaf = p_node.value.l_leaf
        tmp_height = p_node.value.l_height

        # The left child of parent node was on the right of input node.
        in_node.value.r_key = tmp_key
        in_node.value.r_leaf = tmp_leaf
        in_node.value.r_height = tmp_height

        # Now we set the input node as the left child of the parent node.
        p_node.value.l_key = in_node.key
        p_node.value.l_leaf = in_node.leaf
        p_node.value.l_height = 1 + max(in_node.value.l_height, in_node.value.r_height)

        if not is_delete:
        # Switch node stored in local around.
            self._local[index], self._local[index + 1] = self._local[index + 1], self._local[index]
        else:
            self._local[-1], self._local[-2] = self._local[-2], self._local[-1]
        # Return the new parent node.
        return p_node.key, p_node.leaf, 1 + max(p_node.value.l_height, in_node.value.r_height)

    def _rotate_right(self, index: int, in_node: Data, is_delete = False) -> Tuple[Any, int, int]:
        """
        Perform a left rotation at the provided input node.

        :param index: The index of the node to rotate in local.
        :param in_node: Some AVLTreeNode to rotate.
        :return: The (key, leaf, height) of the new parent node after rotation.
        """
        if is_delete:
            # p_node should be sibling node when deleting
            p_node = self._local[-1]
        
        else:
            # The right child should exist in the next location.
            p_node = self._local[index + 1]

        # Verify that this is the right node.
        if p_node.key != in_node.value.l_key:
            raise ValueError("Left node is not loaded when it is supposed to.")

        # Save the left node information to temp.
        tmp_key = p_node.value.r_key
        tmp_leaf = p_node.value.r_leaf
        tmp_height = p_node.value.r_height

        # The left child of parent node was on the right of input node.
        in_node.value.l_key = tmp_key
        in_node.value.l_leaf = tmp_leaf
        in_node.value.l_height = tmp_height

        # Now we set the input node as the left child of the parent node.
        p_node.value.r_key = in_node.key
        p_node.value.r_leaf = in_node.leaf
        p_node.value.r_height = 1 + max(in_node.value.l_height, in_node.value.r_height)

        # Switch node stored in local around.
        if not is_delete:
            self._local[index], self._local[index + 1] = self._local[index + 1], self._local[index]
        else:
            self._local[-1], self._local[-2] = self._local[-2], self._local[-1]
        # Return the new parent node.
        return p_node.key, p_node.leaf, 1 + max(p_node.value.l_height, in_node.value.r_height)

    def _balance_node(self, index: int, node: Data, is_delete= False):
        """Re-balance a node if it is unbalanced."""
        # Get the balance factor.
        balance = node.value.l_height - node.value.r_height
        # Left heavy subtree rotation.
        if balance > 1:
            # The left-right case. Get left node.
            # need to load the extra sibling node when deleting
            if is_delete:
                self._move_node_to_local(key=node.value.l_key, leaf=node.value.l_leaf)
                left_node = self._local[-1]
            else:
                left_node = self._local[index + 1]
            if left_node.key != node.value.l_key:
                raise ValueError("Left node is not loaded when it is supposed to.")

            # Find the balance of the left node.
            if left_node.value.l_height - left_node.value.r_height < 0:
                # need to load the extra sibling node when deleting
                if is_delete:
                    self._move_node_to_local(key=left_node.value.r_key, leaf=left_node.value.r_leaf)
                # Get the updated left node.
                key, leaf, height = self._rotate_left(index=index + 1, in_node=left_node, is_delete=is_delete)
                # Assign left node to the node.
                node.value.l_key, node.value.l_leaf, node.value.l_height = key, leaf, height
                if is_delete:
                    #update the leaf of the sibling node and move to stash
                    self._update_leaves_one(-1)
                    self._stash.append(self._local.pop())
           
            # The left-left case.
            return self._rotate_right(index=index, in_node=node, is_delete=is_delete)

        # Right heavy subtree rotation.
        if balance < -1:
            # The right-left case. Get the right node.
            if is_delete:
                self._move_node_to_local(key=node.value.r_key, leaf=node.value.r_leaf)
                right_node = self._local[-1]
            else:
                right_node = self._local[index + 1]
            if right_node.key != node.value.r_key:
                raise ValueError("Right node is not loaded when it is supposed to.")

            # Find the balance of the right node.
            if right_node.value.l_height - right_node.value.r_height > 0:
                if is_delete:
                    self._move_node_to_local(key=right_node.value.l_key, leaf=right_node.value.l_leaf)
                # Get the updated right node.
                key, leaf, height = self._rotate_right(index=index + 1, in_node=right_node, is_delete = is_delete)
                # Assign the right node to the node.
                node.value.r_key, node.value.r_leaf, node.value.r_height = key, leaf, height
                if is_delete:
                    #update the leaf of the sibling node and move to stash
                    self._update_leaves_one(-1)
                    self._stash.append(self._local.pop())

            # The right-right case.
            return self._rotate_left(index=index, in_node=node, is_delete=is_delete)

        return node.key, node.leaf, 1 + max(node.value.l_height, node.value.r_height)

    def _balance_local(self, is_delete= False):
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
            self.root = (key,leaf)
        else:
            self.root = (self._local[0].key, self._local[0].leaf)

    def insert(self, key: Any, value: Any = None) -> None:
        """
        Given key-value pair, insert the pair to the tree.

        :param key: The search key of interest.
        :param value: The value to insert.
        """
        num_retrieved_nodes = 0
        if key is None:
            self._perform_dummy_operation(num_round=2 * self._max_height + 1 - num_retrieved_nodes)
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
        # If the current root is empty, we can't perform search.
        if self.root is None:
            return None

        # If root is not None, move its content to local.
        self._move_node_to_local_without_eviction(key=self.root[0], leaf=self.root[1])

        # Save the old path and sample a new path.
        old_child_path = self.root[1]
        child_leaf = self._get_new_leaf()
        self.root = (self.root[0], child_leaf)

        # Count the number of visited nodes.
        num_retrieved_nodes = 1

        # Each node is grabbed and then returned, hence we are always getting the first node in local.
        while self._local[0].key != key:
            # Update the number of visited nodes.
            num_retrieved_nodes += 1
            if self._local[0].key < key:
                if self._local[0].value.r_key is not None:
                    # Deep copy needed because of sublist structures.
                    node_to_return = copy.deepcopy(self._local[0])
                    node_to_return.leaf = child_leaf

                    # Sample a new leaf for the next child to read and store it.
                    child_leaf = self._get_new_leaf()
                    # Update the node to write back to server.
                    node_to_return.value.r_leaf = child_leaf

                    # Append the node to stash and write them back to server.
                    self._stash.append(node_to_return)
                    self._client.write_query(
                        label=self._name, leaf=old_child_path, data=self._evict_stash(old_child_path)
                    )

                    # Get the next node to check to local.
                    self._move_node_to_local_without_eviction(
                        key=self._local[0].value.r_key, leaf=self._local[0].value.r_leaf
                    )

                    # Store the old child path and then delete the current node.
                    old_child_path = self._local[0].value.r_leaf
                    del self._local[0]
                else:
                    break
            else:
                if self._local[0].value.l_key is not None:
                    # Deep copy needed because of sublist structures.
                    node_to_return = copy.deepcopy(self._local[0])
                    node_to_return.leaf = child_leaf

                    # Sample a new leaf for the next child to read and store it.
                    child_leaf = self._get_new_leaf()
                    # Update the node to write back to server.
                    node_to_return.value.l_leaf = child_leaf

                    # Append the node to stash and write them back to server.
                    self._stash.append(node_to_return)
                    self._client.write_query(
                        label=self._name, leaf=old_child_path, data=self._evict_stash(old_child_path)
                    )

                    # Get the next node to check to local.
                    self._move_node_to_local_without_eviction(
                        key=self._local[0].value.l_key, leaf=self._local[0].value.l_leaf
                    )

                    # Store the old child path and then delete the current node.
                    old_child_path = self._local[0].value.l_leaf
                    del self._local[0]

                else:
                    break
        if self._local[0].key == key:
        # Per design, the value of interest is the only one stored in local.
            search_value = self._local[0].value.value

        else:
            search_value = None
        # If the provided value is not None, update the data.
        if value is not None:
            self._local[0].value.value = value

        # Deep copy needed because of sublist structures.
        node_to_return = copy.deepcopy(self._local[0])
        node_to_return.leaf = child_leaf

        # Add the node to stash and clear local storage.
        self._stash.append(node_to_return)
        self._local = []
        # Write the new node back to storage.
        self._client.write_query(label=self._name, leaf=old_child_path, data=self._evict_stash(old_child_path))

        # Perform desired number of dummy finds.
        self._perform_dummy_operation(num_round=2 * self._max_height + 1 - num_retrieved_nodes)

        return search_value
    
            
    def delete(self, key: Any) -> Any:
        """
        Given a search key, delete the corresponding node from the tree.

        :param key: The search key of interest.
        :return: The value of the deleted node.
        """
        if key == None:
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

            if node_index != len(self._local) -1:
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
                if node_index != len(self._local) -1:
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
                                      
        # Case 3: Node has two children
        else:
            # Find the inorder successor (largest in the left subtree)
            if len(self._local) == 1:
                # If deleting the root
                gradparent_of_predelete = copy.copy(self._local[node_index-1])

            else:
                gradparent_of_predelete = self._local[node_index-1]

            predelete = copy.copy(self._local[-1])
            left_child_key = node.value.l_key
            left_child_leaf = node.value.l_leaf
            
            # Load the left subtree to find the successor
            self._move_node_to_local(key=left_child_key, leaf=left_child_leaf)
            
            # Find the rightmost node in the left subtree (inorder successor)
            while self._local[-1].value.r_key is not None:
                self._move_node_to_local(key=self._local[-1].value.r_key, leaf=self._local[-1].value.r_leaf)
            
            # The last node in local is the inorder successor
            successor = self._local[-1]
            successor_index = len(self._local) - 1
            
            # replace the node to delete with the successor
            node.key = successor.key
            node.leaf = successor.leaf
            node.value.value = successor.value.value

            if successor_index != len(self._local) -1:
                raise ValueError("successor_index is not the last index in local")

            if successor.value.l_key is None:
                # Successor is a leaf, remove it from its parent
                if successor_index > 0:
                    parent = self._local[successor_index - 1]
                    if parent.value.l_key == successor.key:
                        parent.value.l_key = None
                        parent.value.l_leaf = None
                        parent.value.l_height = 0
                    else:
                        parent.value.r_key = None
                        parent.value.r_leaf = None
                        parent.value.r_height = 0
                self._local.pop()

            else:
                # Successor has a left child, replace successor with its left child
                child_key = successor.value.l_key
                child_leaf = successor.value.l_leaf
                child_height = successor.value.l_height
                
                if successor_index > 0:
                    parent = self._local[successor_index - 1]
                    if parent.value.l_key == successor.key:
                        parent.value.l_key = child_key
                        parent.value.l_leaf = child_leaf
                        parent.value.l_height = child_height
                    else:
                        parent.value.r_key = child_key
                        parent.value.r_leaf = child_leaf
                        parent.value.r_height = child_height
                self._local.pop()

            # Update the parent of the node to delete
            if  gradparent_of_predelete != predelete:
                if gradparent_of_predelete.value.l_key == predelete.key:
                    gradparent_of_predelete.value.l_key = node.key
                    gradparent_of_predelete.value.l_leaf = node.leaf
                        
                else:
                    gradparent_of_predelete.value.r_key = node.key
                    gradparent_of_predelete.value.r_leaf = node.leaf
                       
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