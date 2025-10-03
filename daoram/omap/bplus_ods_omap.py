"""Defines the OMAP constructed with the B+ tree ODS."""

import math
import os
from functools import cached_property
from typing import Any, List, Tuple

from daoram.dependency import BinaryTree, BPlusData, BPlusTree, BPlusTreeNode, Buckets, Data, Helper, InteractServer
from daoram.omap.tree_ods_omap import KV_LIST, ROOT, TreeOdsOmap


class BPlusOdsOmap(TreeOdsOmap):
    def __init__(self,
                 order: int,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 client: InteractServer,
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
        :param filename: The filename to save the oram data to.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param aes_key: The key to use for the AES instance.
        :param num_key_bytes: The number of bytes the aes key should have.
        :param use_encryption: A boolean indicating whether to use encryption.
        """
        # Initialize the parent BaseOmap class.
        super().__init__(
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

        # Save the branching order and middle point.
        self._order: int = order
        self._mid: int = order // 2

        # Set the starting block id to 0.
        self._block_id: int = 0

        # Compute the maximum height of the B+ tree.
        self._max_height: int = math.ceil(math.log(num_data, math.ceil(order / 2)))

    def update_mul_tree_height(self, num_tree: int) -> None:
        """Suppose the ODS is used to store multiple trees, we update each tree's height.

        :param num_tree: Number of trees to store, which is the same as number of data in upper level oram.
        """
        # First compute how many are in the buckets, according to https://eprint.iacr.org/2021/1280.
        tree_size = math.ceil(
            math.e ** (Helper.lambert_w(math.e ** -1 * (math.log(num_tree, 2) + 128 - 1)).real + 1)
        )

        # Update the height accordingly.
        self._max_height = math.ceil(math.log(tree_size, math.ceil(self._order / 2)))

    @cached_property
    def _max_block_size(self) -> int:
        """Get the number of bytes equal to the size of actual data block stored in ORAM."""
        # This is a little tricky since there are two types of data.
        return max(
            len(
                Data(  # This one is the non-leaf data.
                    key=self._num_data - 1,
                    leaf=self._num_data - 1,
                    value=BPlusData(
                        # The keys are actual keys.
                        keys=[os.urandom(self._key_size) for _ in range(self._order - 1)],
                        # The values are (id, leaf), which are integers.
                        values=[(self._num_data - 1, self._num_data - 1) for _ in range(self._order)],
                    ).dump()
                ).dump()
            ),
            len(
                Data(  # This one is the leaf data.
                    key=self._num_data - 1,
                    leaf=self._num_data - 1,
                    value=BPlusData(
                        # The keys are actual keys.
                        keys=[os.urandom(self._key_size) for _ in range(self._order - 1)],
                        # The values are actual values.
                        values=[os.urandom(self._data_size) for _ in range(self._order - 1)],
                    ).dump()
                ).dump()
            )
        )

    def _encrypt_buckets(self, buckets: List[List[Data]]) -> Buckets:
        """
        Encrypt all data in given buckets.

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
                data.value = BPlusData.from_pickle(data=data.value)

            return dec_bucket

        # Return the decrypted list of lists of Data.
        return [_dec_bucket(bucket=bucket) for bucket in buckets] if self._use_encryption else buckets

    def _get_bplus_data(self, keys: Any = None, values: Any = None) -> Data:
        """From the input key and value, create the data should be stored in the B+ tree oram."""
        # Create the data block with input key and value.
        data_block = Data(key=self._block_id, leaf=self._get_new_leaf(), value=BPlusData(keys=keys, values=values))
        # Increment the block id for future use.
        self._block_id += 1
        # Return the data block.
        return data_block

    def _init_ods_storage(self, data: KV_LIST) -> BinaryTree:
        """
        Initialize a binary tree storage to store the B+ tree holding input key-value pairs.

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

        # Insert all the provided KV pairs to the B+ tree.
        if data:
            # Create an empty root and the B+ tree instance.
            root = BPlusTreeNode()
            bplus_tree = BPlusTree(order=self._order, leaf_range=self._leaf_range)

            # Insert the kv pairs to the B+ tree.
            for kv_pair in data:
                root = bplus_tree.insert(root=root, kv_pair=kv_pair)

            # Get node from B+ tree and fill them to the oram storage.
            data_list = bplus_tree.get_data_list(root=root, block_id=self._block_id, encryption=self._use_encryption)

            # Update the block id.
            self._block_id += len(data_list)

            # Fill the oram tree according to the position map.
            for bplus_data in data_list:
                # Node is a list containing leaf info and then details of the node.
                tree.fill_data_to_storage_leaf(data=bplus_data)

            # Store the key and its path in the oram.
            self.root = (root.id, root.leaf)

        # Encryption and fill with dummy data if needed.
        if self._use_encryption:
            tree.storage.encrypt(aes=self._cipher)

        return tree

    def _init_mul_tree_ods_storage(self, data_list: List[KV_LIST]) -> Tuple[BinaryTree, List[ROOT]]:
        """
        Initialize a binary tree storage to store multiple B+ trees holding input lists of key-value pairs.

        :param data_list: A list of lists of key-value pairs.
        :return: The binary tree storage for the input list of key-value pairs and a list of B+ tree roots.
        """
        # Create the binary tree object.
        tree = BinaryTree(
            filename=self._filename,
            num_data=self._num_data,
            data_size=self._max_block_size,
            bucket_size=self._bucket_size,
            enc_key_size=self._num_key_bytes if self._use_encryption else None,
        )

        # Create a root list for each B+ tree.
        root_list = []

        # Enumerate each key-pair list in the input list.
        for index, data in enumerate(data_list):
            # Insert all data to the AVL tree.
            if data:
                # Create an empty root and the B+ tree object.
                root = BPlusTreeNode()
                bplus_tree = BPlusTree(order=self._order, leaf_range=self._leaf_range)

                # Insert the kv pairs to the B+ tree.
                for kv_pair in data:
                    root = bplus_tree.insert(root=root, kv_pair=kv_pair)

                # Get node from B+ tree and fill them to the oram storage.
                data_list = bplus_tree.get_data_list(
                    root=root, block_id=self._block_id, encryption=self._use_encryption
                )

                # Update the block id.
                self._block_id += len(data_list)

                # Fill the oram tree according to the position map.
                for bplus_data in data_list:
                    # Node is a list containing leaf info and then details of the node.
                    tree.fill_data_to_storage_leaf(data=bplus_data)

                # Store the key and its path in the oram.
                root_list.append((root.id, root.leaf))
            else:
                # Otherwise just append None.
                root_list.append(None)

        # Encryption and fill with dummy data if needed.
        if self._use_encryption:
            tree.storage.encrypt(aes=self._cipher)

        return tree, root_list

    def _find_leaf(self, key: Any) -> Tuple[int, int]:
        """Find the leaf containing the desired key, return the leaf node's old path and the number of visited nodes.

        The difference here is that none of the visited nodes is added to local.
        :param key: Search key of interest.
        :return: The leaf node's old path and the total number of visited nodes.
        """
        # Make sure that the local is cleared and is empty at the moment.
        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")

        # Get the node information from oram storage.
        self._move_node_to_local_without_eviction(key=self.root[0], leaf=self.root[1])

        # Get the node from local.
        node = self._local[0]

        # Save node leaf, update it and root.
        old_leaf = node.leaf
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)

        # Assign node visited to 1.
        num_retrieved_node = 1

        # While we do not reach a leaf (whose number of children keys and number of children values are the same).
        while len(node.value.keys) != len(node.value.values):
            # Sample a new leaf for updating the current storage.
            new_leaf = self._get_new_leaf()

            for index, each_key in enumerate(node.value.keys):
                # If key equals, it is on the right.
                if key == each_key:
                    # Get the old child leaf.
                    child_key, child_leaf = node.value.values[index + 1]
                    # Update the current stored value.
                    node.value.values[index + 1] = (node.value.values[index + 1][0], new_leaf)
                    # Add the node to stash and perform eviction before grabbing the next path.
                    self._stash.append(node)
                    self._client.write_query(label="ods", leaf=old_leaf, data=self._evict_stash(leaf=old_leaf))
                    # Move the next node to local.
                    self._move_node_to_local_without_eviction(key=child_key, leaf=child_leaf)
                    break

                # If the key is smaller, it is on the left.
                elif key < each_key:
                    # Get the old child leaf.
                    child_key, child_leaf = node.value.values[index]
                    # Update the current stored value.
                    node.value.values[index] = (node.value.values[index][0], new_leaf)
                    # Add the node to stash and perform eviction before grabbing the next path.
                    self._stash.append(node)
                    self._client.write_query(label="ods", leaf=old_leaf, data=self._evict_stash(leaf=old_leaf))
                    # Move the next node to local.
                    self._move_node_to_local_without_eviction(key=child_key, leaf=child_leaf)
                    break

                # If we reached the end, it is on the right.
                elif index + 1 == len(node.value.keys):
                    # Get the old child leaf.
                    child_key, child_leaf = node.value.values[index + 1]
                    # Update the current stored value.
                    node.value.values[index + 1] = (node.value.values[index + 1][0], new_leaf)
                    # Add the node to stash and perform eviction before grabbing the next path.
                    self._stash.append(node)
                    self._client.write_query(label="ods", leaf=old_leaf, data=self._evict_stash(leaf=old_leaf))
                    # Move the next node to local.
                    self._move_node_to_local_without_eviction(key=child_key, leaf=child_leaf)
                    break

            # Delete last node, store new node leaf and update its leaf.
            del self._local[0]
            node = self._local[0]
            old_leaf = node.leaf
            node.leaf = new_leaf

            # Update the number of retrieved nodes.
            num_retrieved_node += 1

        return old_leaf, num_retrieved_node

    def _find_leaf_to_local(self, key: Any) -> None:
        """Add all nodes we need to visit to local until finding the leaf storing the key.

        :param key: Search key of interest.
        """
        # Make sure that the local is cleared and is empty at the moment.
        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")

        # Get the node information from oram storage.
        self._move_node_to_local(key=self.root[0], leaf=self.root[1])

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
                    # Move the next node to local.
                    self._move_node_to_local(key=node.value.values[index + 1][0], leaf=node.value.values[index + 1][1])
                    # Update the current stored value.
                    node.value.values[index + 1] = (node.value.values[index + 1][0], new_leaf)
                    break
                # If the key is smaller, it is on the left.
                elif key < each_key:
                    # Move the next node to local.
                    self._move_node_to_local(key=node.value.values[index][0], leaf=node.value.values[index][1])
                    # Update the current stored value.
                    node.value.values[index] = (node.value.values[index][0], new_leaf)
                    break
                # If we reached the end, it is on the right.
                elif index + 1 == len(node.value.keys):
                    # Move the next node to local.
                    self._move_node_to_local(key=node.value.values[index + 1][0], leaf=node.value.values[index + 1][1])
                    # Update the current stored value.
                    node.value.values[index + 1] = (node.value.values[index + 1][0], new_leaf)
                    break

            # Update the node and its leaf.
            node = self._local[-1]
            node.leaf = new_leaf

    def _split_node(self, node: Data) -> Tuple[int, int]:
        """
        Given a node that is full, split it depends on whether it is a leaf or not.

        Note that the node itself is modified in place and the new node is directly added to stash.
        :param node: The node whose number of keys is the same as the branching degree.
        :return: The information of the new node, i.e., its key and leaf is returned.
        """
        # We break from the middle and create left child node.
        right_node = self._get_bplus_data()

        # Depending on whether the child node is a leaf node, we break it differently.
        if len(node.value.keys) == len(node.value.values):
            # New leaf gets half of the old leaf.
            right_node.value.keys = node.value.keys[self._mid:]
            right_node.value.values = node.value.values[self._mid:]

            # The old leaf keeps on the first half.
            node.value.keys = node.value.keys[:self._mid]
            node.value.values = node.value.values[:self._mid]

        else:
            # New leaf gets half of the old leaf.
            right_node.value.keys = node.value.keys[self._mid + 1:]
            right_node.value.values = node.value.values[self._mid + 1:]

            # The old leaf keeps on the first half.
            node.value.keys = node.value.keys[:self._mid]
            node.value.values = node.value.values[:self._mid + 1]

        # Add the right node to stash.
        self._stash.append(right_node)

        # Because the nodes are modified in place, we only need to return the right one.
        return right_node.key, right_node.leaf

    def _insert_in_parent(self, child_node: Data, parent_node: Data) -> None:
        """
        Insert the child node into the parent node.

        :param child_node: A B+ tree node whose number of keys is the same as the branching degree.
        :param parent_node: A B+ tree node containing the child node.
        """
        # Store the key to insert to parent.
        insert_key = child_node.value.keys[self._mid]

        # Perform the node split.
        right_node = self._split_node(node=child_node)

        # Now we perform the actual insertion to parent.
        for index, each_key in enumerate(parent_node.value.keys):
            if insert_key < each_key:
                parent_node.value.keys = parent_node.value.keys[:index] + [insert_key] + parent_node.value.keys[index:]
                parent_node.value.values = (
                        parent_node.value.values[:index + 1] + [right_node] + parent_node.value.values[index + 1:]
                )
                # Terminate the loop after insertion.
                return
            elif index + 1 == len(parent_node.value.keys):
                parent_node.value.keys.append(insert_key)
                parent_node.value.values.append(right_node)
                # Terminate the loop after insertion.
                return

    def _create_parent(self, child_node: Data) -> None:
        """
        When a node has no parent and split is required, create a new parent node for them.

        :param child_node: A B+ tree node whose number of keys is the same as the branching degree.
        :return: A B+ tree node containing the split left and right child nodes.
        """
        # Store the key to insert to parent.
        insert_key = child_node.value.keys[self._mid]

        # Perform the node split.
        right_node = self._split_node(node=child_node)

        # Set the values.
        values = [(child_node.key, child_node.leaf), right_node]

        # Create the parent node.
        parent_node = self._get_bplus_data(keys=[insert_key], values=values)

        # Append the parent node to stash.
        self._stash.append(parent_node)

        # Update the root.
        self.root = (parent_node.key, parent_node.leaf)

    def _perform_insertion(self):
        """Perform the insertion to local nodes."""
        # We start from the last node.
        index = len(self._local) - 1

        # Iterate through the leaves.
        while index >= 0:
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

            # We may reach to a point earlier than the root to stop splitting.
            else:
                break

    def insert(self, key: Any, value: Any) -> None:
        """
        Given key-value pair, insert the pair to the tree.

        :param key: The search key of interest.
        :param value: The value to insert.
        """
        # If the current root is empty, we simply set root as this new block.
        if self.root is None:
            # Create a new bplus data block.
            data_block = self._get_bplus_data(keys=[key], values=[value])
            # Append data block to the stash.
            self._stash.append(data_block)
            self.root = (data_block.key, data_block.leaf)
            # Perform at dummy finds and dummy evictions.
            self._perform_dummy_operation(num_round=3 * self._max_height)
            return

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

        # Perform the insertion to local nodes.
        self._perform_insertion()

        # Append local data to stash and clear local.
        self._stash += self._local
        self._local = []

        # Perform the desired number of dummy evictions.
        self._perform_dummy_operation(num_round=3 * self._max_height - num_retrieved_nodes)

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

        # Get all nodes we need to visit until finding the key.
        self._find_leaf_to_local(key=key)

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

        # Save the number of retrieved nodes, move the local nodes to stash and perform dummy evictions.
        num_retrieved_nodes = len(self._local)
        self._stash += self._local
        self._local = []
        self._perform_dummy_operation(num_round=3 * self._max_height - num_retrieved_nodes)

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
            raise ValueError(f"It seems the tree is empty and can't perform search.")

        # Get all nodes we need to visit until finding the key.
        old_leaf, num_retrieved_nodes = self._find_leaf(key=key)

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

        # Save the number of retrieved nodes, move the local nodes to stash and perform dummy evictions.
        self._stash += self._local
        self._local = []
        # Perform one eviction.
        self._client.write_query(label="ods", leaf=old_leaf, data=self._evict_stash(leaf=old_leaf))
        # And then the dummy evictions.
        self._perform_dummy_operation(num_round=self._max_height - num_retrieved_nodes)

        return search_value
