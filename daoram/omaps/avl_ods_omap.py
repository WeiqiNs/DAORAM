"""Defines the OMAP constructed with the AVL tree ODS."""

import copy
import math
import os
import pickle
from typing import Any, List, Tuple

from scipy.special import lambertw

from daoram.dependency.avl_tree import AVLTree, L, R
from daoram.dependency.binary_tree import BinaryTree, KEY, LEAF, VALUE
from daoram.dependency.interact_server import InteractServer
from daoram.omaps.tree_ods_omap import KV_LIST, ROOT, TreeOdsOmap

# We use these codes for better readability of code. For the value stored in a data block, V retrieves the value,
# CK retrieves children node keys, CL retrieves children node leaves, and CH retrieves children node heights.
V = 0
CK = 1
CL = 2
CH = 3


class AVLOdsOmap(TreeOdsOmap):
    def __init__(self,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 client: InteractServer,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True,
                 distinguishable_search: bool = False):
        """
        Initializes the OMAP based on the AVL tree ODS.

        :param num_data: the number of data points the oram should store.
        :param key_size: the number of bytes the random dummy key should have.
        :param data_size: the number of bytes the random dummy data should have.
        :param client: the instance we use to interact with server.
        :param bucket_size: the number of data each bucket should have.
        :param stash_scale: the scaling scale of the stash.
        :param aes_key: the key to use for the AES instance.
        :param num_key_bytes: the number of bytes the aes key should have.
        :param use_encryption: a boolean indicating whether to use encryption.
        :param distinguishable_search: a boolean indicating whether fast search can be used.
        """
        # Initialize the parent BaseOmap class.
        super().__init__(
            client=client,
            aes_key=aes_key,
            num_data=num_data,
            key_size=key_size,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            num_key_bytes=num_key_bytes,
            use_encryption=use_encryption
        )

        # Set whether we use the fast search.
        self.__distinguishable_search = distinguishable_search

        # Compute the maximum height of the AVL tree.
        self.__max_height = math.ceil(1.44 * math.log(self._num_data, 2))

        # Update the padded block size, which depends on the max height.
        self._padded_block_size = self._dummy_block_size

    def update_mul_tree_height(self, num_tree: int) -> None:
        """Suppose the ODS is used to store multiple trees, we update each tree's height.

        :param num_tree: number of tree to store, which is the same as number of data in upper level oram.
        """
        # First compute how many are in the buckets, according to https://eprint.iacr.org/2021/1280.
        tree_size = math.ceil(math.e ** (lambertw(math.e ** -1 * (math.log(num_tree, 2) + 128 - 1)).real + 1))

        # Update the height accordingly.
        self.__max_height = math.ceil(1.44 * math.log(tree_size, 2))

        # Update the padded block size, which depends on the max height.
        self._padded_block_size = self._dummy_block_size

    @property
    def _dummy_block_size(self) -> int:
        """Get the number of bytes equal to the size of actual data block stored in ORAM."""
        return len(pickle.dumps([
            os.urandom(self._key_size),
            self._num_data - 1,
            [
                os.urandom(self._data_size),  # value in the kv pair.
                [os.urandom(self._key_size), os.urandom(self._key_size)],  # children keys.
                [self._num_data, self._num_data],  # children paths.
                [self.__max_height, self.__max_height]  # children heights.
            ]
        ]))

    def __get_avl_data(self, key: Any, value: Any) -> list:
        """
        From the input key and value, create the data should be stored in the AVL tree oram.

        The data is of the following structure:
            - [key, leaf, [value, [left_key, right_key], [left_path, right_path], [left_height, right_height]]]
        """
        return [key, self._get_new_leaf(), [value, [None, None], [None, None], [0, 0]]]

    def _init_ods_storage(self, data: KV_LIST) -> BinaryTree:
        """
        Initialize a binary tree storage to store the AVL tree holding input key-value pairs.

        :param data: a list of key-value pairs.
        :return: the binary tree storage for the input list of key-value pairs.
        """
        # Create an empty root, an empty position map dictionary, an AVL tree instance, and an oram tree instance.
        root = None
        pos_map = {}
        avl_tree = AVLTree(leaf_range=self._leaf_range)
        oram_tree = BinaryTree(num_data=self._num_data, bucket_size=self._bucket_size)

        # Insert all the provided KV pairs to the root.
        if data:
            for kv_pair in data:
                root = avl_tree.recursive_insert(root=root, kv_pair=kv_pair)

            # Fill the position map.
            avl_tree.post_order(root=root, pos_map=pos_map)

            # Fill the oram tree according to the position map.
            for key, node in sorted(pos_map.items()):
                # Node is a list containing leaf info and then details of the node.
                oram_tree.fill_data_to_storage_leaf(data=[key, node[0], node[1]])

            # Store the key and its path in the oram.
            self.root = (root.key, root.path)

        # Fill the storage with dummy data.
        oram_tree.fill_storage_with_dummy_data()

        # Encrypt the tree storage if needed.
        oram_tree.storage = self._encrypt_buckets(buckets=oram_tree.storage)

        return oram_tree

    def _init_mul_tree_ods_storage(self, data_list: List[KV_LIST]) -> Tuple[BinaryTree, List[ROOT]]:
        """
        Initialize a binary tree storage to store multiple AVL trees holding input lists of key-value pairs.

        :param data_list: a list of lists of key-value pairs.
        :return: the binary tree storage for the input list of key-value pairs and a list of AVL tree roots.
        """
        # In this case, we set the max height to be smaller as each subgroup is of O(log n).
        self.__max_height = math.ceil(1.44 * math.log(math.log(self._num_data, 2), 2)) + 1

        # Create a root list, an AVL tree instance, and an oram tree instance.
        root_list = []
        avl_tree = AVLTree(leaf_range=self._leaf_range)
        oram_tree = BinaryTree(num_data=self._num_data, bucket_size=self._bucket_size)

        # Enumerate each key-pair list in the input list.
        for index, data in enumerate(data_list):
            # Set root to None and create an empty pos map for each key-pair list.
            root = None
            pos_map = {}

            # Insert all data to the AVL tree.
            if data:
                for kv_pair in data:
                    root = avl_tree.recursive_insert(root=root, kv_pair=kv_pair)

                # Fill the position map.
                avl_tree.post_order(root=root, pos_map=pos_map)

                # Fill the oram tree according to the position map.
                for key, node in pos_map.items():
                    # Node is a list containing leaf info and then details of the node.
                    oram_tree.fill_data_to_storage_leaf(data=[key, node[0], node[1]])

                # Store the key and its path to the root list.
                root_list.append((root.key, root.path))

            else:
                # Otherwise just append None.
                root_list.append(None)

        # Fill the storage with dummy data.
        oram_tree.fill_storage_with_dummy_data()

        # Encrypt the tree storage if needed.
        oram_tree.storage = self._encrypt_buckets(buckets=oram_tree.storage)

        return oram_tree, root_list

    def __update_heights(self) -> None:
        """Traverse the nodes in local and update their children heights accordingly."""
        for i in range(len(self._local) - 2, -1, -1):
            # Compute the new height of the node.
            height = 1 + max(self._local[i + 1][VALUE][CH][L], self._local[i + 1][VALUE][CH][R])
            # Update the height stored in parent nodes; check the last visited node is right or left.
            if self._local[i][VALUE][CK][L] == self._local[i + 1][KEY]:
                self._local[i][VALUE][CH][L] = height
            else:
                self._local[i][VALUE][CH][R] = height

    def __update_leaves(self) -> None:
        """Traverse the nodes in local and update their children leaves accordingly."""
        for i in range(len(self._local) - 2, -1, -1):
            # Sample a new leaf and store it.
            self._local[i][LEAF] = self._get_new_leaf()
            # Update the children path stored in parent nodes; check the last visited node is right or left.
            if self._local[i][VALUE][CK][L] == self._local[i + 1][KEY]:
                self._local[i][VALUE][CL][L] = self._local[i + 1][LEAF]
            else:
                self._local[i][VALUE][CL][R] = self._local[i + 1][LEAF]

    def __move_stash_node_to_local(self, key: Any) -> None:
        """Given key, move the corresponding data from stash to local."""
        # Iterate though the stash and find the desired key.
        for index, data in enumerate(self._stash):
            # If found, we append the value to local and delete it from stash.
            if data[KEY] == key:
                self._local.append(data)
                del self._stash[index]
                return

        # Since stash is checked after path, raise an error.
        raise KeyError(f"Key {key} not found.")

    def __move_remote_node_to_local(self, key: Any, leaf: int) -> None:
        """
        Given key and path, retrieve the path and move the data block corresponding to the leaf to local.

        :param key: search key of interest.
        :param leaf: indicate which path the data of interest is stored in the ORAM.
        """
        # Set found to false.
        found = False

        # Get the desired path and perform decryption as needed.
        path = self._decrypt_buckets(buckets=self.client.read_query(label="ods", leaf=leaf))

        # Find the desired data in the path.
        for bucket in path:
            for data in bucket:
                if data[KEY] == key:
                    # When we find the desired data, we move it to local storage.
                    self._local.append(data.copy())
                    # Label this data as dummy data; its content will eventually be deleted/replaced.
                    data[KEY] = None
                    # Set found to True.
                    found = True

        # If the required data is not found, we also check the stash.
        if not found:
            self.__move_stash_node_to_local(key=key)

        # Write the path back.
        self.client.write_query(label="ods", leaf=leaf, data=self._encrypt_buckets(buckets=path))

    def __search_move_node_to_local(self, key: Any, leaf: Any) -> None:
        """
        Given key and path, retrieve the path and move the data block corresponding to the leaf to local.

        This is specialized to the fast search method, where the data of interest are moved to local and the rest
        are added to stash.
        :param key: search key of interest.
        :param leaf: indicate which path the data of interest is stored in the ORAM.
        """
        # Set found to false and get existing stash size.
        found = False
        to_index = len(self._stash)

        # Get the desired path and perform decryption as needed.
        path = self._decrypt_buckets(buckets=self.client.read_query(label="ods", leaf=leaf))

        # Find the desired data in the path.
        for bucket in path:
            for data in bucket:
                if data[KEY] is None:
                    continue
                elif data[KEY] == key:
                    # We append the data we want to stash.
                    self._local.append(data)
                    found = True
                else:
                    # Other real data are directly added to stash.
                    self._stash.append(data)

        # Check if stash overflows.
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")

        # If the desired data is not found in the path, we check the stash.
        if not found:
            for i in range(to_index):
                # If we find the data, we add it to local and remove it from stash.
                if self._stash[i][KEY] == key:
                    self._local.append(self._stash[i])
                    del self._stash[i]
                    # Terminate the function.
                    return

            # If also not found in stash, raise an error.
            raise KeyError(f"The search key {key} is not found.")

    def __balance_local(self) -> None:
        """Balance the AVL tree nodes that are added to the local storage."""
        # Start from the third to last node and perform balance.
        for i in range(len(self._local) - 3, 0, -1):
            # Assume is not left child.
            is_left_child = False
            # Compute height balance of this node.
            balance = self._local[i][VALUE][CH][L] - self._local[i][VALUE][CH][R]
            if balance > 1:
                # Check the children node balance.
                if self._local[i + 1][VALUE][CH][L] - self._local[i + 1][VALUE][CH][R] >= 0:
                    if self._local[i - 1][VALUE][CK][L] == self._local[i][KEY]:
                        is_left_child = True
                    self._local[i][VALUE][CK][L] = self._local[i + 1][VALUE][CK][R]
                    self._local[i][VALUE][CL][L] = self._local[i + 1][VALUE][CL][R]
                    self._local[i + 1][VALUE][CK][R] = self._local[i][KEY]
                    self._local[i + 1][VALUE][CL][R] = self._local[i][LEAF]
                    if is_left_child:
                        self._local[i - 1][VALUE][CK][L] = self._local[i + 1][KEY]
                        self._local[i - 1][VALUE][CL][L] = self._local[i + 1][LEAF]
                    else:
                        self._local[i - 1][VALUE][CK][R] = self._local[i + 1][KEY]
                        self._local[i - 1][VALUE][CL][R] = self._local[i + 1][LEAF]

                    self._local[i][VALUE][CH][L] = self._local[i + 1][VALUE][CH][R]
                    self._local[i + 1][VALUE][CH][R] = 1 + max(self._local[i][VALUE][CH][L],
                                                               self._local[i][VALUE][CH][R])
                    if is_left_child:
                        self._local[i - 1][VALUE][CH][L] = 1 + max(self._local[i + 1][VALUE][CH][L],
                                                                   self._local[i + 1][VALUE][CH][R])
                    else:
                        self._local[i - 1][VALUE][CH][R] = 1 + max(self._local[i + 1][VALUE][CH][L],
                                                                   self._local[i + 1][VALUE][CH][R])
                    # Update height of the parent nodes.
                    for j in range(i - 2, -1, -1):
                        if self._local[j][VALUE][CK][L] == self._local[j + 1][KEY]:
                            self._local[j][VALUE][CH][L] = 1 + max(self._local[j + 1][VALUE][CH][L],
                                                                   self._local[j + 1][VALUE][CH][R])
                        else:
                            self._local[j][VALUE][CH][R] = 1 + max(self._local[j + 1][VALUE][CH][L],
                                                                   self._local[j + 1][VALUE][CH][R])
                    self.root = (self._local[0][0], self._local[0][1])
                else:
                    if self._local[i - 1][VALUE][CK][L] == self._local[i][KEY]:
                        is_left_child = True
                    self._local[i][VALUE][CK][L] = self._local[i + 2][VALUE][CK][R]
                    self._local[i][VALUE][CL][L] = self._local[i + 2][VALUE][CL][R]
                    self._local[i][VALUE][CH][L] = self._local[i + 2][VALUE][CH][R]
                    self._local[i + 1][VALUE][CK][R] = self._local[i + 2][VALUE][CK][L]
                    self._local[i + 1][VALUE][CL][R] = self._local[i + 2][VALUE][CL][L]
                    self._local[i + 1][VALUE][CH][R] = self._local[i + 2][VALUE][CH][L]
                    self._local[i + 2][VALUE][CK][R] = self._local[i][KEY]
                    self._local[i + 2][VALUE][CL][R] = self._local[i][LEAF]
                    self._local[i + 2][VALUE][CH][R] = 1 + max(self._local[i][VALUE][CH][L],
                                                               self._local[i][VALUE][CH][R])
                    self._local[i + 2][VALUE][CK][L] = self._local[i + 1][KEY]
                    self._local[i + 2][VALUE][CL][L] = self._local[i + 1][LEAF]
                    self._local[i + 2][VALUE][CH][L] = 1 + max(self._local[i + 1][VALUE][CH][L],
                                                               self._local[i + 1][VALUE][CH][R])
                    if is_left_child:
                        self._local[i - 1][VALUE][CK][L] = self._local[i + 2][KEY]
                        self._local[i - 1][VALUE][CL][L] = self._local[i + 2][LEAF]
                        self._local[i - 1][VALUE][CH][L] = 1 + max(self._local[i + 2][VALUE][CH][L],
                                                                   self._local[i + 2][VALUE][CH][R])
                    else:
                        self._local[i - 1][VALUE][CK][R] = self._local[i + 2][KEY]
                        self._local[i - 1][VALUE][CL][R] = self._local[i + 2][LEAF]
                        self._local[i - 1][VALUE][CH][R] = 1 + max(self._local[i + 2][VALUE][CH][L],
                                                                   self._local[i + 2][VALUE][CH][R])
                    # Update height of the parent nodes.
                    for j in range(i - 2, -1, -1):
                        if self._local[j][VALUE][CK][L] == self._local[j + 1][KEY]:
                            self._local[j][VALUE][CH][L] = 1 + max(self._local[j + 1][VALUE][CH][L],
                                                                   self._local[j + 1][VALUE][CH][R])
                        else:
                            self._local[j][VALUE][CH][R] = 1 + max(self._local[j + 1][VALUE][CH][L],
                                                                   self._local[j + 1][VALUE][CH][R])
                    self.root = (self._local[0][0], self._local[0][1])

            if balance < -1:
                if self._local[i + 1][VALUE][CH][L] - self._local[i + 1][VALUE][CH][R] <= 0:
                    if self._local[i - 1][VALUE][CK][L] == self._local[i][KEY]:
                        is_left_child = True
                    self._local[i][VALUE][CK][R] = self._local[i + 1][VALUE][CK][L]
                    self._local[i][VALUE][CL][R] = self._local[i + 1][VALUE][CL][L]
                    self._local[i + 1][VALUE][CK][L] = self._local[i][KEY]
                    self._local[i + 1][VALUE][CL][L] = self._local[i][LEAF]
                    if is_left_child:
                        self._local[i - 1][VALUE][CK][L] = self._local[i + 1][KEY]
                        self._local[i - 1][VALUE][CL][L] = self._local[i + 1][LEAF]
                    else:
                        self._local[i - 1][VALUE][CK][R] = self._local[i + 1][KEY]
                        self._local[i - 1][VALUE][CL][R] = self._local[i + 1][LEAF]
                    self._local[i][VALUE][CH][R] = self._local[i + 1][VALUE][CH][L]
                    self._local[i + 1][VALUE][CH][L] = 1 + max(self._local[i][VALUE][CH][L],
                                                               self._local[i][VALUE][CH][R])
                    if is_left_child:
                        self._local[i - 1][VALUE][CH][L] = 1 + max(self._local[i + 1][VALUE][CH][L],
                                                                   self._local[i + 1][VALUE][CH][R])
                    else:
                        self._local[i - 1][VALUE][CH][R] = 1 + max(self._local[i + 1][VALUE][CH][L],
                                                                   self._local[i + 1][VALUE][CH][R])
                    # Update height of the parent nodes.
                    for j in range(i - 2, -1, -1):
                        if self._local[j][VALUE][CK][L] == self._local[j + 1][0]:
                            self._local[j][VALUE][CH][L] = 1 + max(self._local[j + 1][VALUE][CH][L],
                                                                   self._local[j + 1][VALUE][CH][R])
                        else:
                            self._local[j][VALUE][CH][R] = 1 + max(self._local[j + 1][VALUE][CH][L],
                                                                   self._local[j + 1][VALUE][CH][R])

                    self.root = (self._local[0][0], self._local[0][1])
                else:
                    if self._local[i - 1][VALUE][CK][L] == self._local[i][KEY]:
                        is_left_child = True
                    self._local[i][VALUE][CK][R] = self._local[i + 2][VALUE][CK][L]
                    self._local[i][VALUE][CL][R] = self._local[i + 2][VALUE][CL][L]
                    self._local[i][VALUE][CH][R] = self._local[i + 2][VALUE][CH][L]
                    self._local[i + 1][VALUE][CK][L] = self._local[i + 2][VALUE][CK][R]
                    self._local[i + 1][VALUE][CL][L] = self._local[i + 2][VALUE][CL][R]
                    self._local[i + 1][VALUE][CH][L] = self._local[i + 2][VALUE][CH][R]
                    self._local[i + 2][VALUE][CK][L] = self._local[i][KEY]
                    self._local[i + 2][VALUE][CL][L] = self._local[i][LEAF]
                    self._local[i + 2][VALUE][CH][L] = 1 + max(self._local[i][VALUE][CH][L],
                                                               self._local[i][VALUE][CH][R])
                    self._local[i + 2][VALUE][CK][R] = self._local[i + 1][KEY]
                    self._local[i + 2][VALUE][CL][R] = self._local[i + 1][LEAF]
                    self._local[i + 2][VALUE][CH][R] = 1 + max(self._local[i + 1][VALUE][CH][L],
                                                               self._local[i + 1][VALUE][CH][R])
                    if is_left_child:
                        self._local[i - 1][VALUE][CK][L] = self._local[i + 2][KEY]
                        self._local[i - 1][VALUE][CL][L] = self._local[i + 2][LEAF]
                        self._local[i - 1][VALUE][CH][L] = 1 + max(self._local[i + 2][VALUE][CH][L],
                                                                   self._local[i + 2][VALUE][CH][R])
                    else:
                        self._local[i - 1][VALUE][CK][R] = self._local[i + 2][KEY]
                        self._local[i - 1][VALUE][CL][R] = self._local[i + 2][LEAF]
                        self._local[i - 1][VALUE][CH][R] = 1 + max(self._local[i + 2][VALUE][CH][L],
                                                                   self._local[i + 2][VALUE][CH][R])
                    # Update height of the parent nodes.
                    for j in range(i - 2, -1, -1):
                        if self._local[j][VALUE][CK][L] == self._local[j + 1][KEY]:
                            self._local[j][VALUE][CH][L] = 1 + max(self._local[j + 1][VALUE][CH][L],
                                                                   self._local[j + 1][VALUE][CH][R])
                        else:
                            self._local[j][VALUE][CH][R] = 1 + max(self._local[j + 1][VALUE][CH][L],
                                                                   self._local[j + 1][VALUE][CH][R])
                    self.root = (self._local[0][0], self._local[0][1])

        # After all sub-nodes are balanced, we balance the root as well.
        self.__balance_root()

    def __balance_root(self) -> None:
        """Perform AVL tree balance at the local root."""
        # Compute height balance of the root.
        root_balance = self._local[0][VALUE][CH][L] - self._local[0][VALUE][CH][R]
        if root_balance > 1:
            if self._local[1][VALUE][CH][L] - self._local[1][VALUE][CH][R] >= 0:
                self._local[0][VALUE][CK][L] = self._local[1][VALUE][CK][R]
                self._local[0][VALUE][CL][L] = self._local[1][VALUE][CL][R]
                self._local[1][VALUE][CK][R] = self._local[0][KEY]
                self._local[1][VALUE][CL][R] = self._local[0][LEAF]
                self._local[0][VALUE][CH][L] = self._local[1][VALUE][CH][R]
                self._local[1][VALUE][CH][R] = 1 + max(self._local[0][VALUE][CH][L], self._local[0][VALUE][CH][R])
                self.root = (self._local[1][0], self._local[1][1])
            else:
                self._local[0][VALUE][CK][L] = self._local[2][VALUE][CK][R]
                self._local[0][VALUE][CL][L] = self._local[2][VALUE][CL][R]
                self._local[0][VALUE][CH][L] = self._local[2][VALUE][CH][R]
                self._local[1][VALUE][CK][R] = self._local[2][VALUE][CK][L]
                self._local[1][VALUE][CL][R] = self._local[2][VALUE][CL][L]
                self._local[1][VALUE][CH][R] = self._local[2][VALUE][CH][L]
                self._local[2][VALUE][CK][R] = self._local[0][KEY]
                self._local[2][VALUE][CL][R] = self._local[0][LEAF]
                self._local[2][VALUE][CH][R] = 1 + max(self._local[0][VALUE][CH][L], self._local[0][VALUE][CH][R])
                self._local[2][VALUE][CK][L] = self._local[1][KEY]
                self._local[2][VALUE][CL][L] = self._local[1][LEAF]
                self._local[2][VALUE][CH][L] = 1 + max(self._local[1][VALUE][CH][L], self._local[1][VALUE][CH][R])
                self.root = (self._local[2][0], self._local[2][1])

        if root_balance < -1:
            if self._local[1][VALUE][CH][L] - self._local[1][VALUE][CH][R] <= 0:
                self._local[0][VALUE][CK][R] = self._local[1][VALUE][CK][L]
                self._local[0][VALUE][CL][R] = self._local[1][VALUE][CL][L]
                self._local[1][VALUE][CK][L] = self._local[0][KEY]
                self._local[1][VALUE][CL][L] = self._local[0][LEAF]
                self._local[0][VALUE][CH][R] = self._local[1][VALUE][CH][L]
                self._local[1][VALUE][CH][L] = 1 + max(self._local[0][VALUE][CH][L], self._local[0][VALUE][CH][R])
                self.root = (self._local[1][0], self._local[1][1])
            else:
                self._local[0][VALUE][CK][R] = self._local[2][VALUE][CK][L]
                self._local[0][VALUE][CL][R] = self._local[2][VALUE][CL][L]
                self._local[0][VALUE][CH][R] = self._local[2][VALUE][CH][L]
                self._local[1][VALUE][CK][L] = self._local[2][VALUE][CK][R]
                self._local[1][VALUE][CL][L] = self._local[2][VALUE][CL][R]
                self._local[1][VALUE][CH][L] = self._local[2][VALUE][CH][R]
                self._local[2][VALUE][CK][L] = self._local[0][KEY]
                self._local[2][VALUE][CL][L] = self._local[0][LEAF]
                self._local[2][VALUE][CH][L] = 1 + max(self._local[0][VALUE][CH][L], self._local[0][VALUE][CH][R])
                self._local[2][VALUE][CK][R] = self._local[1][KEY]
                self._local[2][VALUE][CL][R] = self._local[1][LEAF]
                self._local[2][VALUE][CH][R] = 1 + max(self._local[1][VALUE][CH][L], self._local[1][VALUE][CH][R])
                self.root = (self._local[2][0], self._local[2][1])

    def insert(self, key: Any, value: Any) -> None:
        """
        Given key-value pair, insert the pair to the tree.

        :param key: the search key of interest.
        :param value: the value to insert.
        """
        # Create a new data block that holds the data to insert to tree.
        data_block = self.__get_avl_data(key=key, value=value)

        # If the current root is empty, we simply set root as this new block.
        if self.root is None:
            # Add the data to stash and update root.
            self._stash.append(data_block)
            self.root = (data_block[KEY], data_block[LEAF])
            # Perform dummy finds and evictions.
            self._perform_dummy_finds(num_round=self.__max_height)
            self._perform_dummy_eviction(num_round=2 * self.__max_height + 1)
            # Terminates the function.
            return

        # Get the node information from oram storage.
        self.__move_remote_node_to_local(key=self.root[KEY], leaf=self.root[LEAF])

        # Keep adding node to local until we find a place to insert the new node.
        while True:
            # Save the last node data and its value.
            node = self._local[-1]
            node_value = node[VALUE]

            # If node key is smaller, we go right and check whether a child is already there.
            if node[KEY] < key:
                # If the child is there, we keep grabbing the next node.
                if node_value[CK][R] is not None:
                    self.__move_remote_node_to_local(key=node_value[CK][R], leaf=node_value[CL][R])
                # Else we store the new value.
                else:
                    node_value[CK][R] = data_block[KEY]
                    node_value[CL][R] = data_block[LEAF]
                    self._local.append(data_block)
                    break
            # If key is not smaller, we go left and check the same as above.
            else:
                if node_value[CK][L] is not None:
                    self.__move_remote_node_to_local(key=node_value[CK][L], leaf=node_value[CL][L])
                else:
                    node_value[CK][L] = data_block[KEY]
                    node_value[CL][L] = data_block[LEAF]
                    self._local.append(data_block)
                    break

        # Traverse local and update parent leaves and heights.
        self.__update_leaves()
        self.__update_heights()

        # Update the root to reflect new path.
        self.root = (self._local[0][KEY], self._local[0][LEAF])

        # Perform rotation on the nodes stored in local.
        self.__balance_local()

        # Get the number of retrieved nodes.
        num_retrieved_nodes = len(self._local)

        # Perform desired number of dummy finds.
        self._perform_dummy_finds(num_round=self.__max_height - num_retrieved_nodes)

        # Move the local nodes to stash and perform evictions.
        self._stash += self._local
        self._local = []

        # Perform final dummy evictions.
        self._perform_dummy_eviction(num_round=2 * self.__max_height + 1)

    def __normal_search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value.

        If input value is not None, the value corresponding to the search tree will be updated.
        :param key: the search key of interest.
        :param value: the value to update.
        :return: the (old) value corresponding to the search key.
        """
        # If the current root is empty, we can't perform search.
        if self.root is None:
            raise KeyError(f"The search key {key} is not found.")

        # Otherwise get information about node.
        self.__move_remote_node_to_local(key=self._root[KEY], leaf=self._root[LEAF])

        # Set the node as root.
        node = self._local[-1]

        # Find the desired search key.
        while node[KEY] != key:
            # If node key is smaller, we go right and check whether a child is already there.
            if node[KEY] < key:
                if node[VALUE][CK][R] is not None:
                    self.__move_remote_node_to_local(key=node[VALUE][CK][R], leaf=node[VALUE][CL][R])
                else:
                    raise KeyError(f"The search key {key} is not found.")
            # If key is not smaller, we go left and check the same as above.
            else:
                if node[VALUE][CL][L] is not None:
                    self.__move_remote_node_to_local(key=node[VALUE][CK][L], leaf=node[VALUE][CL][L])
                else:
                    raise KeyError(f"The search key {key} is not found.")
            # Update the node to keep searching.
            node = self._local[-1]

        # Get the desired search value and update its leaf.
        search_value = node[VALUE][V]
        node[LEAF] = self._get_new_leaf()

        # If the input value is not None, we update the value stored.
        if value is not None:
            node[VALUE][V] = value

        # Get the number of retrieved nodes.
        num_retrieved_nodes = len(self._local)
        # Perform desired number of dummy finds.
        self._perform_dummy_finds(num_round=self.__max_height - num_retrieved_nodes)

        # Update new leaves.
        self.__update_leaves()
        # Update the root stored.
        self.root = (self._local[0][KEY], self._local[0][LEAF])

        # Move nodes from local to stash.
        self._stash = self._local
        self._local = []

        # Perform dummy evictions.
        self._perform_dummy_eviction(num_round=2 * self.__max_height)

        return search_value

    def __fast_search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value.

        If input value is not None, the value corresponding to the search tree will be updated.
        :param key: the search key of interest.
        :param value: the value to update.
        :return: the (old) value corresponding to the search key.
        """
        # If the current root is empty, we can't perform search.
        if self.root is None:
            raise KeyError(f"The search key {key} is not found.")

        # If root is not None, move its content to local.
        self.__search_move_node_to_local(key=self.root[0], leaf=self.root[1])

        # Save the old path and sample a new path.
        old_child_path = self.root[1]
        child_leaf = self._get_new_leaf()
        self.root = (self.root[0], child_leaf)

        # Count the number of visited nodes.
        num_retrieved_nodes = 1

        # Each node is grabbed and then returned, hence we are always getting the first node in local.
        while self._local[0][KEY] != key:
            # Update the number of visited nodes.
            num_retrieved_nodes += 1
            if self._local[0][KEY] < key:
                if self._local[0][VALUE][CK][R] is not None:
                    # Deep copy needed because of sublist structures.
                    node_to_return = copy.deepcopy(self._local[0])
                    node_to_return[LEAF] = child_leaf

                    # Sample a new leaf for the next child to read and store it.
                    child_leaf = self._get_new_leaf()
                    # Update the node to write back to server.
                    node_to_return[VALUE][CL][R] = child_leaf

                    # Append the node to stash and write them back to server.
                    self._stash.append(node_to_return)
                    self.client.write_query(label="ods", leaf=old_child_path, data=self._evict_stash(old_child_path))

                    # Get the next node to check to local.
                    self.__search_move_node_to_local(
                        key=self._local[0][VALUE][CK][R], leaf=self._local[0][VALUE][CL][R]
                    )

                    # Store the old child path and then delete current node.
                    old_child_path = self._local[0][VALUE][CL][R]
                    del self._local[0]
                else:
                    raise KeyError(f"The search key {key} is not found.")
            else:
                if self._local[0][VALUE][CK][L] is not None:
                    # Deep copy needed because of sublist structures.
                    node_to_return = copy.deepcopy(self._local[0])
                    node_to_return[LEAF] = child_leaf

                    # Sample a new leaf for the next child to read and store it.
                    child_leaf = self._get_new_leaf()
                    # Update the node to write back to server.
                    node_to_return[VALUE][CL][L] = child_leaf

                    # Append the node to stash and write them back to server.
                    self._stash.append(node_to_return)
                    self.client.write_query(label="ods", leaf=old_child_path, data=self._evict_stash(old_child_path))

                    # Get the next node to check to local.
                    self.__search_move_node_to_local(
                        key=self._local[0][VALUE][CK][L], leaf=self._local[0][VALUE][CL][L]
                    )

                    # Store the old child path and then delete current node.
                    old_child_path = self._local[0][VALUE][CL][L]
                    del self._local[0]
                else:
                    raise KeyError(f"The search key {key} is not found.")

        # Per design, the value of interest is the only one stored in local.
        search_value = self._local[0][VALUE][V]

        # If provided value is not None, update the data.
        if value is not None:
            self._local[0][VALUE][V] = value

        # Deep copy needed because of sublist structures.
        node_to_return = copy.deepcopy(self._local[0])
        node_to_return[LEAF] = child_leaf

        # Add the node to stash and clear local storage.
        self._stash.append(node_to_return)
        self._local = []

        # Write the new node back to storage.
        self.client.write_query(label="ods", leaf=old_child_path, data=self._evict_stash(old_child_path))

        # Perform desired number of dummy finds.
        self._perform_dummy_eviction(num_round=self.__max_height - num_retrieved_nodes)

        return search_value

    def search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value.

        If input value is not None, the value corresponding to the search tree will be updated.
        :param key: the search key of interest.
        :param value: the updated value.
        :return: the (old) value corresponding to the search key.
        """
        if self.__distinguishable_search:
            return self.__fast_search(key=key, value=value)
        else:
            return self.__normal_search(key=key, value=value)
