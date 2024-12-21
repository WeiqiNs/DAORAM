"""Defines the OMAP constructed with the B+ tree ODS."""

import math
import os
import pickle
from typing import Any, List, Tuple

from scipy.special import lambertw

from daoram.dependency.binary_tree import BinaryTree, KEY, LEAF, VALUE
from daoram.dependency.bplus_tree import BPlusTree, BPlusTreeNode
from daoram.dependency.interact_server import InteractServer
from daoram.omaps.tree_ods_omap import KV_LIST, ROOT, TreeOdsOmap

# We use these codes for better readability of code. CK retrieves children node keys and CV retrieves children values.
CK = 0
CV = 1


class BPlusOdsOmap(TreeOdsOmap):
    def __init__(self,
                 order: int,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 client: InteractServer,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True):
        """
        Initializes the OMAP based on the B+ tree ODS.

        :param order: The branching order of the B+ tree.
        :param num_data: the number of data points the oram should store.
        :param key_size: the number of bytes the random dummy key should have.
        :param data_size: the number of bytes the random dummy data should have.
        :param client: the instance we use to interact with server.
        :param bucket_size: the number of data each bucket should have.
        :param stash_scale: the scaling scale of the stash.
        :param aes_key: the key to use for the AES instance.
        :param num_key_bytes: the number of bytes the aes key should have.
        :param use_encryption: a boolean indicating whether to use encryption.
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

        # Save the branching order and middle point.
        self.__order = order
        self.__mid = order // 2

        # Set the starting block id to 0.
        self.__block_id = 0

        # Compute the maximum height of the B+ tree.
        self.__max_height = math.ceil(math.log(num_data, math.ceil(order / 2)))

        # Update the padded block size, which depends on the max height.
        self._padded_block_size = self._dummy_block_size

    def update_mul_tree_height(self, num_tree: int) -> None:
        """Suppose the ODS is used to store multiple trees, we update each tree's height.

        :param num_tree: number of tree to store, which is the same as number of data in upper level oram.
        """
        # First compute how many are in the buckets, according to https://eprint.iacr.org/2021/1280.
        tree_size = math.ceil(math.e ** (lambertw(math.e ** -1 * (math.log(num_tree, 2) + 128 - 1)).real + 1))

        # Update the height accordingly.
        self.__max_height = math.ceil(math.log(tree_size, math.ceil(self.__order / 2)))

        # Update the padded block size, which depends on the max height.
        self._padded_block_size = self._dummy_block_size

    @property
    def _dummy_block_size(self) -> int:
        """Get the number of bytes equal to the size of actual data block stored in ORAM."""
        return len(pickle.dumps([
            self._num_data - 1,
            self._num_data - 1,
            [
                [os.urandom(self._key_size) for _ in range(self.__order)],  # children keys.
                [os.urandom(self._data_size) for _ in range(self.__order)]  # children values.
            ]
        ]))

    def __get_bplus_data(self, key: Any, value: Any) -> list:
        """
        From the input key and value, create the data should be stored in the AVL tree oram.

        The data is of the following structure:
            - [block id, leaf, [[key], [value]]]
        """
        # Create the data block with input key and value.
        data_block = [self.__block_id, self._get_new_leaf(), [[key], [value]]]
        # Increment the block id for future use.
        self.__block_id += 1
        return data_block

    def _init_ods_storage(self, data: KV_LIST) -> BinaryTree:
        """
        Initialize a binary tree storage to store the B+ tree holding input key-value pairs.

        :param data: a list of key-value pairs.
        :return: the binary tree storage for the input list of key-value pairs.
        """
        # Create an empty root, an empty position map dictionary, a B+ tree instance, and an oram tree instance.
        root = BPlusTreeNode()
        pos_map = {}
        bplus_tree = BPlusTree(order=self.__order, leaf_range=self._leaf_range)
        oram_tree = BinaryTree(num_data=self._num_data, bucket_size=self._bucket_size)

        # Insert all the provided KV pairs to the root.
        if data:
            for kv_pair in data:
                root = bplus_tree.insert(root=root, kv_pair=kv_pair)

            # Fill the position map.
            key, path = bplus_tree.post_order(root=root, pos_map=pos_map, block_id=self.__block_id)

            # Fill the oram tree according to the position map.
            for key, node in sorted(pos_map.items()):
                # Node is a list containing leaf info and then details of the node.
                oram_tree.fill_data_to_storage_leaf(data=[key, node[0], node[1]])

            # Store the key and its path in the oram.
            self.root = (key, path)

            # Update the block id.
            self.__block_id = key + 1

        # Fill the storage with dummy data.
        oram_tree.fill_storage_with_dummy_data()

        # Encrypt the tree storage if needed.
        oram_tree.storage = self._encrypt_buckets(buckets=oram_tree.storage)

        return oram_tree

    def _init_mul_tree_ods_storage(self, data_list: List[KV_LIST]) -> Tuple[BinaryTree, List[ROOT]]:
        """
        Initialize a binary tree storage to store multiple B+ trees holding input lists of key-value pairs.

        :param data_list: a list of lists of key-value pairs.
        :return: the binary tree storage for the input list of key-value pairs and a list of B+ tree roots.
        """
        # In this case, we set the max height to be smaller as each subgroup is of O(log n).
        self.__max_height = math.ceil(math.log(math.log(self._num_data, 2), self.__order) + 1)

        # Create a block id, a root list, a B+ tree instance, and an oram tree instance.
        root_list = []
        bplus_tree = BPlusTree(order=self.__order, leaf_range=self._leaf_range)
        oram_tree = BinaryTree(num_data=self._num_data, bucket_size=self._bucket_size)

        # Enumerate each key-pair list in the input list.
        for index, data in enumerate(data_list):
            # Set root to None and create an empty pos map for each key-pair list.
            root = BPlusTreeNode()
            pos_map = {}

            # Insert all data to the AVL tree.
            if data:
                for kv_pair in data:
                    root = bplus_tree.insert(root=root, kv_pair=kv_pair)

                # Fill the position map.
                key, path = bplus_tree.post_order(root=root, pos_map=pos_map, block_id=self.__block_id)

                # Fill the oram tree according to the position map.
                for key, node in pos_map.items():
                    # Node is a list containing leaf info and then details of the node.
                    oram_tree.fill_data_to_storage_leaf(data=[key, node[0], node[1]])

                # Store the key and its path to the root list.
                root_list.append((key, path))

                # Update the block id.
                self.__block_id = key + 1
            else:
                # Otherwise just append None.
                root_list.append(None)

        # Fill the storage with dummy data.
        oram_tree.fill_storage_with_dummy_data()

        # Encrypt the tree storage if needed.
        oram_tree.storage = self._encrypt_buckets(buckets=oram_tree.storage)

        return oram_tree, root_list

    def __move_stash_node_to_local(self, key: Any, new_leaf: int) -> None:
        """Given key, move the corresponding data from stash to local."""
        # Iterate though the stash and find the desired key.
        for index, data in enumerate(self._stash):
            # If found, we append the value to local and delete it from stash.
            if data[KEY] == key:
                # Update the data leaf.
                data[LEAF] = new_leaf
                # When we find the desired data, we move it to local storage.
                self._local.append(data)
                # Delete data from the stash and terminate the function.
                del self._stash[index]
                return

        # Since stash is checked after path, raise an error.
        raise KeyError(f"Key {key} not found.")

    def __move_remote_node_to_local(self, key: Any, leaf: int, new_leaf: int) -> None:
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
                    # Update the data leaf.
                    data[LEAF] = new_leaf
                    # When we find the desired data, we move it to local storage.
                    self._local.append(data.copy())
                    # Label this data as dummy data; its content will eventually be deleted/replaced.
                    data[KEY] = None
                    # Set found to True.
                    found = True

        # If the required data is not found, we also check the stash.
        if not found:
            self.__move_stash_node_to_local(key=key, new_leaf=new_leaf)

        # Write the path back.
        self.client.write_query(label="ods", leaf=leaf, data=self._encrypt_buckets(buckets=path))

    def __find_leaf_to_local(self, key: Any) -> None:
        """Add all nodes we need to visit to local until finding the leaf storing the key.

        :param key: search key of interest.
        """
        # Make sure that the local is cleared and is empty at the moment.
        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")

        # Get the node information from oram storage.
        new_leaf = self._get_new_leaf()
        self.__move_remote_node_to_local(key=self.root[KEY], leaf=self.root[LEAF], new_leaf=new_leaf)
        self.root = (self.root[KEY], new_leaf)

        # Get the node info from retrieved block.
        node = self._local[0][VALUE]

        # While we do not reach a leaf (whose number of children keys and number of children values are the same).
        while len(node[CK]) != len(node[CV]):
            for index, each_key in enumerate(node[CK]):
                # If key equals, it is on the right.
                if key == each_key:
                    new_leaf = self._get_new_leaf()
                    self.__move_remote_node_to_local(
                        key=node[CV][index + 1][KEY], leaf=node[CV][index + 1][LEAF], new_leaf=new_leaf
                    )
                    node[CV][index + 1] = (node[CV][index + 1][KEY], new_leaf)
                    break
                # If key is smaller, it is on the left.
                elif key < each_key:
                    new_leaf = self._get_new_leaf()
                    self.__move_remote_node_to_local(
                        key=node[CV][index][KEY], leaf=node[CV][index][LEAF], new_leaf=new_leaf
                    )
                    node[CV][index] = (node[CV][index][KEY], new_leaf)
                    break
                # If we reached the end, it is on the right.
                elif index + 1 == len(node[CK]):
                    new_leaf = self._get_new_leaf()
                    self.__move_remote_node_to_local(
                        key=node[CV][index + 1][KEY], leaf=node[CV][index + 1][LEAF], new_leaf=new_leaf
                    )
                    node[CV][index + 1] = (node[CV][index + 1][KEY], new_leaf)
                    break

            # Update the node.
            node = self._local[-1][VALUE]

    def __insert_to_parent(self, key: Any, left_node: list, right_node: list, parent_index: int) -> None:
        """
        Inserts a node to the parent node.

        :param key: the key to insert to the parent node.
        :param left_node: the left node of the new parent node.
        :param right_node: the right node of the new parent node.
        :param parent_index: the index of the parent node in the local storage.
        """
        # If the parent node is empty, we create a new parent node.
        if parent_index < 0:
            # Create a new parent leaf block and increment block id.
            parent_block = [
                self.__block_id,
                self._get_new_leaf(),
                [[key], [(left_node[KEY], left_node[LEAF]), (right_node[KEY], right_node[LEAF])]]
            ]
            self.__block_id += 1
            # Insert the new parent block to the local and terminate the function.
            self._local.insert(0, parent_block)
            return
        else:
            # Set the parent block.
            parent_block = self._local[parent_index]

        # If parent node is not empty, we find where to insert.
        for index, each_key in enumerate(parent_block[VALUE][CK]):
            if key < each_key:
                parent_block[VALUE][CK] = parent_block[VALUE][CK][:index] + [key] + parent_block[VALUE][CK][index:]
                parent_block[VALUE][CV] = (
                        parent_block[VALUE][CV][:index + 1] +
                        [(right_node[KEY], right_node[LEAF])] +
                        parent_block[VALUE][CV][index + 1:]
                )
                break
            elif index + 1 == len(parent_block[VALUE][CK]):
                parent_block[VALUE][CK].append(key)
                parent_block[VALUE][CV].append((right_node[KEY], right_node[LEAF]))
                break

        # After insertion, we need to again check whether further insert to parent is needed.
        if len(parent_block[VALUE][CK]) == self.__order:
            # Create an empty leaf block and increment block id.
            new_block = [self.__block_id, self._get_new_leaf(), [[], []]]
            self.__block_id += 1

            # New leaf gets half of the old leaf.
            key = parent_block[VALUE][CK][self.__mid]
            new_block[VALUE][CK] = parent_block[VALUE][CK][self.__mid + 1:]
            new_block[VALUE][CV] = parent_block[VALUE][CV][self.__mid + 1:]
            self._stash.append(new_block)

            # The old leaf keeps on the first half.
            parent_block[VALUE][CK] = parent_block[VALUE][CK][:self.__mid]
            parent_block[VALUE][CV] = parent_block[VALUE][CV][:self.__mid + 1]

            # Insert new key to the parent.
            self.__insert_to_parent(
                key=key, left_node=parent_block, right_node=new_block, parent_index=parent_index - 1
            )

    def insert(self, key: Any, value: Any) -> None:
        """
        Given key-value pair, insert the pair to the tree.

        :param key: the search key of interest.
        :param value: the value to insert.
        """
        # If the current root is empty, we simply set root as this new block.
        if self.root is None:
            # Create a new bplus data block.
            data_block = self.__get_bplus_data(key=key, value=value)
            # Append data block to the stash.
            self._stash.append(data_block)
            self.root = (data_block[KEY], data_block[LEAF])
            # Perform at dummy finds and dummy evictions.
            self._perform_dummy_finds(num_round=self.__max_height)
            self._perform_dummy_eviction(num_round=2 * self.__max_height + 1)
            return

        # Get all nodes we need to visit until finding the key.
        self.__find_leaf_to_local(key=key)

        # Set the last node in local as leaf.
        leaf_block = self._local[-1]

        # Find the proper place to insert the leaf.
        for index, each_key in enumerate(leaf_block[VALUE][CK]):
            if key < each_key:
                leaf_block[VALUE][CK] = leaf_block[VALUE][CK][:index] + [key] + leaf_block[VALUE][CK][index:]
                leaf_block[VALUE][CV] = leaf_block[VALUE][CV][:index] + [value] + leaf_block[VALUE][CV][index:]
                break
            elif index + 1 == len(leaf_block[VALUE][CK]):
                leaf_block[VALUE][CK].append(key)
                leaf_block[VALUE][CV].append(value)
                break

        # We figure out whether split and insert to parent is needed.
        if len(leaf_block[VALUE][CK]) == self.__order:
            # Create an empty leaf block and increment block id.
            new_block = [self.__block_id, self._get_new_leaf(), [[], []]]
            self.__block_id += 1
            # New leaf gets half of the old leaf.
            new_block[VALUE][CK] = leaf_block[VALUE][CK][self.__mid:]
            new_block[VALUE][CV] = leaf_block[VALUE][CV][self.__mid:]
            self._stash.append(new_block)
            # The old leaf keeps on the first half.
            leaf_block[VALUE][CK] = leaf_block[VALUE][CK][:self.__mid]
            leaf_block[VALUE][CV] = leaf_block[VALUE][CV][:self.__mid]
            # Insert new key to the parent.
            self.__insert_to_parent(
                key=new_block[VALUE][CK][0],
                left_node=leaf_block,
                right_node=new_block,
                parent_index=len(self._local) - 2
            )

        # Update the root after insertion is done.
        self.root = (self._local[0][KEY], self._local[0][LEAF])

        # Perform desired number of dummy finds.
        self._perform_dummy_finds(num_round=self.__max_height - len(self._local))

        # Append local data to stash and clear local.
        self._stash += self._local
        self._local = []

        # Perform desired number of dummy evictions.
        self._perform_dummy_eviction(num_round=2 * self.__max_height + 1)

    def search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value.

        If input value is not None, the value corresponding to the search tree will be updated.
        :param key: the search key of interest.
        :param value: the updated value.
        :return: the (old) value corresponding to the search key.
        """
        # If the current root is empty, we can't perform search.
        if self.root is None:
            raise KeyError(f"The search key {key} is not found.")

        # Get all nodes we need to visit until finding the key.
        self.__find_leaf_to_local(key=key)

        # Set the last node in local as leaf and set the return search value to None.
        leaf = self._local[-1][VALUE]
        search_value = None

        # Search the desired key and update its value as needed.
        for index, each_key in enumerate(leaf[CK]):
            if key == each_key:
                search_value = leaf[CV][index]
                if value is not None:
                    leaf[CV][index] = value
                # Terminate the loop after finding the key.
                break

        # Perform desired number of dummy finds.
        self._perform_dummy_finds(num_round=self.__max_height - len(self._local))

        # Append local data to stash and clear local.
        self._stash += self._local
        self._local = []

        # Perform desired number of dummy evictions.
        self._perform_dummy_eviction(num_round=self.__max_height)

        return search_value
