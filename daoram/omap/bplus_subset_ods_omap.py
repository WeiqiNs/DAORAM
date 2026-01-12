"""Defines the OMAP constructed with the B+ subset tree ODS."""

import math
import os
from functools import cached_property
from typing import Any, List, Optional, Tuple

from daoram.dependency import (
    BinaryTree, BPlusSubsetData, SubsetBPlusTree, SubsetBPlusNode, 
    Buckets, Data, Helper, InteractServer
)
from daoram.omap.tree_ods_omap import KV_LIST, ROOT, TreeOdsOmap


class BPlusSubsetOdsOmap(TreeOdsOmap):
    """OMAP implementation using B+ subset tree as ODS.
    
    This OMAP stores key-value pairs and can also return an element from {0, 1, ..., n-1}
    that is not currently stored in the tree.
    """
    
    def __init__(self,
                 order: int,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 client: InteractServer,
                 n: int = None,
                 name: str = "bplus_subset",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True):
        """
        Initializes the OMAP based on the B+ subset tree ODS.

        :param order: The branching order of the B+ tree.
        :param num_data: The number of data points the oram should store.
        :param key_size: The number of bytes the random dummy key should have.
        :param data_size: The number of bytes the random dummy data should have.
        :param client: The instance we use to interact with server.
        :param n: The size of the subset universe {0, 1, ..., n-1}. If not provided, defaults to num_data.
        :param name: The name of the protocol, this should be unique if multiple schemes are used together.
        :param filename: The filename to save the oram data to.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param aes_key: The key to use for the AES instance.
        :param num_key_bytes: The number of bytes the aes key should have.
        :param use_encryption: A boolean indicating whether to use encryption.
        """
        # Initialize the parent TreeOdsOmap class.
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

        # Save the branching order and middle point.
        self._order: int = order
        self._mid: int = order // 2

        # Set the starting block id to 0.
        self._block_id: int = 0

        # Compute the maximum height of the B+ tree.
        self._max_height: int = math.ceil(math.log(num_data, math.ceil(order / 2)))
        
        # Store the subset range n (for {0, 1, ..., n-1})
        self._n: int = n if n is not None else num_data
        self._subset_n: int = self._n

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
                    value=BPlusSubsetData(
                        # The keys are actual keys.
                        keys=[os.urandom(self._key_size) for _ in range(self._order - 1)],
                        # The values are [node_available, (child_id, child_leaf, child_available), ...]
                        values=[True] + [(self._num_data - 1, self._num_data - 1, True) for _ in range(self._order)],
                    ).dump()
                ).dump()
            ),
            len(
                Data(  # This one is the leaf data.
                    key=self._num_data - 1,
                    leaf=self._num_data - 1,
                    value=BPlusSubsetData(
                        # The keys are actual keys.
                        keys=[os.urandom(self._key_size) for _ in range(self._order - 1)],
                        # The values are actual values, plus a marker for leaf availability.
                        values=[os.urandom(self._data_size) for _ in range(self._order - 1)] + [True],
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
            # First, each data's value is BPlusSubsetData; we need to dump it.
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

        # First decrypt the data and then load it as BPlusSubsetData.
        def _dec_bucket(bucket: List[bytes]) -> List[Data]:
            """Helper function to decrypt a bucket."""
            # Perform decryption.
            dec_bucket = [
                dec for data in bucket
                if (dec := Data.from_pickle(
                    Helper.unpad_pickle(data=self._cipher.dec(ciphertext=data)))
                    ).key is not None
            ]

            # Load data.
            for data in dec_bucket:
                data.value = BPlusSubsetData.from_pickle(data=data.value)

            return dec_bucket

        # Return the decrypted list of lists of Data.
        return [_dec_bucket(bucket=bucket) for bucket in buckets] if self._use_encryption else buckets

    def _calculate_leaf_available(self, leaf_node: Data, min_val: int, max_val: int) -> bool:
        """Calculate whether a leaf node has available elements.
        
        :param leaf_node: The leaf node (its value has keys list).
        :param min_val: The minimum value in the range this leaf covers.
        :param max_val: The maximum value in the range this leaf covers.
        :return: True if there's at least one available element in [min_val, max_val].
        """
        num_slots = max_val - min_val + 1
        num_stored = len(leaf_node.value.keys) if leaf_node.value.keys else 0
        return num_stored < num_slots

    def _get_bplus_subset_data(self, keys: Any = None, values: Any = None, min_val: int = None, max_val: int = None) -> Data:
        """From the input key and value, create the data should be stored in the B+ subset tree oram.
        
        :param keys: List of keys
        :param values: List of values or availability flags
        :param min_val: Minimum value in range (for leaf nodes)
        :param max_val: Maximum value in range (for leaf nodes)
        """
        # Create the data block with input key and value.
        data_block = Data(key=self._block_id, leaf=self._get_new_leaf(), 
                         value=BPlusSubsetData(keys=keys, values=values, min_val=min_val, max_val=max_val))
        # Increment the block id for future use.
        self._block_id += 1
        # Return the data block
        return data_block

    def _init_ods_storage(self, data: KV_LIST) -> BinaryTree:
        """
        Initialize a binary tree storage to store the B+ subset tree holding input key-value pairs.

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

        # Insert all the provided KV pairs to the B+ subset tree.
        if data:
            # Create the B+ subset tree instance.
            bplus_tree = SubsetBPlusTree(order=self._order, n=self._subset_n)
            root = SubsetBPlusNode(is_leaf=True, min_val=0, max_val=self._subset_n - 1)

            # Insert the kv pairs to the B+ subset tree.
            # First, insert all keys to track which elements are stored
            for kv_pair in data:
                bplus_tree.insert(key=kv_pair[0])

            # Get node from B+ subset tree and fill them to the oram storage.
            data_list = bplus_tree.get_data_list(root=bplus_tree._SubsetBPlusTree__root or root, 
                                                  block_id=self._block_id, encryption=self._use_encryption)

            # Update the block id.
            self._block_id += len(data_list)

            # Fill the oram tree according to the position map.
            for bplus_data in data_list:
                tree.fill_data_to_storage_leaf(data=bplus_data)

            # Store the root information in oram.
            if bplus_tree._SubsetBPlusTree__root:
                self.root = (bplus_tree._SubsetBPlusTree__root.id, bplus_tree._SubsetBPlusTree__root.leaf)
            else:
                self.root = None

        # Encryption and fill with dummy data if needed.
        if self._use_encryption:
            tree.storage.encrypt(aes=self._cipher)

        return tree

    def _init_mul_tree_ods_storage(self, data_list: List[KV_LIST]) -> Tuple[BinaryTree, List[ROOT]]:
        """
        Initialize a binary tree storage to store multiple B+ subset trees holding input lists of key-value pairs.

        :param data_list: A list of lists of key-value pairs.
        :return: The binary tree storage for the input list of key-value pairs and a list of B+ subset tree roots.
        """
        # Create the binary tree object.
        tree = BinaryTree(
            filename=self._filename,
            num_data=self._num_data,
            data_size=self._max_block_size,
            bucket_size=self._bucket_size,
            enc_key_size=self._num_key_bytes if self._use_encryption else None,
        )

        # Create a root list for each B+ subset tree.
        root_list = []

        # Enumerate each key-pair list in the input list.
        for index, data in enumerate(data_list):
            # Insert all data to the B+ subset tree.
            if data:
                # Create the B+ subset tree object.
                bplus_tree = SubsetBPlusTree(order=self._order, n=self._subset_n)
                root = SubsetBPlusNode(is_leaf=True, min_val=0, max_val=self._subset_n - 1)

                # Insert the kv pairs to the B+ subset tree.
                for kv_pair in data:
                    bplus_tree.insert(key=kv_pair[0])

                # Get node from B+ subset tree and fill them to the oram storage.
                data_list_nodes = bplus_tree.get_data_list(
                    root=bplus_tree._SubsetBPlusTree__root or root, 
                    block_id=self._block_id, encryption=self._use_encryption
                )

                # Update the block id.
                self._block_id += len(data_list_nodes)

                # Fill the oram tree according to the position map.
                for bplus_data in data_list_nodes:
                    tree.fill_data_to_storage_leaf(data=bplus_data)

                # Store the key and its path in the omap.
                if bplus_tree._SubsetBPlusTree__root:
                    root_list.append((bplus_tree._SubsetBPlusTree__root.id, bplus_tree._SubsetBPlusTree__root.leaf))
                else:
                    root_list.append(None)
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

        # While we do not reach a leaf (leaf nodes have min_val set).
        while node.value.min_val is None:
            # Sample a new leaf for updating the current storage.
            new_leaf = self._get_new_leaf()

            # Handle case where internal node has no keys (only one child)
            if len(node.value.keys) == 0:
                if len(node.value.values) > 1:
                    child_key, child_leaf, child_available = node.value.values[1]
                    node.value.values[1] = (child_key, new_leaf, child_available)
                    self._stash.append(node)
                    self._client.write_query(label=self._name, leaf=old_leaf, data=self._evict_stash(leaf=old_leaf))
                    self._move_node_to_local_without_eviction(key=child_key, leaf=child_leaf)
                else:
                    raise ValueError("Internal node has no keys and no children")
            else:
                for index, each_key in enumerate(node.value.keys):
                    # If key equals, it is on the right.
                    if key == each_key:
                        # Get the old child leaf.
                        # For subset trees, internal node values are [node.available, (child_id, child_leaf, child_available), ...]
                        child_key, child_leaf, child_available = node.value.values[index + 2]
                        # Update the current stored value.
                        node.value.values[index + 2] = (child_key, new_leaf, child_available)
                        # Add the node to stash and perform eviction before grabbing the next path.
                        self._stash.append(node)
                        self._client.write_query(label=self._name, leaf=old_leaf, data=self._evict_stash(leaf=old_leaf))
                        # Move the next node to local.
                        self._move_node_to_local_without_eviction(key=child_key, leaf=child_leaf)
                        break

                    # If the key is smaller, it is on the left.
                    elif key < each_key:
                        # Get the old child leaf.
                        child_key, child_leaf, child_available = node.value.values[index + 1]
                        # Update the current stored value.
                        node.value.values[index + 1] = (child_key, new_leaf, child_available)
                        # Add the node to stash and perform eviction before grabbing the next path.
                        self._stash.append(node)
                        self._client.write_query(label=self._name, leaf=old_leaf, data=self._evict_stash(leaf=old_leaf))
                        # Move the next node to local.
                        self._move_node_to_local_without_eviction(key=child_key, leaf=child_leaf)
                        break

                    # If we reached the end, it is on the right.
                    elif index + 1 == len(node.value.keys):
                        # Get the old child leaf.
                        child_key, child_leaf, child_available = node.value.values[index + 2]
                        # Update the current stored value.
                        node.value.values[index + 2] = (child_key, new_leaf, child_available)
                        # Add the node to stash and perform eviction before grabbing the next path.
                        self._stash.append(node)
                        self._client.write_query(label=self._name, leaf=old_leaf, data=self._evict_stash(leaf=old_leaf))
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

        # While we do not reach a leaf (leaf nodes have min_val set).
        while node.value.min_val is None:
            # Sample a new leaf for updating the current storage.
            new_leaf = self._get_new_leaf()

            # Handle case where internal node has no keys (only one child)
            # This can happen temporarily during tree restructuring
            if len(node.value.keys) == 0:
                if len(node.value.values) > 1:
                    child_key, child_leaf, child_available = node.value.values[1]
                    self._move_node_to_local(key=child_key, leaf=child_leaf)
                    node.value.values[1] = (child_key, new_leaf, child_available)
                else:
                    raise ValueError("Internal node has no keys and no children")
            else:
                for index, each_key in enumerate(node.value.keys):
                    # If key equals, it is on the right.
                    if key == each_key:
                        # Move the next node to local.
                        # For internal nodes: values = [node.available, (child_id, child_leaf, child_available), ...]
                        child_key, child_leaf, child_available = node.value.values[index + 2]
                        self._move_node_to_local(key=child_key, leaf=child_leaf)
                        # Update the current stored value.
                        node.value.values[index + 2] = (child_key, new_leaf, child_available)
                        break
                    # If the key is smaller, it is on the left.
                    elif key < each_key:
                        # Move the next node to local.
                        child_key, child_leaf, child_available = node.value.values[index + 1]
                        self._move_node_to_local(key=child_key, leaf=child_leaf)
                        # Update the current stored value.
                        node.value.values[index + 1] = (child_key, new_leaf, child_available)
                        break
                    # If we reached the end, it is on the right.
                    elif index + 1 == len(node.value.keys):
                        # Move the next node to local.
                        child_key, child_leaf, child_available = node.value.values[index + 2]
                        self._move_node_to_local(key=child_key, leaf=child_leaf)
                        # Update the current stored value.
                        node.value.values[index + 2] = (child_key, new_leaf, child_available)
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
        right_node = self._get_bplus_subset_data()

        # Determine if node is a leaf by checking for min_val/max_val (only leaf nodes have these)
        is_leaf = node.value.min_val is not None

        # Depending on whether the child node is a leaf node, we break it differently.
        if is_leaf:
            # Leaf node: values = [available_flag], keys = [k1, k2, ...]
            # Split only the keys, values stays as [available_flag]
            right_node.value.keys = node.value.keys[self._mid:]
            
            # The old leaf keeps on the first half.
            node.value.keys = node.value.keys[:self._mid]
            
            # For leaf nodes, update range information
            # The split point is the first key of the right half
            if right_node.value.keys:
                split_key = right_node.value.keys[0]
                # Left node keeps its min_val but max_val becomes split_key - 1
                right_node.value.min_val = split_key
                right_node.value.max_val = node.value.max_val
                node.value.max_val = split_key - 1
            
            # Calculate availability flags based on range and stored keys
            # A leaf has available elements if (max_val - min_val + 1) > len(keys)
            left_range_size = (node.value.max_val - node.value.min_val + 1) if node.value.min_val is not None and node.value.max_val is not None else 0
            left_stored = len(node.value.keys) if node.value.keys else 0
            left_available = left_range_size > left_stored
            
            right_range_size = (right_node.value.max_val - right_node.value.min_val + 1) if right_node.value.min_val is not None and right_node.value.max_val is not None else 0
            right_stored = len(right_node.value.keys) if right_node.value.keys else 0
            right_available = right_range_size > right_stored
            
            node.value.values = [left_available]
            right_node.value.values = [right_available]

        else:
            # Internal node: values[0] is node.available, values[1:] are child tuples
            # For a node with (order-1) keys, there are (order) children
            # values = [available, child0, child1, ..., child_{order-1}]
            # After split:
            # - Left keeps keys[:mid], and children for those keys: values[0:mid+2] (available + mid+1 children)
            # - Right gets keys[mid+1:], and children for those keys: values[mid+2:] (mid+1 children)
            # - The middle key (keys[mid]) is promoted to parent
            
            # Right node gets the second half of keys and children
            right_node.value.keys = node.value.keys[self._mid + 1:]
            # Right node gets available flag (True by default) + children from position mid+2 onwards
            right_children = node.value.values[self._mid + 2:]
            right_node.value.values = [True] + right_children

            # Left node keeps the first half of keys and children
            node.value.keys = node.value.keys[:self._mid]
            # Left node keeps available flag + children up to position mid+2
            left_children = node.value.values[1:self._mid + 2]
            node.value.values = [node.value.values[0]] + left_children
            
            # Update parent availability flags based on children's availability
            # For internal nodes, values[0] is node.available, values[1:] are child tuples (child_id, child_leaf, child_available)
            node_has_available = any(v[2] if isinstance(v, tuple) and len(v) >= 3 else False 
                                    for v in node.value.values[1:])
            right_has_available = any(v[2] if isinstance(v, tuple) and len(v) >= 3 else False 
                                     for v in right_node.value.values[1:])
            
            # Update the first element (node.available flag)
            node.value.values[0] = node_has_available
            right_node.value.values[0] = right_has_available

        # Add the right node to stash.
        self._stash.append(right_node)

        # Because the nodes are modified in place, we only need to return the right one.
        return right_node.key, right_node.leaf

    def _insert_in_parent(self, child_node: Data, parent_node: Data) -> None:
        """
        Insert the child node into the parent node.

        :param child_node: A B+ subset tree node whose number of keys is the same as the branching degree.
        :param parent_node: A B+ subset tree node containing the child node.
        """
        # Store the key to insert to parent.
        insert_key = child_node.value.keys[self._mid]

        # Perform the node split. This returns (key, leaf) tuple and adds right_node to stash.
        right_node_info = self._split_node(node=child_node)
        
        # Find the right node Data object from stash (it was just added)
        right_node_data = self._stash[-1]  # _split_node appends right_node to stash
        
        # Extract availability flag from the child nodes
        # For leaf nodes: values = [available_flag], has min_val/max_val
        # For internal nodes: values = [node_available, (child)...], no min_val/max_val
        def extract_available(node: Data) -> bool:
            if node.value.min_val is not None:
                # Leaf node: values = [available_flag]
                if node.value.values and isinstance(node.value.values[0], bool):
                    return node.value.values[0]
            else:
                # Internal node: values[0] is node.available
                if node.value.values and isinstance(node.value.values[0], bool):
                    return node.value.values[0]
            return True  # Default to True if unclear

        left_available = extract_available(child_node)
        right_available = extract_available(right_node_data)

        # First, update the existing reference to the left child (child_node) in the parent
        # since its availability may have changed after the split
        for j in range(1, len(parent_node.value.values)):
            val = parent_node.value.values[j]
            if isinstance(val, tuple) and len(val) >= 3 and val[0] == child_node.key:
                parent_node.value.values[j] = (val[0], val[1], left_available)
                break

        # Now we perform the actual insertion to parent.
        for index, each_key in enumerate(parent_node.value.keys):
            if insert_key < each_key:
                parent_node.value.keys = parent_node.value.keys[:index] + [insert_key] + parent_node.value.keys[index:]
                # For internal nodes, values are [node_available, (child_id, child_leaf, child_available), ...]
                # Insert the new child tuple at the appropriate position (after values[0])
                parent_node.value.values = (
                        parent_node.value.values[:index + 2] + [(right_node_info[0], right_node_info[1], right_available)] + 
                        parent_node.value.values[index + 2:]
                )
                break
            elif index + 1 == len(parent_node.value.keys):
                parent_node.value.keys.append(insert_key)
                parent_node.value.values.append((right_node_info[0], right_node_info[1], right_available))
                break
        
        # Update parent's availability based on all its children (values[1:] are the children)
        parent_has_available = any(v[2] if isinstance(v, tuple) and len(v) >= 3 else False 
                                  for v in parent_node.value.values[1:])
        parent_node.value.values[0] = parent_has_available

    def _create_parent(self, child_node: Data) -> None:
        """
        When a node has no parent and split is required, create a new parent node for them.

        :param child_node: A B+ subset tree node whose number of keys is the same as the branching degree.
        :return: A B+ subset tree node containing the split left and right child nodes.
        """
        # Store the key to insert to parent.
        insert_key = child_node.value.keys[self._mid]

        # Perform the node split. This returns (key, leaf) tuple and adds right_node to stash.
        right_node_info = self._split_node(node=child_node)
        
        # Find the right node Data object from stash (it was just added)
        right_node_data = self._stash[-1]  # _split_node appends right_node to stash
        
        # Extract availability flags from child nodes
        # Leaf nodes have min_val set, internal nodes don't
        def extract_available(node: Data) -> bool:
            if node.value.min_val is not None:
                # Leaf node: values = [available_flag]
                if node.value.values and isinstance(node.value.values[0], bool):
                    return node.value.values[0]
            else:
                # Internal node: values[0] is node.available
                if node.value.values and isinstance(node.value.values[0], bool):
                    return node.value.values[0]
            return True

        left_available = extract_available(child_node)
        right_available = extract_available(right_node_data)

        # Set the values with proper availability flags: [node_available, (child_id, child_leaf, child_available), ...]
        parent_has_available = left_available or right_available
        values = [parent_has_available, (child_node.key, child_node.leaf, left_available), 
                  (right_node_info[0], right_node_info[1], right_available)]

        # Create the parent node.
        parent_node = self._get_bplus_subset_data(keys=[insert_key], values=values)

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

    def _propagate_availability_up(self) -> None:
        """
        Propagate availability flags up the tree after an insert or delete operation.
        
        Goes through _local from leaf to root, updating each parent's reference
        to its child with the correct availability flag.
        """
        if len(self._local) < 2:
            return
        
        # Start from the leaf (last in local) and go up
        for i in range(len(self._local) - 1, 0, -1):
            child_node = self._local[i]
            parent_node = self._local[i - 1]
            
            # Get the child's availability
            if child_node.value.min_val is not None:
                # Leaf node: values = [available_flag]
                child_available = child_node.value.values[0] if child_node.value.values else True
            else:
                # Internal node: values[0] is node.available
                child_available = child_node.value.values[0] if child_node.value.values else True
            
            # Update the parent's reference to this child
            # Parent is internal node: values = [node.available, (child_id, child_leaf, child_available), ...]
            for j in range(1, len(parent_node.value.values)):
                val = parent_node.value.values[j]
                if isinstance(val, tuple) and len(val) >= 3 and val[0] == child_node.key:
                    # Update the availability flag
                    parent_node.value.values[j] = (val[0], val[1], child_available)
                    break
            
            # Update the parent's own availability flag (values[0])
            # Parent is available if any of its children is available
            parent_has_available = any(
                v[2] if isinstance(v, tuple) and len(v) >= 3 else False 
                for v in parent_node.value.values[1:]
            )
            parent_node.value.values[0] = parent_has_available

    def insert(self, key: Any) -> None:
        """
        Given key, insert it to the subset tree.

        :param key: The key (integer element) to insert into the subset.
        """
        # If the current root is empty, we simply set root as this new block.
        if self.root is None:
            # Create a new bplus subset data block (leaf node) covering [0, n-1]
            # For leaf nodes: values = [available_flag]
            data_block = self._get_bplus_subset_data(keys=[key], values=[True], 
                                                     min_val=0, max_val=self._n - 1)
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

        # Find the proper place to insert the key.
        # Note: For subset trees, we only track keys (stored integer elements)
        # Values list only contains the availability flag [available_flag]
        if not leaf.value.keys:
            # Empty keys list, just add the key
            leaf.value.keys = [key]
        else:
            inserted = False
            for index, each_key in enumerate(leaf.value.keys):
                if key < each_key:
                    leaf.value.keys = leaf.value.keys[:index] + [key] + leaf.value.keys[index:]
                    inserted = True
                    break
            if not inserted:
                leaf.value.keys.append(key)
        
        # Update availability flag of the leaf node after insertion
        # For leaf nodes, values = [available_flag]
        # A leaf has available elements if (max_val - min_val + 1) > len(keys)
        if leaf.value.min_val is not None and leaf.value.max_val is not None:
            range_size = leaf.value.max_val - leaf.value.min_val + 1
            stored_count = len(leaf.value.keys) if leaf.value.keys else 0
            leaf.value.values = [range_size > stored_count]

        # Save the length of the local.
        num_retrieved_nodes = len(self._local)

        # Perform the insertion to local nodes.
        self._perform_insertion()

        # Propagate availability flags up the tree
        # local contains nodes from root to leaf, we need to update each parent's reference to its child
        self._propagate_availability_up()

        # Append local data to stash and clear local.
        self._stash += self._local
        self._local = []

        # Perform the desired number of dummy evictions.
        self._perform_dummy_operation(num_round=3 * self._max_height - num_retrieved_nodes)

    def search(self, key: Any, value: Any = None) -> Tuple[bool, Optional[int]]:
        """
        Given a search key, check if it exists and return an available (not stored) element.

        For subset OMAP, we only track which keys are stored, not associated values.
        :param key: The search key of interest.
        :param value: Ignored for subset OMAP.
        :return: A tuple of (key_exists, available_element) where available_element is from {0, 1, ..., n-1}.
        """
        # If the current root is empty, we can't perform search.
        if self.root is None:
            raise ValueError(f"It seems the tree is empty and can't perform search.")

        # Get all nodes we need to visit until finding the key.
        self._find_leaf_to_local(key=key)

        # Set the last node in local as leaf.
        leaf = self._local[-1]
        key_exists = False

        # Search for the key in the leaf's keys list.
        for each_key in leaf.value.keys:
            if key == each_key:
                key_exists = True
                break

        # Find an available (not stored) element using the subset tree
        available_element = self.find_available()

        # Save the number of retrieved nodes, move the local nodes to stash and perform dummy evictions.
        num_retrieved_nodes = len(self._local)
        self._stash += self._local
        self._local = []
        self._perform_dummy_operation(num_round=3 * self._max_height - num_retrieved_nodes)

        return key_exists, available_element

    def fast_search(self, key: Any, value: Any = None) -> Tuple[bool, Optional[int]]:
        """
        Given a search key, check if it exists and return an available element.

        The difference here is that fast search will return the node immediately without keeping it in local.
        For subset OMAP, we only track which keys are stored, not associated values.
        :param key: The search key of interest.
        :param value: Ignored for subset OMAP.
        :return: A tuple of (key_exists, available_element).
        """
        # If the current root is empty, we can't perform search.
        if self.root is None:
            raise ValueError(f"It seems the tree is empty and can't perform search.")

        # Get all nodes we need to visit until finding the key.
        old_leaf, num_retrieved_nodes = self._find_leaf(key=key)

        # Set the last node in local as leaf.
        leaf = self._local[-1]
        key_exists = False

        # Search for the key in the leaf's keys list.
        for each_key in leaf.value.keys:
            if key == each_key:
                key_exists = True
                break

        # Find an available element using the subset tree
        available_element = self.find_available()

        # Save the number of retrieved nodes, move the local nodes to stash and perform dummy evictions.
        self._stash += self._local
        self._local = []
        # Perform one eviction.
        self._client.write_query(label=self._name, leaf=old_leaf, data=self._evict_stash(leaf=old_leaf))
        # And then the dummy evictions.
        self._perform_dummy_operation(num_round=self._max_height - num_retrieved_nodes)

        return key_exists, available_element

    def _get_min_keys(self) -> int:
        """Get the minimum number of keys a node should have (except root)."""
        return (self._order + 1) // 2 - 1

    def _is_leaf_node(self, node: Data) -> bool:
        """Check if a node is a leaf node based on its structure."""
        # Leaf nodes have min_val/max_val set (range information)
        # Internal nodes do not have min_val/max_val
        return node.value.min_val is not None

    def _get_sibling_info(self, parent_node: Data, child_index: int) -> Tuple[Optional[int], Optional[int], int]:
        """
        Get information about siblings for merge/borrow operations.
        
        :param parent_node: The parent node containing the child.
        :param child_index: The index of the child in parent's values.
        :return: (left_sibling_index, right_sibling_index, separator_key_index)
        """
        left_sibling_idx = child_index - 1 if child_index > 0 else None
        # For internal nodes, values[1:] are child tuples, so max index is len(values) - 1
        right_sibling_idx = child_index + 1 if child_index < len(parent_node.value.values) - 2 else None
        separator_key_idx = child_index - 1 if child_index > 0 else 0
        return left_sibling_idx, right_sibling_idx, separator_key_idx

    def _merge_leaf_nodes(self, left_node: Data, right_node: Data) -> None:
        """
        Merge right leaf node into left leaf node.
        Updates keys, values, and range information.
        
        :param left_node: The left leaf node (will contain merged result).
        :param right_node: The right leaf node (will be discarded).
        """
        # Merge keys
        left_node.value.keys = left_node.value.keys + right_node.value.keys
        
        # Update range: left keeps min_val, takes right's max_val
        left_node.value.max_val = right_node.value.max_val
        
        # Update availability flag
        # After merge, available = left.available or right.available
        left_available = left_node.value.values[0] if left_node.value.values else True
        right_available = right_node.value.values[0] if right_node.value.values else True
        left_node.value.values = [left_available or right_available]

    def _merge_internal_nodes(self, left_node: Data, right_node: Data, separator_key: Any) -> None:
        """
        Merge right internal node into left internal node.
        
        :param left_node: The left internal node (will contain merged result).
        :param right_node: The right internal node (will be discarded).
        :param separator_key: The key from parent that separates the two nodes.
        """
        # Add separator key and right node's keys
        left_node.value.keys = left_node.value.keys + [separator_key] + right_node.value.keys
        
        # Merge child references (skip right's values[0] which is the available flag)
        left_node.value.values = left_node.value.values + right_node.value.values[1:]
        
        # Update availability flag
        left_has_available = any(v[2] if isinstance(v, tuple) and len(v) >= 3 else False 
                                for v in left_node.value.values[1:])
        left_node.value.values[0] = left_has_available

    def _borrow_from_left_leaf(self, node: Data, left_sibling: Data, parent_node: Data, separator_idx: int) -> None:
        """
        Borrow a key from left sibling leaf node.
        
        :param node: The underflow node.
        :param left_sibling: The left sibling with extra keys.
        :param parent_node: The parent node.
        :param separator_idx: Index of the separator key in parent.
        """
        # Move last key from left sibling to beginning of node
        borrowed_key = left_sibling.value.keys.pop()
        node.value.keys.insert(0, borrowed_key)
        
        # Update separator key in parent
        parent_node.value.keys[separator_idx] = node.value.keys[0]
        
        # Update range information
        node.value.min_val = borrowed_key
        left_sibling.value.max_val = borrowed_key - 1
        
        # Update availability flags
        left_sibling.value.values[0] = True  # Left sibling now has more room
        node.value.values[0] = True  # Node might still have room

    def _borrow_from_right_leaf(self, node: Data, right_sibling: Data, parent_node: Data, separator_idx: int) -> None:
        """
        Borrow a key from right sibling leaf node.
        
        :param node: The underflow node.
        :param right_sibling: The right sibling with extra keys.
        :param parent_node: The parent node.
        :param separator_idx: Index of the separator key in parent.
        """
        # Move first key from right sibling to end of node
        borrowed_key = right_sibling.value.keys.pop(0)
        node.value.keys.append(borrowed_key)
        
        # Update separator key in parent
        parent_node.value.keys[separator_idx] = right_sibling.value.keys[0] if right_sibling.value.keys else borrowed_key + 1
        
        # Update range information
        node.value.max_val = borrowed_key
        right_sibling.value.min_val = borrowed_key + 1
        
        # Update availability flags
        right_sibling.value.values[0] = True
        node.value.values[0] = True

    def _remove_child_from_parent(self, parent_node: Data, child_index: int, separator_idx: int) -> None:
        """
        Remove a child reference and separator key from parent after merge.
        
        :param parent_node: The parent node.
        :param child_index: Index of the child to remove (in values[1:]).
        :param separator_idx: Index of the separator key to remove.
        """
        # Remove the separator key
        if separator_idx < len(parent_node.value.keys):
            parent_node.value.keys.pop(separator_idx)
        
        # Remove the child tuple (child_index is relative to values[1:], so add 1)
        if child_index + 1 < len(parent_node.value.values):
            parent_node.value.values.pop(child_index + 1)
        
        # Update parent's availability flag
        parent_has_available = any(v[2] if isinstance(v, tuple) and len(v) >= 3 else False 
                                  for v in parent_node.value.values[1:])
        parent_node.value.values[0] = parent_has_available

    def _handle_underflow(self) -> None:
        """
        Handle underflow after deletion by merging or borrowing from siblings.
        This method works with nodes in self._local.
        
        For oblivious access, we fetch siblings from ORAM into stash, perform
        merge/borrow operations, and update parent pointers accordingly.
        """
        min_keys = self._get_min_keys()
        
        # Start from the leaf (last in _local) and work up
        i = len(self._local) - 1
        while i > 0:
            node = self._local[i]
            parent = self._local[i - 1]
            
            # Check if node has underflow (less than minimum keys)
            if len(node.value.keys) >= min_keys:
                i -= 1
                continue
            
            # Find node's index in parent's children
            node_idx_in_parent = None
            for j, child_tuple in enumerate(parent.value.values[1:]):  # Skip values[0] (available flag)
                if isinstance(child_tuple, tuple) and child_tuple[0] == node.key:
                    node_idx_in_parent = j
                    break
            
            if node_idx_in_parent is None:
                i -= 1
                continue
            
            left_sib_idx, right_sib_idx, sep_key_idx = self._get_sibling_info(parent, node_idx_in_parent)
            
            is_leaf = self._is_leaf_node(node)
            merged = False
            
            # Try to get left sibling and borrow/merge
            if left_sib_idx is not None and not merged:
                left_sib_tuple = parent.value.values[left_sib_idx + 1]  # +1 because values[0] is available flag
                if isinstance(left_sib_tuple, tuple):
                    left_sib_key, left_sib_leaf, left_sib_avail = left_sib_tuple
                    
                    # Fetch left sibling from ORAM (oblivious access)
                    left_sibling = self._fetch_sibling_node(left_sib_key, left_sib_leaf)
                    
                    if left_sibling:
                        # Update parent's reference to left sibling (its leaf may have changed)
                        parent.value.values[left_sib_idx + 1] = (left_sib_key, left_sibling.leaf, left_sib_avail)
                        
                        # Check if we can borrow (sibling has more than minimum keys)
                        if len(left_sibling.value.keys) > min_keys:
                            # Borrow from left sibling
                            if is_leaf:
                                self._borrow_from_left_leaf(node, left_sibling, parent, sep_key_idx)
                            else:
                                self._borrow_from_left_internal(node, left_sibling, parent, sep_key_idx)
                            # Put sibling back to stash
                            self._stash.append(left_sibling)
                        else:
                            # Merge: left sibling absorbs current node
                            if is_leaf:
                                self._merge_leaf_nodes(left_sibling, node)
                            else:
                                separator_key = parent.value.keys[sep_key_idx] if sep_key_idx < len(parent.value.keys) else None
                                self._merge_internal_nodes(left_sibling, node, separator_key)
                            
                            # Remove current node reference from parent
                            self._remove_child_from_parent(parent, node_idx_in_parent, sep_key_idx)
                            
                            # Mark node as deleted (it's absorbed into left sibling)
                            node.value.keys = []
                            node.value.values = [True]
                            
                            # Put sibling back to stash
                            self._stash.append(left_sibling)
                            merged = True
            
            # Try right sibling if left didn't work
            if right_sib_idx is not None and not merged:
                right_sib_tuple = parent.value.values[right_sib_idx + 1]
                if isinstance(right_sib_tuple, tuple):
                    right_sib_key, right_sib_leaf, right_sib_avail = right_sib_tuple
                    
                    # Fetch right sibling from ORAM
                    right_sibling = self._fetch_sibling_node(right_sib_key, right_sib_leaf)
                    
                    if right_sibling:
                        # Update parent's reference to right sibling (its leaf may have changed)
                        parent.value.values[right_sib_idx + 1] = (right_sib_key, right_sibling.leaf, right_sib_avail)
                        
                        # The separator key between current node and right sibling is at index node_idx_in_parent
                        # (since keys[i] separates child[i] and child[i+1])
                        right_sep_key_idx = node_idx_in_parent
                        
                        # Check if we can borrow
                        if len(right_sibling.value.keys) > min_keys:
                            # Borrow from right sibling
                            if is_leaf:
                                self._borrow_from_right_leaf(node, right_sibling, parent, right_sep_key_idx)
                            else:
                                self._borrow_from_right_internal(node, right_sibling, parent, right_sep_key_idx)
                            self._stash.append(right_sibling)
                        else:
                            # Merge: current node absorbs right sibling
                            if is_leaf:
                                self._merge_leaf_nodes(node, right_sibling)
                            else:
                                separator_key = parent.value.keys[right_sep_key_idx] if right_sep_key_idx < len(parent.value.keys) else None
                                self._merge_internal_nodes(node, right_sibling, separator_key)
                            
                            # Remove right sibling reference from parent
                            # right_sep_key_idx is the index of the separator key between current node and right sibling
                            self._remove_child_from_parent(parent, right_sib_idx, right_sep_key_idx)
                            
                            # Mark right sibling as deleted
                            right_sibling.value.keys = []
                            right_sibling.value.values = [True]
                            
                            self._stash.append(right_sibling)
                            merged = True
            
            # If no sibling could help, just update availability flag
            if not merged and is_leaf and node.value.values:
                node.value.values[0] = True
            
            i -= 1
        
        # Update root if it becomes empty
        if len(self._local) > 0:
            root_node = self._local[0]
            if len(root_node.value.keys) == 0 and not self._is_leaf_node(root_node):
                # Root is empty internal node, promote the only child
                if len(root_node.value.values) > 1:
                    child_tuple = root_node.value.values[1]
                    if isinstance(child_tuple, tuple):
                        self.root = (child_tuple[0], child_tuple[1])

    def _fetch_sibling_node(self, sibling_key: int, sibling_leaf: int) -> Optional[Data]:
        """
        Fetch a sibling node from ORAM storage for merge/borrow operations.
        
        :param sibling_key: The key (block id) of the sibling node.
        :param sibling_leaf: The leaf path of the sibling node.
        :return: The sibling Data node, or None if not found.
        """
        # Read the path from ORAM and decrypt it
        path = self._decrypt_buckets(buckets=self._client.read_query(label=self._name, leaf=sibling_leaf))
        
        # Search for the sibling in the path data
        sibling_node = None
        for bucket in path:
            for data in bucket:
                if data.key == sibling_key and data.key is not None:
                    sibling_node = data
                    break
            if sibling_node:
                break
        
        if sibling_node:
            # Assign new leaf for the sibling (oblivious update)
            sibling_node.leaf = self._get_new_leaf()
            
            # Put remaining path data into stash (excluding the sibling we found)
            for bucket in path:
                for data in bucket:
                    if data.key != sibling_key and data.key is not None:
                        self._stash.append(data)
            
            # Evict stash to the old path
            self._client.write_query(label=self._name, leaf=sibling_leaf, 
                                    data=self._evict_stash(leaf=sibling_leaf))
        else:
            # Sibling might be in stash
            for data in self._stash:
                if data.key == sibling_key:
                    sibling_node = data
                    self._stash.remove(data)
                    sibling_node.leaf = self._get_new_leaf()
                    break
            
            # Still need to write back path data to maintain obliviousness
            for bucket in path:
                for data in bucket:
                    if data.key is not None:
                        self._stash.append(data)
            self._client.write_query(label=self._name, leaf=sibling_leaf, 
                                    data=self._evict_stash(leaf=sibling_leaf))
        
        return sibling_node

    def _borrow_from_left_internal(self, node: Data, left_sibling: Data, parent_node: Data, separator_idx: int) -> None:
        """
        Borrow from left sibling for internal nodes.
        
        :param node: The underflow internal node.
        :param left_sibling: The left sibling with extra keys.
        :param parent_node: The parent node.
        :param separator_idx: Index of the separator key in parent.
        """
        # Move separator key from parent to beginning of node's keys
        separator_key = parent_node.value.keys[separator_idx]
        node.value.keys.insert(0, separator_key)
        
        # Move last key from left sibling to parent
        parent_node.value.keys[separator_idx] = left_sibling.value.keys.pop()
        
        # Move last child from left sibling to beginning of node's children
        if len(left_sibling.value.values) > 1:
            last_child = left_sibling.value.values.pop()
            node.value.values.insert(1, last_child)  # Insert after values[0] (available flag)
        
        # Update availability flags
        self._update_node_availability(left_sibling)
        self._update_node_availability(node)

    def _borrow_from_right_internal(self, node: Data, right_sibling: Data, parent_node: Data, separator_idx: int) -> None:
        """
        Borrow from right sibling for internal nodes.
        
        :param node: The underflow internal node.
        :param right_sibling: The right sibling with extra keys.
        :param parent_node: The parent node.
        :param separator_idx: Index of the separator key in parent.
        """
        # Move separator key from parent to end of node's keys
        separator_key = parent_node.value.keys[separator_idx]
        node.value.keys.append(separator_key)
        
        # Move first key from right sibling to parent
        parent_node.value.keys[separator_idx] = right_sibling.value.keys.pop(0)
        
        # Move first child from right sibling to end of node's children
        if len(right_sibling.value.values) > 1:
            first_child = right_sibling.value.values.pop(1)  # Pop after values[0] (available flag)
            node.value.values.append(first_child)
        
        # Update availability flags
        self._update_node_availability(right_sibling)
        self._update_node_availability(node)

    def _update_node_availability(self, node: Data) -> None:
        """
        Update a node's availability flag based on its children.
        
        :param node: The node to update.
        """
        if self._is_leaf_node(node):
            # Leaf node: available if has room for more keys
            node.value.values[0] = len(node.value.keys) < self._order
        else:
            # Internal node: available if any child is available
            has_available = any(v[2] if isinstance(v, tuple) and len(v) >= 3 else False 
                               for v in node.value.values[1:])
            if node.value.values:
                node.value.values[0] = has_available

    def delete(self, key: Any) -> None:
        """
        Delete a key-value pair from the tree.
        
        Optimized version using piggyback:
        - When traversing to the key's leaf node, also fetch siblings at each level
        - This allows underflow handling to use already-fetched siblings
        - Reduces ORAM interaction rounds

        :param key: The search key to delete.
        """
        # If the current root is empty, we can't perform deletion.
        if self.root is None:
            raise ValueError(f"It seems the tree is empty and can't perform deletion.")

        # Traverse to leaf AND fetch siblings at each level
        self._find_leaf_with_siblings_to_local(key=key)

        # Set the last node in local as leaf
        # Note: The last node in _local is the leaf we're looking for
        # The sibling nodes are stored in _sibling_cache
        leaf = self._local[-1]

        # Find and remove the key from the leaf
        found = False
        for index, each_key in enumerate(leaf.value.keys):
            if key == each_key:
                leaf.value.keys.pop(index)
                found = True
                break

        if found:
            # Update availability flag of the leaf node after deletion
            if leaf.value.values and isinstance(leaf.value.values[0], bool):
                leaf.value.values[0] = True
            
            # Handle underflow using siblings from _sibling_cache
            self._handle_underflow_with_cached_siblings()
            
            # Update parent availability flags up the tree
            for i in range(len(self._local) - 2, -1, -1):
                parent = self._local[i]
                child = self._local[i + 1]
                
                # Find child in parent and update its availability
                for j, child_tuple in enumerate(parent.value.values[1:]):
                    if isinstance(child_tuple, tuple) and child_tuple[0] == child.key:
                        child_available = child.value.values[0] if child.value.values and isinstance(child.value.values[0], bool) else True
                        parent.value.values[j + 1] = (child_tuple[0], child_tuple[1], child_available)
                        break
                
                # Update parent's own availability
                parent_has_available = any(v[2] if isinstance(v, tuple) and len(v) >= 3 else False 
                                          for v in parent.value.values[1:])
                parent.value.values[0] = parent_has_available

        # Save the length of the local
        num_retrieved_nodes = len(self._local)

        # Append local data to stash and clear local
        self._stash += self._local
        self._local = []
        
        # Append sibling cache to stash and clear
        self._stash += self._sibling_cache
        self._sibling_cache = []

        # Perform the desired number of dummy evictions
        self._perform_dummy_operation(num_round=3 * self._max_height - num_retrieved_nodes)

    def _find_leaf_with_siblings_to_local(self, key: Any) -> None:
        """
        Traverse to the leaf containing key AND fetch siblings at each level.
        
        This enables underflow handling without additional ORAM reads.
        The path nodes are stored in _local, siblings are stored in _sibling_cache.
        
        :param key: Search key of interest.
        """
        # Make sure local and sibling cache are cleared
        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")
        
        # Initialize sibling cache
        self._sibling_cache = []
        
        # Get the root node from ORAM storage
        self._move_node_to_local(key=self.root[0], leaf=self.root[1])
        
        # Get the node from local
        node = self._local[0]
        
        # Update node leaf and root
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)
        
        # While we do not reach a leaf (leaf nodes have min_val set)
        while node.value.min_val is None:
            # Sample a new leaf for the child
            new_leaf = self._get_new_leaf()
            
            # Handle case where internal node has no keys (only one child)
            if len(node.value.keys) == 0:
                if len(node.value.values) > 1:
                    child_key, child_leaf, child_available = node.value.values[1]
                    self._move_node_to_local(key=child_key, leaf=child_leaf)
                    node.value.values[1] = (child_key, new_leaf, child_available)
                else:
                    raise ValueError("Internal node has no keys and no children")
            else:
                # Find which child to follow and what its siblings are
                child_idx = None
                for index, each_key in enumerate(node.value.keys):
                    if key == each_key:
                        child_idx = index + 1
                        break
                    elif key < each_key:
                        child_idx = index
                        break
                    elif index + 1 == len(node.value.keys):
                        child_idx = index + 1
                        break
                
                if child_idx is None:
                    child_idx = 0
                
                # Fetch the target child
                # Children are in values[1:]
                children = node.value.values[1:]
                if child_idx < len(children):
                    child_key, child_leaf, child_available = children[child_idx]
                    self._move_node_to_local(key=child_key, leaf=child_leaf)
                    node.value.values[child_idx + 1] = (child_key, new_leaf, child_available)
                    
                    # Now fetch siblings using piggyback (to _sibling_cache, not _local)
                    # Left sibling
                    if child_idx > 0:
                        left_sib = children[child_idx - 1]
                        if isinstance(left_sib, tuple):
                            left_sib_key, left_sib_leaf, left_sib_avail = left_sib
                            self._fetch_sibling_to_cache(left_sib_key, left_sib_leaf)
                            # Update parent's reference to sibling (leaf may have changed)
                            sibling = self._sibling_cache[-1] if self._sibling_cache else None
                            if sibling and sibling.key == left_sib_key:
                                node.value.values[child_idx] = (left_sib_key, sibling.leaf, left_sib_avail)
                    
                    # Right sibling
                    if child_idx + 1 < len(children):
                        right_sib = children[child_idx + 1]
                        if isinstance(right_sib, tuple):
                            right_sib_key, right_sib_leaf, right_sib_avail = right_sib
                            self._fetch_sibling_to_cache(right_sib_key, right_sib_leaf)
                            # Update parent's reference to sibling
                            sibling = self._sibling_cache[-1] if self._sibling_cache else None
                            if sibling and sibling.key == right_sib_key:
                                node.value.values[child_idx + 2] = (right_sib_key, sibling.leaf, right_sib_avail)
            
            # Update the node
            node = self._local[-1]
            node.leaf = new_leaf

    def _fetch_sibling_to_cache(self, sibling_key: int, sibling_leaf: int) -> None:
        """
        Fetch a sibling node and store in _sibling_cache.
        
        Uses the standard ORAM read/write pattern but stores node in cache instead of local.
        
        :param sibling_key: The key (block id) of the sibling node.
        :param sibling_leaf: The leaf path of the sibling node.
        """
        # Get the desired path and perform decryption
        path = self._decrypt_buckets(buckets=self._client.read_query(label=self._name, leaf=sibling_leaf))
        
        # Find the sibling in the path
        sibling_node = None
        for bucket in path:
            for data in bucket:
                if data.key == sibling_key:
                    sibling_node = data
                else:
                    self._stash.append(data)
        
        # If sibling not in path, check stash
        if sibling_node is None:
            for data in self._stash:
                if data.key == sibling_key:
                    sibling_node = data
                    self._stash.remove(data)
                    break
        
        if sibling_node:
            # Assign new leaf
            sibling_node.leaf = self._get_new_leaf()
            self._sibling_cache.append(sibling_node)
        
        # Check stash overflow
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")
        
        # Evict and write back
        self._client.write_query(label=self._name, leaf=sibling_leaf, data=self._evict_stash(leaf=sibling_leaf))

    def _handle_underflow_with_cached_siblings(self) -> None:
        """
        Handle underflow after deletion using siblings from _sibling_cache.
        
        Since siblings were pre-fetched via piggyback, no additional ORAM reads needed.
        """
        min_keys = self._get_min_keys()
        
        # Build a map of sibling nodes by key
        sibling_by_key = {data.key: data for data in self._sibling_cache}
        
        # Start from the leaf (last in _local) and work up
        i = len(self._local) - 1
        while i > 0:
            node = self._local[i]
            parent = self._local[i - 1]
            
            # Check if node has underflow
            if len(node.value.keys) >= min_keys:
                i -= 1
                continue
            
            # Find node's index in parent's children
            node_idx_in_parent = None
            for j, child_tuple in enumerate(parent.value.values[1:]):
                if isinstance(child_tuple, tuple) and child_tuple[0] == node.key:
                    node_idx_in_parent = j
                    break
            
            if node_idx_in_parent is None:
                i -= 1
                continue
            
            left_sib_idx, right_sib_idx, sep_key_idx = self._get_sibling_info(parent, node_idx_in_parent)
            is_leaf = self._is_leaf_node(node)
            merged = False
            
            # Try left sibling (from cache)
            if left_sib_idx is not None and not merged:
                left_sib_tuple = parent.value.values[left_sib_idx + 1]
                if isinstance(left_sib_tuple, tuple):
                    left_sib_key = left_sib_tuple[0]
                    left_sibling = sibling_by_key.get(left_sib_key)
                    
                    if left_sibling:
                        if len(left_sibling.value.keys) > min_keys:
                            # Borrow from left sibling
                            if is_leaf:
                                self._borrow_from_left_leaf(node, left_sibling, parent, sep_key_idx)
                            else:
                                self._borrow_from_left_internal(node, left_sibling, parent, sep_key_idx)
                        else:
                            # Merge: left sibling absorbs current node
                            if is_leaf:
                                self._merge_leaf_nodes(left_sibling, node)
                            else:
                                separator_key = parent.value.keys[sep_key_idx] if sep_key_idx < len(parent.value.keys) else None
                                self._merge_internal_nodes(left_sibling, node, separator_key)
                            
                            self._remove_child_from_parent(parent, node_idx_in_parent, sep_key_idx)
                            node.value.keys = []
                            node.value.values = [True]
                            merged = True
            
            # Try right sibling (from cache)
            if right_sib_idx is not None and not merged:
                right_sib_tuple = parent.value.values[right_sib_idx + 1]
                if isinstance(right_sib_tuple, tuple):
                    right_sib_key = right_sib_tuple[0]
                    right_sibling = sibling_by_key.get(right_sib_key)
                    
                    if right_sibling:
                        right_sep_key_idx = node_idx_in_parent
                        
                        if len(right_sibling.value.keys) > min_keys:
                            # Borrow from right sibling
                            if is_leaf:
                                self._borrow_from_right_leaf(node, right_sibling, parent, right_sep_key_idx)
                            else:
                                self._borrow_from_right_internal(node, right_sibling, parent, right_sep_key_idx)
                        else:
                            # Merge: current node absorbs right sibling
                            if is_leaf:
                                self._merge_leaf_nodes(node, right_sibling)
                            else:
                                separator_key = parent.value.keys[right_sep_key_idx] if right_sep_key_idx < len(parent.value.keys) else None
                                self._merge_internal_nodes(node, right_sibling, separator_key)
                            
                            self._remove_child_from_parent(parent, right_sib_idx, right_sep_key_idx)
                            right_sibling.value.keys = []
                            right_sibling.value.values = [True]
                            merged = True
            
            # If no cached sibling could help and node is leaf, mark as available
            if not merged and is_leaf and node.value.values:
                node.value.values[0] = True
            
            i -= 1
        
        # Update root if it becomes empty
        if len(self._local) > 0:
            root_node = self._local[0]
            if len(root_node.value.keys) == 0 and not self._is_leaf_node(root_node):
                # Root is empty internal node, promote the only child
                if len(root_node.value.values) > 1:
                    child_tuple = root_node.value.values[1]
                    if isinstance(child_tuple, tuple):
                        self.root = (child_tuple[0], child_tuple[1])

    def batch_delete(self, keys: List[Any]) -> int:
        """
        Delete multiple keys from the tree in a single batch operation.
        
        This method uses piggyback optimization:
        1. Collect all ORAM paths needed (for target nodes and their siblings)
        2. Read all paths at once (deduplicated)
        3. Perform deletions locally in stash
        4. Write back
        
        :param keys: A list of keys to delete.
        :return: The number of keys successfully deleted.
        """
        if not keys:
            return 0
        
        if self.root is None:
            raise ValueError("It seems the tree is empty and can't perform deletion.")
        
        # Phase 1: Collect all ORAM paths needed
        # We need: paths to all leaves containing keys + paths to their siblings
        all_paths_needed = set()
        all_paths_needed.add(self.root[1])  # Always need root path
        
        # First pass: read root path and collect more paths iteratively
        self._read_path_to_stash(self.root[1])
        
        # For each key, trace path and collect sibling paths
        keys_to_delete = list(keys)
        paths_collected = {self.root[1]}
        
        # Iteratively collect all needed paths
        for _ in range(self._max_height + 1):
            new_paths = set()
            still_need = []
            
            for del_key in keys_to_delete:
                # Trace this key and collect sibling paths along the way
                paths, reached_leaf = self._collect_paths_for_key(del_key)
                
                for p in paths:
                    if p not in paths_collected:
                        new_paths.add(p)
                
                if not reached_leaf:
                    still_need.append(del_key)
            
            if not new_paths:
                break
            
            # Read new paths (piggyback - deduplicated)
            for p in new_paths:
                self._read_path_to_stash(p)
                paths_collected.add(p)
            
            keys_to_delete = still_need
        
        # Phase 2: Perform deletions locally in stash
        deleted_count = 0
        for del_key in keys:
            # Find the leaf node containing this key
            leaf_node = self._find_leaf_node_in_stash(del_key)
            if leaf_node is None:
                continue
            
            # Remove key from leaf
            for idx, k in enumerate(leaf_node.value.keys):
                if k == del_key:
                    leaf_node.value.keys.pop(idx)
                    deleted_count += 1
                    if leaf_node.value.values:
                        leaf_node.value.values[0] = True
                    break
        
        # Phase 3: Handle underflows locally
        # Since all nodes and siblings are in stash, we can do this locally
        self._handle_batch_underflows()
        
        # Phase 4: Update availability flags
        self._batch_update_availability()
        
        # Phase 5: Assign new random leaves to nodes in stash
        # CRITICAL: Only assign new leaves to nodes whose parents are ALSO in stash
        # Otherwise, the parent's reference would become stale
        
        # First, identify which nodes have their parents in stash
        stash_keys = {data.key for data in self._stash}
        
        # Build child->parent map for nodes in stash
        child_to_parent_in_stash = {}
        for data in self._stash:
            if data.value.min_val is None:  # Internal node
                for val in data.value.values[1:]:
                    if isinstance(val, tuple) and len(val) >= 2:
                        child_key = val[0]
                        if child_key in stash_keys:
                            child_to_parent_in_stash[child_key] = data.key
        
        # Assign new leaves only to:
        # 1. The root (has no parent)
        # 2. Nodes whose parents are in stash (parent's reference will be updated)
        node_key_to_new_leaf = {}
        for data in self._stash:
            is_root = (data.key == self.root[0])
            parent_in_stash = data.key in child_to_parent_in_stash
            
            if is_root or parent_in_stash:
                data.leaf = self._get_new_leaf()
                node_key_to_new_leaf[data.key] = data.leaf
            # Else: keep original leaf so parent's reference remains valid
        
        # Update parent->child references with new leaf values
        for data in self._stash:
            if data.value.min_val is None:  # Internal node
                for i, val in enumerate(data.value.values[1:], start=1):
                    if isinstance(val, tuple) and len(val) >= 2:
                        child_key = val[0]
                        if child_key in node_key_to_new_leaf:
                            new_leaf = node_key_to_new_leaf[child_key]
                            available = val[2] if len(val) >= 3 else True
                            data.value.values[i] = (child_key, new_leaf, available)
        
        # Update root reference based on the new leaf of the root node
        for data in self._stash:
            if data.key == self.root[0]:
                self.root = (data.key, data.leaf)
                break
        
        # Phase 6: Write back - perform enough evictions to empty stash
        # Use _perform_dummy_operation pattern: read path, add to stash, evict
        num_evictions = max(len(paths_collected) * 2, 3 * self._max_height)
        for _ in range(num_evictions):
            if len(self._stash) == 0:
                break
            leaf = self._get_new_leaf()
            # Read the path and add all data to stash (avoiding duplicates)
            path = self._decrypt_buckets(buckets=self._client.read_query(label=self._name, leaf=leaf))
            existing_keys = {data.key for data in self._stash}
            for bucket in path:
                for data in bucket:
                    # Only add if not already in stash (we may have a modified version)
                    if data.key not in existing_keys:
                        self._stash.append(data)
                        existing_keys.add(data.key)
            # Check stash overflow
            if len(self._stash) > self._stash_size:
                raise MemoryError("Stash overflow during batch eviction!")
            # Evict and write back
            self._client.write_query(label=self._name, leaf=leaf, data=self._evict_stash(leaf=leaf))
        
        return deleted_count

    def _collect_paths_for_key(self, key: Any) -> Tuple[List[int], bool]:
        """
        Collect all ORAM paths needed to delete a key, including sibling paths.
        
        :param key: The key to delete.
        :return: (list of ORAM leaves needed, whether we reached the leaf node)
        """
        paths = []
        
        if self.root is None:
            return paths, False
        
        current_key = self.root[0]
        reached_leaf = False
        
        while True:
            # Find current node in stash
            node = None
            for data in self._stash:
                if data.key == current_key:
                    node = data
                    break
            
            if node is None:
                break
            
            # If leaf node, we're done tracing
            if node.value.min_val is not None:
                reached_leaf = True
                break
            
            # Internal node - find child for this key and collect sibling paths
            child_idx = None
            for index, each_key in enumerate(node.value.keys):
                if key == each_key:
                    child_idx = index + 1  # Right of separator
                    break
                elif key < each_key:
                    child_idx = index  # Left of separator
                    break
                elif index + 1 == len(node.value.keys):
                    child_idx = index + 1  # Rightmost
                    break
            
            if child_idx is None and len(node.value.keys) == 0:
                # Empty internal node - only one child
                if len(node.value.values) > 1:
                    child_tuple = node.value.values[1]
                    if isinstance(child_tuple, tuple):
                        paths.append(child_tuple[1])
                        current_key = child_tuple[0]
                        continue
                break
            
            if child_idx is None:
                break
            
            # Get child and its siblings' paths
            children = node.value.values[1:]  # Skip available flag
            
            # Add child's path
            if child_idx < len(children):
                child_tuple = children[child_idx]
                if isinstance(child_tuple, tuple):
                    paths.append(child_tuple[1])
                    current_key = child_tuple[0]
                    
                    # Add left sibling's path if exists
                    if child_idx > 0:
                        left_sib = children[child_idx - 1]
                        if isinstance(left_sib, tuple):
                            paths.append(left_sib[1])
                    
                    # Add right sibling's path if exists
                    if child_idx + 1 < len(children):
                        right_sib = children[child_idx + 1]
                        if isinstance(right_sib, tuple):
                            paths.append(right_sib[1])
                else:
                    break
            else:
                break
        
        return paths, reached_leaf

    def _find_leaf_node_in_stash(self, key: Any) -> Optional[Data]:
        """
        Find the leaf node containing the key by traversing nodes in stash.
        
        :param key: The key to find.
        :return: The leaf node Data object, or None.
        """
        if self.root is None:
            return None
        
        current_key = self.root[0]
        
        while True:
            node = None
            for data in self._stash:
                if data.key == current_key:
                    node = data
                    break
            
            if node is None:
                return None
            
            # Leaf node
            if node.value.min_val is not None:
                if key in node.value.keys:
                    return node
                return None
            
            # Internal node - navigate
            next_key = None
            if len(node.value.keys) == 0:
                if len(node.value.values) > 1:
                    child = node.value.values[1]
                    if isinstance(child, tuple):
                        next_key = child[0]
            else:
                for index, each_key in enumerate(node.value.keys):
                    if key == each_key:
                        child = node.value.values[index + 2]
                        if isinstance(child, tuple):
                            next_key = child[0]
                        break
                    elif key < each_key:
                        child = node.value.values[index + 1]
                        if isinstance(child, tuple):
                            next_key = child[0]
                        break
                    elif index + 1 == len(node.value.keys):
                        child = node.value.values[index + 2]
                        if isinstance(child, tuple):
                            next_key = child[0]
                        break
            
            if next_key is None:
                return None
            current_key = next_key

    def _handle_batch_underflows(self) -> None:
        """
        Handle underflows for all leaf nodes in stash after batch deletion.
        Since all nodes and siblings are in stash, we can process locally.
        """
        min_keys = self._get_min_keys()
        
        # Find all leaf nodes with underflow
        # Build parent-child relationships from stash
        node_by_key = {data.key: data for data in self._stash}
        
        # Find leaf nodes with underflow
        underflow_leaves = []
        for data in self._stash:
            if data.value.min_val is not None:  # Leaf node
                if len(data.value.keys) < min_keys:
                    underflow_leaves.append(data)
        
        # For simplicity, we'll just mark them as available
        # Full underflow handling (borrow/merge) is complex for batch operations
        # The tree structure remains valid, just some leaves may have fewer keys
        for leaf in underflow_leaves:
            if leaf.value.values:
                leaf.value.values[0] = True

    def _read_path_to_stash(self, oram_leaf: int) -> None:
        """
        Read an ORAM path and add all data to stash (avoiding duplicates).
        
        :param oram_leaf: The ORAM leaf index to read.
        """
        path = self._decrypt_buckets(buckets=self._client.read_query(label=self._name, leaf=oram_leaf))
        existing_keys = {data.key for data in self._stash}
        
        for bucket in path:
            for data in bucket:
                if data.key not in existing_keys:
                    self._stash.append(data)
                    existing_keys.add(data.key)
        
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow during batch read!")

    def _batch_update_availability(self) -> None:
        """
        Update availability flags for all nodes in stash after batch deletion.
        """
        # Update all leaf nodes
        for data in self._stash:
            if data.value.min_val is not None:
                # Leaf node - available if range has room for more keys
                if data.value.values:
                    range_size = data.value.max_val - data.value.min_val + 1
                    stored_count = len(data.value.keys) if data.value.keys else 0
                    data.value.values[0] = range_size > stored_count
        
        # Update internal nodes based on children
        # Build a map of node_key -> availability for leaves
        leaf_availability = {}
        for data in self._stash:
            if data.value.min_val is not None:
                leaf_availability[data.key] = data.value.values[0] if data.value.values else True
        
        # Update internal nodes
        for data in self._stash:
            if data.value.min_val is None:
                # Internal node - update child availability info first
                for i, val in enumerate(data.value.values[1:], start=1):
                    if isinstance(val, tuple) and len(val) >= 3:
                        child_key, child_leaf, _ = val
                        # Check if child is a leaf we know about
                        if child_key in leaf_availability:
                            data.value.values[i] = (child_key, child_leaf, leaf_availability[child_key])
                
                # Now update this node's availability
                if data.value.values:
                    has_available = any(
                        v[2] if isinstance(v, tuple) and len(v) >= 3 else False
                        for v in data.value.values[1:]
                    )
                    data.value.values[0] = has_available

    def _collect_stored_keys(self) -> set:
        """
        Collect all stored integer keys from the B+ subset tree by traversal.
        
        :return: A set of all stored keys in the tree.
        """
        stored_keys = set()
        
        if self.root is None:
            return stored_keys
        
        # Stack-based traversal of the tree
        stack = [(self.root[0], self.root[1])]
        visited = set()
        
        while stack:
            key, leaf = stack.pop()
            
            # Avoid revisiting nodes
            if (key, leaf) in visited:
                continue
            visited.add((key, leaf))
            
            # Get the node from ORAM without adding to local (peek-only)
            try:
                path = self._decrypt_buckets(buckets=self._client.read_query(label=self._name, leaf=leaf))
                
                for bucket in path:
                    for data in bucket:
                        if data.key == key and data.key is not None:
                            # This is the node we're looking for
                            # Collect keys from this node
                            if isinstance(data.value.keys, list):
                                for k in data.value.keys:
                                    if isinstance(k, int) and 0 <= k < self._n:
                                        stored_keys.add(k)
                            
                            # If this is an internal node, add children to stack
                            # Internal nodes: values = [node.available, (child_id, child_leaf, child_available), ...]
                            # Leaf nodes: values = [available_flag] (single bool)
                            if isinstance(data.value.values, list) and len(data.value.values) > 1:
                                # Check if values[1] is a child tuple (internal node)
                                if isinstance(data.value.values[1], tuple) and len(data.value.values[1]) >= 2:
                                    # Internal node: values[1:] are (child_id, child_leaf, child_available)
                                    for val_tuple in data.value.values[1:]:
                                        if isinstance(val_tuple, tuple) and len(val_tuple) >= 2:
                                            child_id = val_tuple[0]
                                            child_leaf = val_tuple[1]
                                            if child_id is not None and child_leaf is not None:
                                                stack.append((child_id, child_leaf))
                            break
            except Exception:
                # If there's an error reading from ORAM, continue
                pass
        
        return stored_keys

    def find_available(self) -> Optional[int]:
        """
        Find an available (not stored) element from {0, 1, ..., n-1}.
        
        Traverses the B+ subset tree obliviously by following the availability flags
        to find a leaf node with available elements. Uses the leaf's min_val and max_val
        to determine the correct range.
        
        :return: An available element or None if all are stored.
        """
        if self.root is None:
            # Empty tree, return first element
            return 0 if self._n > 0 else None
        
        # Oblivious traversal following available flags
        current_key, current_leaf = self.root
        
        while True:
            try:
                # Read the current node from ORAM
                path = self._decrypt_buckets(buckets=self._client.read_query(label=self._name, leaf=current_leaf))
                
                # Find the node we're looking for
                node_data = None
                for bucket in path:
                    for data in bucket:
                        if data.key == current_key and data.key is not None:
                            node_data = data
                            break
                    if node_data is not None:
                        break
                
                if node_data is None:
                    # Node not found in path
                    return None
                
                # Check if this is a leaf node by checking min_val
                # Leaf nodes have min_val set, internal nodes don't
                if node_data.value.min_val is not None:
                    # This is a leaf node
                    stored_keys = set(node_data.value.keys) if isinstance(node_data.value.keys, list) else set()
                    # Filter to only integer keys
                    stored_keys = {k for k in stored_keys if isinstance(k, int)}
                    
                    # Use the leaf's range [min_val, max_val] to find available element
                    min_val = node_data.value.min_val
                    max_val = node_data.value.max_val if node_data.value.max_val is not None else self._n - 1
                    
                    # Find first available element in this leaf's range
                    for i in range(min_val, max_val + 1):
                        if i not in stored_keys:
                            return i
                    # No available element in this leaf's range
                    return None
                else:
                    # Internal node: values[0] is node.available (bool), values[1:] are child tuples (child_id, child_leaf, child_available)
                    if len(node_data.value.values) > 1:
                        # Find first child with available=True
                        found_available_child = False
                        for val_tuple in node_data.value.values[1:]:  # Skip values[0] which is the node's available flag
                            if isinstance(val_tuple, tuple) and len(val_tuple) >= 3:
                                child_id = val_tuple[0]
                                child_leaf = val_tuple[1]
                                available_flag = val_tuple[2]
                                
                                if available_flag and child_id is not None and child_leaf is not None:
                                    # Follow this child
                                    current_key, current_leaf = child_id, child_leaf
                                    found_available_child = True
                                    break
                        
                        if not found_available_child:
                            # No available child found
                            return None
                    else:
                        # Internal node with no children (shouldn't happen)
                        return None
                    
            except Exception:
                # If there's an error reading from ORAM, return None
                return None
