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
                 name: str = "bplus",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 100,
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
        # Add +1 because root node has minimum 2 children (not ceil(order/2)),
        # so worst-case tree height is 1 level higher than the ideal formula.
        self._max_height: int = math.ceil(math.log(num_data, math.ceil(order / 2))) + 1

    def restore_client_state(self, force_reset_caches: bool = False) -> None:
        """
        Restore client state (root pointer) from server metadata if available.
        """
        # Try to fetch root from server
        try:
            root_key = f"{self._name}_root"
            # InteractServer.list_all returns the value directly for memory storage
            # Assuming server stores simple values (like root tuple) directly in storage dict
            # InteractRemoteServer usage is trickier: list_all sends 'la' query.
            # LocalServer sends back self.__storage[label]
            stored_root = self._client.list_all(label=root_key)
            if stored_root:
                self._root = stored_root
                print(f"  [BPlusOdsOmap] Restored root for {self._name}: {self._root}")
        except Exception:
            # Metadata might not exist if using old storage file or fresh init
            if not force_reset_caches:
                 print(f"  [BPlusOdsOmap] Warning: Could not restore root for {self._name} from server.")

    def update_mul_tree_height(self, num_tree: int) -> None:
        """Suppose the ODS is used to store multiple trees, we update each tree's height.

        :param num_tree: Number of trees to store, which is the same as number of data in upper level oram.
        """
        # First compute how many are in the buckets, according to https://eprint.iacr.org/2021/1280.
        tree_size = math.ceil(
            math.e ** (Helper.lambert_w(math.e ** -1 * (math.log(num_tree, 2) + 128 - 1)).real + 1)
        )

        # Update the height accordingly.
        # Add +1 because root node has minimum 2 children (not ceil(order/2)),
        # so worst-case tree height is 1 level higher than the ideal formula.
        self._max_height = math.ceil(math.log(tree_size, math.ceil(self._order / 2))) + 1

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

            # Get nodes from B+ tree
            data_list = bplus_tree.get_data_list(root=root, block_id=self._block_id, encryption=self._use_encryption)

            # Update the block id
            self._block_id += len(data_list)

            # Fill the ORAM tree according to the position map, collecting overflow into a local stash
            init_stash = []
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

                # Get nodes from B+ tree
                data_list = bplus_tree.get_data_list(
                    root=root, block_id=self._block_id, encryption=self._use_encryption
                )

                # Update the block id
                self._block_id += len(data_list)

                # Fill storage and collect overflow
                init_stash = []
                for bplus_data in data_list:
                    if not tree.fill_data_to_storage_leaf(data=bplus_data):
                        init_stash.append(bplus_data)

                # Evict overflowed blocks
                max_attempts = max(2 * self._leaf_range, len(init_stash) * (tree.level))
                attempts = 0
                while init_stash and attempts < max_attempts:
                    attempts += 1
                    leaf = secrets.randbelow(self._leaf_range)
                    path = tree.read_path(leaf=leaf)
                    remaining = []
                    for data_block in init_stash:
                        inserted = BinaryTree.fill_data_to_path(
                            data=data_block,
                            path=path,
                            leaf=leaf,
                            level=tree.level,
                            bucket_size=self._bucket_size
                        )
                        if not inserted:
                            remaining.append(data_block)
                    tree.write_path(leaf=leaf, data=path)
                    init_stash = remaining

                if init_stash:
                    raise MemoryError("Initialization stash overflow in multi-tree storage.")

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
                    self._client.write_query(label=self._name, leaf=old_leaf, data=self._evict_stash(leaf=old_leaf))
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
                    self._client.write_query(label=self._name, leaf=old_leaf, data=self._evict_stash(leaf=old_leaf))
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

    def _evict_stash_mul_path(self, leaves: List[int]) -> Buckets:
        """
        Evict data blocks in the stash to multiple paths simultaneously.
        
        For each item in stash, find the deepest position it can reach across ALL paths.
        This solves the shared bucket problem by considering all paths together.
        
        :param leaves: List of leaf labels for the paths we are evicting data to.
        :return: The encrypted buckets for all paths combined.
        """
        # Create a dictionary for the multi-path structure
        path_dict = BinaryTree.get_mul_path_dict(level=self._level, indices=leaves)
        
        # Create a temporary stash
        temp_stash = []
        
        # Now we evict the stash by going through all real data in it
        for data in self._stash:
            # Attempt to insert actual data to the multi-path
            inserted = BinaryTree.fill_data_to_mul_path(
                data=data, path=path_dict, leaves=leaves, level=self._level, bucket_size=self._bucket_size
            )
            
            # If we were not able to insert data, overflow happened, put the block to the temp stash
            if not inserted:
                temp_stash.append(data)
        
        # Update the stash
        self._stash = temp_stash

        self._update_peak_client_size()
        
        # Convert the path_dict back to the bucket format expected by the server
        # The keys in path_dict are tree indices, we need to convert them to the correct order
        path_indices = BinaryTree.get_mul_path_dict(level=self._level, indices=leaves).keys()
        path_indices = sorted(path_indices, reverse=True)  # From leaves to root
        
        path = [path_dict[idx] for idx in path_indices]
        
        # Return the encrypted buckets
        return self._encrypt_buckets(buckets=path)

    def _find_leaf_to_local_optimized(self, key: Any) -> int:
        """
        Optimized version of _find_leaf_to_local that uses 2h path accesses in h rounds.
        
        In each round, we:
        1. Read 2 paths simultaneously as a combined path (using read_query with [leaf1, leaf2])
        2. Find and keep the target node in _local, add others to stash
        3. Evict simultaneously to both paths using _evict_stash_mul_path
        4. Write back the combined path using write_query with [leaf1, leaf2]
        
        This ensures each round processes 2 paths with proper handling of shared buckets.
        Using combined read/write avoids duplicating data from shared buckets.
        
        :param key: Search key of interest.
        :return: The number of rounds performed.
        """
        # Make sure that the local is cleared and is empty at the moment.
        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")
        
        round_count = 0
        
        # First round: get root node
        real_leaf = self.root[1]
        real_key = self.root[0]
        dummy_leaf = self._get_new_leaf()
        
        # Read both paths as a combined path (shared buckets are read only once)
        combined_path = self._decrypt_buckets(
            buckets=self._client.read_query(label=self._name, leaf=[real_leaf, dummy_leaf])
        )
        
        # Process combined path: find target node, add others to stash
        found = False
        for bucket in combined_path:
            for data in bucket:
                if data.key == real_key:
                    self._local.append(data)
                    found = True
                else:
                    self._stash.append(data)

        self._update_peak_client_size()
        
        # If not found in path, check stash
        if not found:
            for i, data in enumerate(self._stash):
                if data.key == real_key:
                    self._local.append(data)
                    del self._stash[i]
                    found = True
                    break
            if not found:
                raise KeyError(f"The search key {real_key} is not found.")

        self._update_peak_client_size()
        
        # Evict to both paths simultaneously and write back using combined leaves
        evicted = self._evict_stash_mul_path(leaves=[real_leaf, dummy_leaf])
        self._client.write_query(label=self._name, leaf=[real_leaf, dummy_leaf], data=evicted)
        
        # Check stash overflow
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")
        
        round_count += 1
        
        # Get the root node from local and update its leaf
        node = self._local[0]
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)
        
        # Continue traversing while not at a leaf node
        while len(node.value.keys) != len(node.value.values):
            # Find the next child to visit
            new_leaf = self._get_new_leaf()
            child_key = None
            child_leaf = None
            child_index = None
            
            for index, each_key in enumerate(node.value.keys):
                if key == each_key:
                    child_key, child_leaf = node.value.values[index + 1]
                    child_index = index + 1
                    break
                elif key < each_key:
                    child_key, child_leaf = node.value.values[index]
                    child_index = index
                    break
                elif index + 1 == len(node.value.keys):
                    child_key, child_leaf = node.value.values[index + 1]
                    child_index = index + 1
                    break
            
            # Generate dummy leaf for this round
            dummy_leaf = self._get_new_leaf()
            
            # Read both paths as a combined path
            combined_path = self._decrypt_buckets(
                buckets=self._client.read_query(label=self._name, leaf=[child_leaf, dummy_leaf])
            )
            
            # Process combined path: find target node
            found = False
            for bucket in combined_path:
                for data in bucket:
                    if data.key == child_key:
                        self._local.append(data)
                        found = True
                    else:
                        self._stash.append(data)
            
            # If not found in path, check stash
            if not found:
                for i, data in enumerate(self._stash):
                    if data.key == child_key:
                        self._local.append(data)
                        del self._stash[i]
                        found = True
                        break
                if not found:
                    raise KeyError(f"The search key {child_key} is not found.")
            
            # Evict to both paths simultaneously and write back using combined leaves
            evicted = self._evict_stash_mul_path(leaves=[child_leaf, dummy_leaf])
            self._client.write_query(label=self._name, leaf=[child_leaf, dummy_leaf], data=evicted)
            
            # Check stash overflow
            if len(self._stash) > self._stash_size:
                raise MemoryError("Stash overflow!")
            
            round_count += 1
            
            # Update the parent's reference to the child with new leaf
            node.value.values[child_index] = (child_key, new_leaf)
            
            # Move to the new node
            node = self._local[-1]
            node.leaf = new_leaf
        
        return round_count

    def _perform_dummy_rounds(self, num_rounds: int) -> None:
        """
        Perform dummy rounds to pad the access pattern.
        
        Each dummy round reads 2 paths as a combined path, evicts them together to
        the deepest positions across both paths, and writes them back together.
        Using combined read/write avoids duplicating data from shared buckets.
        
        :param num_rounds: Number of dummy rounds to perform.
        """
        if num_rounds <= 0:
            return
        
        for _ in range(num_rounds):
            # Generate two random leaves
            leaf1 = self._get_new_leaf()
            leaf2 = self._get_new_leaf()
            
            # Read both paths as a combined path (shared buckets are read only once)
            combined_path = self._decrypt_buckets(
                buckets=self._client.read_query(label=self._name, leaf=[leaf1, leaf2])
            )
            
            # Add all data from combined path to stash
            for bucket in combined_path:
                for data in bucket:
                    self._stash.append(data)

            self._update_peak_client_size()
            
            # Evict to both paths simultaneously and write back using combined leaves
            evicted = self._evict_stash_mul_path(leaves=[leaf1, leaf2])
            self._client.write_query(label=self._name, leaf=[leaf1, leaf2], data=evicted)
            
            # Check stash overflow
            if len(self._stash) > self._stash_size:
                raise MemoryError("Stash overflow!")

            self._update_peak_client_size()

    def _find_leaf_to_local_with_siblings_optimized(self, key: Any) -> int:
        """
        Optimized traversal that fetches both target nodes AND their siblings in h rounds.
        
        This is designed for delete operations where we need sibling nodes for 
        potential borrow/merge operations.
        
        In each round, we:
        1. Read 2 paths: target child's path + one sibling's path (or dummy if no sibling)
        2. Keep target node in _local, keep sibling in _sibling_cache, add others to stash
        3. Evict to both paths simultaneously
        4. Write back both paths together
        
        After h rounds:
        - _local contains the traversal path from root to leaf
        - _sibling_cache contains siblings at each level (for borrow/merge)
        
        :param key: Search key of interest.
        :return: The number of rounds performed.
        """
        # Make sure that the local is cleared and is empty at the moment.
        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")
        
        # Initialize sibling cache
        self._sibling_cache = []
        
        round_count = 0
        
        # First round: get root node (root has no sibling, use dummy path)
        real_leaf = self.root[1]
        real_key = self.root[0]
        dummy_leaf = self._get_new_leaf()
        
        # Read both paths as a combined path
        combined_path = self._decrypt_buckets(
            buckets=self._client.read_query(label=self._name, leaf=[real_leaf, dummy_leaf])
        )
        
        # Process combined path: find target node, add others to stash
        found = False
        for bucket in combined_path:
            for data in bucket:
                if data.key == real_key:
                    self._local.append(data)
                    found = True
                else:
                    self._stash.append(data)
        
        # If not found in path, check stash
        if not found:
            for i, data in enumerate(self._stash):
                if data.key == real_key:
                    self._local.append(data)
                    del self._stash[i]
                    found = True
                    break
            if not found:
                raise KeyError(f"The search key {real_key} is not found.")
        
        # Evict and write back
        evicted = self._evict_stash_mul_path(leaves=[real_leaf, dummy_leaf])
        self._client.write_query(label=self._name, leaf=[real_leaf, dummy_leaf], data=evicted)
        
        # Check stash overflow
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")
        
        round_count += 1
        
        # Get the root node from local and update its leaf
        node = self._local[0]
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)
        
        # Continue traversing while not at a leaf node
        while len(node.value.keys) != len(node.value.values):
            # Find the next child to visit
            new_leaf = self._get_new_leaf()
            child_key = None
            child_leaf = None
            child_index = None
            
            for index, each_key in enumerate(node.value.keys):
                if key == each_key:
                    child_key, child_leaf = node.value.values[index + 1]
                    child_index = index + 1
                    break
                elif key < each_key:
                    child_key, child_leaf = node.value.values[index]
                    child_index = index
                    break
                elif index + 1 == len(node.value.keys):
                    child_key, child_leaf = node.value.values[index + 1]
                    child_index = index + 1
                    break
            
            # Determine sibling to fetch (prefer left sibling, fallback to right, else dummy)
            sibling_key = None
            sibling_leaf = None
            sibling_index = None
            sibling_new_leaf = self._get_new_leaf()
            
            if child_index > 0:
                # Left sibling exists
                sibling_key, sibling_leaf = node.value.values[child_index - 1]
                sibling_index = child_index - 1
            elif child_index + 1 < len(node.value.values):
                # Right sibling exists
                sibling_key, sibling_leaf = node.value.values[child_index + 1]
                sibling_index = child_index + 1
            else:
                # No sibling, use dummy leaf
                sibling_leaf = self._get_new_leaf()
            
            # Read both paths as a combined path (child + sibling/dummy)
            combined_path = self._decrypt_buckets(
                buckets=self._client.read_query(label=self._name, leaf=[child_leaf, sibling_leaf])
            )
            
            # Process combined path: find target node and sibling
            found_child = False
            found_sibling = False
            for bucket in combined_path:
                for data in bucket:
                    if data.key == child_key:
                        self._local.append(data)
                        found_child = True
                    elif sibling_key is not None and data.key == sibling_key:
                        # Update sibling's leaf and add to cache
                        data.leaf = sibling_new_leaf
                        self._sibling_cache.append(data)
                        found_sibling = True
                    else:
                        self._stash.append(data)

            self._update_peak_client_size()
            
            # If child not found in path, check stash
            if not found_child:
                for i, data in enumerate(self._stash):
                    if data.key == child_key:
                        self._local.append(data)
                        del self._stash[i]
                        found_child = True
                        break
                if not found_child:
                    raise KeyError(f"The search key {child_key} is not found.")
            
            # If sibling not found in path, check stash
            if sibling_key is not None and not found_sibling:
                for i, data in enumerate(self._stash):
                    if data.key == sibling_key:
                        data.leaf = sibling_new_leaf
                        self._sibling_cache.append(data)
                        del self._stash[i]
                        found_sibling = True
                        break
            
            # Evict and write back
            evicted = self._evict_stash_mul_path(leaves=[child_leaf, sibling_leaf])
            self._client.write_query(label=self._name, leaf=[child_leaf, sibling_leaf], data=evicted)
            
            # Check stash overflow
            if len(self._stash) > self._stash_size:
                raise MemoryError("Stash overflow!")
            
            round_count += 1

            self._update_peak_client_size()
            
            # Update the parent's reference to the child with new leaf
            node.value.values[child_index] = (child_key, new_leaf)
            
            # Update the parent's reference to the sibling with new leaf (if sibling exists)
            if sibling_index is not None:
                node.value.values[sibling_index] = (sibling_key, sibling_new_leaf)
            
            # Move to the new node
            node = self._local[-1]
            node.leaf = new_leaf
        
            self._update_peak_client_size()

        return round_count

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

    def insert(self, key: Any, value: Any = None) -> None:
        """
        Given key-value pair, insert the pair to the tree.
        
        Uses optimized traversal: each round reads 2 paths (1 real + 1 random),
        performs immediate eviction, and keeps traversal nodes in local.
        Total rounds = max_height (padded with dummy rounds if needed).

        :param key: The search key of interest.
        :param value: The value to insert.
        """
        if key is None:
            self._perform_dummy_rounds(num_rounds=self._max_height)
            return

        # If the current root is empty, we simply set root as this new block.
        if self.root is None:
            # Create a new bplus data block.
            data_block = self._get_bplus_data(keys=[key], values=[value])
            # Append data block to the stash.
            self._stash.append(data_block)
            self.root = (data_block.key, data_block.leaf)
            # Perform dummy rounds to maintain consistent access pattern.
            self._perform_dummy_rounds(num_rounds=self._max_height)
            return

        # Get all nodes we need to visit until finding the key (optimized version).
        try:
            rounds_used = self._find_leaf_to_local_optimized(key=key)
        except KeyError:
            # Tree structure issue during traversal - perform dummy rounds and return
            self._local = []
            self._perform_dummy_rounds(num_rounds=self._max_height)
            return

        self._update_peak_client_size()

        # Set the last node in local as leaf.
        leaf = self._local[-1]

        # Handle empty leaf node (can happen after delete operations)
        if len(leaf.value.keys) == 0:
            leaf.value.keys = [key]
            leaf.value.values = [value]
        else:
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

        # Perform the insertion to local nodes (splitting if needed).
        self._perform_insertion()

        self._update_peak_client_size()

        # Append local data to stash and clear local.
        self._stash += self._local
        self._local = []

        # Perform dummy rounds to pad to max_height.
        self._perform_dummy_rounds(num_rounds=self._max_height - rounds_used)

    def search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value.
        
        Uses optimized traversal: each round reads 2 paths (1 real + 1 random),
        performs immediate eviction, and keeps traversal nodes in local.
        Total rounds = max_height (padded with dummy rounds if needed).

        If the input value is not None, the value corresponding to the search tree will be updated.
        :param key: The search key of interest.
        :param value: The updated value.
        :return: The (old) value corresponding to the search key, or None if not found.
        """
        if key is None:
            self._perform_dummy_rounds(num_rounds=self._max_height)
            return None

        # If the current root is empty, we can't perform search.
        if self.root is None:
            self._perform_dummy_rounds(num_rounds=self._max_height)
            return None

        # Get all nodes we need to visit until finding the key (optimized version).
        try:
            rounds_used = self._find_leaf_to_local_optimized(key=key)
        except KeyError:
            # Key not found during traversal - perform dummy rounds and return None
            self._local = []
            self._perform_dummy_rounds(num_rounds=self._max_height)
            return None

        self._update_peak_client_size()

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

        # Move the local nodes to stash.
        self._stash += self._local
        self._local = []

        self._update_peak_client_size()
        
        # Perform dummy rounds to pad to max_height.
        self._perform_dummy_rounds(num_rounds=self._max_height - rounds_used)

        return search_value

    def search_local(self, key: Any, value: Any = None) -> Any:
        """
        Search for a key in stash (after parallel_search has loaded the path).
        
        This method assumes the path from root to the target leaf is already
        in the stash after a parallel_search operation. It traverses the stash
        to find the key and optionally updates its value.
        
        Performs dummy ORAM rounds to maintain consistent access pattern.
        
        :param key: The search key of interest.
        :param value: If provided, update the key's value to this.
        :return: The (old) value corresponding to the search key, or None if not found.
        """
        if key is None:
            self._perform_dummy_rounds(num_rounds=self._max_height)
            return None

        if self.root is None:
            self._perform_dummy_rounds(num_rounds=self._max_height)
            return None

        search_value = None
        
        # Traverse stash from root to leaf to find the key
        current_key = self.root[0]
        current_node = None
        
        # Build local path by traversing through stash
        for _ in range(self._max_height):
            # Find current node in stash
            for data in self._stash:
                if data.key == current_key:
                    current_node = data
                    break
            
            if current_node is None:
                break
                
            # Check if leaf node (keys.length == values.length)
            if len(current_node.value.keys) == len(current_node.value.values):
                # This is a leaf, search for key
                for index, each_key in enumerate(current_node.value.keys):
                    if key == each_key:
                        search_value = current_node.value.values[index]
                        if value is not None:
                            current_node.value.values[index] = value
                        break
                break
            else:
                # Internal node, find next child
                child_key = None
                for index, each_key in enumerate(current_node.value.keys):
                    if key == each_key:
                        child_key, _ = current_node.value.values[index + 1]
                        break
                    elif key < each_key:
                        child_key, _ = current_node.value.values[index]
                        break
                    elif index + 1 == len(current_node.value.keys):
                        child_key, _ = current_node.value.values[index + 1]
                        break
                current_key = child_key
                current_node = None
        
        # Perform dummy ORAM rounds to maintain access pattern
        # OPTIMIZATION: In Top-Down protocol, 'search_local' is called after 'parallel_search'
        # which already performed h rounds. We shouldn't add another h rounds here.
        # self._perform_dummy_rounds(num_rounds=self._max_height)

        self._update_peak_client_size()
        
        return search_value

    def insert_local(self, key: Any, value: Any = None) -> None:
        """
        Insert a key-value pair using path already in stash (after parallel_search).
        
        This method assumes the path from root to the target leaf is already
        in the stash after a parallel_search operation. It finds the correct
        leaf in the stash and performs the insertion locally.
        
        Performs dummy ORAM rounds to maintain consistent access pattern.
        
        :param key: The key to insert.
        :param value: The value to insert.
        """
        if key is None:
            self._perform_dummy_rounds(num_rounds=self._max_height)
            return

        # If tree is empty, create root
        if self.root is None:
            data_block = self._get_bplus_data(keys=[key], values=[value])
            self._stash.append(data_block)
            self.root = (data_block.key, data_block.leaf)
            # OPTIMIZATION: Remove forced dummy rounds
            # self._perform_dummy_rounds(num_rounds=self._max_height)
            return

        # Traverse stash from root to leaf, building _local path
        self._local = []
        current_key = self.root[0]
        
        for _ in range(self._max_height):
            # Find current node in stash and move to local
            found_idx = None
            for i, data in enumerate(self._stash):
                if data.key == current_key:
                    found_idx = i
                    break
            
            if found_idx is None:
                break
            
            current_node = self._stash.pop(found_idx)
            self._local.append(current_node)
            
            # Check if leaf node
            if len(current_node.value.keys) == len(current_node.value.values):
                # This is a leaf, stop traversing
                break
            else:
                # Internal node, find next child
                child_key = None
                for index, each_key in enumerate(current_node.value.keys):
                    if key == each_key:
                        child_key, _ = current_node.value.values[index + 1]
                        break
                    elif key < each_key:
                        child_key, _ = current_node.value.values[index]
                        break
                    elif index + 1 == len(current_node.value.keys):
                        child_key, _ = current_node.value.values[index + 1]
                        break
                current_key = child_key

        # Get the leaf node (last in local)
        if self._local:
            leaf = self._local[-1]
            
            # Handle empty leaf
            if len(leaf.value.keys) == 0:
                leaf.value.keys = [key]
                leaf.value.values = [value]
            else:
                # Find proper place to insert
                for index, each_key in enumerate(leaf.value.keys):
                    if key < each_key:
                        leaf.value.keys = leaf.value.keys[:index] + [key] + leaf.value.keys[index:]
                        leaf.value.values = leaf.value.values[:index] + [value] + leaf.value.values[index:]
                        break
                    elif index + 1 == len(leaf.value.keys):
                        leaf.value.keys.append(key)
                        leaf.value.values.append(value)
                        break
            
            # Perform insertion (splitting if needed)
            self._perform_insertion()
        
        # Move local back to stash
        self._stash += self._local
        self._local = []
        
        # Perform dummy ORAM rounds to maintain access pattern
        # OPTIMIZATION: Remove forced dummy rounds
        # self._perform_dummy_rounds(num_rounds=self._max_height)

        self._update_peak_client_size()

    @staticmethod
    def parallel_search(omap1: 'BPlusOdsOmap', key1: Any, 
                        omap2: 'BPlusOdsOmap', key2: Any,
                        value1: Any = None, value2: Any = None) -> Tuple[Any, Any]:
        """
        Perform search on two OMAPs in parallel, reducing 2h rounds to h rounds.
        
        In each round, we batch read paths from both OMAPs, process them locally,
        and batch write back. This halves the WAN interaction rounds.
        
        The key insight is that we only update a parent's pointer to a child
        AFTER we have actually read the child and assigned it a new leaf.
        This ensures consistency between parent pointers and child locations.
        
        :param omap1: First OMAP instance (e.g., O_W)
        :param key1: Search key for first OMAP
        :param omap2: Second OMAP instance (e.g., O_R)  
        :param key2: Search key for second OMAP
        :param value1: Optional value to update in omap1
        :param value2: Optional value to update in omap2
        :return: Tuple of (search_value1, search_value2)
        """
        # Ensure both OMAPs share the same client
        if omap1._client is not omap2._client:
            raise ValueError("Both OMAPs must share the same client for parallel access.")
        
        client = omap1._client
        max_height = max(omap1._max_height, omap2._max_height)
        
        # Handle None keys (dummy operations)
        key1_is_none = key1 is None
        key2_is_none = key2 is None
        
        # Handle empty roots
        root1_is_none = omap1.root is None
        root2_is_none = omap2.root is None
        
        # Ensure _local is cleared for both OMAPs at the start
        if omap1._local:
            raise MemoryError("omap1._local was not emptied before parallel_search.")
        if omap2._local:
            raise MemoryError("omap2._local was not emptied before parallel_search.")
        
        # Initialize traversal states
        # For each OMAP, we track:
        # - node_key, node_leaf: the next node to read
        # - traversing: whether we're still traversing
        # - pending_child_index: the index in parent's values to update after reading child
        if not key1_is_none and not root1_is_none:
            omap1._local = []
            node1_key = omap1.root[0]
            node1_leaf = omap1.root[1]
            traversing1 = True
            pending_child_index1 = None  # No parent update needed for root
        else:
            traversing1 = False
            node1_key = None
            node1_leaf = None
            pending_child_index1 = None
        
        # State for omap2
        if not key2_is_none and not root2_is_none:
            omap2._local = []
            node2_key = omap2.root[0]
            node2_leaf = omap2.root[1]
            traversing2 = True
            pending_child_index2 = None  # No parent update needed for root
        else:
            traversing2 = False
            node2_key = None
            node2_leaf = None
            pending_child_index2 = None
        
        round_count = 0
        
        # Perform exactly max_height rounds
        max_height = max(omap1._max_height, omap2._max_height)
        for _ in range(max_height):
            # Prepare read queries for this round
            labels = []
            leaves = []
            omap_indices = []  # Track which omap each query belongs to
            
            # For omap1: read target path + dummy path
            if traversing1:
                # FIX for BottomUp: Check if leaf is within range of this omap
                if node1_leaf >= omap1._leaf_range:
                    traversing1 = False
                    # Create dummy round instead
                    leaf1_a = omap1._get_new_leaf()
                    leaf1_b = omap1._get_new_leaf()
                    labels.append(omap1._name)
                    leaves.append([leaf1_a, leaf1_b])
                    omap_indices.append((1, leaf1_a, leaf1_b, None))
                else:
                    dummy_leaf1 = omap1._get_new_leaf()
                    labels.append(omap1._name)
                    leaves.append([node1_leaf, dummy_leaf1])
                    omap_indices.append((1, node1_leaf, dummy_leaf1, node1_key))
            else:
                # Dummy round for omap1
                leaf1_a = omap1._get_new_leaf()
                leaf1_b = omap1._get_new_leaf()
                labels.append(omap1._name)
                leaves.append([leaf1_a, leaf1_b])
                omap_indices.append((1, leaf1_a, leaf1_b, None))
            
            # For omap2: read target path + dummy path
            if traversing2:
                # FIX for BottomUp: Check if leaf is within range of this omap
                if node2_leaf >= omap2._leaf_range:
                    traversing2 = False
                    # Create dummy round instead
                    leaf2_a = omap2._get_new_leaf()
                    leaf2_b = omap2._get_new_leaf()
                    labels.append(omap2._name)
                    leaves.append([leaf2_a, leaf2_b])
                    omap_indices.append((2, leaf2_a, leaf2_b, None))
                else: 
                    dummy_leaf2 = omap2._get_new_leaf()
                    labels.append(omap2._name)
                    leaves.append([node2_leaf, dummy_leaf2])
                    omap_indices.append((2, node2_leaf, dummy_leaf2, node2_key))
            else:
                # Dummy round for omap2
                leaf2_a = omap2._get_new_leaf()
                leaf2_b = omap2._get_new_leaf()
                labels.append(omap2._name)
                leaves.append([leaf2_a, leaf2_b])
                omap_indices.append((2, leaf2_a, leaf2_b, None))
            
            # Batch read from both OMAPs
            all_paths = client.read_mul_query(label=labels, leaf=leaves)
            
            # Process each OMAP's path
            evicted_data = []
            write_labels = []
            write_leaves = []
            
            for i, (omap_idx, real_leaf, dummy_leaf, target_key) in enumerate(omap_indices):
                omap = omap1 if omap_idx == 1 else omap2
                path = omap._decrypt_buckets(buckets=all_paths[i])
                
                if target_key is not None:
                    # Find and extract target node
                    found = False
                    for bucket in path:
                        for data in bucket:
                            if data.key == target_key:
                                omap._local.append(data)
                                found = True
                            else:
                                omap._stash.append(data)
                    
                    # Check stash if not found
                    if not found:
                        for j, data in enumerate(omap._stash):
                            if data.key == target_key:
                                omap._local.append(data)
                                del omap._stash[j]
                                found = True
                                break
                        if not found:
                            raise KeyError(f"The search key {target_key} is not found in omap{omap_idx}.")
                else:
                    # Dummy round: just add everything to stash
                    for bucket in path:
                        for data in bucket:
                            omap._stash.append(data)

                omap._update_peak_client_size()
                
                # Evict and prepare write-back
                evicted = omap._evict_stash_mul_path(leaves=[real_leaf, dummy_leaf])
                evicted_data.append(evicted)
                write_labels.append(omap._name)
                write_leaves.append([real_leaf, dummy_leaf])
                
                # Check stash overflow
                if len(omap._stash) > omap._stash_size:
                    raise MemoryError(f"Stash overflow in omap{omap_idx}!")

                omap._update_peak_client_size()
            
            # Batch write back to both OMAPs
            client.write_mul_query(label=write_labels, leaf=write_leaves, data=evicted_data)
            
            round_count += 1
            
            # Update traversal state for omap1
            if traversing1 and omap1._local:
                node1 = omap1._local[-1]
                
                # Assign new leaf to the node we just read
                node1.leaf = omap1._get_new_leaf()
                
                # Update root or parent's pointer
                if len(omap1._local) == 1:
                    # This is root, update root pointer
                    omap1.root = (node1.key, node1.leaf)
                else:
                    # Update parent's reference using the pending_child_index
                    # This ensures we only update after reading the child
                    if pending_child_index1 is not None:
                        parent1 = omap1._local[-2]
                        parent1.value.values[pending_child_index1] = (node1.key, node1.leaf)
                
                # Check if we've reached a leaf node
                if len(node1.value.keys) == len(node1.value.values):
                    traversing1 = False
                    pending_child_index1 = None
                else:
                    # Find next child to visit
                    child_index1 = None
                    for index, each_key in enumerate(node1.value.keys):
                        if key1 == each_key:
                            node1_key, node1_leaf = node1.value.values[index + 1]
                            child_index1 = index + 1
                            break
                        elif key1 < each_key:
                            node1_key, node1_leaf = node1.value.values[index]
                            child_index1 = index
                            break
                        elif index + 1 == len(node1.value.keys):
                            node1_key, node1_leaf = node1.value.values[index + 1]
                            child_index1 = index + 1
                            break
                    
                    # Save the child index for next round to update parent's pointer
                    pending_child_index1 = child_index1
            
            # Update traversal state for omap2
            if traversing2 and omap2._local:
                node2 = omap2._local[-1]
                
                # Assign new leaf to the node we just read
                node2.leaf = omap2._get_new_leaf()
                
                # Update root or parent's pointer
                if len(omap2._local) == 1:
                    # This is root, update root pointer
                    omap2.root = (node2.key, node2.leaf)
                else:
                    # Update parent's reference using the pending_child_index
                    # This ensures we only update after reading the child
                    if pending_child_index2 is not None:
                        parent2 = omap2._local[-2]
                        parent2.value.values[pending_child_index2] = (node2.key, node2.leaf)
                
                # Check if we've reached a leaf node
                if len(node2.value.keys) == len(node2.value.values):
                    traversing2 = False
                    pending_child_index2 = None
                else:
                    # Find next child to visit
                    child_index2 = None
                    for index, each_key in enumerate(node2.value.keys):
                        if key2 == each_key:
                            node2_key, node2_leaf = node2.value.values[index + 1]
                            child_index2 = index + 1
                            break
                        elif key2 < each_key:
                            node2_key, node2_leaf = node2.value.values[index]
                            child_index2 = index
                            break
                        elif index + 1 == len(node2.value.keys):
                            node2_key, node2_leaf = node2.value.values[index + 1]
                            child_index2 = index + 1
                            break
                    
                    # Save the child index for next round to update parent's pointer
                    pending_child_index2 = child_index2
        
        # Extract search results
        search_value1 = None
        search_value2 = None
        
        # Process omap1 result
        if not key1_is_none and not root1_is_none and omap1._local:
            leaf1 = omap1._local[-1]
            for index, each_key in enumerate(leaf1.value.keys):
                if key1 == each_key:
                    search_value1 = leaf1.value.values[index]
                    if value1 is not None:
                        leaf1.value.values[index] = value1
                    break
            omap1._stash += omap1._local
            omap1._local = []
        
        # Process omap2 result
        if not key2_is_none and not root2_is_none and omap2._local:
            leaf2 = omap2._local[-1]
            for index, each_key in enumerate(leaf2.value.keys):
                if key2 == each_key:
                    search_value2 = leaf2.value.values[index]
                    if value2 is not None:
                        leaf2.value.values[index] = value2
                    break
            omap2._stash += omap2._local
            omap2._local = []
        
        return search_value1, search_value2

    @staticmethod
    def parallel_search_with_delete(omap1: 'BPlusOdsOmap', search_key1: Any, delete_key1: Any,
                                     omap2: 'BPlusOdsOmap', search_key2: Any, delete_key2: Any) -> Tuple[Any, Any, Any]:
        """
        Perform search on two OMAPs in parallel, while also executing pending deletions.
        
        This method combines:
        1. Search for search_key1 in omap1 and search_key2 in omap2
        2. Delete delete_key1 from omap1 (if not None) 
        3. Delete delete_key2 from omap2 (if not None)
        
        All operations share the same h rounds of WAN interaction.
        
        :param omap1: First OMAP (O_W)
        :param search_key1: Key to search in omap1
        :param delete_key1: Key to delete from omap1 (None for dummy deletion)
        :param omap2: Second OMAP (O_R)  
        :param search_key2: Key to search in omap2
        :param delete_key2: Key to delete from omap2 (None for dummy deletion)
        :return: Tuple of (search_value1, search_value2, deleted_value_from_omap1)
        """
        # For now, implement as sequential operations
        # A full optimization would interleave all 4 operations in the same h rounds
        
        # First, do the parallel search
        search_value1, search_value2 = BPlusOdsOmap.parallel_search(
            omap1=omap1, key1=search_key1,
            omap2=omap2, key2=search_key2
        )
        
        # Then do the deletions (these add more rounds, but maintain correctness)
        deleted_value1 = None
        if delete_key1 is not None:
            deleted_value1 = omap1.delete(delete_key1)
        else:
            # Dummy deletion to maintain access pattern
            omap1.delete(None)
        
        if delete_key2 is not None:
            omap2.delete(delete_key2)
        else:
            # Dummy deletion to maintain access pattern
            omap2.delete(None)
        
        return search_value1, search_value2, deleted_value1

    @staticmethod
    def parallel_search_and_delete(omap1: 'BPlusOdsOmap', search_key1: Any, delete_key1: Any,
                                    omap2: 'BPlusOdsOmap', search_key2: Any, delete_key2: Any,
                                    insert_key2: Any = None, insert_value2: Any = None,
                                    observer: Any = None) -> Tuple[Any, Any, Any]:
        """
        Perform search and delete on two OMAPs, and optional insert on Omap2, in parallel (h rounds).
        """
        if omap1._client is not omap2._client:
            raise ValueError("Both OMAPs must share the same client for parallel access.")
            
        # Re-verify root integrity before starting
        if omap1.root is not None and omap2.root is not None:
             pass
        
        client = omap1._client
        max_height = max(omap1._max_height, omap2._max_height)
        
        class TraversalState:
            def __init__(self, omap, key, name, needs_sibling=False, is_insert=False):
                self.omap = omap
                self.key = key
                self.name = name
                self.needs_sibling = needs_sibling
                self.is_insert = is_insert
                self.local = []
                self.sibling_cache = [] if needs_sibling else None
                
                # Special handling for insert on empty tree
                if is_insert and omap.root is None:
                    self.traversing = False # Will handle insert-on-empty post-loop
                else:
                    self.traversing = (key is not None) and (omap.root is not None)
                
                if self.traversing:
                    self.node_key = omap.root[0]
                    self.node_leaf = omap.root[1]
                else:
                    self.node_key = None
                    self.node_leaf = None
                self.pending_child_index = None
                self.sibling_key = None
                self.sibling_leaf = None
                self.sibling_index = None
        
        # 5 states: search1, delete1, search2, delete2, insert2
        search1 = TraversalState(omap1, search_key1, "search1", needs_sibling=False)
        delete1 = TraversalState(omap1, delete_key1, "delete1", needs_sibling=True)
        search2 = TraversalState(omap2, search_key2, "search2", needs_sibling=False)
        delete2 = TraversalState(omap2, delete_key2, "delete2", needs_sibling=True)
        insert2 = TraversalState(omap2, insert_key2, "insert2", needs_sibling=False, is_insert=True)
        
        states = [search1, delete1, search2, delete2, insert2]

        # h rounds of parallel traversal (each round: read + write)
        for round_num in range(max_height):
            labels = []
            leaves = []
            state_infos = []  # (state, real_leaf, aux_leaf, target_key, aux_key, is_sibling_aux)
            for state in states:
                omap = state.omap
                # 强制所有round都走真实的read/write（即使traversing=False）
                if state.traversing:
                    # ...原有traversing逻辑...
                    if state.node_leaf is not None and state.node_leaf >= omap._leaf_range:
                        state.traversing = False
                        leaf_a = omap._get_new_leaf()
                        leaf_b = omap._get_new_leaf()
                        labels.append(omap._name)
                        leaves.append([leaf_a, leaf_b])
                        state_infos.append((state, leaf_a, leaf_b, None, None, False))
                    elif state.needs_sibling and len(state.local) > 0:
                        parent = state.local[-1]
                        if len(parent.value.keys) != len(parent.value.values):
                            child_idx = None
                            for idx, each_key in enumerate(parent.value.keys):
                                if state.key == each_key:
                                    child_idx = idx + 1
                                    break
                                elif state.key < each_key:
                                    child_idx = idx
                                    break
                                elif idx + 1 == len(parent.value.keys):
                                    child_idx = idx + 1
                                    break
                            if child_idx is not None and child_idx > 0:
                                sib_key, sib_leaf = parent.value.values[child_idx - 1]
                                state.sibling_key = sib_key
                                state.sibling_leaf = sib_leaf
                                state.sibling_index = child_idx - 1
                            elif child_idx is not None and child_idx + 1 < len(parent.value.values):
                                sib_key, sib_leaf = parent.value.values[child_idx + 1]
                                state.sibling_key = sib_key
                                state.sibling_leaf = sib_leaf
                                state.sibling_index = child_idx + 1
                            else:
                                state.sibling_key = None
                                state.sibling_leaf = omap._get_new_leaf()
                                state.sibling_index = None
                        else:
                            state.sibling_key = None
                            state.sibling_leaf = omap._get_new_leaf()
                            state.sibling_index = None
                        labels.append(omap._name)
                        leaves.append([state.node_leaf, state.sibling_leaf])
                        state_infos.append((state, state.node_leaf, state.sibling_leaf, state.node_key, state.sibling_key, True))
                    else:
                        dummy_leaf = omap._get_new_leaf()
                        labels.append(omap._name)
                        leaves.append([state.node_leaf, dummy_leaf])
                        state_infos.append((state, state.node_leaf, dummy_leaf, state.node_key, None, False))
                else:
                    # Dummy round也强制真实读写（用随机叶子）
                    leaf_a = omap._get_new_leaf()
                    leaf_b = omap._get_new_leaf()
                    labels.append(omap._name)
                    leaves.append([leaf_a, leaf_b])
                    # 这里target_key和aux_key都为None，is_sibling_aux=False
                    state_infos.append((state, leaf_a, leaf_b, None, None, False))

            # 每一轮都强制执行真实的read/write
            all_paths = client.read_mul_query(label=labels, leaf=leaves)
            
            # Track leaves for this round's write-back
            round_accessed_paths = {}
            
            for i, (state, real_leaf, aux_leaf, target_key, aux_key, is_sibling_aux) in enumerate(state_infos):
                omap = state.omap
                path = omap._decrypt_buckets(buckets=all_paths[i])
                if target_key is not None:
                    # ...原有target_key逻辑...
                    found_target = False
                    found_sibling = False
                    sibling_new_leaf = omap._get_new_leaf() if aux_key else None
                    if sibling_new_leaf is None and aux_key is not None:
                        sibling_new_leaf = omap._get_new_leaf()
                    for bucket in path:
                        for data in bucket:
                            if data.key == target_key:
                                already_in_local = any(d.key == target_key for d in state.local)
                                if not already_in_local:
                                    state.local.append(data)
                                    found_target = True
                            elif is_sibling_aux and aux_key is not None and data.key == aux_key:
                                if data.leaf is None:
                                    data.leaf = sibling_new_leaf
                                state.sibling_cache.append(data)
                                found_sibling = True
                            else:
                                if not any(d.key == data.key for d in omap._stash):
                                    omap._stash.append(data)
                    if not found_target:
                        for j, data in enumerate(omap._stash):
                            if data.key == target_key:
                                already_in_local = any(d.key == target_key for d in state.local)
                                if not already_in_local:
                                    state.local.append(data)
                                    del omap._stash[j]
                                    found_target = True
                                break
                        if not found_target:
                            for other_state in states:
                                if other_state.omap is omap and other_state is not state:
                                    for d in other_state.local:
                                        if d.key == target_key:
                                            state.local.append(d)
                                            found_target = True
                                            break
                                if found_target:
                                    break
                        if not found_target:
                            state.traversing = False
                            continue
                    if is_sibling_aux and aux_key is not None and not found_sibling:
                        for j, data in enumerate(omap._stash):
                            if data.key == aux_key:
                                if data.leaf is None:
                                    data.leaf = sibling_new_leaf
                                state.sibling_cache.append(data)
                                del omap._stash[j]
                                found_sibling = True
                                break
                else:
                    # Dummy round也强制真实读写（但不会动local/stash）
                    pass
                if observer is not None:
                    total_nodes = sum(len(s.local) for s in states) + sum(len(s.sibling_cache or []) for s in states)
                    observer(total_nodes)
                omap._update_peak_client_size()
                
            # Track accessed paths for this round
            for i, (state, real_leaf, aux_leaf, _, _, _) in enumerate(state_infos):
                omap = state.omap
                if omap not in round_accessed_paths:
                    round_accessed_paths[omap] = []
                round_accessed_paths[omap].extend([real_leaf, aux_leaf])
            for state in states:
                if state.traversing and state.local:
                    node = state.local[-1]
                    omap = state.omap
                    if node.leaf is None:
                        node.leaf = omap._get_new_leaf()
                    if len(state.local) == 1:
                        omap.root = (node.key, node.leaf)
                    elif state.pending_child_index is not None:
                        parent = state.local[-2]
                        parent.value.values[state.pending_child_index] = (node.key, node.leaf)
                        if state.sibling_index is not None and state.sibling_cache:
                            sib = state.sibling_cache[-1] if state.sibling_cache else None
                            if sib:
                                if sib.leaf is None:
                                    sib.leaf = omap._get_new_leaf()
                                parent.value.values[state.sibling_index] = (sib.key, sib.leaf)
                    if len(node.value.keys) == len(node.value.values):
                        state.traversing = False
                        state.pending_child_index = None
                    else:
                        key = state.key
                        child_index = None
                        for index, each_key in enumerate(node.value.keys):
                            if key == each_key:
                                child_ptr = node.value.values[index + 1]
                                # Handle both tuple and other formats
                                if isinstance(child_ptr, tuple) and len(child_ptr) == 2:
                                    state.node_key, state.node_leaf = child_ptr
                                else:
                                    # Invalid format - stop traversing
                                    state.traversing = False
                                    break
                                child_index = index + 1
                                break
                            elif key < each_key:
                                child_ptr = node.value.values[index]
                                if isinstance(child_ptr, tuple) and len(child_ptr) == 2:
                                    state.node_key, state.node_leaf = child_ptr
                                else:
                                    state.traversing = False
                                    break
                                child_index = index
                                break
                            elif index + 1 == len(node.value.keys):
                                child_ptr = node.value.values[index + 1]
                                if isinstance(child_ptr, tuple) and len(child_ptr) == 2:
                                    state.node_key, state.node_leaf = child_ptr
                                else:
                                    state.traversing = False
                                    break
                                child_index = index + 1
                                break
                        state.pending_child_index = child_index
                        state.omap._update_peak_client_size()
            
            # Write-back for this round (immediate, not deferred)
            write_labels = []
            write_leaves = []
            evicted_data = []
            
            for omap, round_leaves in round_accessed_paths.items():
                unique_leaves = list(set(round_leaves))
                evicted = omap._evict_stash_mul_path(leaves=unique_leaves)
                evicted_data.append(evicted)
                write_labels.append(omap._name)
                write_leaves.append(unique_leaves)
                omap._update_peak_client_size()
            
            # Always execute write_mul_query for oblivious security (even if empty)
            # If no accessed paths, use dummy writes for both OMAPs
            if not write_labels:
                for omap in [omap1, omap2]:
                    dummy_leaf = omap._get_new_leaf()
                    evicted = omap._evict_stash_mul_path(leaves=[dummy_leaf])
                    evicted_data.append(evicted)
                    write_labels.append(omap._name)
                    write_leaves.append([dummy_leaf])
            
            client.write_mul_query(label=write_labels, leaf=write_leaves, data=evicted_data)

        # Extract results
        search_value1 = None
        search_value2 = None
        deleted_value1 = None
        
        # Process search1
        if search_key1 is not None and omap1.root is not None and search1.local:
            leaf = search1.local[-1]
            for index, each_key in enumerate(leaf.value.keys):
                if search_key1 == each_key:
                    search_value1 = leaf.value.values[index]
                    break
            omap1._stash += search1.local
        
        # Process delete1
        if delete1.local:
            if delete_key1 is not None:
                leaf = delete1.local[-1]
                for index, each_key in enumerate(leaf.value.keys):
                    if delete_key1 == each_key:
                        deleted_value1 = leaf.value.values[index]
                        leaf.value.keys.pop(index)
                        leaf.value.values.pop(index)
                        break
                if delete1.sibling_cache:
                    omap1._local = delete1.local
                    omap1._sibling_cache = delete1.sibling_cache
                    omap1._handle_underflow_with_cached_siblings()
                    delete1.local = omap1._local
                    delete1.sibling_cache = omap1._sibling_cache
                    omap1._local = []
                    omap1._sibling_cache = []
            
            for d in delete1.local:
                if d.leaf is None: d.leaf = omap1._get_new_leaf()
                if not any(s.key == d.key for s in omap1._stash):
                    omap1._stash.append(d)
            if delete1.sibling_cache:
                for d in delete1.sibling_cache:
                    if d.leaf is None: d.leaf = omap1._get_new_leaf()
                    if not any(s.key == d.key for s in omap1._stash):
                        omap1._stash.append(d)
        
        # Process search2
        if search_key2 is not None and omap2.root is not None and search2.local:
            leaf = search2.local[-1]
            for index, each_key in enumerate(leaf.value.keys):
                if search_key2 == each_key:
                    search_value2 = leaf.value.values[index]
                    break
            omap2._stash += search2.local
        
        # Process delete2
        if delete2.local:
            if delete_key2 is not None:
                leaf = delete2.local[-1]
                for index, each_key in enumerate(leaf.value.keys):
                    if delete_key2 == each_key:
                        leaf.value.keys.pop(index)
                        leaf.value.values.pop(index)
                        break
                if delete2.sibling_cache:
                    omap2._local = delete2.local
                    omap2._sibling_cache = delete2.sibling_cache
                    omap2._handle_underflow_with_cached_siblings()
                    delete2.local = omap2._local
                    delete2.sibling_cache = omap2._sibling_cache
                    omap2._local = []
                    omap2._sibling_cache = []
            
            for d in delete2.local:
                if d.leaf is None: d.leaf = omap2._get_new_leaf()
                if not any(s.key == d.key for s in omap2._stash):
                    omap2._stash.append(d)
            if delete2.sibling_cache:
                for d in delete2.sibling_cache:
                    if d.leaf is None: d.leaf = omap2._get_new_leaf()
                    if not any(s.key == d.key for s in omap2._stash):
                        omap2._stash.append(d)
                        
        # Process insert2
        if insert_key2 is not None:
            # Case A: Tree was empty
            if omap2.root is None:
                data_block = omap2._get_bplus_data(keys=[insert_key2], values=[insert_value2])
                omap2._stash.append(data_block)
                omap2.root = (data_block.key, data_block.leaf)
            
            # Case B: Standard insert into traversed path
            elif insert2.local:
                # 1. Update/Add to Leaf
                leaf = insert2.local[-1]
                if len(leaf.value.keys) == 0:
                    leaf.value.keys = [insert_key2]
                    leaf.value.values = [insert_value2]
                else:
                    inserted = False
                    for index, each_key in enumerate(leaf.value.keys):
                        if insert_key2 == each_key: # Update
                            leaf.value.values[index] = insert_value2
                            inserted = True
                            break
                        elif insert_key2 < each_key: # Insert New
                            leaf.value.keys = leaf.value.keys[:index] + [insert_key2] + leaf.value.keys[index:]
                            leaf.value.values = leaf.value.values[:index] + [insert_value2] + leaf.value.values[index:]
                            inserted = True
                            break
                    if not inserted: # Append
                        leaf.value.keys.append(insert_key2)
                        leaf.value.values.append(insert_value2)
                
                # 2. Perform Insertion (Split Propagation)
                omap2._local = insert2.local
                omap2._perform_insertion()
                
                omap2._stash += omap2._local
                omap2._local = []
        
        return search_value1, search_value2, deleted_value1

    @staticmethod
    def parallel_search_and_delete_deprecated(omap1: 'BPlusOdsOmap', search_key1: Any, delete_key1: Any,
                                    omap2: 'BPlusOdsOmap', search_key2: Any, delete_key2: Any,
                                    observer: Any = None) -> Tuple[Any, Any, Any]:
        """
        Perform search and delete on two OMAPs in parallel, all in h rounds.
        """
        if omap1._client is not omap2._client:
            raise ValueError("Both OMAPs must share the same client for parallel access.")
            
        # Re-verify root integrity before starting
        if omap1.root is not None and omap2.root is not None:
            # Check if root keys exist in stashes or are "in" the tree
            # This is hard to check fully, but let's assume if root is not None, it should be somewhere.
            pass
        
        client = omap1._client
        max_height = max(omap1._max_height, omap2._max_height)
        
        class TraversalState:
            def __init__(self, omap, key, name, needs_sibling=False):
                self.omap = omap
                self.key = key
                self.name = name
                self.needs_sibling = needs_sibling
                self.local = []
                self.sibling_cache = [] if needs_sibling else None
                self.traversing = (key is not None) and (omap.root is not None)
                if self.traversing:
                    self.node_key = omap.root[0]
                    self.node_leaf = omap.root[1]
                else:
                    self.node_key = None
                    self.node_leaf = None
                self.pending_child_index = None
                # For sibling tracking
                self.sibling_key = None
                self.sibling_leaf = None
                self.sibling_index = None
        
        # 4 traversal states: search needs dummy, delete needs sibling
        search1 = TraversalState(omap1, search_key1, "search1", needs_sibling=False)
        delete1 = TraversalState(omap1, delete_key1, "delete1", needs_sibling=True)
        search2 = TraversalState(omap2, search_key2, "search2", needs_sibling=False)
        delete2 = TraversalState(omap2, delete_key2, "delete2", needs_sibling=True)
        
        states = [search1, delete1, search2, delete2]
        
        # h rounds of parallel traversal
        for round_num in range(max_height):
            labels = []
            leaves = []
            state_infos = []  # (state, real_leaf, aux_leaf, target_key, aux_key, is_sibling_aux)
            
            for state in states:
                omap = state.omap
                if state.traversing:
                    # FIX for BottomUp: Check if leaf is within range of this omap
                    if state.node_leaf is not None and state.node_leaf >= omap._leaf_range:
                        state.traversing = False
                        # Convert to dummy
                        leaf_a = omap._get_new_leaf()
                        leaf_b = omap._get_new_leaf()
                        labels.append(omap._name)
                        leaves.append([leaf_a, leaf_b])
                        state_infos.append((state, leaf_a, leaf_b, None, None, False))
                    elif state.needs_sibling and len(state.local) > 0:
                        # Delete traversal: get sibling
                        parent = state.local[-1]
                        if len(parent.value.keys) != len(parent.value.values):  # Internal node
                            # Find child index for current key
                            child_idx = None
                            for idx, each_key in enumerate(parent.value.keys):
                                if state.key == each_key:
                                    child_idx = idx + 1
                                    break
                                elif state.key < each_key:
                                    child_idx = idx
                                    break
                                elif idx + 1 == len(parent.value.keys):
                                    child_idx = idx + 1
                                    break
                            
                            # Find sibling
                            if child_idx is not None and child_idx > 0:
                                sib_key, sib_leaf = parent.value.values[child_idx - 1]
                                state.sibling_key = sib_key
                                state.sibling_leaf = sib_leaf
                                state.sibling_index = child_idx - 1
                            elif child_idx is not None and child_idx + 1 < len(parent.value.values):
                                sib_key, sib_leaf = parent.value.values[child_idx + 1]
                                state.sibling_key = sib_key
                                state.sibling_leaf = sib_leaf
                                state.sibling_index = child_idx + 1
                            else:
                                state.sibling_key = None
                                state.sibling_leaf = omap._get_new_leaf()
                                state.sibling_index = None
                        else:
                            state.sibling_key = None
                            state.sibling_leaf = omap._get_new_leaf()
                            state.sibling_index = None
                        
                        labels.append(omap._name)
                        leaves.append([state.node_leaf, state.sibling_leaf])
                        state_infos.append((state, state.node_leaf, state.sibling_leaf, 
                                          state.node_key, state.sibling_key, True))
                    else:
                        # Search traversal or first round of delete: target + dummy
                        dummy_leaf = omap._get_new_leaf()
                        labels.append(omap._name)
                        leaves.append([state.node_leaf, dummy_leaf])
                        state_infos.append((state, state.node_leaf, dummy_leaf, 
                                          state.node_key, None, False))
                else:
                    # Dummy round
                    leaf_a = omap._get_new_leaf()
                    leaf_b = omap._get_new_leaf()
                    labels.append(omap._name)
                    leaves.append([leaf_a, leaf_b])
                    state_infos.append((state, leaf_a, leaf_b, None, None, False))
            
            # Batch read all paths
            all_paths = client.read_mul_query(label=labels, leaf=leaves)
            
            # Process each path
            evicted_data = []
            write_labels = []
            write_leaves = []
            
            for i, (state, real_leaf, aux_leaf, target_key, aux_key, is_sibling_aux) in enumerate(state_infos):
                omap = state.omap
                path = omap._decrypt_buckets(buckets=all_paths[i])
                
                if target_key is not None:
                    # Find and extract target node
                    found_target = False
                    found_sibling = False
                    sibling_new_leaf = omap._get_new_leaf() if aux_key else None
                    if sibling_new_leaf is None and aux_key is not None:
                        sibling_new_leaf = omap._get_new_leaf()
                    
                    for bucket in path:
                        for data in bucket:
                            if data.key == target_key:
                                already_in_local = any(d.key == target_key for d in state.local)
                                if not already_in_local:
                                    state.local.append(data)
                                    found_target = True
                            elif is_sibling_aux and aux_key is not None and data.key == aux_key:
                                if data.leaf is None:
                                    data.leaf = sibling_new_leaf
                                state.sibling_cache.append(data)
                                found_sibling = True
                            else:
                                if not any(d.key == data.key for d in omap._stash):
                                    omap._stash.append(data)
                    
                    # Check stash for target if not found
                    if not found_target:
                        for j, data in enumerate(omap._stash):
                            if data.key == target_key:
                                already_in_local = any(d.key == target_key for d in state.local)
                                if not already_in_local:
                                    state.local.append(data)
                                    del omap._stash[j]
                                found_target = True
                                break
                        if not found_target:
                            # Check other states' local
                            for other_state in states:
                                if other_state.omap is omap and other_state is not state:
                                    for d in other_state.local:
                                        if d.key == target_key:
                                            state.local.append(d)
                                            found_target = True
                                            break
                                if found_target:
                                    break
                        if not found_target:
                            # raise KeyError(f"Key {target_key} not found in {state.name}.")
                            # FIX: If not found, it implies the structure is incomplete or the key is truly missing (and we reached a dead end in the tree).
                            # We should treat this as "search failed" and stop traversing.
                            state.traversing = False
                            continue  # Skip to next state
                    
                    # Check stash for sibling if not found
                    if is_sibling_aux and aux_key is not None and not found_sibling:
                        for j, data in enumerate(omap._stash):
                            if data.key == aux_key:
                                if data.leaf is None:
                                    data.leaf = sibling_new_leaf
                                state.sibling_cache.append(data)
                                del omap._stash[j]
                                found_sibling = True
                                break
                else:
                    # Dummy round
                    for bucket in path:
                        for data in bucket:
                            if not any(d.key == data.key for d in omap._stash):
                                omap._stash.append(data)

                # Call observer if provided (reporting total traversing node count)
                if observer is not None:
                    # Calculate total size of all traversal buffers
                    total_local = sum(len(s.local) for s in states)
                    total_sibling = sum(len(s.sibling_cache) for s in states if s.sibling_cache is not None)
                    observer(total_local + total_sibling)

                omap._update_peak_client_size()
            
            # Group leaves by OMAP for unified eviction
            # This prevents overwriting shared buckets (like root) when multiple operations target the same OMAP
            omap_leaves_map = {} 
            for i, (state, real_leaf, aux_leaf, _, _, _) in enumerate(state_infos):
                omap = state.omap
                if omap not in omap_leaves_map:
                    omap_leaves_map[omap] = []
                omap_leaves_map[omap].extend([real_leaf, aux_leaf])

            for omap, leaves in omap_leaves_map.items():
                evicted = omap._evict_stash_mul_path(leaves=leaves)
                evicted_data.append(evicted)
                write_labels.append(omap._name)
                write_leaves.append(leaves)
                
                if len(omap._stash) > omap._stash_size:
                    raise MemoryError("Stash overflow!")
                omap._update_peak_client_size()
            
            # Batch write all paths
            if write_labels:
                client.write_mul_query(label=write_labels, leaf=write_leaves, data=evicted_data)
            
            # Update traversal state for each operation
            for state in states:
                if state.traversing and state.local:
                    node = state.local[-1]
                    # if round_num == 0:
                    #     print(f"DEBUG: Root Node name={state.name}. KeysLen: {len(node.value.keys) if node.value.keys else 'None'}, ValsLen: {len(node.value.values) if node.value.values else 'None'}")
                    
                    omap = state.omap
                    
                    # Assign new leaf (defensive: some nodes may arrive without a leaf)
                    if node.leaf is None:
                        node.leaf = omap._get_new_leaf()
                    
                    # Update root or parent's pointer
                    if len(state.local) == 1:
                        omap.root = (node.key, node.leaf)
                    elif state.pending_child_index is not None:
                        parent = state.local[-2]
                        parent.value.values[state.pending_child_index] = (node.key, node.leaf)
                        # Also update sibling pointer if applicable
                        if state.sibling_index is not None and state.sibling_cache:
                            sib = state.sibling_cache[-1] if state.sibling_cache else None
                            if sib:
                                if sib.leaf is None:
                                    sib.leaf = omap._get_new_leaf()
                                parent.value.values[state.sibling_index] = (sib.key, sib.leaf)
                    
                    # Check if leaf
                    if len(node.value.keys) == len(node.value.values):
                        state.traversing = False
                        state.pending_child_index = None
                    else:
                        # Find next child
                        key = state.key
                        child_index = None
                        for index, each_key in enumerate(node.value.keys):
                            if key == each_key:
                                state.node_key, state.node_leaf = node.value.values[index + 1]
                                child_index = index + 1
                                break
                            elif key < each_key:
                                state.node_key, state.node_leaf = node.value.values[index]
                                child_index = index
                                break
                            elif index + 1 == len(node.value.keys):
                                state.node_key, state.node_leaf = node.value.values[index + 1]
                                child_index = index + 1
                                break
                        state.pending_child_index = child_index
                        state.omap._update_peak_client_size()
        
        # Extract results
        search_value1 = None
        search_value2 = None
        deleted_value1 = None
        
        # Process search1 result
        if search_key1 is not None and omap1.root is not None and search1.local:
            leaf = search1.local[-1]
            for index, each_key in enumerate(leaf.value.keys):
                if search_key1 == each_key:
                    search_value1 = leaf.value.values[index]
                    break
            omap1._stash += search1.local
        
        # Process delete1: remove key from leaf and handle underflow
        if delete1.local:
            if delete_key1 is not None:
                leaf = delete1.local[-1]
                for index, each_key in enumerate(leaf.value.keys):
                    if delete_key1 == each_key:
                        deleted_value1 = leaf.value.values[index]
                        leaf.value.keys.pop(index)
                        leaf.value.values.pop(index)
                        break
                
                # Handle underflow with sibling cache
                if delete1.sibling_cache:
                    omap1._local = delete1.local
                    omap1._sibling_cache = delete1.sibling_cache
                    omap1._handle_underflow_with_cached_siblings()
                    delete1.local = omap1._local
                    delete1.sibling_cache = omap1._sibling_cache
                    omap1._local = []
                    omap1._sibling_cache = []
            
            # Add to stash (ensure all nodes have valid leaves)
            for d in delete1.local:
                if d.leaf is None:
                    d.leaf = omap1._get_new_leaf()
                if not any(s.key == d.key for s in omap1._stash):
                    omap1._stash.append(d)
            if delete1.sibling_cache:
                for d in delete1.sibling_cache:
                    if d.leaf is None:
                        d.leaf = omap1._get_new_leaf()
                    if not any(s.key == d.key for s in omap1._stash):
                        omap1._stash.append(d)
        
        # Process search2 result
        if search_key2 is not None and omap2.root is not None and search2.local:
            leaf = search2.local[-1]
            for index, each_key in enumerate(leaf.value.keys):
                if search_key2 == each_key:
                    search_value2 = leaf.value.values[index]
                    break
            omap2._stash += search2.local
        
        # Process delete2: remove key from leaf and handle underflow
        if delete2.local:
            if delete_key2 is not None:
                leaf = delete2.local[-1]
                for index, each_key in enumerate(leaf.value.keys):
                    if delete_key2 == each_key:
                        leaf.value.keys.pop(index)
                        leaf.value.values.pop(index)
                        break
                
                # Handle underflow with sibling cache
                if delete2.sibling_cache:
                    omap2._local = delete2.local
                    omap2._sibling_cache = delete2.sibling_cache
                    omap2._handle_underflow_with_cached_siblings()
                    delete2.local = omap2._local
                    delete2.sibling_cache = omap2._sibling_cache
                    omap2._local = []
                    omap2._sibling_cache = []
            
            # Add to stash (ensure all nodes have valid leaves)
            for d in delete2.local:
                if d.leaf is None:
                    d.leaf = omap2._get_new_leaf()
                if not any(s.key == d.key for s in omap2._stash):
                    omap2._stash.append(d)
            if delete2.sibling_cache:
                for d in delete2.sibling_cache:
                    if d.leaf is None:
                        d.leaf = omap2._get_new_leaf()
                    if not any(s.key == d.key for s in omap2._stash):
                        omap2._stash.append(d)
        
        return search_value1, search_value2, deleted_value1

    def fast_search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value.

        The difference here is that fast search will return the node immediately without keeping it in local.
        If the input value is not None, the value corresponding to the search tree will be updated.
        :param key: The search key of interest.
        :param value: The value to update.
        :return: The (old) value corresponding to the search key.
        """
        if key is None:
            self._perform_dummy_operation(num_round=self._max_height)

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
        self._client.write_query(label=self._name, leaf=old_leaf, data=self._evict_stash(leaf=old_leaf))
        # And then the dummy evictions.
        self._perform_dummy_operation(num_round=self._max_height - num_retrieved_nodes)

        return search_value

    def _get_min_keys(self) -> int:
        """Get the minimum number of keys a node should have (except root)."""
        return (self._order + 1) // 2 - 1

    def _is_leaf_node(self, node: Data) -> bool:
        """Check if a node is a leaf node based on its structure."""
        # Leaf nodes have len(keys) == len(values) (key-value pairs)
        # Internal nodes have len(keys) != len(values) (children pointers)
        return len(node.value.keys) == len(node.value.values)

    def _get_sibling_info(self, parent_node: Data, child_index: int) -> Tuple[int, int, int]:
        """
        Get information about siblings for merge/borrow operations.
        
        :param parent_node: The parent node containing the child.
        :param child_index: The index of the child in parent's values.
        :return: (left_sibling_index, right_sibling_index, separator_key_index)
        """
        left_sibling_idx = child_index - 1 if child_index > 0 else None
        right_sibling_idx = child_index + 1 if child_index < len(parent_node.value.values) - 1 else None
        separator_key_idx = child_index - 1 if child_index > 0 else 0
        return left_sibling_idx, right_sibling_idx, separator_key_idx

    def _merge_leaf_nodes(self, left_node: Data, right_node: Data) -> None:
        """
        Merge right leaf node into left leaf node.
        
        :param left_node: The left leaf node (will contain merged result).
        :param right_node: The right leaf node (will be discarded).
        """
        left_node.value.keys = left_node.value.keys + right_node.value.keys
        left_node.value.values = left_node.value.values + right_node.value.values

    def _merge_internal_nodes(self, left_node: Data, right_node: Data, separator_key: Any) -> None:
        """
        Merge right internal node into left internal node.
        
        :param left_node: The left internal node (will contain merged result).
        :param right_node: The right internal node (will be discarded).
        :param separator_key: The key from parent that separates the two nodes.
        """
        left_node.value.keys = left_node.value.keys + [separator_key] + right_node.value.keys
        left_node.value.values = left_node.value.values + right_node.value.values

    def _borrow_from_left_leaf(self, node: Data, left_sibling: Data, parent_node: Data, separator_idx: int) -> None:
        """
        Borrow a key-value pair from left sibling leaf node.
        """
        borrowed_key = left_sibling.value.keys.pop()
        borrowed_value = left_sibling.value.values.pop()
        node.value.keys.insert(0, borrowed_key)
        node.value.values.insert(0, borrowed_value)
        # Update separator key in parent
        parent_node.value.keys[separator_idx] = node.value.keys[0]

    def _borrow_from_right_leaf(self, node: Data, right_sibling: Data, parent_node: Data, separator_idx: int) -> None:
        """
        Borrow a key-value pair from right sibling leaf node.
        """
        borrowed_key = right_sibling.value.keys.pop(0)
        borrowed_value = right_sibling.value.values.pop(0)
        node.value.keys.append(borrowed_key)
        node.value.values.append(borrowed_value)
        # Update separator key in parent
        parent_node.value.keys[separator_idx] = right_sibling.value.keys[0] if right_sibling.value.keys else borrowed_key

    def _borrow_from_left_internal(self, node: Data, left_sibling: Data, parent_node: Data, separator_idx: int) -> None:
        """
        Borrow from left sibling for internal nodes.
        """
        # Move separator key from parent to beginning of node's keys
        separator_key = parent_node.value.keys[separator_idx]
        node.value.keys.insert(0, separator_key)
        # Move last key from left sibling to parent
        parent_node.value.keys[separator_idx] = left_sibling.value.keys.pop()
        # Move last child from left sibling to beginning of node's children
        if left_sibling.value.values:
            last_child = left_sibling.value.values.pop()
            node.value.values.insert(0, last_child)

    def _borrow_from_right_internal(self, node: Data, right_sibling: Data, parent_node: Data, separator_idx: int) -> None:
        """
        Borrow from right sibling for internal nodes.
        """
        # Move separator key from parent to end of node's keys
        separator_key = parent_node.value.keys[separator_idx]
        node.value.keys.append(separator_key)
        # Move first key from right sibling to parent
        parent_node.value.keys[separator_idx] = right_sibling.value.keys.pop(0)
        # Move first child from right sibling to end of node's children
        if right_sibling.value.values:
            first_child = right_sibling.value.values.pop(0)
            node.value.values.append(first_child)

    def _remove_child_from_parent(self, parent_node: Data, child_index: int, separator_idx: int) -> None:
        """
        Remove a child reference and separator key from parent after merge.
        """
        if separator_idx < len(parent_node.value.keys):
            parent_node.value.keys.pop(separator_idx)
        if child_index < len(parent_node.value.values):
            parent_node.value.values.pop(child_index)

    def delete(self, key: Any) -> Any:
        """
        Delete a key-value pair from the tree.
        
        Uses optimized traversal with siblings: each round reads 2 paths 
        (1 target child + 1 sibling), performs immediate eviction.
        Total rounds = max_height (padded with dummy rounds if needed).
        
        After traversal, both the path nodes and sibling nodes are in local/cache,
        allowing complete B+ tree deletion with borrow/merge operations locally.
        
        :param key: The search key to delete.
        :return: The deleted value, or None if key not found.
        """
        # Handle dummy deletion requests
        if key is None:
            # Perform dummy rounds to preserve access pattern
            self._perform_dummy_rounds(num_rounds=self._max_height)
            return None

        if self.root is None:
            raise ValueError("It seems the tree is empty and can't perform deletion.")

        # Traverse to leaf with siblings (optimized version)
        try:
            rounds_used = self._find_leaf_to_local_with_siblings_optimized(key=key)
        except KeyError:
            # Key not found during traversal - perform dummy rounds and return None
            self._local = []
            self._sibling_cache = []
            self._perform_dummy_rounds(num_rounds=self._max_height)
            return None

        # Set the last node in local as leaf
        leaf = self._local[-1]
        deleted_value = None

        # Find and remove the key-value pair from the leaf
        for index, each_key in enumerate(leaf.value.keys):
            if key == each_key:
                deleted_value = leaf.value.values[index]
                leaf.value.keys.pop(index)
                leaf.value.values.pop(index)
                break

        # Handle underflow using cached siblings
        self._handle_underflow_with_cached_siblings()

        # Append local data and siblings to stash and clear
        self._stash += self._local
        self._stash += self._sibling_cache
        self._local = []
        self._sibling_cache = []

        # Perform dummy rounds to pad to max_height
        self._perform_dummy_rounds(num_rounds=self._max_height - rounds_used)

        return deleted_value

    def _find_leaf_with_siblings_to_local(self, key: Any) -> None:
        """
        Traverse to the leaf containing key AND fetch siblings at each level.
        
        This method uses batch read/write (read_mul_query/write_mul_query) to 
        piggyback the fetching of target child and its siblings in a single 
        network round-trip per level, reducing WAN interaction overhead.
        
        The path nodes are stored in _local, siblings are stored in _sibling_cache.
        """
        if self._local:
            raise MemoryError("The local storage was not emptied before this operation.")
        
        # Initialize sibling cache
        self._sibling_cache = []
        
        # Get the root node from ORAM storage (single read for root)
        self._move_node_to_local(key=self.root[0], leaf=self.root[1])
        
        # Get the node from local
        node = self._local[0]
        
        # Update node leaf and root
        node.leaf = self._get_new_leaf()
        self.root = (node.key, node.leaf)
        
        # While we do not reach a leaf
        while not self._is_leaf_node(node):
            # Find which child to follow
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
            
            if child_idx >= len(node.value.values):
                break
            
            # Collect all nodes to read in this level: target child + siblings
            nodes_to_read = []  # List of (node_key, oram_leaf, is_target)
            
            # Target child
            child_key, child_leaf = node.value.values[child_idx]
            nodes_to_read.append((child_key, child_leaf, True))
            
            # Left sibling (if exists)
            if child_idx > 0:
                left_sib_key, left_sib_leaf = node.value.values[child_idx - 1]
                nodes_to_read.append((left_sib_key, left_sib_leaf, False))
            
            # Right sibling (if exists)
            if child_idx + 1 < len(node.value.values):
                right_sib_key, right_sib_leaf = node.value.values[child_idx + 1]
                nodes_to_read.append((right_sib_key, right_sib_leaf, False))
            
            # Batch read all paths in one network round-trip
            self._batch_read_nodes_to_local_and_cache(nodes_to_read, node, child_idx)
            
            # Update the node to the newly fetched child
            node = self._local[-1]

    def _batch_read_nodes_to_local_and_cache(
        self, 
        nodes_to_read: list, 
        parent_node: Any, 
        child_idx: int
    ) -> None:
        """
        Batch read multiple ORAM paths and process them.
        
        :param nodes_to_read: List of (node_key, oram_leaf, is_target) tuples.
        :param parent_node: The parent node whose child references need updating.
        :param child_idx: The index of the target child in parent's values.
        """
        if not nodes_to_read:
            return
        
        # Prepare labels and leaves for batch read
        labels = [self._name] * len(nodes_to_read)
        leaves = [item[1] for item in nodes_to_read]
        
        # Batch read all paths in one network round-trip
        all_paths = self._client.read_mul_query(label=labels, leaf=leaves)
        
        # Decrypt all paths
        decrypted_paths = [self._decrypt_buckets(buckets=path) for path in all_paths]
        
        # Process each path
        new_leaves = []  # Store new leaves for write-back
        evicted_data = []  # Store evicted buckets for write-back
        
        for i, (node_key, old_leaf, is_target) in enumerate(nodes_to_read):
            path = decrypted_paths[i]
            new_leaf = self._get_new_leaf()
            new_leaves.append(new_leaf)
            
            # Find the node in the path
            found_node = None
            for bucket in path:
                for data in bucket:
                    if data.key == node_key:
                        found_node = data
                    else:
                        self._stash.append(data)
            
            # If not found in path, check stash
            if found_node is None:
                for j, data in enumerate(self._stash):
                    if data.key == node_key:
                        found_node = data
                        del self._stash[j]
                        break
            
            if found_node is None:
                raise KeyError(f"Node with key {node_key} not found.")
            
            # Update the node's leaf
            found_node.leaf = new_leaf
            
            if is_target:
                # Add to _local for path traversal
                self._local.append(found_node)
            else:
                # Add to sibling cache
                self._sibling_cache.append(found_node)
            
            # Check stash overflow
            if len(self._stash) > self._stash_size:
                raise MemoryError("Stash overflow!")
            
            # Prepare eviction for this path
            evicted_data.append(self._evict_stash(leaf=old_leaf))
        
        # Batch write all paths in one network round-trip
        self._client.write_mul_query(label=labels, leaf=leaves, data=evicted_data)
        
        # Update parent's references to children with new leaves
        for i, (node_key, old_leaf, is_target) in enumerate(nodes_to_read):
            new_leaf = new_leaves[i]
            
            if is_target:
                # Update target child reference
                parent_node.value.values[child_idx] = (node_key, new_leaf)
            else:
                # Find which sibling this is and update reference
                if child_idx > 0 and parent_node.value.values[child_idx - 1][0] == node_key:
                    parent_node.value.values[child_idx - 1] = (node_key, new_leaf)
                elif child_idx + 1 < len(parent_node.value.values) and parent_node.value.values[child_idx + 1][0] == node_key:
                    parent_node.value.values[child_idx + 1] = (node_key, new_leaf)

    def _fetch_sibling_to_cache(self, sibling_key: int, sibling_leaf: int) -> None:
        """
        Fetch a sibling node and store in _sibling_cache.
        
        Note: This method is kept for backward compatibility but the optimized
        _find_leaf_with_siblings_to_local now uses batch reads instead.
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
        """
        min_keys = self._get_min_keys()
        
        # Build a map of sibling nodes by key
        sibling_by_key = {data.key: data for data in self._sibling_cache}
        
        # Start from the leaf and work up
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
            for j, child_tuple in enumerate(parent.value.values):
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
                left_sib_tuple = parent.value.values[left_sib_idx]
                if isinstance(left_sib_tuple, tuple):
                    left_sib_key = left_sib_tuple[0]
                    left_sibling = sibling_by_key.get(left_sib_key)
                    
                    if left_sibling:
                        if len(left_sibling.value.keys) > min_keys:
                            if is_leaf:
                                self._borrow_from_left_leaf(node, left_sibling, parent, sep_key_idx)
                            else:
                                self._borrow_from_left_internal(node, left_sibling, parent, sep_key_idx)
                            # Update parent's pointer to left sibling with its new leaf
                            if left_sibling.leaf is not None:
                                parent.value.values[left_sib_idx] = (left_sibling.key, left_sibling.leaf)
                        else:
                            if is_leaf:
                                self._merge_leaf_nodes(left_sibling, node)
                            else:
                                separator_key = parent.value.keys[sep_key_idx] if sep_key_idx < len(parent.value.keys) else None
                                self._merge_internal_nodes(left_sibling, node, separator_key)
                            self._remove_child_from_parent(parent, node_idx_in_parent, sep_key_idx)
                            # After removing child, left_sib_idx might have shifted if node was after it
                            # Update parent's pointer to left sibling with its new leaf
                            if left_sibling.leaf is not None:
                                # Find left_sibling's new position in parent
                                for new_idx, child_tuple in enumerate(parent.value.values):
                                    if isinstance(child_tuple, tuple) and child_tuple[0] == left_sibling.key:
                                        parent.value.values[new_idx] = (left_sibling.key, left_sibling.leaf)
                                        break
                            node.value.keys = []
                            node.value.values = []
                            merged = True
            
            # Try right sibling (from cache)
            if right_sib_idx is not None and not merged:
                right_sib_tuple = parent.value.values[right_sib_idx]
                if isinstance(right_sib_tuple, tuple):
                    right_sib_key = right_sib_tuple[0]
                    right_sibling = sibling_by_key.get(right_sib_key)
                    
                    if right_sibling:
                        right_sep_key_idx = node_idx_in_parent
                        
                        if len(right_sibling.value.keys) > min_keys:
                            if is_leaf:
                                self._borrow_from_right_leaf(node, right_sibling, parent, right_sep_key_idx)
                            else:
                                self._borrow_from_right_internal(node, right_sibling, parent, right_sep_key_idx)
                            # Update parent's pointer to right sibling with its new leaf
                            if right_sibling.leaf is not None:
                                parent.value.values[right_sib_idx] = (right_sibling.key, right_sibling.leaf)
                            # Also update parent's pointer to node with its new leaf
                            if node.leaf is not None:
                                parent.value.values[node_idx_in_parent] = (node.key, node.leaf)
                        else:
                            if is_leaf:
                                self._merge_leaf_nodes(node, right_sibling)
                            else:
                                separator_key = parent.value.keys[right_sep_key_idx] if right_sep_key_idx < len(parent.value.keys) else None
                                self._merge_internal_nodes(node, right_sibling, separator_key)
                            self._remove_child_from_parent(parent, right_sib_idx, right_sep_key_idx)
                            # Update parent's pointer to node with its new leaf
                            if node.leaf is not None:
                                # Find node's new position in parent after removal
                                for new_idx, child_tuple in enumerate(parent.value.values):
                                    if isinstance(child_tuple, tuple) and child_tuple[0] == node.key:
                                        parent.value.values[new_idx] = (node.key, node.leaf)
                                        break
                            right_sibling.value.keys = []
                            right_sibling.value.values = []
                            merged = True
            
            i -= 1

        # Ensure all nodes and child pointers have valid leaves
        for node in self._local:
            if node.leaf is None:
                node.leaf = self._get_new_leaf()
            for idx, child in enumerate(node.value.values):
                if isinstance(child, tuple):
                    child_key, child_leaf = child
                    if child_leaf is None:
                        child_leaf = self._get_new_leaf()
                        node.value.values[idx] = (child_key, child_leaf)

        # Ensure sibling cache nodes also carry valid leaves
        for sib in self._sibling_cache:
            if sib.leaf is None:
                sib.leaf = self._get_new_leaf()
        
        # Update root if it becomes empty
        if len(self._local) > 0:
            root_node = self._local[0]
            if len(root_node.value.keys) == 0 and not self._is_leaf_node(root_node):
                if len(root_node.value.values) > 0:
                    child_tuple = root_node.value.values[0]
                    if isinstance(child_tuple, tuple):
                        self.root = (child_tuple[0], child_tuple[1])
