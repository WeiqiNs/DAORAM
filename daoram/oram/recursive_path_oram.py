"""
This module defines the recursive path oram class.

Recursive path oram has three public methods:
    - init_storage_on_pos_map: this should be called first after the class object is created. This method constructs the
        storage the server should hold for the client.
    - compress_pos_map: this should be called after the storage is initialized as this will destroy the initial position
        map and compress it to a list of oram. The compression rate by default is set to 1/4.
    - operate_on_key: after the server gets the created storage, the client can use this function to obliviously access
        data points stored in the storage.
"""

import math
import pickle
import secrets
from functools import cached_property
from typing import Any, List, Optional, Tuple

from daoram.dependency import BinaryTree, Buckets, Data, InteractServer, ServerStorage
from daoram.oram.tree_base_oram import TreeBaseOram


class RecursivePathOram(TreeBaseOram):
    def __init__(self,
                 num_data: int,
                 data_size: int,
                 name: str = "rc",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 on_chip_mem: int = 10,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 compression_ratio: int = 4,
                 use_encryption: bool = True,
                 client: Optional[InteractServer] = None):
        """
        Initialize the recursive path oram with the following parameters.

        :param num_data: The number of data points the oram should store.
        :param data_size: The number of bytes the random dummy data should have.
        :param name: The name of the protocol, this should be unique if multiple schemes are used together.
        :param filename: The filename to save the oram data to.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param on_chip_mem: The number of data points the client can store.
        :param aes_key: The key to use for the AES instance.
        :param num_key_bytes: The number of bytes the aes key should have.
        :param compression_ratio: The amount of leaves each position maps data point stores.
        :param use_encryption: A boolean indicating whether to use encryption.
        :param client: The instance we use to interact with server; maybe None for pos map oram.
        """
        # Initialize the parent BaseOram class.
        super().__init__(
            name=name,
            client=client,
            aes_key=aes_key,
            filename=filename,
            num_data=num_data,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            num_key_bytes=num_key_bytes,
            use_encryption=use_encryption
        )

        # Add children class attributes.
        self._on_chip_mem = on_chip_mem
        self._compression_ratio = compression_ratio

        # Position maps will be a list of RecursivePathOram.
        self._pos_maps: List[RecursivePathOram] = []

        # This attribute is used to store a leaf temporarily for reading a path without evicting it immediately.
        self._tmp_leaf: Optional[int] = None

        # Initialize the position map upon creation.
        self._init_pos_map()

    @cached_property
    def _num_oram_pos_map(self) -> int:
        """Get the number of oram pos maps needed."""
        return int(math.ceil(math.log(self._num_data / self._on_chip_mem, self._compression_ratio)))

    @cached_property
    def _pos_map_oram_dummy_size(self) -> int:
        """Get the byte size of the random dummy data to store in position maps."""
        return len(pickle.dumps([self._num_data - 1 - i for i in range(self._compression_ratio)]))

    def _get_pos_map_keys(self, key: int) -> List[Tuple[int, int]]:
        """
        Given a key, find what key we should use for each position map oram.

        We also return its index in the values.
        :param key: Key to some data of interest.
        :return: A list of (key, index) pairs.
        """
        # Create an empty list to hold the result.
        pos_map_keys = []

        # For each position map, compute which block the key should be in, and its index in value list.
        for i in range(self._num_oram_pos_map):
            index = key % self._compression_ratio
            key = key // self._compression_ratio
            pos_map_keys.append((key, index))

        # Reverse the list so we can go backwards.
        pos_map_keys.reverse()

        # Return the result.
        return pos_map_keys

    def _compress_pos_map(self) -> ServerStorage:
        """Compress the large position map to a list of position map oram. """
        # Storage binary trees to send to the server.
        server_storage = {}

        # Set the position map size.
        last_pos_map = self._pos_map
        pos_map_size = self._num_data

        # Get the series of pos maps.
        for i in range(self._num_oram_pos_map):
            # Store the amount of data last oram has.
            last_pos_map_size = pos_map_size
            # Compute how many blocks this position map needs to store.
            pos_map_size = math.ceil(pos_map_size / self._compression_ratio)

            # Each position map now is an oram.
            cur_pos_map_oram = RecursivePathOram(
                aes_key=self._cipher.key if self._cipher else None,
                num_data=pos_map_size,
                data_size=self._pos_map_oram_dummy_size,
                bucket_size=self._bucket_size,
                stash_scale=self._stash_scale,
                num_key_bytes=self._num_key_bytes,
                use_encryption=self._use_encryption
            )

            # For the current position map, get its corresponding binary tree.
            tree = BinaryTree(
                filename=f"{self._filename}_pos_map_{self._num_oram_pos_map - i - 1}.bin" if self._filename else None,
                num_data=pos_map_size,
                data_size=cur_pos_map_oram._max_block_size,
                bucket_size=self._bucket_size,
                enc_key_size=self._num_key_bytes if self._use_encryption else None,
            )

            # Since key is set to range from 0 to num_data - 1.
            for key, leaf in cur_pos_map_oram._pos_map.items():
                # Note that we simply store some random values, when pos_map_size % compression_ratio != 0.
                value = [
                    last_pos_map[i] if i < last_pos_map_size else secrets.randbelow(pos_map_size)
                    for i in range(key * self._compression_ratio, (key + 1) * self._compression_ratio)
                ]
                # Fill the value to the correct place of the tree.
                tree.fill_data_to_storage_leaf(data=Data(key=key, leaf=leaf, value=value))

            # Encryption and fill with dummy data if needed.
            if self._use_encryption:
                tree.storage.encrypt(aes=self._cipher)

            # We update the last position map to the current one and delete it by setting it to an empty dict.
            last_pos_map = cur_pos_map_oram._pos_map
            cur_pos_map_oram._pos_map = {}

            # Save the binary tree and the position map oram.
            server_storage[f"{self._name}_pos_map_{self._num_oram_pos_map - i - 1}"] = tree
            self._pos_maps.append(cur_pos_map_oram)

        # Finally, only store the on chip position maps.
        self._pos_map = last_pos_map

        # Reverse the pos map oram.
        self._pos_maps.reverse()

        # Return the storage.
        return server_storage

    def init_server_storage(self, data_map: dict = None) -> None:
        """
        Initialize the server storage based on the data map for this oram.

        :param data_map: A dictionary storing {key: data}.
        """
        # Initialize the storage.
        storage = self._init_storage_on_pos_map(data_map=data_map)

        # Compress the position map.
        pos_map_storage_dict = self._compress_pos_map()

        # Add the oram storage to the dictionary.
        pos_map_storage_dict[self._name] = storage

        # Let the server hold these storages.
        self.client.init(storage=pos_map_storage_dict)

    def _retrieve_pos_map_stash(self, key: int, value: int, offset: int, new_leaf: int, to_index: int) -> int:
        """
        Given a key and an operation, retrieve the block from the stash and apply the operation to it.

        :param key: The key of the data block of interest.
        :param value: If the operation is "write", this is the new value for data block.
        :param offset: The offset of where the value should be written to.
        :param to_index: Up to which index we should be checking.
        :param new_leaf: If a new leaf value is provided, store the accessed data to that leaf.
        :return: The leaf of the data block we found, and a value if the operation is "read".
        """
        # Set a value for whether the key is found.
        found = False
        # Temp holder for the value to read.
        read_value = None

        # Read all buckets in the path and add real data to stash.
        for data in self._stash[:to_index]:
            # If we find the data of interest, perform operation, otherwise skip over.
            if data.key == key:
                read_value = data.value[offset]
                data.value[offset] = value
                data.leaf = new_leaf
                # Set found to true.
                found = True
                # Break for loop.
                continue

        # If the key was never found, raise an error, since the stash is always searched after a path.
        if not found:
            raise KeyError(f"Key {key} not found.")

        return read_value

    def _retrieve_pos_map_block(self, key: int, offset: int, new_leaf: int, value: int, path: Buckets) -> Any:
        """
        Given a key and an operation, retrieve the block and apply the operation to it.

        :param key: A key to a data block.
        :param offset: The offset of where the value should be written to.
        :param new_leaf: The leaf of where the block should be written to.
        :param value: By default, write this value to key block at offset.
        :param path: A list of buckets of data.
        :return: The leaf of the data block we found.
        """
        # Temp holder for the value to read.
        read_value = None
        # Store the current stash length.
        to_index = len(self._stash)

        # Decrypt data blocks if we use encryption.
        path = self._decrypt_buckets(buckets=path)

        # Read all buckets in the path and add real data to stash.
        for bucket in path:
            for data in bucket:
                # If dummy data, we skip it.
                if data.key is None:
                    continue
                # If it's the data of interest, we both read and write it, and give it a new path.
                elif data.key == key:
                    read_value = data.value[offset]
                    data.value[offset] = value
                    data.leaf = new_leaf

                # And all real data to the stash.
                self._stash.append(data)

        # Check if the stash overflows.
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")

        # If unable to read a value, something should be wrong.
        if read_value is None:
            read_value = self._retrieve_pos_map_stash(
                key=key, value=value, offset=offset, new_leaf=new_leaf, to_index=to_index
            )

        return read_value

    def _retrieve_data_stash(self, op: str, key: int, to_index: int, new_leaf: int, value: Any = None) -> int:
        """
        Given a key and an operation, retrieve the block from the stash and apply the operation to it.

        :param op: An operation, which can be "r", "w" or "rw".
        :param key: The key of the data block of interest.
        :param to_index: Up to which index we should be checking.
        :param value: If the operation is "write", this is the new value for data block.
        :param new_leaf: If a new leaf value is provided, store the accessed data to that leaf.
        :return: The leaf of the data block we found, and a value if the operation is "read".
        """
        # Set a value for whether the key is found.
        found = False
        # Temp holder for the value to read.
        read_value = None

        # Read all buckets in the path and add real data to stash.
        for data in self._stash[:to_index]:
            # If we find the data of interest, perform operation, otherwise skip over.
            if data.key == key:
                if op == "r":
                    read_value = data.value
                elif op == "w":
                    data.value = value
                elif op == "rw":
                    read_value = data.value
                    data.value = value
                else:
                    raise ValueError("The provided operation is not valid.")
                # Get a new path and update the position map.
                data.leaf = new_leaf
                # Set found to true.
                found = True
                # Break the for loop.
                continue

        # If the key was never found, raise an error, since the stash is always searched after a path.
        if not found:
            raise KeyError(f"Key {key} not found.")

        return read_value

    def _retrieve_data_block(self, op: str, key: int, new_leaf: int, path: Buckets, value: Any = None) -> Any:
        """
        Given a key and an operation, retrieve the block and apply the operation to it.

        :param op: An operation, which can be "r", "w" or "rw".
        :param key: The key of the data block of interest.
        :param path: A list of buckets of data.
        :param value: If the operation is "write", this is the new value for data block.
        :param new_leaf: If a new leaf value is provided, store the accessed data to that leaf.
        :return: The leaf of the data block we found, and a value if the operation is "read".
        """
        # Set a value for whether the key is found.
        found = False
        # Temp holder for the value to read.
        read_value = None
        # Store the current stash length.
        to_index = len(self._stash)

        # Decrypt the path if needed.
        path = self._decrypt_buckets(buckets=path)

        # Read all buckets in the path and add real data to stash.
        for bucket in path:
            for data in bucket:
                # If dummy data, we skip it.
                if data.key is None:
                    continue
                # If it's the data of interest, we read/write it, and give it a new path.
                elif data.key == key:
                    if op == "r":
                        read_value = data.value
                    elif op == "w":
                        data.value = value
                    elif op == "rw":
                        read_value = data.value
                        data.value = value
                    else:
                        raise ValueError("The provided operation is not valid.")
                    # Get a new path and update the position map.
                    data.leaf = new_leaf
                    # Set found to True.
                    found = True

                # And all real data to the stash.
                self._stash.append(data)

        # Check if the stash overflows.
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")

        # If the value is not found, it might be in the stash.
        if not found:
            read_value = self._retrieve_data_stash(op=op, key=key, to_index=to_index, value=value, new_leaf=new_leaf)

        return read_value

    def _evict_stash(self, leaf: int) -> Buckets:
        """
        Evict data blocks in the stash while maintaining correctness.

        :param leaf: The leaf label of the path we are evicting data to.
        :return: The leaf label and the path we should write there.
        """
        # Create a temporary stash.
        temp_stash = []
        # Create a placeholder for the new path.
        path = [[] for _ in range(self._level)]

        # Now we evict the stash by going through all real data in it.
        for data in self._stash:
            # Attempt to insert actual data to the path.
            inserted = BinaryTree.fill_data_to_path(
                data=data, path=path, leaf=leaf, level=self._level, bucket_size=self._bucket_size
            )

            # If we were not able to insert data, overflow happened, put the block to the temp stash.
            if not inserted:
                temp_stash.append(data)

        # Update the stash.
        self._stash = temp_stash

        # Note that we return the path in the reversed order because we write a path from bottom up.
        return self._encrypt_buckets(buckets=path[::-1])

    def _get_leaf_from_pos_map(self, key: int) -> Tuple[int, int]:
        """
        Provide a key to some data, iterate through all position map oram to find where it is stored.

        :param key: The key of the data block of interest.
        :return: Which path the data block is on and the new path it should be stored to.
        """
        # Set up some variables.
        cur_leaf, new_cur_leaf = None, None

        # Iterate through all position maps to retrieve the key for the data of interest.
        for pos_map_index, (cur_key, cur_index) in enumerate(self._get_pos_map_keys(key=key)):
            # Get the first leaf from the on chip mem, otherwise it should have been set.
            if pos_map_index == 0:
                cur_leaf = self._pos_map[cur_key]
                # In this case, we also sample the new current leaf.
                new_cur_leaf = self._pos_maps[pos_map_index]._get_new_leaf()
                # Update the on-chip position map.
                self._pos_map[cur_key] = new_cur_leaf

            # Sample a new leaf to replace the stored leaf; if we hit the last one, we sample from the data oram.
            new_next_leaf = self._pos_maps[pos_map_index + 1]._get_new_leaf() \
                if pos_map_index < self._num_oram_pos_map - 1 else self._get_new_leaf()

            # Base on the current leaf, get the desired path.
            path_data = self.client.read_query(label=f"{self._name}_pos_map_{pos_map_index}", leaf=cur_leaf)

            # Get the next leaf.
            next_leaf = self._pos_maps[pos_map_index]._retrieve_pos_map_block(
                key=cur_key,
                offset=cur_index,
                path=path_data,
                new_leaf=new_cur_leaf,
                value=new_next_leaf
            )
            # Evict stash to current leaf.
            path_data = self._pos_maps[pos_map_index]._evict_stash(leaf=cur_leaf)

            self.client.write_query(label=f"{self._name}_pos_map_{pos_map_index}", leaf=cur_leaf, data=path_data)

            # Finally, set the current leaf to the next leaf.
            cur_leaf, new_cur_leaf = next_leaf, new_next_leaf

        return cur_leaf, new_cur_leaf

    def operate_on_key(self, op: str, key: int, value: Any = None) -> Any:
        """
        Perform operation on a given key.

        :param op: An operation, which can be "r", "w" or "rw".
        :param key: The key of the data block of interest.
        :param value: If the operation is "w", this is the new value for data block.
        :return: The leaf of the data block we found, and a value if the operation is "r" or "rw".
        """
        # Find which path the data of interest lies on.
        leaf, new_leaf = self._get_leaf_from_pos_map(key=key)

        # We read the path from the server.
        path = self.client.read_query(label=self._name, leaf=leaf)

        # Retrieve value from the path, or write to it.
        value = self._retrieve_data_block(op=op, key=key, path=path, value=value, new_leaf=new_leaf)

        # Perform an eviction and get a new path.
        path = self._evict_stash(leaf=leaf)

        # Write the path back to the server.
        self.client.write_query(label=self._name, leaf=leaf, data=path)

        return value

    def operate_on_key_without_eviction(self, op: str, key: int, value: Any = None) -> Any:
        """
        Perform operation on a given key without writing the data added to the stash back to the server.

        :param op: An operation, which can be "r", "w" or "rw".
        :param key: The key of the data block of interest.
        :param value: If the operation is "w", this is the new value for data block.
        :return: The leaf of the data block we found, and a value if the operation is "r" or "rw".
        """
        # Find which path the data of interest lies on.
        leaf, new_leaf = self._get_leaf_from_pos_map(key=key)

        # We read the path from the server.
        path = self.client.read_query(label=self._name, leaf=leaf)

        # Retrieve value from the path, or write to it.
        value = self._retrieve_data_block(op=op, key=key, path=path, value=value, new_leaf=new_leaf)

        # Temporarily save the leaf for future eviction.
        self._tmp_leaf = leaf

        return value

    def eviction_with_update_stash(self, key: int, value: Any) -> None:
        """Update a data block stored in the stash and then perform eviction.

        :param key: The key of the data block of interest.
        :param value: The value to update the data block of interest.
        """
        # Set found the key to False.
        found = False

        # Read all buckets stored in the stash and find the desired data block of interest.
        for data in self._stash:
            # If we find the data of interest, update value and set found to True.
            if data.key == key:
                data.value = value
                found = True

        # If the data was never found, we raise an error.
        if not found:
            raise KeyError(f"Key {key} not found.")

        # Perform an eviction and get a new path.
        path = self._evict_stash(leaf=self._tmp_leaf)

        # Write the path back to the server.
        self.client.write_query(label=self._name, leaf=self._tmp_leaf, data=path)

        # Set temporary leaf to None.
        self._tmp_leaf = None
