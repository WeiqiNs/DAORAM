"""
This module defines the recursive path oram class.

Recursive path oram has three public methods:
    - init_storage_on_pos_map: this should be called first after the class object is created. This method constructs the
        storage the server should hold for the client.
    - compress_pos_map: this should be called after the storage is initialized as this will destroy the initial position
        map and compress it to a list of orams. The compression rate by default is set to 1/4.
    - operate_on_key: after the server get the created storage, the client can use this function to obliviously access
        data points stored in the storage.
"""

import math
import pickle
import secrets
from typing import Any, List, Optional, Tuple

from daoram.dependency.binary_tree import BinaryTree, Buckets, KEY, LEAF, VALUE
from daoram.dependency.interact_server import InteractServer
from daoram.orams.tree_base_oram import TreeBaseOram


class RecursivePathOram(TreeBaseOram):
    def __init__(self,
                 num_data: int,
                 data_size: int,
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

        :param num_data: the number of data points the oram should store.
        :param data_size: the number of bytes the random dummy data should have.
        :param bucket_size: the number of data each bucket should have.
        :param stash_scale: the scaling scale of the stash.
        :param on_chip_mem: the number of data points the client can store.
        :param aes_key: the key to use for the AES instance.
        :param num_key_bytes: the number of bytes the aes key should have.
        :param compression_ratio: the amount of leaves each position map data point stores.
        :param use_encryption: a boolean indicating whether to use encryption.
        :param client: the instance we use to interact with server; maybe None for pos map orams.
        """
        # Initialize the parent BaseOram class.
        super().__init__(
            client=client,
            aes_key=aes_key,
            num_data=num_data,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            num_key_bytes=num_key_bytes,
            use_encryption=use_encryption
        )

        # Add children class attributes.
        self.__on_chip_mem = on_chip_mem
        self.__compression_ratio = compression_ratio

        # Position maps will be a list of RecursivePathOram.
        self.__pos_maps: List[RecursivePathOram] = []

        # This attribute is used to store a leaf temporarily for reading a path without evicting it immediately.
        self.__tmp_leaf: Optional[int] = None

        # Initialize the position map upon creation.
        self._init_pos_map()

    @property
    def __num_oram_pos_map(self) -> int:
        """Get the number of oram pos maps needed."""
        return int(math.ceil(math.log(self._num_data / self.__on_chip_mem, self.__compression_ratio)))

    @property
    def __pos_map_oram_dummy_size(self) -> int:
        """Get the byte size of the random dummy data to store in position maps."""
        # This maybe a little hacky but this seems to incur a constant oversize of 15 bytes.
        return len(pickle.dumps([self._num_data - 1 - i for i in range(self.__compression_ratio)])) - 15

    def __get_pos_map_keys(self, key: int) -> List[Tuple[int, int]]:
        """
        Given a key, find what key we should use for each position map oram.

        We also return its index in the values.
        :param key: key to some data of interest.
        :return: a list of key, index pairs.
        """
        # Create an empty list to hold result.
        pos_map_keys = []

        # For each position map, compute which block the key should be in, and it's index in value list.
        for i in range(self.__num_oram_pos_map):
            index = key % self.__compression_ratio
            key = key // self.__compression_ratio
            pos_map_keys.insert(0, (key, index))

        # Return the result.
        return pos_map_keys

    def __compress_pos_map(self) -> List[BinaryTree]:
        """Compress the large position map to a list of position map orams. """
        # Storage binary trees to send to the server.
        server_storage = []

        # Set the position map size.
        last_pos_map = self._pos_map
        pos_map_size = self._num_data

        # Get the series of pos maps.
        for i in range(self.__num_oram_pos_map):
            # Store the amount of data last oram has.
            last_pos_map_size = pos_map_size
            # Compute how many blocks this position map needs to store.
            pos_map_size = int(math.ceil(pos_map_size / self.__compression_ratio))

            # Each position map now is an oram.
            cur_pos_map_oram = RecursivePathOram(
                aes_key=self._cipher.key if self._cipher else None,
                num_data=pos_map_size,
                data_size=self.__pos_map_oram_dummy_size,
                bucket_size=self._bucket_size,
                stash_scale=self._stash_scale,
                num_key_bytes=self._num_key_bytes,
                use_encryption=self._use_encryption,
            )

            # For current position map, get its corresponding binary tree.
            tree = BinaryTree(num_data=pos_map_size, bucket_size=self._bucket_size)

            # Since key is set to range from 0 to num_data - 1.
            for key, leaf in cur_pos_map_oram._pos_map.items():
                # Note that we simply store some random values, when pos_map_size % compression_ratio != 0.
                value = [
                    last_pos_map[i] if i < last_pos_map_size else secrets.randbelow(pos_map_size)
                    for i in range(key * self.__compression_ratio, (key + 1) * self.__compression_ratio)
                ]
                # Fill the value to the correct place of the tree.
                tree.fill_data_to_storage_leaf([key, leaf, value])

            # Before storing, fill tree with dummy data.
            tree.fill_storage_with_dummy_data()

            # Perform encryption if needed.
            tree.storage = cur_pos_map_oram._encrypt_buckets(buckets=tree.storage)

            # We update the last position map to the current one and delete it by setting it to an empty dict.
            last_pos_map = cur_pos_map_oram._pos_map
            cur_pos_map_oram._pos_map = {}

            # Save the binary tree and the position map oram.
            server_storage.insert(0, tree)
            self.__pos_maps.insert(0, cur_pos_map_oram)

        # Finally only store the on chip position maps.
        self._pos_map = last_pos_map

        # Return the storage.
        return server_storage

    def init_server_storage(self, data_map: dict = None) -> None:
        """
        Initialize the server storage based on the data map for this oram.

        :param data_map: a dictionary storing {key: data}.
        """
        # Initialize the storage.
        storage = self._init_storage_on_pos_map(data_map=data_map)

        # Compress the position map.
        pos_map_storage = self.__compress_pos_map()

        # Let server hold these storages.
        self.client.init_query(label="oram", storage=storage)
        self.client.init_query(label="pos_map", storage=pos_map_storage)

    def __retrieve_pos_map_stash(self, key: int, value: int, offset: int, new_leaf: int, to_index: int) -> int:
        """
        Given a key and an operation, retrieve the block from stash and apply the operation to it.

        :param key: the key of the data block of interest.
        :param value: If the operation is "write", this is the new value for data block.
        :param offset: the offset of where the value should be written to.
        :param to_index: up to which index we should be checking.
        :param new_leaf: If new leaf value is provided, store the accessed data to that leaf.
        :return: The leaf of the data block we found, and a value if the operation is "read".
        """
        # Set a value for whether the key is found.
        found = False
        # Temp holder for the value to read.
        read_value = None

        # Read all buckets in the path and add real data to stash.
        for data in self._stash[:to_index]:
            # If we find the data of interest, perform operation, otherwise just skip over.
            if data[KEY] == key:
                read_value = data[VALUE][offset]
                data[VALUE][offset] = value
                data[LEAF] = new_leaf
                # Set found to true.
                found = True
                # Break the for loop.
                continue

        # If the key was never found, raise an error, since the stash is always searched after path.
        if not found:
            raise KeyError(f"Key {key} not found.")

        return read_value

    def __retrieve_pos_map_block(self, key: int, offset: int, new_leaf: int, value: int, path: Buckets) -> Any:
        """
        Given a key and an operation, retrieve the block and apply the operation to it.

        :param key: a key to a data block.
        :param offset: the offset of where the value should be written to.
        :param new_leaf: the leaf of where the block should be written to.
        :param value: By default, write this value to key block at offset.
        :param path: a list of buckets of data.
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
                if data[KEY] is None:
                    continue
                # If it's the data of interest, we both read and write it, and give it a new path.
                elif data[KEY] == key:
                    read_value = data[VALUE][offset]
                    data[VALUE][offset] = value
                    data[LEAF] = new_leaf

                # And all real data to the stash.
                self._stash.append(data)

        # Check if the stash overflows.
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")

        # If unable to read a value, something should be wrong.
        if read_value is None:
            read_value = self.__retrieve_pos_map_stash(
                key=key, value=value, offset=offset, new_leaf=new_leaf, to_index=to_index
            )

        return read_value

    def __retrieve_data_stash(self, op: str, key: int, to_index: int, new_leaf: int, value: Any = None) -> int:
        """
        Given a key and an operation, retrieve the block from stash and apply the operation to it.

        :param op: an operation, can be "r", "w" or "rw".
        :param key: the key of the data block of interest.
        :param to_index: up to which index we should be checking.
        :param value: If the operation is "write", this is the new value for data block.
        :param new_leaf: If new leaf value is provided, store the accessed data to that leaf.
        :return: The leaf of the data block we found, and a value if the operation is "read".
        """
        # Set a value for whether the key is found.
        found = False
        # Temp holder for the value to read.
        read_value = None

        # Read all buckets in the path and add real data to stash.
        for data in self._stash[:to_index]:
            # If we find the data of interest, perform operation, otherwise just skip over.
            if data[KEY] == key:
                if op == "r":
                    read_value = data[VALUE]
                elif op == "w":
                    data[VALUE] = value
                elif op == "rw":
                    read_value = data[VALUE]
                    data[VALUE] = value
                else:
                    raise ValueError("The provided operation is not valid.")
                # Get new path and update the position map.
                data[LEAF] = new_leaf
                # Set found to true.
                found = True
                # Break the for loop.
                continue

        # If the key was never found, raise an error, since the stash is always searched after path.
        if not found:
            raise KeyError(f"Key {key} not found.")

        return read_value

    def __retrieve_data_block(self, op: str, key: int, new_leaf: int, path: Buckets, value: Any = None) -> Any:
        """
        Given a key and an operation, retrieve the block and apply the operation to it.

        :param op: an operation, can be "r", "w" or "rw".
        :param key: the key of the data block of interest.
        :param path: a list of buckets of data.
        :param value: If the operation is "write", this is the new value for data block.
        :param new_leaf: If new leaf value is provided, store the accessed data to that leaf.
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
                if data[KEY] is None:
                    continue
                # If it's the data of interest, we read/write it, and give it a new path.
                elif data[KEY] == key:
                    if op == "r":
                        read_value = data[VALUE]
                    elif op == "w":
                        data[VALUE] = value
                    elif op == "rw":
                        read_value = data[VALUE]
                        data[VALUE] = value
                    else:
                        raise ValueError("The provided operation is not valid.")
                    # Get new path and update the position map.
                    data[LEAF] = new_leaf
                    # Set found to True.
                    found = True

                # And all real data to the stash.
                self._stash.append(data)

        # Check if the stash overflows.
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")

        # If the value is not found, it might be in the stash.
        if not found:
            read_value = self.__retrieve_data_stash(op=op, key=key, to_index=to_index, value=value, new_leaf=new_leaf)

        return read_value

    def __evict_stash(self, leaf: int) -> Buckets:
        """
        Evict data blocks in the stash while maintaining correctness.

        :param leaf: the leaf label of the path we are evicting data to.
        :return: The leaf label and the path we should write there.
        """
        # Create a temporary stash.
        temp_stash = []
        # Create a placeholder for the new path.
        path = [[] for _ in range(self._level)]

        # Now we evict the stash by going through all real data in it.
        for data in self._stash:
            # Attempt to insert actual data to path.
            inserted = BinaryTree.fill_data_to_path(
                data=data, path=path, leaf=leaf, level=self._level, bucket_size=self._bucket_size
            )

            # If we were not able to insert data, overflow happened, put the block to the temp stash.
            if not inserted:
                temp_stash.append(data)

        # After we are done with all real data, complete the path with dummy data.
        BinaryTree.fill_buckets_with_dummy_data(buckets=path, bucket_size=self._bucket_size)

        # Update the stash.
        self._stash = temp_stash

        # Note that we return the path in the reversed order because we write path from bottom up.
        return self._encrypt_buckets(buckets=path[::-1])

    def __get_leaf_from_pos_map(self, key: int) -> Tuple[int, int]:
        """
        Provide a key to some data, iterate through all position map orams to find where it is stored.

        :param key: the key of the data block of interest.
        :return: which path the data block is on and the new path it should be stored to.
        """
        # Set up some variables.
        cur_leaf, new_cur_leaf = None, None

        # Iterate through all position map to retrieve the key for the data of interest.
        for pos_map_index, (cur_key, cur_index) in enumerate(self.__get_pos_map_keys(key=key)):
            # Get the first leaf from the on chip mem, otherwise it should have been set.
            if pos_map_index == 0:
                cur_leaf = self._pos_map[cur_key]
                # In this case, we also sample the new current leaf.
                new_cur_leaf = self.__pos_maps[pos_map_index]._get_new_leaf()
                # Update the on-chip position map.
                self._pos_map[cur_key] = new_cur_leaf

            # Sample a new leaf to replace the stored leaf; if we hit the last one, we sample from the data oram.
            new_next_leaf = self.__pos_maps[pos_map_index + 1]._get_new_leaf() \
                if pos_map_index < self.__num_oram_pos_map - 1 else self._get_new_leaf()

            # Base on the current leaf, get the desired path.
            path_data = self.client.read_query(label="pos_map", leaf=cur_leaf, index=pos_map_index)

            # Get the next leaf.
            next_leaf = self.__pos_maps[pos_map_index].__retrieve_pos_map_block(
                key=cur_key,
                offset=cur_index,
                path=path_data,
                new_leaf=new_cur_leaf,
                value=new_next_leaf
            )
            # Evict stash to current leaf.
            path_data = self.__pos_maps[pos_map_index].__evict_stash(leaf=cur_leaf)

            self.client.write_query(label="pos_map", leaf=cur_leaf, data=path_data, index=pos_map_index)

            # Finally set the current leaf to the next leaf.
            cur_leaf, new_cur_leaf = next_leaf, new_next_leaf

        return cur_leaf, new_cur_leaf

    def operate_on_key(self, op: str, key: int, value: Any = None) -> Any:
        """
        Perform operation on a given key.

        :param op: an operation, can be "r", "w" or "rw".
        :param key: the key of the data block of interest.
        :param value: If the operation is "w", this is the new value for data block.
        :return: The leaf of the data block we found, and a value if the operation is "r" or "rw".
        """
        # Find which path the data of interest lies on.
        leaf, new_leaf = self.__get_leaf_from_pos_map(key=key)

        # We read the path from the server.
        path = self.client.read_query(label="oram", leaf=leaf)

        # Retrieve value from the path, or write to it.
        value = self.__retrieve_data_block(op=op, key=key, path=path, value=value, new_leaf=new_leaf)

        # Perform an eviction and get a new path.
        path = self.__evict_stash(leaf=leaf)

        # Write the path back to the server.
        self.client.write_query(label="oram", leaf=leaf, data=path)

        return value

    def operate_on_key_without_eviction(self, op: str, key: int, value: Any = None) -> Any:
        """
        Perform operation on a given key without writing the data added to the stash back to the server.

        :param op: an operation, can be "r", "w" or "rw".
        :param key: the key of the data block of interest.
        :param value: If the operation is "w", this is the new value for data block.
        :return: The leaf of the data block we found, and a value if the operation is "r" or "rw".
        """
        # Find which path the data of interest lies on.
        leaf, new_leaf = self.__get_leaf_from_pos_map(key=key)

        # We read the path from the server.
        path = self.client.read_query(label="oram", leaf=leaf)

        # Retrieve value from the path, or write to it.
        value = self.__retrieve_data_block(op=op, key=key, path=path, value=value, new_leaf=new_leaf)

        # Temporarily save the leaf for future eviction.
        self.__tmp_leaf = leaf

        return value

    def eviction_with_update_stash(self, key: int, value: Any) -> None:
        """Update a data block stored in the stash and then perform eviction.

        :param key: the key of the data block of interest.
        :param value: the value to update the data block of interest.
        """
        # Set found the key to False.
        found = False

        # Read all buckets stored in the stash and find the desired data block of interest.
        for data in self._stash:
            # If we find the data of interest, update value and set found to True.
            if data[KEY] == key:
                data[VALUE] = value
                found = True

        # If the data was never found, we raise an error.
        if not found:
            raise KeyError(f"Key {key} not found.")

        # Perform an eviction and get a new path.
        path = self.__evict_stash(leaf=self.__tmp_leaf)

        # Write the path back to the server.
        self.client.write_query(label="oram", leaf=self.__tmp_leaf, data=path)

        # Set temporary leaf to None.
        self.__tmp_leaf = None
