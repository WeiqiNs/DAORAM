"""
This module defines the path oram class.

Path oram has two public methods:
    - init_storage_on_pos_map: this should be called first after the class object is created. This method constructs the
        storage the server should hold for the client.
    - operate_on_key: after the server gets the created storage, the client can use this function to obliviously access
        data points stored in the storage.
"""

from typing import Any, Optional

from daoram.dependency import BinaryTree, Buckets, InteractServer
from daoram.oram.tree_base_oram import TreeBaseOram


class PathOram(TreeBaseOram):
    def __init__(self,
                 num_data: int,
                 data_size: int,
                 client: InteractServer,
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True):
        """
        Defines the path oram, including its attributes and methods.

        :param num_data: The number of data points the oram should store.
        :param data_size: The number of bytes the random dummy data should have.
        :param client: The instance we use to interact with server.
        :param filename: The filename to save the oram data to.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param aes_key: The key to use for the AES instance.
        :param num_key_bytes: The number of bytes the aes key should have.
        :param use_encryption: A boolean indicating whether to use encryption.
        """
        # Initialize the parent BaseOram class.
        super().__init__(
            client=client,
            aes_key=aes_key,
            num_data=num_data,
            filename=filename,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            num_key_bytes=num_key_bytes,
            use_encryption=use_encryption
        )

        # This attribute is used to store a leaf temporarily for reading a path without evicting it immediately.
        self.__tmp_leaf: Optional[int] = None

        # In path oram, we initialize the position map.
        self._init_pos_map()

    def init_server_storage(self, data_map: dict = None) -> None:
        """
        Initialize the server storage based on the data map for this oram.

        :param data_map: A dictionary storing {key: data}.
        """
        # Get the storage.
        storage = {"oram": self._init_storage_on_pos_map(data_map=data_map)}

        # Initialize the storage and send it to the server.
        self.client.init_query(storage=storage)

    def __retrieve_stash(self, op: str, key: int, to_index: int, value: Any = None) -> int:
        """
        Given a key and an operation, retrieve the block from the stash and apply the operation to it.

        :param op: An operation, which can be "r", "w" or "rw".
        :param key: The key of the data block of interest.
        :param to_index: Up to which index we should be checking.
        :param value: If the operation is "write", this is the new value for data block.
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
                data.leaf = self._get_new_leaf()
                # Update the position map.
                self._pos_map[key] = data.leaf
                # Set found to true.
                found = True
                # Break the for loop.
                continue

        # If the key was never found, raise an error, since the stash is always searched after a path.
        if not found:
            raise KeyError(f"Key {key} not found.")

        return read_value

    def __retrieve_block(self, op: str, key: int, path: Buckets, value: Any = None) -> Any:
        """
        Given a key and an operation, retrieve the block and apply the operation to it.

        :param op: An operation, which can be "r", "w" or "rw".
        :param key: The key of the data block of interest.
        :param path: A list of buckets of data where the data block of interest is stored.
        :param value: If the operation is "w", this is the new value for data block.
        :return: The leaf of the data block we found, and a value if the operation is "r" or "rw".
        """
        # Set a value for whether the key is found.
        found = False
        # Temp holder for the value to read.
        read_value = None
        # Store the current stash length.
        to_index = len(self._stash)

        # Get the path data from the server and decrypt it if needed.
        path = self._decrypt_buckets(buckets=path)

        # Read all buckets in the path and add real data to stash.
        for bucket in path:
            for data in bucket:
                # Real data is always placed in front of dummy data, once we read dummy, we skip it.
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
                    data.leaf = self._get_new_leaf()
                    # Update the position map.
                    self._pos_map[key] = data.leaf
                    # Set found to True.
                    found = True

                # And all real data to the stash.
                self._stash.append(data)

        # Check if the stash overflows.
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")

        # If the value is not found, it might be in the stash.
        if not found:
            read_value = self.__retrieve_stash(op=op, key=key, to_index=to_index, value=value)

        # Return the read value, which maybe None in case of write operation.
        return read_value

    def __evict_stash(self, leaf: int) -> Buckets:
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
            # Attempt to insert actual data to a path.
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

    def operate_on_key(self, op: str, key: int, value: Any = None) -> Any:
        """
        Perform operation on a given key.

        :param op: An operation, which can be "r", "w" or "rw".
        :param key: The key of the data block of interest.
        :param value: If the operation is "w", this is the new value for data block.
        :return: The leaf of the data block we found, and a value if the operation is "r" or "rw".
        """
        # Find which path the data of interest lies on.
        leaf = self._look_up_pos_map(key=key)

        # We read the path from the server.
        path = self.client.read_query(label="oram", leaf=leaf)

        # Retrieve value from the path, or write to it.
        value = self.__retrieve_block(op=op, key=key, path=path, value=value)

        # Perform an eviction and get a new path.
        path = self.__evict_stash(leaf=leaf)

        # Write the path back to the server.
        self.client.write_query(label="oram", leaf=leaf, data=path)

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
        leaf = self._look_up_pos_map(key=key)

        # We read the path from the server.
        path = self.client.read_query(label="oram", leaf=leaf)

        # Retrieve value from the path, or write to it.
        value = self.__retrieve_block(op=op, key=key, path=path, value=value)

        # Temporarily save the leaf for future eviction.
        self.__tmp_leaf = leaf

        return value

    def eviction_with_update_stash(self, key: int, value: Any) -> None:
        """
        Update a data block stored in the stash and then perform eviction.

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
        path = self.__evict_stash(leaf=self.__tmp_leaf)

        # Write the path back to the server.
        self.client.write_query(label="oram", leaf=self.__tmp_leaf, data=path)

        # Set temporary leaf to None.
        self.__tmp_leaf = None
