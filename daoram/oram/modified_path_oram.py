import math
import os
from typing import Any, Optional, Tuple, Union
from daoram.dependency import BinaryTree, Buckets, InteractServer
from daoram.dependency.flexible_binary_tree import FlexibleBinaryTree
from daoram.dependency.helper import Data
from daoram.oram.tree_base_oram import TreeBaseOram


class ModifiedPathOram(TreeBaseOram):
    def __init__(self,
                 num_data: int,
                 data_size: int,
                 client: InteractServer,
                 name: str = "po",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 4,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = False):
        """
        Defines the path oram, including its attributes and methods.

        :param num_data: The number of data points the oram should store.
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
        # Initialize the parent BaseOram class.
        super().__init__(
            name=name,
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
        self.__tmp_leaf: Optional[Any] = None
        # Recalculate _max_block_size to account for tuple in leaf field
        self._max_block_size: int = len(
            Data(key=self._num_data - 1, leaf=(self._num_data - 1, self._level), value=os.urandom(self._data_size)).dump()
        )
        
        # Compute the level of the binary tree needed.
        self._level = int(math.ceil(math.log(num_data, 2))) + 1
        # In path oram, we initialize the position map.
        self._init_pos_map()
        self.label = name
    def scale_up(self) -> bool:
        if(self.client.scale_up(label = self.label)):
            self._level +=1
            self._stash_size = self._stash_scale * (self._level - 1) if self._level > 1 else self._stash_scale
            self._leaf_range = pow(2, self._level - 1) 
            return True
        return False
    
    def scale_down(self) -> bool:
        if(self.client.scale_down(label = self.label)):
            self._level -=1
            self._stash_size = self._stash_scale * (self._level - 1) if self._level > 1 else self._stash_scale
            self._leaf_range = pow(2, self._level - 1)    
            return True
        return False
    def _init_pos_map(self) -> None:
        """Initialize the default position map where {i : random_leaf}, for i in [0, leaf_range)."""
        self._pos_map = {i: (self._get_new_leaf(), self._level) for i in range(self._num_data)}

    def init_server_storage(self, data_map: dict = None) -> None:
        """
        Initialize the server storage based on the data map for this oram.

        :param data_map: A dictionary storing {key: data}.
        """
        # Get the storage.
        storage = {self._name: self._init_storage_on_pos_map(data_map=data_map)}

        # Initialize the storage and send it to the server.
        self.client.init(storage=storage)

    def _init_storage_on_pos_map(self, data_map: dict = None) -> FlexibleBinaryTree:
        """
        Initialize a binary tree storage based on the data map.

        :param data_map: A dictionary storing {key: data}.
        :return: The binary tree storage based on the data map.
        """
        # Create the binary tree object.
        tree = FlexibleBinaryTree(
            filename=self._filename,
            num_data=self._num_data,
            data_size=self._max_block_size,
            bucket_size=self._bucket_size,
            enc_key_size=self._num_key_bytes if self._use_encryption else None,
        )

        # Fill the data to leaf according to the provided data_map.
        if data_map:
            for key, leaf in self._pos_map.items():
                tree.fill_data_to_storage_leaf(data=Data(key=key, leaf=leaf, value=data_map[key]))

        # Otherwise, fill dummy data at the correct places.
        else:
            for key, leaf in self._pos_map.items():
                tree.fill_data_to_storage_leaf(data=Data(key=key, leaf=leaf, value=os.urandom(self._data_size)))

        # Encrypt the tree storage if needed.
        if self._use_encryption:
            tree.storage.encrypt(aes=self._cipher)

        return tree
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
                new_leaf = (self._get_new_leaf(), self._level)
                data.leaf = new_leaf
                # Update the position map.
                self._pos_map[key] = new_leaf
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
                    new_leaf = (self._get_new_leaf(), self._level)
                    data.leaf = new_leaf
                    # Update the position map.
                    self._pos_map[key] = new_leaf
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

    def _evict_stash(self, leaf: Tuple[int, int]) -> Buckets:
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
            inserted = FlexibleBinaryTree.fill_data_to_path(
                data=data, path=path, leaf=leaf, bucket_size=self._bucket_size, level=self._level
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
        path = self.client.read_query(label=self._name, leaf=leaf)

        # Retrieve value from the path, or write to it.
        value = self.__retrieve_block(op=op, key=key, path=path, value=value)

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
        leaf = self._look_up_pos_map(key=key)

        # We read the path from the server.
        path = self.client.read_query(label=self._name, leaf=leaf)

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
        self.client.write_query(label=self._name, leaf=self.__tmp_leaf, data=path)

        # Set temporary leaf to None.
        self.__tmp_leaf = None
