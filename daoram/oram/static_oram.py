"""
This module defines the static oram class, which inherits from PathOram.

StaticOram overrides the __retrieve_stash and __retrieve_block methods to not call self._get_new_leaf().
"""
import os
import random
from typing import Any, Optional

from daoram.dependency import Buckets
from daoram.dependency.binary_tree import BinaryTree
from daoram.dependency.crypto import Prf
from daoram.dependency.helper import Data
from daoram.oram.path_oram import PathOram


class StaticOram(PathOram):
    def __init__(self,
                 num_data: int,
                 data_size: int,
                 client,
                 name: str = "static_oram",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 4,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True):
        """
        Defines the static oram, which inherits from PathOram but doesn't update leaf positions.
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
        # Initialize the parent PathOram class.
        super().__init__(
            num_data=num_data,
            data_size=data_size,
            client=client,
            name=name,
            filename=filename,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            aes_key=aes_key,
            num_key_bytes=num_key_bytes,
            use_encryption=use_encryption
        )
        self._hash_func = Prf(key=os.urandom(16))

    
    def __retrieve_stash(self, op: str, key: int, to_index: int, value: Any = None) -> int:
        """
        Given a key and an operation, retrieve the block from the stash and apply the operation to it.
        This method overrides the parent method to not call self._get_new_leaf().

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
                # Note: We don't call self._get_new_leaf() here, keeping the leaf position static
                # Set found to true.
                found = True
                # Break the for loop.
                continue
        return read_value

    def __retrieve_block(self, op: str, key: int, path: Buckets, value: Any = None) -> Any:
        """
        Given a key and an operation, retrieve the block and apply the operation to it.
        This method overrides the parent method to not call self._get_new_leaf().

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
                # If it's the data of interest, we read/write it, but don't give it a new path.
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
                    # Note: We don't call self._get_new_leaf() here, keeping the leaf position static
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
    

    def _init_storage_on_pos_map(self, data_map: dict = None) -> BinaryTree:
        """
        Initialize a binary tree storage based on the data map. This method overrides the parent method to 
        call _get_path_number() to get the path number for each key.

        :param data_map: A dictionary storing {key: data}.
        :return: The binary tree storage based on the data map.
        """
        # Create the binary tree object.
        tree = BinaryTree(
            filename=self._filename,
            num_data=self._num_data,
            data_size=self._max_block_size,
            bucket_size=self._bucket_size,
            enc_key_size=self._num_key_bytes if self._use_encryption else None,
        )

        # Fill the data to leaf according to the provided data_map.
        if data_map:
            for key, leaf in self._pos_map.items():
                tree.fill_data_to_storage_leaf(data=Data(key=key, leaf=self._get_path_number(key), value=data_map[key]))

        # Encrypt the tree storage if needed.
        if self._use_encryption:
            tree.storage.encrypt(aes=self._cipher)

        return tree

    def operate_on_key(self, op: str, key: int, value: Any = None) -> Any:
        """
        Perform operation on a given key. This method overrides the parent method to call _get_path_number() to 
        get the path number for the key.

        :param op: An operation, which can be "r", "w" or "rw".
        :param key: The key of the data block of interest.
        :param value: If the operation is "w", this is the new value for data block.
        :return: The leaf of the data block we found, and a value if the operation is "r" or "rw".
        """
        # Find which path the data of interest lies on.
        leaf = self._get_path_number(key)

        # We read the path from the server.
        path = self.client.read_query(label=self._name, leaf=leaf)

        # Retrieve value from the path, or write to it.
        value = self.__retrieve_block(op=op, key=key, path=path, value=value)

        # Perform an eviction and get a new path.
        path = self._evict_stash(leaf=leaf)

        # Write the path back to the server.
        self.client.write_query(label=self._name, leaf=leaf, data=path)

        return value

    def _get_path_number(self, key: int) -> int:
        """
        generate a fixed path number for the given key.
        
        :param key: The key of the data
        :return: path number
        """
        if key is None:
            return random.randint(0, self._num_data-1)
        return self._hash_func.digest_mod_n(str(key).encode(), self._num_data)
