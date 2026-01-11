"""
Module for defining a parent class for binary tree-based oram.

The BaseOram class defines a set of attributes that an oram should have and some basic methods such as encryption, etc.
Note that we don't use double underscores (name mangling) in this file for private methods because all things defined
here should be accessible to its children classes.
"""

import math
import os
import secrets
from abc import ABC, abstractmethod
from typing import Any, List, Optional

from daoram.dependency import BinaryTree, Data, Helper, InteractServer, Encryptor, PathData, UNSET


class TreeBaseOram(ABC):
    def __init__(self,
                 name: str,
                 num_data: int,
                 data_size: int,
                 client: InteractServer,
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 encryptor: Encryptor = None):
        """
        Defines the base oram, including its attributes and methods.

        :param name: The name of the protocol, this should be unique if multiple schemes are used together.
        :param num_data: The number of data points the oram should store.
        :param data_size: The number of bytes the random dummy data should have.
        :param client: The instance we use to interact with server.
        :param filename: The filename to save the oram data to.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param encryptor: The encryptor to use for encryption.
        """
        # Store the useful input values.
        self._name: str = name
        self._filename: str = filename
        self._num_data: int = num_data
        self._data_size: int = data_size
        self._bucket_size: int = bucket_size
        self._stash_scale: int = stash_scale
        self._encryptor: Encryptor = encryptor

        # Compute the level of the binary tree needed.
        self._level: int = int(math.ceil(math.log(num_data, 2))) + 1

        # Compute the range of possible leafs [0, leaf_range).
        self._leaf_range: int = pow(2, self._level - 1)

        # Compute the stash size.
        self._stash_size: int = stash_scale * (self._level - 1) if self._level > 1 else stash_scale

        # Claim the variables for stash and the position map.
        self._stash: list = []
        self._pos_map: dict = {}

        # Set dumpled data size and disk size for use of encryption or file storage.
        self._dumped_data_size: Optional[int] = None
        self._disk_size: Optional[int] = None

        # Compute the padded total data size (needed for encryption or file storage).
        if self._encryptor or self._filename:
            self._dumped_data_size = len(
                Data(key=self._num_data - 1, leaf=self._num_data - 1, value=os.urandom(data_size)).dump()
            ) + Helper.LENGTH_HEADER_SIZE

        # Compute the disk length if needed.
        if self._filename:
            if encryptor:
                self._disk_size = encryptor.ciphertext_length(self._dumped_data_size)
            else:
                self._disk_size = self._dumped_data_size

        # Initialize the client connection.
        self._client: InteractServer = client

        # Variable to save the max stash. (Note that this variable is here only for experimental purposes.)
        self.max_stash: int = 0

    @property
    def client(self) -> InteractServer:
        """Return the client object."""
        return self._client

    @property
    def stash(self) -> list:
        """Return the stash."""
        return self._stash

    @stash.setter
    def stash(self, value: list):
        """Update the stash with input."""
        self._stash = value

    @property
    def stash_size(self) -> int:
        """Return the stash size."""
        return len(self._stash)

    @property
    def total_stash_size(self) -> int:
        """Compute the total size of the stashed, including the ones in position maps."""
        return self.stash_size + sum([pos_map.stash_size for pos_map in self._pos_maps]) \
            if hasattr(self, '_pos_maps') else self.stash_size

    def _init_pos_map(self) -> None:
        """Initialize the default position map where {i : random_leaf}, for i in [0, leaf_range)."""
        self._pos_map = {i: self._get_new_leaf() for i in range(self._num_data)}

    def _encrypt_path_data(self, path: PathData) -> PathData:
        """
        Encrypt all buckets in a PathData dict.

        :param path: PathData dict mapping storage index to bucket.
        :return: PathData dict with encrypted buckets.
        """

        def _enc_bucket(bucket: List[Data]) -> List[bytes]:
            """Helper function to add dummy data and encrypt a bucket."""
            enc_bucket = [
                self._encryptor.enc(plaintext=data.dump_pad(self._dumped_data_size))
                for data in bucket
            ]

            # Compute if dummy block is needed.
            dummy_needed = self._bucket_size - len(bucket)
            # If needed, perform padding.
            if dummy_needed > 0:
                enc_bucket.extend([
                    self._encryptor.enc(plaintext=Data().dump_pad(self._dumped_data_size))
                    for _ in range(dummy_needed)
                ])

            return enc_bucket

        return {idx: _enc_bucket(bucket) for idx, bucket in path.items()} if self._encryptor else path

    def _decrypt_path_data(self, path: PathData) -> PathData:
        """
        Decrypt all buckets in a PathData dict.

        :param path: PathData dict mapping storage index to encrypted bucket.
        :return: PathData dict with decrypted buckets (dummy blocks filtered out).
        """

        def _dec_bucket(bucket: List[bytes]) -> List[Data]:
            """Helper function to decrypt a bucket and filter out dummy data."""
            return [
                dec for data in bucket
                if (dec := Data.load_unpad(self._encryptor.dec(ciphertext=data))).key is not None
            ]

        return {idx: _dec_bucket(bucket) for idx, bucket in path.items()} if self._encryptor else path

    def _get_new_leaf(self) -> int:
        """Get a random leaf label within the range."""
        return secrets.randbelow(self._leaf_range)

    def _look_up_pos_map(self, key: int) -> int:
        """
        Look up key of data and get the leaf for where the data is stored.

        :param key: A key of a data block.
        :return: The corresponding leaf if found.
        """
        # If key can't be found, raise an error.
        if key not in self._pos_map:
            raise KeyError(f"Key {key} not found in position map.")

        return self._pos_map[key]

    def _init_storage_on_pos_map(self, data_map: dict = None) -> BinaryTree:
        """
        Initialize a binary tree storage based on the data map.

        :param data_map: A dictionary storing {key: data}.
        :return: The binary tree storage based on the data map.
        """
        # Create the binary tree object.
        tree = BinaryTree(
            filename=self._filename,
            num_data=self._num_data,
            disk_size=self._disk_size,
            bucket_size=self._bucket_size,
            data_size=self._dumped_data_size,
            encryption=True if self._encryptor else False,
        )

        # Fill the data to leaf according to the position map.
        for key, leaf in self._pos_map.items():
            value = data_map[key] if data_map else os.urandom(self._data_size)
            tree.fill_data_to_storage_leaf(data=Data(key=key, leaf=leaf, value=value))

        # Encrypt the tree storage if needed.
        if self._encryptor:
            tree.storage.encrypt(encryptor=self._encryptor)

        return tree

    def _evict_stash(self, leaves: List[int]) -> PathData:
        """
        Evict data blocks in the stash to one or more paths while maintaining correctness.

        :param leaves: A list of leaf labels of the paths we are evicting data to.
        :return: PathData dict mapping storage index to encrypted bucket.
        """
        # Create a temporary stash.
        temp_stash = []

        # Create a dictionary contains locations for where all leaves' paths touch.
        path = BinaryTree.get_mul_path_dict(level=self._level, indices=leaves)

        # Now we evict the stash by going through all real data in it.
        for data in self._stash:
            # Attempt to insert actual data to the path.
            inserted = BinaryTree.fill_data_to_path(
                data=data, path=path, leaves=leaves, level=self._level, bucket_size=self._bucket_size
            )
            # If we were not able to insert data, overflow happened, put the block to the temp stash.
            if not inserted:
                temp_stash.append(data)

        # Update the stash.
        self._stash = temp_stash

        return self._encrypt_path_data(path=path)

    def _retrieve_data_stash(self, key: int, to_index: int, new_leaf: int, value: Any = UNSET) -> Any:
        """
        Given a key, retrieve the block from the stash. If value is provided, write it.

        :param key: The key of the data block of interest.
        :param to_index: Up to which index we should be checking.
        :param new_leaf: The new leaf to store the accessed data to.
        :param value: If provided (not UNSET), write this value to the data block.
        :return: The current value of the data block.
        """
        # Read all buckets in the path and add real data to stash.
        for data in self._stash[:to_index]:
            # If we find the data of interest, perform operation, otherwise skip over.
            if data.key == key:
                # Always read the current value.
                read_value = data.value
                # Write if value is provided.
                if value is not UNSET:
                    data.value = value
                # Update the leaf.
                data.leaf = new_leaf
                return read_value

        # If the key was never found, raise an error, since the stash is always searched after the path.
        raise KeyError(f"Key {key} not found.")

    def _retrieve_data_block(self, key: int, new_leaf: int, path: PathData, value: Any = UNSET) -> Any:
        """
        Given a key, retrieve the block. If value is provided, write it.

        :param key: The key of the data block of interest.
        :param new_leaf: The new leaf to store the accessed data to.
        :param path: PathData dict mapping storage index to bucket.
        :param value: If provided (not UNSET), write this value to the data block.
        :return: The current value of the data block.
        """
        # Set a value for whether the key is found.
        found = False
        # Temp holder for the value to read.
        read_value = None
        # Store the current stash length.
        to_index = len(self._stash)

        # Decrypt the path if needed.
        path = self._decrypt_path_data(path=path)

        # Read all buckets in the path and add real data to stash.
        for bucket in path.values():
            for data in bucket:
                # If dummy data, we skip it.
                if data.key is None:
                    continue
                # If it's the data of interest, we read/write it, and give it a new path.
                elif data.key == key:
                    # Always read the current value.
                    read_value = data.value
                    # Write if value is provided.
                    if value is not UNSET:
                        data.value = value
                    # Update the leaf.
                    data.leaf = new_leaf
                    # Set found to True.
                    found = True
                # Add all real data to the stash.
                self._stash.append(data)

        # Check if the stash overflows.
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")

        # If the value is not found, it might be in the stash.
        if not found:
            read_value = self._retrieve_data_stash(key=key, to_index=to_index, value=value, new_leaf=new_leaf)

        return read_value

    @abstractmethod
    def init_server_storage(self, data_map: dict = None) -> None:
        """
        Initialize the server storage based on the data map for this oram.

        :param data_map: A dictionary storing {key: data}.
        """
        raise NotImplementedError

    @abstractmethod
    def operate_on_key(self, key: int, value: Any = UNSET) -> Any:
        """
        Perform operation on a given key. Always returns the current value.
        If value is provided (not UNSET), writes the new value.

        :param key: The key of the data block of interest.
        :param value: If provided (not UNSET), write this value to the data block.
        :return: The current value of the data block (before write if writing).
        """
        raise NotImplementedError

    @abstractmethod
    def operate_on_key_without_eviction(self, key: int, value: Any = UNSET) -> Any:
        """
        Perform operation on a given key without writing the data added to the stash back to the server.

        :param key: The key of the data block of interest.
        :param value: If provided (not UNSET), write this value to the data block.
        :return: The current value of the data block (before write if writing).
        """
        raise NotImplementedError

    @abstractmethod
    def eviction_with_update_stash(self, key: int, value: Any, execute: bool = True) -> None:
        """Update a data block stored in the stash and then perform eviction.

        :param key: The key of the data block of interest.
        :param value: The value to update the data block of interest.
        :param execute: If True, execute immediately. If False, queue write for batching.
        """
        raise NotImplementedError
