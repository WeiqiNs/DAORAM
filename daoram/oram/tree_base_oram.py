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
from typing import Any, List

from daoram.dependency import BinaryTree, Buckets, Data, Helper, InteractServer, Encryptor, PathData


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

        # Compute the padded total data size, which at largest would be (biggest_key, biggest_leaf, biggest_data).
        self._dumped_data_size: int = len(
            Data(key=self._num_data - 1, leaf=self._num_data - 1, value=os.urandom(data_size)).dump()
        ) + Helper.LENGTH_HEADER_SIZE if self._encryptor else None

        # Compute the disk length if needed.
        self._disk_size: int = encryptor.ciphertext_length(self._dumped_data_size) if self._filename else None

        # Initialize the client connection.
        self._client: InteractServer = client

        # Variable to save the max stash. (Note that this variable is here only for experimental purposes.)
        self.max_stash: int = 0

    @property
    def client(self) -> InteractServer:
        """Return the client object."""
        return self._client

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

    def _encrypt_buckets(self, buckets: List[List[Data]]) -> Buckets:
        """
        Encrypt all data in given buckets.

        Note that we first pad data to the desired length and then perform the encryption. This encryption also fills
        the bucket with the desired amount of dummy data.
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

        # Return the encrypted list of lists of bytes.
        return [_enc_bucket(bucket=bucket) for bucket in buckets] if self._encryptor else buckets

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

    def _decrypt_buckets(self, buckets: Buckets) -> List[List[Data]]:
        """Given encrypted buckets, decrypt all data in it."""
        # Return the decrypted list of data.
        return [[
            dec for data in bucket
            if (dec := Data.load_unpad(Helper.unpad_pickle(data=self._encryptor.dec(ciphertext=data)))).key is not None
        ] for bucket in buckets] if self._encryptor else buckets

    def _path_data_to_buckets(self, leaf: int, path_data: PathData) -> Buckets:
        """
        Convert PathData (dict) to Buckets (list) for a given leaf.

        :param leaf: The leaf label.
        :param path_data: PathData dict mapping storage index to bucket.
        :return: List of buckets from leaf to root.
        """
        # Compute path indices from leaf to root.
        start_leaf = pow(2, self._level - 1) - 1
        path_indices = BinaryTree.get_path_indices(index=leaf + start_leaf)
        # Extract buckets in order.
        return [path_data[idx] for idx in path_indices]

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
            bucket_size=self._bucket_size,
            data_size=self._dumped_data_size,
            disk_size=self._dumped_data_size,
            encryption=True if self._encryptor else False,
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
        if self._encryptor:
            tree.storage.encrypt(encryptor=self._encryptor)

        return tree

    @abstractmethod
    def init_server_storage(self, data_map: dict = None) -> None:
        """
        Initialize the server storage based on the data map for this oram.

        :param data_map: A dictionary storing {key: data}.
        """
        raise NotImplementedError

    @abstractmethod
    def operate_on_key(self, op: str, key: int, value: Any = None) -> Any:
        """
        Perform operation on a given key.

        :param op: An operation, which can be "r", "w" or "rw".
        :param key: The key of the data block of interest.
        :param value: If the operation is "w", this is the new value for data block.
        :return: The leaf of the data block we found, and a value if the operation is "r" or "rw".
        """
        raise NotImplementedError

    @abstractmethod
    def operate_on_key_without_eviction(self, op: str, key: int, value: Any = None) -> Any:
        """
        Perform operation on a given key without writing the data added to the stash back to the server.

        :param op: An operation, which can be "r", "w" or "rw".
        :param key: The key of the data block of interest.
        :param value: If the operation is "w", this is the new value for data block.
        :return: The leaf of the data block we found, and a value if the operation is "r" or "rw".
        """
        raise NotImplementedError

    @abstractmethod
    def eviction_with_update_stash(self, key: int, value: Any) -> None:
        """Update a data block stored in the stash and then perform eviction.

        :param key: The key of the data block of interest.
        :param value: The value to update the data block of interest.
        """
        raise NotImplementedError
