"""
Module for defining a parent class for binary tree based oram.

The BaseOram class defines a set of attributes that an oram should have and some basic methods such as encryption, etc.
Note that we don't use double underscores (name mangling) in this file for private methods because all things defined
here should be accessible to its children classes.
"""

import math
import os
import pickle
import secrets
from abc import ABC, abstractmethod
from typing import Any, Optional

from daoram.dependency.binary_tree import BinaryTree, Buckets, VALUE
from daoram.dependency.crypto import Aes, pad_pickle, unpad_pickle
from daoram.dependency.interact_server import InteractServer

# Set variables to extract encrypted metadata and encrypted real value.
ENC_META = 0
ENC_VALUE = 1


class TreeBaseOram(ABC):
    def __init__(self,
                 num_data: int,
                 data_size: int,
                 client: InteractServer,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 split_meta: bool = False,
                 use_encryption: bool = True):
        """
        Defines the base oram, including its attributes and methods.

        :param num_data: the number of data points the oram should store.
        :param data_size: the number of bytes the random dummy data should have.
        :param client: the instance we use to interact with server.
        :param bucket_size: the number of data each bucket should have.
        :param stash_scale: the scaling scale of the stash.
        :param aes_key: the key to use for the AES instance.
        :param num_key_bytes: the number of bytes the aes key should have.
        :param split_meta: whether to split the data into metadata and real data.
        :param use_encryption: a boolean indicating whether to use encryption.
        """
        # Store the useful input values.
        self._num_data: int = num_data
        self._data_size: int = data_size
        self._bucket_size: int = bucket_size
        self._stash_scale: int = stash_scale
        self._split_meta: bool = split_meta
        self._use_encryption: bool = use_encryption

        # Compute the level of the binary tree needed.
        self._level: int = int(math.ceil(math.log(num_data, 2))) + 1

        # Compute the range of possible leafs [0, leaf_range).
        self._leaf_range: int = pow(2, self._level - 1)

        # Compute the stash size.
        self._stash_size: int = stash_scale * (self._level - 1) if self._level > 1 else stash_scale

        # Claim the variables for stash and the position map.
        self._stash: list = []
        self._pos_map: dict = {}

        # Compute the padded total data size, which at largest would be [biggest_key, biggest_leaf, biggest_data].
        self._padded_block_size: int = len(
            pickle.dumps([self._num_data - 1, self._num_data - 1, os.urandom(data_size)]))
        # Compute the padded metadata size, which at largest would be [biggest_key, biggest_leaf].
        self._padded_meta_data_size: int = len(pickle.dumps([self._num_data - 1, self._num_data - 1]))

        # Use encryption if required.
        self._aes_key: bytes = aes_key
        self._num_key_bytes: int = num_key_bytes
        self._cipher: Optional[Aes] = Aes(key=aes_key, key_byte_length=num_key_bytes) if use_encryption else None

        # Initialize the client connection.
        self.__client: InteractServer = client

        # Variable to save the max stash. (Note that this variable is here only for experimental purposes.)
        self.max_stash: int = 0

    @property
    def client(self) -> InteractServer:
        return self.__client

    @property
    def stash_size(self) -> int:
        return len(self._stash)

    @property
    def get_stash_size(self) -> int:
        """Compute the total size of the stashed, including the ones in position maps."""
        return self.stash_size + sum([len(pos_map.stash_size) for pos_map in self.__pos_maps]) \
            if hasattr(self, '__pos_maps') else self.stash_size

    def _init_pos_map(self) -> None:
        """Initialize the default position map where {i : random_leaf}, for i in [0, leaf_range)."""
        self._pos_map = {i: self._get_new_leaf() for i in range(self._num_data)}

    def _encrypt_buckets(self, buckets: Buckets) -> Buckets:
        """
        Given buckets, encrypt all data in it.

        Note that we first pad data to desired length and then perform the encryption.
        """
        return [[self._cipher.enc(plaintext=pad_pickle(data=pickle.dumps(data), length=self._padded_block_size))
                 for data in bucket] for bucket in buckets] if self._use_encryption else buckets

    def _encrypt_meta_buckets(self, buckets: Buckets) -> Buckets:
        """
        Given buckets, encrypt all data in it.

        Assuming each data contains [key, leaf, data] where key and leaf are the metadata.
        The output is a tuple of Enc(metadata), Enc(data).
        """
        return [[
            (
                self._cipher.enc(plaintext=pickle.dumps(data[:VALUE])),
                self._cipher.enc(plaintext=pickle.dumps(data[VALUE]))
            ) for data in bucket
        ] for bucket in buckets] if self._use_encryption else buckets

    def _decrypt_buckets(self, buckets: Buckets) -> Buckets:
        """Given encrypted buckets, decrypt all data in it."""
        return [[pickle.loads(unpad_pickle(data=self._cipher.dec(ciphertext=data)))
                 for data in bucket] for bucket in buckets] if self._use_encryption else buckets

    def _decrypt_meta_buckets(self, buckets: Buckets) -> Buckets:
        """Given encrypted buckets, decrypt all data in it."""
        return [[
            [
                *pickle.loads(self._cipher.dec(ciphertext=data[ENC_META])),
                pickle.loads(self._cipher.dec(ciphertext=data[ENC_VALUE]))
            ] for data in bucket
        ] for bucket in buckets] if self._use_encryption else buckets

    def _get_new_leaf(self) -> int:
        """Get a random leaf label within the range."""
        return secrets.randbelow(self._leaf_range)

    def _look_up_pos_map(self, key: int) -> int:
        """
        Look up key of a data and get the leaf for where the data is stored.

        :param key: a key of a data block.
        :return: the corresponding leaf if found.
        """
        # If key can't be found, raise an error.
        if key not in self._pos_map:
            raise KeyError(f"Key {key} not found in position map.")

        return self._pos_map[key]

    def _init_storage_on_pos_map(self, data_map: dict = None) -> BinaryTree:
        """
        Initialize a binary tree storage based on the data map.

        :param data_map: a dictionary storing {key: data}.
        :return: the binary tree storage based on the data map.
        """
        # Create the binary tree object.
        tree = BinaryTree(num_data=self._num_data, bucket_size=self._bucket_size)

        # Fill the data to leaf according to the provided data_map.
        if data_map:
            for key, leaf in self._pos_map.items():
                tree.fill_data_to_storage_leaf(data=[key, leaf, data_map[key]])

        # Otherwise just fill dummy data at correct places.
        else:
            for key, leaf in self._pos_map.items():
                tree.fill_data_to_storage_leaf(data=[key, leaf, os.urandom(self._data_size)])

        # Fill the storage with dummy data.
        tree.fill_storage_with_dummy_data()

        # Encrypt the tree storage if needed.
        if self._use_encryption:
            if self._split_meta:
                tree.storage = self._encrypt_meta_buckets(buckets=tree.storage)
            else:
                tree.storage = self._encrypt_buckets(buckets=tree.storage)

        return tree

    @abstractmethod
    def init_server_storage(self, data_map: dict = None) -> None:
        """
        Initialize the server storage based on the data map for this oram.

        :param data_map: a dictionary storing {key: data}.
        """
        raise NotImplementedError

    @abstractmethod
    def operate_on_key(self, op: str, key: int, value: Any = None) -> Any:
        """
        Perform operation on a given key.

        :param op: an operation, can be "r", "w" or "rw".
        :param key: the key of the data block of interest.
        :param value: If the operation is "w", this is the new value for data block.
        :return: The leaf of the data block we found, and a value if the operation is "r" or "rw".
        """
        raise NotImplementedError

    @abstractmethod
    def operate_on_key_without_eviction(self, op: str, key: int, value: Any = None) -> Any:
        """
        Perform operation on a given key without writing the data added to the stash back to the server.

        :param op: an operation, can be "r", "w" or "rw".
        :param key: the key of the data block of interest.
        :param value: If the operation is "w", this is the new value for data block.
        :return: The leaf of the data block we found, and a value if the operation is "r" or "rw".
        """
        raise NotImplementedError

    @abstractmethod
    def eviction_with_update_stash(self, key: int, value: Any) -> None:
        """Update a data block stored in the stash and then perform eviction.

        :param key: the key of the data block of interest.
        :param value: the value to update the data block of interest.
        """
        raise NotImplementedError
