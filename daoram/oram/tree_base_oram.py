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

from daoram.dependency import Aes, BinaryTree, Buckets, Data, Helper, InteractServer


class TreeBaseOram(ABC):
    def __init__(self,
                 name: str,
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
        Defines the base oram, including its attributes and methods.

        :param name: The name of the protocol, this should be unique if multiple schemes are used together.
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
        # Store the useful input values.
        self._name: str = name
        self._filename: str = filename
        self._num_data: int = num_data
        self._data_size: int = data_size
        self._bucket_size: int = bucket_size
        self._stash_scale: int = stash_scale
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

        # Compute the padded total data size, which at largest would be (biggest_key, biggest_leaf, biggest_data).
        self._max_block_size: int = len(
            Data(key=self._num_data - 1, leaf=self._num_data - 1, value=os.urandom(data_size)).dump_pad()
        )

        # Use encryption if required.
        self._aes_key: bytes = aes_key
        self._num_key_bytes: int = num_key_bytes
        self._cipher: Optional[Aes] = Aes(key=aes_key, key_byte_length=num_key_bytes) if use_encryption else None

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
                self._cipher.enc(plaintext=Helper.pad_pickle(data=data.dump_pad(), length=self._max_block_size))
                for data in bucket
            ]

            # Compute if dummy block is needed.
            dummy_needed = self._bucket_size - len(bucket)
            # If needed, perform padding.
            if dummy_needed > 0:
                enc_bucket.extend([
                    self._cipher.enc(plaintext=Helper.pad_pickle(data=Data().dump_pad(), length=self._max_block_size))
                    for _ in range(dummy_needed)
                ])

            return enc_bucket

        # Return the encrypted list of lists of bytes.
        return [_enc_bucket(bucket=bucket) for bucket in buckets] if self._use_encryption else buckets

    def _decrypt_buckets(self, buckets: Buckets) -> List[List[Data]]:
        """Given encrypted buckets, decrypt all data in it."""
        # Return the decrypted list of data.
        return [[
            dec for data in bucket
            if (dec := Data.load_unpad(Helper.unpad_pickle(data=self._cipher.dec(ciphertext=data)))).key is not None
        ] for bucket in buckets] if self._use_encryption else buckets

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
            tree.storage.encrypt(encryptor=self._cipher)

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
