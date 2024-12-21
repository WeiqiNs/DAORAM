"""
Module for defining a parent class for tree-structure ods omap.

The BaseOmap class defines a set of attributes that an omap should have and some basic methods such as encryption, etc.
Note that we don't use double underscores (name mangling) in this file for private methods because all things defined
here should be accessible to its children classes.
"""

import math
import pickle
import secrets
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from daoram.dependency.binary_tree import BinaryTree, Buckets, KEY
from daoram.dependency.crypto import Aes, pad_pickle, unpad_pickle
from daoram.dependency.interact_server import InteractServer

# Root contains the data key and the path.
ROOT = Tuple[Any, int]
# Define the input key-value pair types.
KV_LIST = List[Tuple[Any, Any]]


class TreeOdsOmap(ABC):
    def __init__(self,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 client: InteractServer,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True):
        """
        Defines the base omap, including its attributes and methods.

        :param num_data: the number of data points the oram should store.
        :param key_size: the number of bytes the random key should have.
        :param data_size: the number of bytes the random dummy data should have.
        :param client: the instance we use to interact with server.
        :param bucket_size: the number of data each bucket should have.
        :param stash_scale: the scaling scale of the stash.
        :param aes_key: the key to use for the AES instance.
        :param num_key_bytes: the number of bytes the aes key should have.
        :param use_encryption: a boolean indicating whether to use encryption.
        """
        # Store the useful input values.
        self._num_data: int = num_data
        self._key_size: int = key_size
        self._data_size: int = data_size
        self._bucket_size: int = bucket_size
        self._use_encryption: bool = use_encryption

        # Compute the level of the binary tree needed.
        self._level: int = int(math.ceil(math.log(num_data, 2))) + 1

        # Compute the range of possible leafs [0, leaf_range).
        self._leaf_range: int = pow(2, self._level - 1)

        # Compute the stash size.
        self._stash_size: int = stash_scale * (self._level - 1) if self._level > 1 else stash_scale

        # Claim the variables for root, local, stash, and the position map.
        self._root: Optional[ROOT] = None
        self._local: list = []
        self._stash: list = []

        # Compute the padded total data size, which at largest would be [biggest_key, biggest_leaf, biggest_data].
        self._padded_block_size: int = 0

        # Use encryption if required.
        self._cipher: Optional[Aes] = Aes(key=aes_key, key_byte_length=num_key_bytes) if use_encryption else None

        # Initialize the client connection.
        self.client: InteractServer = client

    @property
    def root(self) -> ROOT:
        """Get the root stored in the local."""
        return self._root

    @root.setter
    def root(self, root: ROOT) -> None:
        """Set the root stored in the local."""
        self._root = root

    @abstractmethod
    def update_mul_tree_height(self, num_tree: int) -> None:
        """Suppose the ODS is used to store multiple trees, we update each tree's height.
        
        :param num_tree: number of tree to store, which is the same as number of data in upper level oram.
        """
        raise NotImplementedError

    @abstractmethod
    def _dummy_block_size(self) -> int:
        """Get the number of bytes equal to the size of actual data block stored in ORAM."""
        raise NotImplementedError

    def _encrypt_buckets(self, buckets: Buckets) -> Buckets:
        """Given buckets, encrypt all data in it."""
        return [[self._cipher.enc(plaintext=pad_pickle(data=pickle.dumps(data), length=self._padded_block_size))
                 for data in bucket] for bucket in buckets] if self._use_encryption else buckets

    def _decrypt_buckets(self, buckets: Buckets) -> Buckets:
        """Given encrypted buckets, decrypt all data in it."""
        return [[pickle.loads(unpad_pickle(data=self._cipher.dec(ciphertext=data)))
                 for data in bucket] for bucket in buckets] if self._use_encryption else buckets

    def _get_new_leaf(self) -> int:
        """Get a random leaf label within the range."""
        return secrets.randbelow(self._leaf_range)

    def _evict_stash(self, leaf: int) -> Buckets:
        """
        Evict data blocks in the stash while maintaining correctness.

        :param leaf: the leaf label of the path we are evicting data to.
        :return: The leaf label and the path we should write.
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

        # Note that we return the path in the reversed order because we copy path from bottom up.
        return self._encrypt_buckets(buckets=path[::-1])

    def _perform_dummy_finds(self, num_round: int) -> None:
        """Perform desired number of dummy finds."""
        # Check if the number of round is less than needed.
        if num_round < 0:
            raise ValueError("The height is not enough.")

        # Perform desired number of dummy finds.
        for i in range(num_round):
            # Generate a random path.
            leaf = self._get_new_leaf()

            # Read a path from the ODS storage.
            path = self.client.read_query(label="ods", leaf=leaf)

            # Perform dummy decryption and encryption.
            path = self._encrypt_buckets(buckets=self._decrypt_buckets(buckets=path))

            # Write the path back to the ODS storage.
            self.client.write_query(label="ods", leaf=leaf, data=path)

    def _perform_dummy_eviction(self, num_round: int) -> None:
        """Perform desired number of dummy evictions."""
        # Check if the number of round is less than needed.
        if num_round < 0:
            raise ValueError("The height is not enough.")

        # Perform desired number of dummy evictions.
        for i in range(num_round):
            # Generate a random path.
            leaf = self._get_new_leaf()

            # Read a path from the ODS storage.
            path = self.client.read_query(label="ods", leaf=leaf)

            # Decrypt the path as needed.
            path = self._decrypt_buckets(buckets=path)

            # Add all real data to the stash.
            for bucket in path:
                for data in bucket:
                    if data[KEY] is None:
                        continue
                    else:
                        self._stash.append(data)

            # Check if stash overflows.
            if len(self._stash) > self._stash_size:
                raise MemoryError("Stash overflow!")

            # Evict stash and get a new path.
            path = self._evict_stash(leaf=leaf)

            # Write the path back to the ODS storage.
            self.client.write_query(label="ods", leaf=leaf, data=path)

    @abstractmethod
    def _init_ods_storage(self, data: KV_LIST) -> BinaryTree:
        """
        Initialize a binary tree storage to store the B+ tree holding input key-value pairs.

        :param data: a list of key-value pairs.
        :return: the binary tree storage for the input list of key-value pairs.
        """
        raise NotImplementedError

    def init_server_storage(self, data: KV_LIST = None) -> None:
        """
        Initialize the server storage for the input list of key-value pairs.

        :param data: a list of key-value pairs.
        """
        # Let server store the binary tree.
        self.client.init_query(label="ods", storage=self._init_ods_storage(data=data))

    @abstractmethod
    def _init_mul_tree_ods_storage(self, data_list: List[KV_LIST]) -> Tuple[BinaryTree, List[ROOT]]:
        """
        Initialize a binary tree storage to store multiple B+ trees holding input lists of key-value pairs.

        :param data_list: a list of lists of key-value pairs.
        :return: the binary tree storage for the input list of key-value pairs and a list of B+ tree roots.
        """
        raise NotImplementedError

    def init_mul_tree_server_storage(self, data_list: List[KV_LIST] = None) -> List[ROOT]:
        """
        Send server the tree storage storing multiple AVL trees holding input lists of key-value pairs.

        :param data_list: a list of lists of key-value pairs.:
        :return: a list of AVL tree roots.
        """
        # Initialize the server binary tree storage and get list of roots of AVL trees.
        oram_tree, root_list = self._init_mul_tree_ods_storage(data_list=data_list)
        # Let the server store the binary tree.
        self.client.init_query(label="ods", storage=oram_tree)
        # Return list of roots.
        return root_list

    @abstractmethod
    def insert(self, key: Any, value: Any) -> None:
        """
        Given key-value pair, insert the pair to the tree.

        :param key: the search key of interest.
        :param value: the value to insert.
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value.

        If input value is not None, the value corresponding to the search tree will be updated.
        :param key: the search key of interest.
        :param value: the updated value.
        :return: the (old) value corresponding to the search key.
        """
        raise NotImplementedError
