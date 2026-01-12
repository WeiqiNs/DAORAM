"""
Module for defining a parent class for tree-structure ODS omap.

The BaseOmap class defines a set of attributes that an omap should have and some basic methods such as encryption, etc.
"""

import math
import os
import secrets
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from daoram.dependency import BinaryTree, Data, Encryptor, Helper, InteractServer, PathData

# Root contains the data key and the path.
ROOT = Tuple[Any, int]
# Define the input key-value pair types.
KV_LIST = List[Tuple[Any, Any]]


class ObliviousSearchTree(ABC):
    def __init__(self,
                 name: str,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 client: InteractServer,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 filename: str = None,
                 encryptor: Encryptor = None):
        """
        Defines the base omap, including its attributes and methods.

        :param name: The name of the protocol, this should be unique if multiple schemes are used together.
        :param num_data: The number of data points the oram should store.
        :param key_size: The number of bytes the random key should have.
        :param data_size: The number of bytes the random dummy data should have.
        :param client: The instance we use to interact with the server.
        :param filename: The filename to save the oram data to.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param encryptor: The encryptor to use for encryption.
        """
        # Store the useful input values.
        self._name: str = name
        self._filename: str = filename
        self._num_data: int = num_data
        self._key_size: int = key_size
        self._data_size: int = data_size
        self._bucket_size: int = bucket_size
        self._encryptor: Encryptor = encryptor

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

        # Set dumped data size and disk size for use of encryption or file storage.
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

    @property
    def root(self) -> ROOT:
        """Get the root stored in the local."""
        return self._root

    @root.setter
    def root(self, root: ROOT) -> None:
        """Set the root stored in the local."""
        self._root = root

    @property
    def client(self) -> InteractServer:
        """Return the client object."""
        return self._client

    @property
    def stash_size(self) -> int:
        """Return the stash size."""
        return len(self._stash)

    @abstractmethod
    def update_mul_tree_height(self, num_tree: int) -> None:
        """Suppose the ODS is used to store multiple trees, we update each tree's height.
        
        :param num_tree: Number of trees to store, which is the same as number of data in upper level oram.
        """
        raise NotImplementedError

    @abstractmethod
    def _max_block_size(self) -> int:
        """Get the number of bytes equal to the size of actual data block stored in ORAM."""
        raise NotImplementedError

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

    def _find_in_stash(self, key: Any) -> int:
        """
        Check if a node with the given key is in the stash.

        :param key: The key to search for.
        :return: Index in stash if found, -1 otherwise.
        """
        for i, node in enumerate(self._stash):
            if node.key == key:
                return i
        return -1

    def _flush_local_to_stash(self) -> None:
        """Move all nodes from local to stash and clear local."""
        self._stash += self._local
        self._local = []

    def _move_node_to_local(self, key: Any, leaf: int) -> None:
        """
        Given key and path, retrieve the path and move the data block corresponding to the leaf to local.

        :param key: Search key of interest.
        :param leaf: Indicate which path the data of interest is stored in the ORAM.
        """
        # Move the node to local without doing eviction.
        self._move_node_to_local_without_eviction(key=key, leaf=leaf)

        # Perform eviction and write the path back.
        self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=[leaf]))
        self._client.execute()

    def _move_node_to_local_without_eviction(self, key: Any, leaf: Any) -> None:
        """
        Given key and path, retrieve the path and move the data block corresponding to the leaf to local.

        This is specialized in the fast search method, where the data of interest are moved to local and the rest
        are added to the stash.
        :param key: Search key of interest.
        :param leaf: Indicate which path the data of interest is stored in the ORAM.
        """
        # Set found to false and get existing stash size.
        found = False
        to_index = len(self._stash)

        # Read the path from the server.
        self._client.add_read_path(label=self._name, leaves=[leaf])
        result = self._client.execute()
        path_data = result.results[self._name]

        # Decrypt the path.
        path = self._decrypt_path_data(path=path_data)

        # Find the desired data in the path.
        for bucket in path.values():
            for data in bucket:
                if data.key == key:
                    # We append the data we want to stash.
                    self._local.append(data)
                    found = True
                else:
                    # Other real data are directly added to the stash.
                    self._stash.append(data)

        # Check if stash overflows.
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")

        # If the desired data is not found in the path, we check the stash.
        if not found:
            stash_idx = self._find_in_stash(key)
            if stash_idx >= 0 and stash_idx < to_index:
                # Found in stash - move to local and remove from stash.
                self._local.append(self._stash[stash_idx])
                del self._stash[stash_idx]
                return

            # If also not found in the stash, raise an error.
            raise KeyError(f"The search key {key} is not found.")

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

    def _perform_dummy_operation(self, num_round: int) -> None:
        """Perform the desired number of dummy evictions."""
        # Check if the number of rounds is lower than needed.
        if num_round < 0:
            raise ValueError("The height is not enough, as the number of dummy operation required is negative.")

        # Perform the desired number of dummy evictions.
        for _ in range(num_round):
            # Generate a random path.
            leaf = self._get_new_leaf()

            # Read a path from the ODS storage.
            self._client.add_read_path(label=self._name, leaves=[leaf])
            result = self._client.execute()
            path_data = result.results[self._name]

            # Decrypt the path.
            path = self._decrypt_path_data(path=path_data)

            # Add all real data to the stash.
            for bucket in path.values():
                for data in bucket:
                    self._stash.append(data)

            # Check if stash overflows.
            if len(self._stash) > self._stash_size:
                raise MemoryError("Stash overflow!")

            # Evict the stash and write the path back to the ODS storage.
            self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=[leaf]))
            self._client.execute()

    @abstractmethod
    def _init_ods_storage(self, data: KV_LIST) -> BinaryTree:
        """
        Initialize a binary tree storage to store the B+ tree holding input key-value pairs.

        :param data: A list of key-value pairs.
        :return: The binary tree storage for the input list of key-value pairs.
        """
        raise NotImplementedError

    def init_server_storage(self, data: KV_LIST = None) -> None:
        """
        Initialize the server storage for the input list of key-value pairs.

        :param data: A list of key-value pairs.
        """
        # Let the server store the binary tree.
        self._client.init_storage(storage={self._name: self._init_ods_storage(data=data)})

    @abstractmethod
    def _init_mul_tree_ods_storage(self, data_list: List[KV_LIST]) -> Tuple[BinaryTree, List[ROOT]]:
        """
        Initialize a binary tree storage to store multiple B+ trees holding input lists of key-value pairs.

        :param data_list: A list of lists of key-value pairs.
        :return: The binary tree storage for the input list of key-value pairs and a list of B+ tree roots.
        """
        raise NotImplementedError

    def init_mul_tree_server_storage(self, data_list: List[KV_LIST] = None) -> List[ROOT]:
        """
        Send server the tree storage storing multiple AVL trees holding input lists of key-value pairs.

        :param data_list: A list of lists of key-value pairs.:
        :return: a list of AVL tree roots.
        """
        # Initialize the server binary tree storage and get a list of roots of AVL trees.
        tree, root_list = self._init_mul_tree_ods_storage(data_list=data_list)
        # Let the server store the binary tree.
        self._client.init_storage(storage={self._name: tree})
        # Return list of roots.
        return root_list

    @abstractmethod
    def insert(self, key: Any, value: Any) -> None:
        """
        Given key-value pair, insert the pair to the tree.

        :param key: The search key of interest.
        :param value: The value to insert.
        """
        raise NotImplementedError

    @abstractmethod
    def search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value.

        If the input value is not None, the value corresponding to the search tree will be updated.
        :param key: The search key of interest.
        :param value: The updated value.
        :return: The (old) value corresponding to the search key.
        """
        raise NotImplementedError

    @abstractmethod
    def fast_search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value.

        Note: This search is allowed to be distinguished from insert and can be sped up.
        The difference here is that fast search will return the node immediately without keeping it in local.
        If the input value is not None, the value corresponding to the search tree will be updated.
        :param key: The search key of interest.
        :param value: The updated value.
        :return: The (old) value corresponding to the search key.
        """
        raise NotImplementedError
