"""
This module defines the static oram class, which inherits from PathOram.

StaticOram uses fixed leaf positions based on a PRF of the key, rather than randomly reassigning leaves on each access.
"""
import random
from typing import Any, Optional

from daoram.dependency import BinaryTree, Data, Encryptor, InteractServer, PseudoRandomFunction, Blake2Prf, UNSET
from daoram.oram.path_oram import PathOram


class StaticOram(PathOram):
    def __init__(self,
                 num_data: int,
                 data_size: int,
                 client: InteractServer,
                 name: str = "so",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 encryptor: Encryptor = None,
                 prf: PseudoRandomFunction = None):
        """
        Defines the static oram, which inherits from PathOram but doesn't update leaf positions.

        :param num_data: The number of data points the oram should store.
        :param data_size: The number of bytes the random dummy data should have.
        :param client: The instance we use to interact with server.
        :param name: The name of the protocol, this should be unique if multiple schemes are used together.
        :param filename: The filename to save the oram data to.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param encryptor: The encryptor to use for encryption.
        :param prf: The PRF for generating fixed leaf positions; a Blake2Prf is created if not provided.
        """
        # Initialize the parent PathOram class.
        super().__init__(
            name=name,
            client=client,
            num_data=num_data,
            filename=filename,
            encryptor=encryptor,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
        )

        # PRF for generating fixed path numbers from keys.
        self._prf = prf if prf else Blake2Prf()

    def _get_path_number(self, key: Optional[int]) -> int:
        """
        Generate a fixed path number for the given key using PRF.

        :param key: The key of the data, or None to get a random path.
        :return: Fixed path number for this key, or random if key is None.
        """
        # If key is None, return a random leaf for dummy accesses.
        if key is None:
            return random.randint(0, pow(2, self._level - 1) - 1)
        return self._prf.digest_mod_n(str(key).encode(), pow(2, self._level - 1))

    def _init_storage_on_pos_map(self, data_map: dict = None) -> BinaryTree:
        """
        Initialize a binary tree storage based on the data map.
        Overrides parent to use _get_path_number() for fixed leaf positions.

        :param data_map: A dictionary storing {key: data}.
        :return: The binary tree storage based on the data map.
        """
        # Create the binary tree object.
        tree = BinaryTree(
            filename=self._filename,
            num_data=self._num_data,
            data_size=self._dumped_data_size,
            bucket_size=self._bucket_size,
            disk_size=self._disk_size,
            encryption=True if self._encryptor else False,
        )

        # Fill data blocks using fixed paths for all keys.
        for key in range(self._num_data):
            leaf = self._get_path_number(key)
            value = data_map.get(key) if data_map else None
            tree.fill_data_to_storage_leaf(data=Data(key=key, leaf=leaf, value=value))

        # Encrypt the tree storage if needed.
        if self._encryptor:
            tree.storage.encrypt(encryptor=self._encryptor)

        return tree

    def operate_on_key(self, key: int, value: Any = UNSET) -> Any:
        """
        Perform operation on a given key using fixed leaf position.

        :param key: The key of the data block of interest.
        :param value: If provided (not UNSET), write this value to the data block.
        :return: The current value of the data block (before write if writing).
        """
        # Get the fixed path for this key (doesn't change).
        leaf = self._get_path_number(key)

        # Read the path from the server.
        self.client.add_read_path(label=self._name, leaves=[leaf])
        result = self.client.execute()
        path_data = result.results[self._name]

        # Retrieve value from the path - use same leaf as new_leaf since position is fixed.
        read_value = self._retrieve_data_block(key=key, new_leaf=leaf, path=path_data, value=value)

        # Perform an eviction and get a new path.
        evicted_path = self._evict_stash(leaves=[leaf])

        # Write the path back to the server.
        self.client.add_write_path(label=self._name, data=evicted_path)
        self.client.execute()

        return read_value

    def operate_on_key_without_eviction(self, key: int, value: Any = UNSET) -> Any:
        """
        Perform operation on a given key without writing the data added to the stash back to the server.

        :param key: The key of the data block of interest.
        :param value: If provided (not UNSET), write this value to the data block.
        :return: The current value of the data block (before write if writing).
        """
        # Get the fixed path for this key.
        leaf = self._get_path_number(key)

        # Read the path from the server.
        self.client.add_read_path(label=self._name, leaves=[leaf])
        result = self.client.execute()
        path_data = result.results[self._name]

        # Retrieve value from the path - use same leaf as new_leaf since position is fixed.
        read_value = self._retrieve_data_block(key=key, new_leaf=leaf, path=path_data, value=value)

        # Temporarily save the leaf for future eviction.
        self._tmp_leaf = leaf

        return read_value
