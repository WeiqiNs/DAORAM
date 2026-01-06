"""
This module defines the path oram class.

Path oram has two public methods:
    - init_storage_on_pos_map: this should be called first after the class object is created. This method constructs the
        storage the server should hold for the client.
    - operate_on_key: after the server gets the created storage, the client can use this function to obliviously access
        data points stored in the storage.
"""
from typing import Any, Optional

from daoram.dependency import InteractServer, Encryptor, UNSET
from daoram.oram.tree_base_oram import TreeBaseOram


class PathOram(TreeBaseOram):
    def __init__(self,
                 num_data: int,
                 data_size: int,
                 client: InteractServer,
                 name: str = "po",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 encryptor: Encryptor = None):
        """
        Defines the path oram, including its attributes and methods.

        :param name: The name of the protocol, this should be unique if multiple schemes are used together.
        :param client: The instance we use to interact with server.
        :param num_data: The number of data points the oram should store.
        :param filename: The filename to save the oram data to.
        :param encryptor: The encryptor to use for encryption.
        :param data_size: The number of bytes the random dummy data should have.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        """
        # Initialize the parent BaseOram class.
        super().__init__(
            name=name,
            client=client,
            num_data=num_data,
            filename=filename,
            encryptor=encryptor,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale
        )

        # This attribute is used to store a leaf temporarily for reading a path without evicting it immediately.
        self._tmp_leaf: Optional[int] = None

        # In path oram, we initialize the position map.
        self._init_pos_map()

    def init_server_storage(self, data_map: dict = None) -> None:
        """
        Initialize the server storage based on the data map for this oram.

        :param data_map: A dictionary storing {key: data}.
        """
        # Get the storage.
        storage = {self._name: self._init_storage_on_pos_map(data_map=data_map)}

        # Initialize the storage and send it to the server.
        self.client.init_storage(storage=storage)

    def operate_on_key(self, key: int, value: Any = UNSET) -> Any:
        """
        Perform operation on a given key. Always returns the current value.
        If value is provided (not UNSET), writes the new value.

        :param key: The key of the data block of interest.
        :param value: If provided (not UNSET), write this value to the data block.
        :return: The current value of the data block (before write if writing).
        """
        # Find which path the data of interest lies on.
        leaf = self._look_up_pos_map(key=key)

        # Generate a new leaf and update the position map.
        new_leaf = self._get_new_leaf()
        self._pos_map[key] = new_leaf

        # Read the path from the server.
        self.client.add_read_path(label=self._name, leaves=[leaf])
        result = self.client.execute()
        path_data = result.results[self._name]

        # Retrieve value from the path, and optionally write to it.
        read_value = self._retrieve_data_block(key=key, new_leaf=new_leaf, path=path_data, value=value)

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
        # Find which path the data of interest lies on.
        leaf = self._look_up_pos_map(key=key)

        # Generate a new leaf and update the position map.
        new_leaf = self._get_new_leaf()
        self._pos_map[key] = new_leaf

        # Read the path from the server.
        self.client.add_read_path(label=self._name, leaves=[leaf])
        result = self.client.execute()
        path_data = result.results[self._name]

        # Retrieve value from the path, and optionally write to it.
        read_value = self._retrieve_data_block(key=key, new_leaf=new_leaf, path=path_data, value=value)

        # Temporarily save the leaf for future eviction.
        self._tmp_leaf = leaf

        return read_value

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
        evicted_path = self._evict_stash(leaves=[self._tmp_leaf])

        # Write the path back to the server.
        self.client.add_write_path(label=self._name, data=evicted_path)
        self.client.execute()

        # Set temporary leaf to None.
        self._tmp_leaf = None
