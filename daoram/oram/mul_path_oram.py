"""
This module defines the multi-path oram class.

MulPathOram extends PathOram to support batch operations where multiple paths
are read and written at once, improving efficiency for operations that need
to access multiple keys.
"""
from typing import Any, Dict, List, Optional, Tuple

from daoram.dependency import BinaryTree, Data, Encryptor, InteractServer, PathData, UNSET
from daoram.oram.path_oram import PathOram


class MulPathOram(PathOram):
    def __init__(self,
                 num_data: int,
                 data_size: int,
                 client: InteractServer,
                 name: str = "mp",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 encryptor: Encryptor = None):
        """
        Defines the multi-path oram for batch operations.

        :param num_data: The number of data points the oram should store.
        :param data_size: The number of bytes the random dummy data should have.
        :param client: The instance we use to interact with server.
        :param name: The name of the protocol, this should be unique if multiple schemes are used together.
        :param filename: The filename to save the oram data to.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param encryptor: The encryptor to use for encryption.
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
            stash_scale=stash_scale
        )

        # Store temporary leaves for batch operations without immediate eviction.
        self._tmp_leaves: List[int] = []

    @property
    def stash(self) -> list:
        """Return the stash."""
        return self._stash

    @stash.setter
    def stash(self, value: list):
        """Update the stash with input."""
        self._stash = value

    def _retrieve_mul_data_blocks(
        self,
        key_leaf_map: Dict[int, int],
        path: PathData,
        values: Dict[int, Any] = None
    ) -> Dict[int, Any]:
        """
        Retrieve multiple data blocks from the path. If values provided, write them.

        :param key_leaf_map: Dict mapping key to its new leaf.
        :param path: PathData dict mapping storage index to bucket.
        :param values: If provided, dict mapping key to value to write.
        :return: Dict mapping key to its current value.
        """
        # Track which keys we've found and their values.
        found_keys = set()
        read_values: Dict[int, Any] = {}

        # Store the current stash length for searching stash later.
        to_index = len(self._stash)

        # Decrypt the path if needed.
        path = self._decrypt_path_data(path=path)

        # Read all buckets in the path and add real data to stash.
        for bucket in path.values():
            for data in bucket:
                # If dummy data, skip it.
                if data.key is None:
                    continue

                # Check if this is one of the keys we're looking for.
                if data.key in key_leaf_map:
                    # Read the current value.
                    read_values[data.key] = data.value
                    # Write if value is provided.
                    if values and data.key in values:
                        data.value = values[data.key]
                    # Update the leaf.
                    data.leaf = key_leaf_map[data.key]
                    found_keys.add(data.key)

                # Add all real data to the stash.
                self._stash.append(data)

        # Check if the stash overflows.
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")

        # Check stash for any keys not found in path.
        for key, new_leaf in key_leaf_map.items():
            if key not in found_keys:
                # Search in the existing stash (before we added new data).
                value_to_write = values.get(key, UNSET) if values else UNSET
                read_values[key] = self._retrieve_data_stash(
                    key=key, to_index=to_index, new_leaf=new_leaf, value=value_to_write
                )

        return read_values

    def operate_on_keys(
        self,
        key_value_map: Dict[int, Any] = None,
        keys: List[int] = None
    ) -> Dict[int, Any]:
        """
        Perform batch operations on multiple keys. Reads all paths at once,
        performs operations, and evicts all paths at once.

        Either provide key_value_map for read/write operations, or keys for read-only.

        :param key_value_map: Dict mapping key to value to write. Use UNSET for read-only.
        :param keys: List of keys to read (alternative to key_value_map for read-only).
        :return: Dict mapping key to its value (before write if writing).
        """
        # Build the key list from either parameter.
        if key_value_map is not None:
            key_list = list(key_value_map.keys())
            values = {k: v for k, v in key_value_map.items() if v is not UNSET}
        elif keys is not None:
            key_list = keys
            values = None
        else:
            raise ValueError("Must provide either key_value_map or keys")

        if not key_list:
            return {}

        # Look up current leaves and generate new leaves for all keys.
        old_leaves: List[int] = []
        key_leaf_map: Dict[int, int] = {}

        for key in key_list:
            # Find which path the data lies on.
            old_leaf = self._look_up_pos_map(key=key)
            old_leaves.append(old_leaf)

            # Generate a new leaf and update position map.
            new_leaf = self._get_new_leaf()
            self._pos_map[key] = new_leaf
            key_leaf_map[key] = new_leaf

        # Read all paths at once.
        self._client.add_read_path(label=self._name, leaves=old_leaves)
        result = self._client.execute()
        path_data = result.results[self._name]

        # Retrieve values from paths and optionally write to them.
        read_values = self._retrieve_mul_data_blocks(
            key_leaf_map=key_leaf_map, path=path_data, values=values
        )

        # Evict stash to all paths at once.
        evicted_path = self._evict_stash(leaves=old_leaves)

        # Write all paths back.
        self._client.add_write_path(label=self._name, data=evicted_path)
        self._client.execute()

        return read_values

    def operate_on_keys_without_eviction(
        self,
        key_value_map: Dict[int, Any] = None,
        keys: List[int] = None
    ) -> Dict[int, Any]:
        """
        Perform batch operations on multiple keys without eviction.
        Call eviction_for_mul_keys() later to complete the operation.

        :param key_value_map: Dict mapping key to value to write. Use UNSET for read-only.
        :param keys: List of keys to read (alternative to key_value_map for read-only).
        :return: Dict mapping key to its value (before write if writing).
        """
        # Build the key list from either parameter.
        if key_value_map is not None:
            key_list = list(key_value_map.keys())
            values = {k: v for k, v in key_value_map.items() if v is not UNSET}
        elif keys is not None:
            key_list = keys
            values = None
        else:
            raise ValueError("Must provide either key_value_map or keys")

        if not key_list:
            return {}

        # Look up current leaves and generate new leaves for all keys.
        old_leaves: List[int] = []
        key_leaf_map: Dict[int, int] = {}

        for key in key_list:
            # Find which path the data lies on.
            old_leaf = self._look_up_pos_map(key=key)
            old_leaves.append(old_leaf)

            # Generate a new leaf and update position map.
            new_leaf = self._get_new_leaf()
            self._pos_map[key] = new_leaf
            key_leaf_map[key] = new_leaf

        # Read all paths at once.
        self._client.add_read_path(label=self._name, leaves=old_leaves)
        result = self._client.execute()
        path_data = result.results[self._name]

        # Retrieve values from paths and optionally write to them.
        read_values = self._retrieve_mul_data_blocks(
            key_leaf_map=key_leaf_map, path=path_data, values=values
        )

        # Store leaves for later eviction.
        self._tmp_leaves = old_leaves

        return read_values

    def eviction_for_mul_keys(self, updates: Dict[int, Any] = None) -> None:
        """
        Complete the batch operation by updating stash and evicting.

        :param updates: Optional dict mapping key to new value to update in stash.
        """
        # Apply any updates to stash.
        if updates:
            for key, value in updates.items():
                for data in self._stash:
                    if data.key == key:
                        data.value = value
                        break

        # Evict stash to all paths.
        evicted_path = self._evict_stash(leaves=self._tmp_leaves)

        # Write all paths back.
        self._client.add_write_path(label=self._name, data=evicted_path)
        self._client.execute()

        # Clear temporary leaves.
        self._tmp_leaves = []
