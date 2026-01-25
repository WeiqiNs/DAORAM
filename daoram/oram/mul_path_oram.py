"""
This module defines the multi-path oram class.

MulPathOram extends PathOram to support batch operations where multiple paths are read and written at once,
improving efficiency for operations that need to access multiple keys.
"""
from typing import Any, Dict, List

from daoram.dependency import Encryptor, InteractServer, PathData, UNSET
from daoram.oram.path_oram import PathOram


class MulPathOram(PathOram):
    def __init__(self,
                 num_data: int,
                 data_size: int,
                 client: InteractServer,
                 name: str = "mul_path_oram",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 stash_scale_multiplier: int = 1,
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
        :param stash_scale_multiplier: Multiplier for the stash size (default 1).
        :param encryptor: The encryptor to use for encryption.
        """
        # Initialize the parent PathOram class.
        super().__init__(
            num_data=num_data,
            data_size=data_size,
            client=client,
            name=name,
            filename=filename,
            bucket_size=bucket_size,
            stash_scale=stash_scale * stash_scale_multiplier,
            encryptor=encryptor,
        )

        # Store temporary leaves for batch operations without immediate eviction.
        self._tmp_leaves: List[int] = []

    def _init_pos_map(self) -> None:
        """Initialize an empty position map to support noninteger keys."""
        self._pos_map = {}

    def _look_up_pos_map(self, key: Any) -> int:
        """Look up key and get the leaf for where the data is stored.

        :param key: A key of a data block (can be any hashable type).
        :return: The corresponding leaf if found.
        """
        if key not in self._pos_map:
            raise KeyError(f"Key {key} not found in position map.")
        return self._pos_map[key]

    def init_server_storage(self, data_map: Dict[Any, Any] = None, path_map: Dict[Any, int] = None) -> None:
        """
        Initialize the server storage based on the data map for this oram.

        :param data_map: A dictionary storing {key: data}.
        :param path_map: Optional dictionary mapping {key: leaf}. If provided, uses these
                         keys and leaves. If not provided, falls back to integer keys (0 to num_data-1).
        """
        if path_map is not None:
            # Use provided path_map for noninteger keys (can be empty)
            for key, leaf in path_map.items():
                self._pos_map[key] = leaf
        else:
            # Fall back to integer keys for backwards compatibility
            self._pos_map = {i: self._get_new_leaf() for i in range(self._num_data)}

        # Call parent implementation
        super().init_server_storage(data_map=data_map)

    def queue_read(self, leaves: List[int]) -> None:
        """Queue path read and store leaves for later eviction."""
        self._client.add_read_path(label=self._name, leaves=leaves)
        self._tmp_leaves = leaves

    def queue_write(self, leaves: List[int] = None) -> None:
        """Evict stash and queue path write. Uses stored leaves if not specified."""
        evict_leaves = leaves if leaves is not None else self._tmp_leaves
        evicted_path = self._evict_stash(leaves=evict_leaves)
        self._client.add_write_path(label=self._name, data=evicted_path)
        self._tmp_leaves = []

    def process_read_result(self, result: Any) -> None:
        """
        Process the read result from the client and add data to stash.

        :param result: The ExecuteResult object from client.execute().
        """
        if self._name not in result.results:
            return

        path_data = result.results[self._name]

        # Decrypt the path if needed.
        path = self.decrypt_path_data(path=path_data)

        # Read all buckets in the path and add real data to stash.
        for bucket in path.values():
            for data in bucket:
                # If dummy data, skip it.
                if data.key is None:
                    continue

                # Add all real data to the stash.
                self._stash.append(data)

        # Check if the stash overflows.
        if len(self._stash) > self._stash_size:
            raise OverflowError(
                f"Stash overflow in {self._name}: size {len(self._stash)} exceeds max {self._stash_size}.")

    def _retrieve_mul_data_blocks(
            self,
            path: PathData,
            key_leaf_map: Dict[Any, int],
            values: Dict[Any, Any] = None,
    ) -> Dict[Any, Any]:
        """
        Retrieve multiple data blocks from the path. If values provided, write them.

        :param path: PathData dict mapping storage index to bucket.
        :param key_leaf_map: Dict mapping key to its new leaf.
        :param values: If provided, dict mapping key to value to write.
        :return: Dict mapping key to its current value.
        """
        # Track which keys we've found and their values.
        found_keys = set()
        read_values: Dict[Any, Any] = {}

        # Store the current stash length for searching stash later.
        to_index = len(self._stash)

        # Decrypt the path if needed.
        path = self.decrypt_path_data(path=path)

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
            raise OverflowError(
                f"Stash overflow in {self._name}: size {len(self._stash)} exceeds max {self._stash_size}.")

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
            key_value_map: Dict[Any, Any],
            key_path_map: Dict[Any, int] = None,
            new_path_map: Dict[Any, int] = None,
    ) -> Dict[Any, Any]:
        """
        Perform batch operations on multiple keys. Reads all paths at once,
        performs operations, and evicts all paths at once.

        :param key_value_map: Dict mapping key to value to write. Use UNSET for read-only.
        :param key_path_map: Optional dict mapping key to old path (where to read from).
            If None, paths are retrieved from the position map.
        :param new_path_map: Optional dict mapping key to new path (where to write to).
            If None, new paths are generated randomly.
        :return: Dict mapping key to its value (before write if writing).
        """
        key_list = list(key_value_map.keys())
        values = {k: v for k, v in key_value_map.items() if v is not UNSET}

        if not key_list:
            return {}

        # Look up current leaves and determine new leaves for all keys.
        old_leaves: List[int] = []
        key_leaf_map: Dict[Any, int] = {}

        for key in key_list:
            # Find which path the data lies on.
            if key_path_map is not None:
                old_leaf = key_path_map[key]
            else:
                old_leaf = self._look_up_pos_map(key=key)
            old_leaves.append(old_leaf)

            # Determine new leaf: use provided map or generate randomly.
            if new_path_map is not None and key in new_path_map:
                new_leaf = new_path_map[key]
            else:
                new_leaf = self._get_new_leaf()
            self._pos_map[key] = new_leaf
            key_leaf_map[key] = new_leaf

        # Read all paths at once.
        self._client.add_read_path(label=self._name, leaves=old_leaves)
        result = self._client.execute()
        path_data = result.results[self._name]

        # Retrieve values from paths and optionally write to them.
        read_values = self._retrieve_mul_data_blocks(
            path=path_data, key_leaf_map=key_leaf_map, values=values
        )

        # Evict stash to all paths at once.
        evicted_path = self._evict_stash(leaves=old_leaves)

        # Write all paths back.
        self._client.add_write_path(label=self._name, data=evicted_path)
        self._client.execute()

        return read_values

    def operate_on_keys_without_eviction(
            self,
            key_value_map: Dict[Any, Any],
            key_path_map: Dict[Any, int] = None,
            new_path_map: Dict[Any, int] = None,
    ) -> Dict[Any, Any]:
        """
        Perform batch operations on multiple keys without eviction.
        Call eviction_for_mul_keys() later to complete the operation.

        :param key_value_map: Dict mapping key to value to write. Use UNSET for read-only.
        :param key_path_map: Optional dict mapping key to old path (where to read from).
            If None, paths are retrieved from the position map.
        :param new_path_map: Optional dict mapping key to new path (where to write to).
            If None, new paths are generated randomly.
        :return: Dict mapping key to its value (before write if writing).
        """
        key_list = list(key_value_map.keys())
        values = {k: v for k, v in key_value_map.items() if v is not UNSET}

        if not key_list:
            return {}

        # Look up current leaves and determine new leaves for all keys.
        old_leaves: List[int] = []
        key_leaf_map: Dict[Any, int] = {}

        for key in key_list:
            # Find which path the data lies on.
            if key_path_map is not None:
                old_leaf = key_path_map[key]
            else:
                old_leaf = self._look_up_pos_map(key=key)
            old_leaves.append(old_leaf)

            # Determine new leaf: use provided map or generate randomly.
            if new_path_map is not None and key in new_path_map:
                new_leaf = new_path_map[key]
            else:
                new_leaf = self._get_new_leaf()
            self._pos_map[key] = new_leaf
            key_leaf_map[key] = new_leaf

        # Read all paths at once.
        self._client.add_read_path(label=self._name, leaves=old_leaves)
        result = self._client.execute()
        path_data = result.results[self._name]

        # Retrieve values from paths and optionally write to them.
        read_values = self._retrieve_mul_data_blocks(
            path=path_data, key_leaf_map=key_leaf_map, values=values
        )

        # Store leaves for later eviction.
        self._tmp_leaves = old_leaves

        return read_values

    def eviction_for_mul_keys(self, updates: Dict[Any, Any] = None, execute: bool = True) -> None:
        """
        Complete the batch operation by updating stash and evicting.

        :param updates: Optional dict mapping key to new value to update in stash.
        :param execute: If True, execute immediately. If False, queue write for batching.
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

        # Add write to client queue.
        self._client.add_write_path(label=self._name, data=evicted_path)

        # Execute if requested.
        if execute:
            self._client.execute()

        # Clear temporary leaves.
        self._tmp_leaves = []
