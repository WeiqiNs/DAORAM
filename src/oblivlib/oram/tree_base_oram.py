"""Abstract base class for binary-tree-based ORAMs.

Adds the ORAM-specific layer (position map, eviction, the operate_on_key protocol) on top of the
shared ``TreeStorageBase`` (config accessors, level/stash math, random leaves, path encryption).
Private members use single underscores (not name-mangled ``__``) so subclasses can access them.
"""

import os
from abc import ABC, abstractmethod
from typing import Any

from oblivlib.dependency import UNSET, BinaryTree, Data, DataMap, OramConfig, PathData, PosMap
from oblivlib.dependency.tree_storage_base import TreeStorageBase


class TreeBaseOram[ConfigT: OramConfig](TreeStorageBase[ConfigT], ABC):
    def __init__(self, config: ConfigT):
        super().__init__(config)
        self._pos_map: PosMap = {}
        self._max_stash: int = 0

    @property
    def total_stash_size(self) -> int:
        """Stash size including the stashes of any recursive position-map orams."""
        # Flat schemes define no _pos_maps, so the sum is empty and this is just stash_size.
        pos_maps: list[TreeBaseOram] = getattr(self, "_pos_maps", [])
        return self.stash_size + sum(pos_map.stash_size for pos_map in pos_maps)

    @property
    def max_stash(self) -> int:
        """Peak stash size observed, including any recursive position-map orams, for sizing client
        storage. (An upper bound: the per-oram peaks need not occur at the same time.)"""
        pos_maps: list[TreeBaseOram] = getattr(self, "_pos_maps", [])
        return self._max_stash + sum(pos_map.max_stash for pos_map in pos_maps)

    def _init_pos_map(self) -> None:
        self._pos_map = {i: self._get_new_leaf() for i in range(self._num_data)}

    def _look_up_pos_map(self, key: int) -> int:
        if key not in self._pos_map:
            raise KeyError(f"Key {key} not found in position map.")

        return self._pos_map[key]

    def _init_storage_on_pos_map(self, data_map: DataMap | None = None) -> BinaryTree:
        """Build the binary tree storage from the position map (and optional {key: data} map)."""
        tree = BinaryTree(
            filename=self._filename,
            num_data=self._num_data,
            disk_size=self._disk_size,
            bucket_size=self._bucket_size,
            data_size=self._dumped_data_size,
            encryption=self._encryptor is not None,
        )

        for key, leaf in self._pos_map.items():
            value = data_map[key] if data_map else os.urandom(self._data_size)
            tree.fill_data_to_storage_leaf(data=Data(key=key, leaf=leaf, value=value))

        if self._encryptor:
            tree.storage.encrypt(encryptor=self._encryptor)

        return tree

    def _evict_stash(self, leaves: list[int]) -> PathData:
        """Evict stash blocks onto the given paths; blocks that don't fit stay in the stash."""
        temp_stash = []

        path = BinaryTree.get_mul_path_dict(level=self._level, indices=leaves)

        for data in self._stash:
            inserted = BinaryTree.fill_data_to_path(
                data=data, path=path, leaves=leaves, level=self._level, bucket_size=self._bucket_size
            )
            if not inserted:
                temp_stash.append(data)

        self._stash = temp_stash

        return self._encrypt_path_data(path=path)

    def _check_stash(self) -> None:
        """Record this oram's peak stash size and raise if the stash overflowed. Called right after a
        path is read into the stash -- its largest point, before eviction shrinks it again."""
        self._max_stash = max(self._max_stash, self.stash_size)
        if self.stash_size > self._stash_size:
            raise MemoryError("Stash overflow!")

    def _retrieve_data_stash(self, key: int, to_index: int, new_leaf: int, value: Any = UNSET) -> Any:
        """Find key in the stash (searched after the path); read it, optionally write value, remap to new_leaf."""
        for data in self._stash[:to_index]:
            if data.key == key:
                read_value = data.value
                if value is not UNSET:
                    data.value = value
                data.leaf = new_leaf
                return read_value

        raise KeyError(f"Key {key} not found.")

    def _retrieve_data_block(self, key: int, new_leaf: int, path: PathData, value: Any = UNSET) -> Any:
        """Pull the path into the stash, read key (optionally writing value), and remap it to new_leaf."""
        found = False
        read_value = None
        to_index = len(self._stash)

        decrypted = self._decrypt_path_data(path=path)

        for bucket in decrypted.values():
            for data in bucket:
                if data.key is None:
                    continue
                elif data.key == key:
                    read_value = data.value
                    if value is not UNSET:
                        data.value = value
                    data.leaf = new_leaf
                    found = True
                self._stash.append(data)

        self._check_stash()

        if not found:
            read_value = self._retrieve_data_stash(key=key, to_index=to_index, value=value, new_leaf=new_leaf)

        return read_value

    @abstractmethod
    def init_server_storage(self, data_map: DataMap | None = None) -> None:
        """Initialize the server storage for this oram from an optional {key: data} map."""
        raise NotImplementedError

    @abstractmethod
    def operate_on_key(self, key: int, value: Any = UNSET) -> Any:
        """Read key and return its current value; if value is not UNSET, write it."""
        raise NotImplementedError

    @abstractmethod
    def operate_on_key_without_eviction(self, key: int, value: Any = UNSET) -> Any:
        """Like operate_on_key but defers writing the stash back to the server."""
        raise NotImplementedError

    @abstractmethod
    def eviction_with_update_stash(self, key: int, value: Any, execute: bool = True) -> None:
        """Update key's block in the stash then evict; if execute is False, queue the write."""
        raise NotImplementedError
