"""Abstract base class for binary-tree-based ORAMs.

Adds the ORAM-specific layer (position map, eviction, the operate_on_key protocol) on top of the
shared ``TreeStorageBase`` (config accessors, level/stash math, random leaves, path encryption).
Private members use single underscores (not name-mangled ``__``) so subclasses can access them.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, override

from oblivlib.dependency import UNSET, Data, DataMap, OramConfig, PathData, PosMap
from oblivlib.dependency.tree_storage_base import TreeStorageBase


class TreeBaseOram[ConfigT: OramConfig](TreeStorageBase[ConfigT], ABC):
    def __init__(self, config: ConfigT):
        super().__init__(config)
        self._pos_map: PosMap = {}
        self._max_stash: int = 0

    @property
    def total_stash_size(self) -> int:
        """Stash size including the stashes of any recursive position-map orams."""
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

    def _initial_blocks(self, data_map: DataMap | None = None) -> list[Data]:
        return [
            Data(key=key, leaf=leaf, value=data_map[key] if data_map else os.urandom(self._data_size))
            for key, leaf in self._pos_map.items()
        ]

    @override
    def _check_stash(self) -> None:
        """Record this oram's peak stash size (called when the stash is largest) and raise on overflow."""
        self._max_stash = max(self._max_stash, self.stash_size)
        super()._check_stash()

    def _absorb_path(self, path: PathData) -> None:
        for bucket in self._decrypt_path_data(path=path).values():
            self._stash.extend(data for data in bucket if data.key is not None)
        self._check_stash()

    def _require_in_stash(self, key: int) -> Data:
        for data in self._stash:
            if data.key == key:
                return data
        raise KeyError(f"Key {key} not found.")

    def _retrieve_data_block(self, key: int, new_leaf: int, path: PathData, value: Any = UNSET) -> Any:
        """Pull the path into the stash, read key (optionally writing value), and remap it to new_leaf."""
        self._absorb_path(path=path)
        data = self._require_in_stash(key=key)
        read_value = data.value
        if value is not UNSET:
            data.value = value
        data.leaf = new_leaf
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
