"""Abstract base class for binary-tree-based ORAMs.

Adds the ORAM-specific layer (position map, eviction, the operate_on_key protocol) on top of the
shared ``TreeStorageBase`` (config accessors, level/stash math, random leaves, path encryption).
Private members use single underscores (not name-mangled ``__``) so subclasses can access them.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, override

from oblivlib.dependency import UNSET, Data, InitData, OramConfig, PathRows, PosMap
from oblivlib.dependency.contract import require_oram_key, require_value
from oblivlib.dependency.tree_storage_base import TreeStorageBase


def _initial_pairs(data: InitData | None) -> Iterable[tuple[int, Any]]:
    if data is None:
        return ()
    if isinstance(data, Mapping):
        mapping: Mapping[Any, Any] = data
        return mapping.items()
    return data


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

    def _initial_leaf(self, key: int) -> int:
        return self._pos_map[key]

    def _initial_blocks(self, data: InitData | None) -> Iterator[Data]:
        supplied = bytearray(self._num_data)
        for key, value in _initial_pairs(data):
            require_oram_key(self._identity, key, self._num_data)
            require_value(self._identity, value, self._data_size)
            if supplied[key]:
                raise ValueError(f"{self._identity}: initial key {key} is given twice.")
            supplied[key] = 1
            yield Data(key=key, leaf=self._initial_leaf(key), value=value)
        for key in range(self._num_data):
            if not supplied[key]:
                yield Data(key=key, leaf=self._initial_leaf(key), value=b"")

    @override
    def _check_stash(self) -> None:
        """Record this oram's peak stash size (called when the stash is largest) and raise on overflow."""
        self._max_stash = max(self._max_stash, self.stash_size)
        super()._check_stash()

    def _absorb_path(self, path: PathRows) -> None:
        for bucket in self._cipher.open_path(path).values():
            self._stash.extend(data for data in bucket if data.key is not None)
        self._check_stash()

    def _require_in_stash(self, key: int) -> Data:
        for data in self._stash:
            if data.key == key:
                return data
        raise KeyError(f"Key {key} not found.")

    def _retrieve_data_block(self, key: int, new_leaf: int, path: PathRows, value: Any = UNSET) -> Any:
        """Pull the path into the stash, read key (optionally writing value), and remap it to new_leaf."""
        self._absorb_path(path=path)
        data = self._require_in_stash(key=key)
        read_value = data.value
        if value is not UNSET:
            data.value = value
        data.leaf = new_leaf
        return read_value

    @abstractmethod
    def init_server_storage(self, data: InitData | None = None) -> None:
        """Build and host this oram's server storage. ``data`` gives initial values, as a mapping or a
        one-shot stream of ``(key, value)`` pairs; every other key starts as ``b""``."""
        raise NotImplementedError

    def _require_access(self, key: Any, value: Any) -> None:
        require_oram_key(self._identity, key, self._num_data)
        if value is not UNSET:
            require_value(self._identity, value, self._data_size)

    def operate_on_key(self, key: int, value: Any = UNSET) -> bytes:
        """Return key's current value (``b""`` until written); write ``value`` when one is given."""
        self._require_access(key, value)
        return self._operate_on_key(key, value)

    def operate_on_key_without_eviction(self, key: int, value: Any = UNSET) -> bytes:
        """Like ``operate_on_key`` but leaves the path's write-back to ``eviction_with_update_stash``."""
        self._require_access(key, value)
        return self._operate_on_key_without_eviction(key, value)

    def eviction_with_update_stash(self, key: int, value: bytes, execute: bool = True) -> None:
        """Set key's value in the stash, then evict; with ``execute=False`` the write-back is only staged."""
        require_oram_key(self._identity, key, self._num_data)
        require_value(self._identity, value, self._data_size)
        self._eviction_with_update_stash(key, value, execute)

    @abstractmethod
    def _operate_on_key(self, key: int, value: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _operate_on_key_without_eviction(self, key: int, value: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _eviction_with_update_stash(self, key: int, value: Any, execute: bool) -> None:
        raise NotImplementedError
