"""Multi-path ORAM: extends PathOram to read and evict several paths in one batch."""

from dataclasses import replace
from typing import Any

from oblivlib.dependency import UNSET, Data, DataMap, PathRows, PosMap
from oblivlib.dependency.config import MulPathOramConfig
from oblivlib.dependency.contract import require_oram_key, require_value
from oblivlib.oram.path_oram import PathOram


class MulPathOram(PathOram[MulPathOramConfig]):
    def __init__(self, config: MulPathOramConfig):
        super().__init__(replace(config, stash_scale=config.stash_scale * config.stash_scale_multiplier))

        self._tmp_leaves: list[int] = []

    def _retrieve_mul_data_blocks(
        self,
        path: PathRows,
        key_leaf_map: dict[Any, int],
        values: dict[Any, Any] | None = None,
    ) -> dict[Any, Any]:
        """Pull the path into the stash, then read every key's block, remap each to ``key_leaf_map[key]``,
        and optionally write ``values``; returns {key: value before any write}."""
        self._absorb_path(path=path)

        read_values: dict[Any, Any] = {}
        for key, new_leaf in key_leaf_map.items():
            data = self._require_in_stash(key=key)
            read_values[key] = data.value
            if values and key in values:
                data.value = values[key]
            data.leaf = new_leaf

        return read_values

    def _read_keys(
        self,
        key_value_map: dict[Any, Any],
        key_path_map: dict[Any, int] | None,
        new_path_map: dict[Any, int] | None,
    ) -> tuple[dict[Any, Any], list[int]]:
        """Read every requested key's path in one batch, remap each to a new leaf, apply any writes;
        returns ({key: value before write}, old_leaves). old_leaves is empty for an empty request."""
        values = {k: v for k, v in key_value_map.items() if v is not UNSET}
        if not key_value_map:
            return {}, []

        old_leaves: list[int] = []
        key_leaf_map: dict[Any, int] = {}
        for key in key_value_map:
            old_leaves.append(key_path_map[key] if key_path_map is not None else self._look_up_pos_map(key=key))
            new_leaf = new_path_map[key] if new_path_map is not None and key in new_path_map else self._get_new_leaf()
            self._pos_map[key] = new_leaf
            key_leaf_map[key] = new_leaf

        self._client.add_read_path(label=self._name, leaves=old_leaves)
        path_data = self._client.execute().require(self._name)
        read_values = self._retrieve_mul_data_blocks(path=path_data, key_leaf_map=key_leaf_map, values=values)
        return read_values, old_leaves

    def _check_batch(self, key_value_map: dict[int, Any]) -> None:
        for key, value in key_value_map.items():
            self._require_access(key, value)

    def operate_on_keys(
        self,
        key_value_map: DataMap,
        key_path_map: PosMap | None = None,
        new_path_map: PosMap | None = None,
    ) -> DataMap:
        """Batch read/write: read all paths at once, operate, then evict all paths at once.

        ``key_value_map`` is {key: value to write} (UNSET for read-only); ``key_path_map`` overrides the
        leaves read from (default: the position map); ``new_path_map`` overrides the new leaves (default:
        random). Returns {key: value before any write}.
        """
        self._check_batch(key_value_map)
        return self._operate_on_keys(key_value_map, key_path_map, new_path_map)

    def _operate_on_keys(
        self,
        key_value_map: dict[Any, Any],
        key_path_map: dict[Any, int] | None = None,
        new_path_map: dict[Any, int] | None = None,
    ) -> dict[Any, Any]:
        read_values, old_leaves = self._read_keys(key_value_map, key_path_map, new_path_map)
        if not old_leaves:
            return {}

        evicted_path = self._evict_stash(leaves=old_leaves)
        self._client.add_write_path(label=self._name, data=evicted_path)
        self._client.execute()
        return read_values

    def operate_on_keys_without_eviction(
        self,
        key_value_map: DataMap,
        key_path_map: PosMap | None = None,
        new_path_map: PosMap | None = None,
    ) -> DataMap:
        """Like operate_on_keys but defers eviction to a later eviction_for_mul_keys call."""
        self._check_batch(key_value_map)
        return self._operate_on_keys_without_eviction(key_value_map, key_path_map, new_path_map)

    def _operate_on_keys_without_eviction(
        self,
        key_value_map: dict[Any, Any],
        key_path_map: dict[Any, int] | None = None,
        new_path_map: dict[Any, int] | None = None,
    ) -> dict[Any, Any]:
        read_values, old_leaves = self._read_keys(key_value_map, key_path_map, new_path_map)
        if not old_leaves:
            return {}

        self._tmp_leaves = old_leaves
        return read_values

    def eviction_for_mul_keys(self, updates: DataMap | None = None, execute: bool = True) -> None:
        """Apply optional {key: value} updates to the stash, then evict the batch's paths."""
        for key, value in (updates or {}).items():
            require_oram_key(self._identity, key, self._num_data)
            require_value(self._identity, value, self._data_size)
        self._eviction_for_mul_keys(updates, execute)

    def _eviction_for_mul_keys(self, updates: dict[Any, Any] | None = None, execute: bool = True) -> None:
        if updates:
            for key, value in updates.items():
                for data in self._stash:
                    if data.key == key:
                        data.value = value
                        break

        evicted_path = self._evict_stash(leaves=self._tmp_leaves)

        self._client.add_write_path(label=self._name, data=evicted_path)

        if execute:
            self._client.execute()

        self._tmp_leaves = []

    def _insert_block(self, block: Data) -> None:
        """Add a block that is not stored yet: read one random path, stash the block, and evict that path,
        so the server sees exactly a single-key access."""
        leaf = self._get_new_leaf()
        self._client.add_read_path(label=self._name, leaves=[leaf])
        self._absorb_path(self._client.execute().require(self._name))
        self._stash.append(block)
        self._check_stash()
        self._client.add_write_path(label=self._name, data=self._evict_stash(leaves=[leaf]))
        self._client.execute()
