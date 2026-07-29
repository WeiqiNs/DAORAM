"""Multi-path ORAM: extends PathOram to read and evict several paths in one batch."""

from dataclasses import replace
from typing import Any, override

from oblivlib.dependency import UNSET, DataMap, PathData, PosMap
from oblivlib.dependency.config import MulPathOramConfig
from oblivlib.oram.path_oram import PathOram


class MulPathOram(PathOram[MulPathOramConfig]):
    def __init__(self, config: MulPathOramConfig):
        # stash_scale_multiplier widens the stash for the larger batch; bake it into stash_scale.
        super().__init__(replace(config, stash_scale=config.stash_scale * config.stash_scale_multiplier))

        self._tmp_leaves: list[int] = []

    @override
    def init_server_storage(self, data_map: DataMap | None = None, path_map: PosMap | None = None) -> None:
        """``path_map`` overrides the random position-map leaves for the given keys."""
        if path_map:
            for key, leaf in path_map.items():
                self._pos_map[key] = leaf

        super().init_server_storage(data_map=data_map)

    def _retrieve_mul_data_blocks(
        self,
        path: PathData,
        key_leaf_map: PosMap,
        values: DataMap | None = None,
    ) -> DataMap:
        """Read every key's block off the path, remap each to ``key_leaf_map[key]``, optionally write
        ``values``; returns {key: current value}. Keys not on the path are read from the prior stash."""
        found_keys = set()
        read_values: dict[int, Any] = {}

        to_index = len(self._stash)

        decrypted = self._decrypt_path_data(path=path)

        for bucket in decrypted.values():
            for data in bucket:
                if data.key is None:
                    continue

                if data.key in key_leaf_map:
                    read_values[data.key] = data.value
                    if values and data.key in values:
                        data.value = values[data.key]
                    data.leaf = key_leaf_map[data.key]
                    found_keys.add(data.key)

                self._stash.append(data)

        self._check_stash()

        for key, new_leaf in key_leaf_map.items():
            if key not in found_keys:
                value_to_write = values.get(key, UNSET) if values else UNSET
                read_values[key] = self._retrieve_data_stash(
                    key=key, to_index=to_index, new_leaf=new_leaf, value=value_to_write
                )

        return read_values

    def _read_keys(
        self,
        key_value_map: dict[int, Any],
        key_path_map: dict[int, int] | None,
        new_path_map: dict[int, int] | None,
    ) -> tuple[dict[int, Any], list[int]]:
        """Read every requested key's path in one batch, remap each to a new leaf, apply any writes;
        returns ({key: value before write}, old_leaves). old_leaves is empty for an empty request."""
        values = {k: v for k, v in key_value_map.items() if v is not UNSET}
        if not key_value_map:
            return {}, []

        old_leaves: list[int] = []
        key_leaf_map: dict[int, int] = {}
        for key in key_value_map:
            old_leaves.append(key_path_map[key] if key_path_map is not None else self._look_up_pos_map(key=key))
            new_leaf = new_path_map[key] if new_path_map is not None and key in new_path_map else self._get_new_leaf()
            self._pos_map[key] = new_leaf
            key_leaf_map[key] = new_leaf

        self._client.add_read_path(label=self._name, leaves=old_leaves)
        path_data = self._client.execute().require(self._name)
        read_values = self._retrieve_mul_data_blocks(path=path_data, key_leaf_map=key_leaf_map, values=values)
        return read_values, old_leaves

    def operate_on_keys(
        self,
        key_value_map: dict[int, Any],
        key_path_map: dict[int, int] | None = None,
        new_path_map: dict[int, int] | None = None,
    ) -> dict[int, Any]:
        """Batch read/write: read all paths at once, operate, then evict all paths at once.

        ``key_value_map`` is {key: value to write} (UNSET for read-only); ``key_path_map`` overrides the
        leaves read from (default: the position map); ``new_path_map`` overrides the new leaves (default:
        random). Returns {key: value before any write}.
        """
        read_values, old_leaves = self._read_keys(key_value_map, key_path_map, new_path_map)
        if not old_leaves:
            return {}

        evicted_path = self._evict_stash(leaves=old_leaves)
        self._client.add_write_path(label=self._name, data=evicted_path)
        self._client.execute()
        return read_values

    def operate_on_keys_without_eviction(
        self,
        key_value_map: dict[int, Any],
        key_path_map: dict[int, int] | None = None,
        new_path_map: dict[int, int] | None = None,
    ) -> dict[int, Any]:
        """Like operate_on_keys but defers eviction to a later eviction_for_mul_keys call."""
        read_values, old_leaves = self._read_keys(key_value_map, key_path_map, new_path_map)
        if not old_leaves:
            return {}

        self._tmp_leaves = old_leaves
        return read_values

    def eviction_for_mul_keys(self, updates: dict[int, Any] | None = None, execute: bool = True) -> None:
        """Apply optional {key: value} updates to the stash, then evict the batch's paths."""
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
