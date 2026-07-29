"""Path ORAM.

Call ``init_server_storage`` once after construction to build the server storage, then use
``operate_on_key`` to obliviously read/write data points.
"""

from typing import Any, override

from oblivlib.dependency import UNSET, DataMap, OramConfig, ServerStorage
from oblivlib.oram.tree_base_oram import TreeBaseOram


class PathOram[ConfigT: OramConfig](TreeBaseOram[ConfigT]):
    """Path ORAM. Generic over its config type so subclasses (StaticOram, MulPathOram) can
    specialize ``_config`` to their own config while reusing this scheme's logic."""

    def __init__(self, config: ConfigT):
        super().__init__(config)

        self._tmp_leaf: int | None = None

        self._init_pos_map()

    @override
    def init_server_storage(self, data_map: DataMap | None = None) -> None:
        storage: ServerStorage = {self._name: self._init_storage_on_pos_map(data_map=data_map)}
        self._client.init_storage(storage=storage)

    @override
    def operate_on_key(self, key: int, value: Any = UNSET) -> Any:
        leaf = self._look_up_pos_map(key=key)

        new_leaf = self._get_new_leaf()
        self._pos_map[key] = new_leaf

        self._client.add_read_path(label=self._name, leaves=[leaf])
        result = self._client.execute()
        path_data = result.require(self._name)

        read_value = self._retrieve_data_block(key=key, new_leaf=new_leaf, path=path_data, value=value)

        evicted_path = self._evict_stash(leaves=[leaf])
        self._client.add_write_path(label=self._name, data=evicted_path)
        self._client.execute()

        return read_value

    @override
    def operate_on_key_without_eviction(self, key: int, value: Any = UNSET) -> Any:
        leaf = self._look_up_pos_map(key=key)

        new_leaf = self._get_new_leaf()
        self._pos_map[key] = new_leaf

        self._client.add_read_path(label=self._name, leaves=[leaf])
        result = self._client.execute()
        path_data = result.require(self._name)

        read_value = self._retrieve_data_block(key=key, new_leaf=new_leaf, path=path_data, value=value)

        self._tmp_leaf = leaf

        return read_value

    @override
    def eviction_with_update_stash(self, key: int, value: Any, execute: bool = True) -> None:
        found = False
        for data in self._stash:
            if data.key == key:
                data.value = value
                found = True

        if not found:
            raise KeyError(f"Key {key} not found.")

        assert self._tmp_leaf is not None
        evicted_path = self._evict_stash(leaves=[self._tmp_leaf])
        self._client.add_write_path(label=self._name, data=evicted_path)

        if execute:
            self._client.execute()

        self._tmp_leaf = None
