"""Static ORAM: a PathOram whose leaf positions are fixed by a PRF of the key rather than
reassigned on each access."""

from typing import Any, override

from oblivlib.dependency import Blake2Prf, PseudoRandomFunction
from oblivlib.dependency.config import StaticOramConfig
from oblivlib.oram.path_oram import PathOram


class StaticOram(PathOram[StaticOramConfig]):
    def __init__(self, config: StaticOramConfig, prf: PseudoRandomFunction | None = None):
        super().__init__(config)
        self._prf = prf if prf else Blake2Prf()

    def _get_path_number(self, key: int | None) -> int:
        """Fixed PRF-derived leaf for key, or a random leaf when key is None (dummy access)."""
        if key is None:
            return self._get_new_leaf()
        return self._prf.digest_mod_n(str(key).encode(), pow(2, self._level - 1))

    @override
    def _initial_leaf(self, key: int) -> int:
        return self._get_path_number(key)

    @override
    def _operate_on_key(self, key: int, value: Any) -> Any:
        leaf = self._get_path_number(key)

        self._client.add_read_path(label=self._name, leaves=[leaf])
        result = self._client.execute()
        path_data = result.require(self._name)

        read_value = self._retrieve_data_block(key=key, new_leaf=leaf, path=path_data, value=value)

        evicted_path = self._evict_stash(leaves=[leaf])
        self._client.add_write_path(label=self._name, data=evicted_path)
        self._client.execute()

        return read_value

    @override
    def _operate_on_key_without_eviction(self, key: int, value: Any) -> Any:
        leaf = self._get_path_number(key)

        self._client.add_read_path(label=self._name, leaves=[leaf])
        result = self._client.execute()
        path_data = result.require(self._name)

        read_value = self._retrieve_data_block(key=key, new_leaf=leaf, path=path_data, value=value)

        self._tmp_leaf = leaf

        return read_value
