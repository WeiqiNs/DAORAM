"""OMAP combining an ORAM with an Oblivious Search Tree (the VLDB 2025 framework).

Each logical key is hashed (via a PRF) to a slot in the ORAM, which stores the root pointer of that
slot's ODS tree. An op fetches the root from the ORAM, runs the ODS op against it, then writes the
(possibly updated) root back -- so any TreeBaseOram composes with any OstBaseOmap.
"""

from typing import Any, override

import msgpack

from oblivlib.dependency import Blake2Prf, hash_data_to_leaf, hash_data_to_map
from oblivlib.dependency.config import OramOstOmapConfig
from oblivlib.dependency.contract import require_omap_key, require_value
from oblivlib.omap.base_omap import BaseOmap
from oblivlib.omap.ost_base_omap import ROOT, OstBaseOmap
from oblivlib.oram.tree_base_oram import TreeBaseOram

_MAX_PACKED_UINT = 2**64 - 1


def _encode_root(root: ROOT | None) -> bytes:
    return b"" if root is None else msgpack.packb(list(root))


def _decode_root(value: bytes) -> ROOT | None:
    if value == b"":
        return None
    key, leaf = msgpack.unpackb(value)
    return key, leaf


class OramOstOmap(BaseOmap):
    def __init__(self, config: OramOstOmapConfig, ost: OstBaseOmap[Any, Any], oram: TreeBaseOram):
        required = self.oram_data_size(ost._key_size)
        if oram._data_size < required:
            raise ValueError(
                f"OramOstOmap ORAM data_size={oram._data_size} is too small for an ODS root pointer; it must be at "
                + f"least {required}. Size it with OramOstOmap.oram_data_size(key_size)."
            )

        self._config: OramOstOmapConfig = config
        self._ost: OstBaseOmap[Any, Any] = ost
        self._oram: TreeBaseOram = oram

        self._ost.update_mul_tree_height(num_tree=self._num_data)

        self._prf: Blake2Prf = Blake2Prf()

    @staticmethod
    def oram_data_size(key_size: int) -> int:
        """Minimum ``data_size`` of the ORAM, which stores each slot's encoded root pointer."""
        return len(_encode_root((bytes(key_size), _MAX_PACKED_UINT)))

    @property
    def _num_data(self) -> int:
        return self._config.num_data

    @property
    def _identity(self) -> str:
        return f"{type(self).__name__} {self._ost._name!r}"

    @override
    def init_server_storage(self, data: list[tuple[bytes, bytes]] | None = None) -> None:
        for key, value in data or []:
            require_omap_key(self._identity, key, self._ost._key_size)
            require_value(self._identity, value, self._ost._data_size)

        data_map = hash_data_to_map(prf=self._prf, data=data or [], map_size=self._num_data)
        data_list = [data_map[key] for key in range(self._num_data)]

        roots = self._ost.init_mul_tree_server_storage(data_list=data_list)
        self._oram.init_server_storage((key, _encode_root(root)) for key, root in enumerate(roots))

    @override
    def search(self, key: bytes, value: bytes | None = None) -> bytes | None:
        """Search for ``key``, writing ``value`` first when given; returns the old value."""
        require_omap_key(self._identity, key, self._ost._key_size)
        if value is not None:
            require_value(self._identity, value, self._ost._data_size)
        oram_key = hash_data_to_leaf(prf=self._prf, data=key, map_size=self._num_data)
        self._ost.root = _decode_root(self._oram.operate_on_key_without_eviction(key=oram_key))
        old_value = self._ost._search(key, value)
        self._oram.eviction_with_update_stash(key=oram_key, value=_encode_root(self._ost.root))
        return old_value

    @override
    def insert(self, key: bytes, value: bytes) -> None:
        require_omap_key(self._identity, key, self._ost._key_size)
        require_value(self._identity, value, self._ost._data_size)
        oram_key = hash_data_to_leaf(prf=self._prf, data=key, map_size=self._num_data)
        self._ost.root = _decode_root(self._oram.operate_on_key_without_eviction(key=oram_key))
        self._ost._insert(key, value)
        self._oram.eviction_with_update_stash(key=oram_key, value=_encode_root(self._ost.root))
