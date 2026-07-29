"""OMAP combining an ORAM with an Oblivious Search Tree (the VLDB 2025 framework).

Each logical key is hashed (via a PRF) to a slot in the ORAM, which stores the root pointer of that
slot's ODS tree. An op fetches the root from the ORAM, runs the ODS op against it, then writes the
(possibly updated) root back -- so any TreeBaseOram composes with any OstBaseOmap.
"""

from typing import Any, override

from oblivlib.dependency import Blake2Prf, Helper
from oblivlib.dependency.config import OramOstOmapConfig
from oblivlib.omap.base_omap import BaseOmap
from oblivlib.omap.ost_base_omap import OstBaseOmap
from oblivlib.oram.tree_base_oram import TreeBaseOram


class OramOstOmap(BaseOmap):
    def __init__(self, config: OramOstOmapConfig, ost: OstBaseOmap[Any, Any], oram: TreeBaseOram):
        # The frozen config is the single source of truth for the construction parameters.
        self._config: OramOstOmapConfig = config
        self._ost: OstBaseOmap[Any, Any] = ost
        self._oram: TreeBaseOram = oram

        # The ODS only needs its per-tree height adjusted for the number of trees it now holds.
        self._ost.update_mul_tree_height(num_tree=self._num_data)

        # PRF used to hash input keys into the ORAM.
        self._prf: Blake2Prf = Blake2Prf()

    @property
    def _num_data(self) -> int:
        return self._config.num_data

    @override
    def init_server_storage(self, data: list[tuple[str | int | bytes, Any]] | None = None) -> None:
        if data is None:
            data = []

        # Group the pairs by their hashed ORAM slot, one value-list per slot.
        data_map = Helper.hash_data_to_map(prf=self._prf, data=data, map_size=self._num_data)
        data_list = [data_map[key] for key in range(self._num_data)]

        # Build one ODS tree per slot, then store each tree's root in the ORAM.
        roots = self._ost.init_mul_tree_server_storage(data_list=data_list)
        self._oram.init_server_storage(data_map={key: root for key, root in enumerate(roots)})

    @override
    def search(self, key: str | int | bytes, value: Any = None) -> Any:
        """Search for ``key``, writing ``value`` first when given; returns the old value."""
        oram_key = Helper.hash_data_to_leaf(prf=self._prf, data=key, map_size=self._num_data)
        # Fetch the slot's ODS root, run the op, write the (possibly updated) root back.
        root = self._oram.operate_on_key_without_eviction(key=oram_key)
        self._ost.root = root
        value = self._ost.search(key=key, value=value)
        self._oram.eviction_with_update_stash(key=oram_key, value=self._ost.root)
        return value

    @override
    def insert(self, key: str | int | bytes, value: Any):
        oram_key = Helper.hash_data_to_leaf(prf=self._prf, data=key, map_size=self._num_data)
        root = self._oram.operate_on_key_without_eviction(key=oram_key)
        self._ost.root = root
        self._ost.insert(key=key, value=value)
        self._oram.eviction_with_update_stash(key=oram_key, value=self._ost.root)
