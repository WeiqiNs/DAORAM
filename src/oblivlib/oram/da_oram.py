"""De-Amortized ORAM (DAOram).

The position map is compressed into a chain of position-map orams whose leaves are derived from a
PRF of (key, group count, individual count); de-amortized resets keep counter overflow bounded.
Call ``init_server_storage`` once, then use ``operate_on_key``. Defaults use a 1/64 compression ratio.
"""

import math
import secrets
from dataclasses import replace
from functools import cached_property
from typing import Any, NamedTuple, override

from oblivlib.dependency import Blake2Prf, Data, InitData, PathRows
from oblivlib.dependency.config import DaOramConfig
from oblivlib.oram.bit_string import binary_str_to_bytes, bytes_to_binary_str
from oblivlib.oram.tree_base_oram import TreeBaseOram


class ResetLeaf(NamedTuple):
    """A pending counter reset. ``offset`` is -1 (with a random ``cur_leaf`` and ``new_leaf=None``) when
    no reset is due; otherwise it is the ic offset whose backup leaf moves from ``cur_leaf`` to
    ``new_leaf``. (Named ``offset`` rather than ``index`` because NamedTuple fields may not shadow the
    built-in ``tuple.index`` method.)"""

    offset: int
    cur_leaf: int
    new_leaf: int | None


class ProcessedData(NamedTuple):
    """Result of processing a counter block plus any piggybacked reset: the looked-up key's current
    and new leaf (``cur_leaf``/``new_leaf``), and the reset key's offset/current/new leaf
    (``r_index`` is -1 when no reset is carried)."""

    cur_leaf: int
    new_leaf: int
    r_index: int
    r_cur_leaf: int
    r_new_leaf: int | None


class DAOram(TreeBaseOram[DaOramConfig]):
    def __init__(
        self,
        config: DaOramConfig,
        *,
        _is_pos_map: bool = False,
        _last_oram_data: int | None = None,
        _last_oram_level: int | None = None,
    ):
        if not _is_pos_map:
            if config.client is None:
                raise ValueError("Client is required for main ORAM.")
            if config.num_data <= config.on_chip_mem:
                raise ValueError(
                    f"num_data ({config.num_data}) must be greater than on_chip_mem ({config.on_chip_mem})."
                )

        super().__init__(config)

        self._last_oram_data: int | None = _last_oram_data
        self._last_oram_level: int | None = _last_oram_level

        self._tmp_leaves: list[int] | None = None

        self._prf: Blake2Prf = Blake2Prf(key=config.prf_key)

        self._on_chip_storage: list[bytes] = []
        self._pos_maps: list[DAOram] = []

        self._init_pos_map()

    @property
    def _num_ic(self) -> int:
        return self._config.num_ic

    @property
    def _ic_length(self) -> int:
        return self._config.ic_length

    @property
    def _gc_length(self) -> int:
        return self._config.gc_length

    @property
    def _on_chip_mem(self) -> int:
        return self._config.on_chip_mem

    @cached_property
    def _count_length(self) -> int:
        """Bit length of the counter value a position-map block stores."""
        return self._gc_length + (self._ic_length + 1) * self._num_ic

    @cached_property
    def _num_oram_pos_map(self) -> int:
        """Number of position-map orams needed; the last is kept on chip, hence the -1."""
        return math.ceil(math.log(self._num_data / self._on_chip_mem, self._num_ic)) - 1

    @cached_property
    def _pos_map_oram_dummy_size(self) -> int:
        """Byte size of the dummy value stored in position maps."""
        return math.ceil(self._count_length / 8)

    def _get_pos_map_keys(self, key: int) -> list[tuple[int, int]]:
        """For each position map (outermost first), the (block key, offset within block) for this key."""
        pos_map_keys = []

        for _ in range(self._num_oram_pos_map + 1):
            index = key % self._num_ic
            key = key // self._num_ic
            pos_map_keys.append((key, index))

        pos_map_keys.reverse()

        return pos_map_keys

    def _get_leaf_from_prf(self, key: int, gc: int, ic: int) -> int:
        """Leaf for (key, gc, ic) in this oram, computed as PRF(KEY||GC||IC) mod 2^L."""
        return self._prf.digest_mod_n(
            binary_str_to_bytes(
                bin(key)[2:].zfill(self._level - 1)
                + bin(gc)[2:].zfill(self._gc_length)
                + bin(ic)[2:].zfill(self._ic_length)
            ),
            pow(2, self._level - 1),
        )

    def _get_previous_leaf_from_prf(self, key: int, gc: int, ic: int) -> int:
        """Leaf for (key, gc, ic) in the previous (larger) oram, as PRF(KEY||GC||IC) mod 2^LAST_L."""
        assert self._last_oram_level is not None
        return self._prf.digest_mod_n(
            binary_str_to_bytes(
                bin(key)[2:].zfill(self._last_oram_level - 1)
                + bin(gc)[2:].zfill(self._gc_length)
                + bin(ic)[2:].zfill(self._ic_length)
            ),
            pow(2, self._last_oram_level - 1),
        )

    @override
    def _init_pos_map(self) -> None:
        self._pos_map = {i: self._get_leaf_from_prf(key=i, gc=0, ic=0) for i in range(self._num_data)}

    def _compress_pos_map(self) -> None:
        """Compress the flat position map into a chain of position-map orams, hosting each as it is built."""
        self._pos_map = {}

        value = binary_str_to_bytes("0" * self._count_length)

        last_oram_data = self._num_data
        last_oram_level = self._level

        for i in range(self._num_oram_pos_map):
            pos_map_size = math.ceil(last_oram_data / self._num_ic)

            pos_map_name = f"{self._name}_pos_map_{self._num_oram_pos_map - i - 1}"

            cur_pos_map_oram = DAOram(
                replace(
                    self._config,
                    prf_key=self._prf.key,
                    num_data=pos_map_size,
                    name=pos_map_name,
                    data_size=self._pos_map_oram_dummy_size,
                ),
                _is_pos_map=True,
                _last_oram_data=last_oram_data,
                _last_oram_level=last_oram_level,
            )

            tree = cur_pos_map_oram._build_tree(
                [Data(key=key, leaf=leaf, value=value) for key, leaf in cur_pos_map_oram._pos_map.items()]
            )

            last_oram_data = pos_map_size
            last_oram_level = cur_pos_map_oram._level

            cur_pos_map_oram._host_tree(tree)

            cur_pos_map_oram._pos_map = {}
            self._pos_maps.append(cur_pos_map_oram)

        self._last_oram_data = last_oram_data
        self._last_oram_level = last_oram_level

        self._on_chip_storage = [value for _ in range(math.ceil(last_oram_data / self._num_ic) + 1)]

        self._pos_maps.reverse()

    @override
    def init_server_storage(self, data: InitData | None = None) -> None:
        self._host_tree(self._build_tree(self._initial_blocks(data)))
        self._compress_pos_map()

    def _update_stash_leaf(self, key: int | None, new_leaf: int | None) -> None:
        if key is None:
            return

        for data in self._stash:
            if data.key == key:
                data.leaf = new_leaf
                return

        raise KeyError(f"Key {key} not found.")

    def _perform_reset_on_chip(self, key: int) -> ResetLeaf:
        """Check the on-chip ic indicators for a pending reset; returns (offset, cur_leaf, new_leaf).

        offset is -1 (and a random cur_leaf, None new_leaf) when no reset is due.
        """
        assert self._last_oram_data is not None and self._last_oram_level is not None
        data = bytes_to_binary_str(self._on_chip_storage[key]).zfill(self._count_length)

        ic_indicators = data[-self._num_ic :]
        offset = ic_indicators.find("1")

        if key * self._num_ic + offset >= self._last_oram_data:
            offset = -1

        if offset != -1:
            cur_leaf, new_leaf = self._update_data_on_chip(key=key, offset=offset)
        else:
            cur_leaf, new_leaf = secrets.randbelow(pow(2, self._last_oram_level - 1)), None

        return ResetLeaf(offset, cur_leaf, new_leaf)

    def _perform_reset(self, key: int, data: Data) -> ResetLeaf:
        """Like _perform_reset_on_chip but reads the indicators from a stored Data block."""
        assert self._last_oram_data is not None and self._last_oram_level is not None
        data.value = bytes_to_binary_str(data.value).zfill(self._count_length)

        ic_indicators = data.value[-self._num_ic :]
        offset = ic_indicators.find("1")

        data.value = binary_str_to_bytes(data.value)

        if key * self._num_ic + offset >= self._last_oram_data:
            offset = -1

        if offset != -1:
            cur_leaf, new_leaf = self._update_data(key=key, data=data, offset=offset)
        else:
            cur_leaf, new_leaf = secrets.randbelow(pow(2, self._last_oram_level - 1)), None

        return ResetLeaf(offset, cur_leaf, new_leaf)

    def _update_data_on_chip(self, key: int, offset: int) -> tuple[int, int]:
        """Advance the on-chip counter at offset and return (cur_leaf, new_leaf) for the previous oram."""
        data = bytes_to_binary_str(self._on_chip_storage[key]).zfill(self._count_length)

        gc = int(data[: self._gc_length], 2)

        ic_start = self._gc_length + offset * self._ic_length
        ic_end = self._gc_length + (offset + 1) * self._ic_length
        ic = int(data[ic_start:ic_end], 2)

        ic_ind_start = self._gc_length + self._num_ic * self._ic_length

        if data[ic_ind_start + offset] == "1":
            next_ic = 0
            next_gc = gc
            gc = gc - 1
            data = f"{data[: ic_ind_start + offset]}{'0'}{data[ic_ind_start + offset + 1 :]}"

        elif ic + 1 >= pow(2, self._ic_length):
            next_ic = 0
            next_gc = gc + 1
            data = (
                f"{bin(next_gc)[2:].zfill(self._gc_length)}"
                f"{data[self._gc_length : -self._num_ic]}"
                f"{'1' * offset + '0' + '1' * (self._num_ic - offset - 1)}"
            )

        else:
            next_ic = ic + 1
            next_gc = gc

        self._on_chip_storage[key] = binary_str_to_bytes(
            f"{data[:ic_start]}{bin(next_ic)[2:].zfill(self._ic_length)}{data[ic_end:]}"
        )

        cur_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic)
        new_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=next_gc, ic=next_ic)

        return cur_leaf, new_leaf

    def _update_data(self, key: int, data: Data, offset: int) -> tuple[int, int]:
        """Like _update_data_on_chip but operates on a stored Data block's counter value."""
        data.value = bytes_to_binary_str(data.value).zfill(self._count_length)

        gc = int(data.value[: self._gc_length], 2)

        ic_start = self._gc_length + offset * self._ic_length
        ic_end = self._gc_length + (offset + 1) * self._ic_length
        ic = int(data.value[ic_start:ic_end], 2)

        ic_ind_start = self._gc_length + self._num_ic * self._ic_length

        if data.value[ic_ind_start + offset] == "1":
            next_ic = 0
            next_gc = gc
            gc = gc - 1
            data.value = f"{data.value[: ic_ind_start + offset]}{'0'}{data.value[ic_ind_start + offset + 1 :]}"

        elif ic + 1 >= pow(2, self._ic_length):
            next_ic = 0
            next_gc = gc + 1
            data.value = (
                f"{bin(next_gc)[2:].zfill(self._gc_length)}"
                f"{data.value[self._gc_length : -self._num_ic]}"
                f"{'1' * offset + '0' + '1' * (self._num_ic - offset - 1)}"
            )

        else:
            next_ic = ic + 1
            next_gc = gc

        data.value = binary_str_to_bytes(
            f"{data.value[:ic_start]}{bin(next_ic)[2:].zfill(self._ic_length)}{data.value[ic_end:]}"
        )

        cur_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic)
        new_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=next_gc, ic=next_ic)

        return cur_leaf, new_leaf

    def _retrieve_pos_map_block_with_reset(
        self, key: int, offset: int, new_leaf: int, r_key: int | None, r_new_leaf: int | None, path: PathRows
    ) -> ProcessedData:
        """Pull the path into the stash, advance key's counter and remap it to new_leaf, and remap the
        optional reset key r_key to r_new_leaf. The returned reset fields are r_index=-1, a random cur
        leaf, and None new leaf when no reset is carried to the next level."""
        self._absorb_path(path=path)

        data = self._require_in_stash(key=key)
        next_cur_leaf, next_new_leaf = self._update_data(key=key, data=data, offset=offset)
        r_index, r_next_cur_leaf, r_next_new_leaf = self._perform_reset(key=key, data=data)
        data.leaf = new_leaf

        self._update_stash_leaf(key=r_key, new_leaf=r_new_leaf)

        return ProcessedData(next_cur_leaf, next_new_leaf, r_index, r_next_cur_leaf, r_next_new_leaf)

    def _access_pos_map_level(
        self, cur_key: int, cur_index: int, new_leaf: int, r_key: int | None, r_new_leaf: int | None, leaves: list[int]
    ) -> ProcessedData:
        """Drive this position-map level's own server I/O and process its block plus any reset.

        Reads the data path and reset path on the shared client (under its own name), updates the
        counter for ``cur_key`` (and remaps the optional reset key ``r_key``), then evicts and writes
        the paths back. Returns ``(next_cur_leaf, next_new_leaf, r_index, r_cur_leaf, r_new_leaf)``.
        """
        self._client.add_read_path(label=self._name, leaves=leaves)
        result = self._client.execute()
        path = result.require(self._name)

        next_cur_leaf, next_new_leaf, r_index, r_cur_leaf, r_new_leaf = self._retrieve_pos_map_block_with_reset(
            key=cur_key, r_key=r_key, offset=cur_index, new_leaf=new_leaf, r_new_leaf=r_new_leaf, path=path
        )

        evicted_path = self._evict_stash(leaves=leaves)

        self._client.add_write_path(label=self._name, data=evicted_path)
        self._client.execute()

        return ProcessedData(next_cur_leaf, next_new_leaf, r_index, r_cur_leaf, r_new_leaf)

    def _get_leaf_from_pos_map(self, key: int) -> ProcessedData:
        """Walk the position-map chain to find key's leaf and new leaf, performing resets along the way."""
        pos_map_keys = self._get_pos_map_keys(key=key)

        cur_leaf, new_leaf = self._update_data_on_chip(key=pos_map_keys[0][0], offset=pos_map_keys[0][1])

        r_index, r_cur_leaf, r_new_leaf = self._perform_reset_on_chip(key=pos_map_keys[0][0])

        for pos_map_index, (cur_key, cur_index) in enumerate(pos_map_keys[1:]):
            r_key = None if r_index == -1 else cur_key // self._num_ic * self._num_ic + r_index

            leaves = [cur_leaf, r_cur_leaf]

            next_cur_leaf, next_new_leaf, r_index, r_cur_leaf, r_new_leaf = self._pos_maps[
                pos_map_index
            ]._access_pos_map_level(
                cur_key=cur_key,
                cur_index=cur_index,
                new_leaf=new_leaf,
                r_key=r_key,
                r_new_leaf=r_new_leaf,
                leaves=leaves,
            )

            cur_leaf, new_leaf = next_cur_leaf, next_new_leaf

        return ProcessedData(cur_leaf, new_leaf, r_index, r_cur_leaf, r_new_leaf)

    @override
    def _operate_on_key(self, key: int, value: Any) -> Any:
        cur_leaf, new_leaf, r_index, r_cur_leaf, r_new_leaf = self._get_leaf_from_pos_map(key=key)

        r_key = None if r_index == -1 else key // self._num_ic * self._num_ic + r_index

        leaves = [cur_leaf, r_cur_leaf]

        self._client.add_read_path(label=self._name, leaves=leaves)
        result = self._client.execute()
        path = result.require(self._name)

        read_value = self._retrieve_data_block(key=key, value=value, new_leaf=new_leaf, path=path)

        self._update_stash_leaf(key=r_key, new_leaf=r_new_leaf)

        evicted_path = self._evict_stash(leaves=leaves)

        self._client.add_write_path(label=self._name, data=evicted_path)
        self._client.execute()

        return read_value

    @override
    def _operate_on_key_without_eviction(self, key: int, value: Any) -> Any:
        cur_leaf, new_leaf, r_index, r_cur_leaf, r_new_leaf = self._get_leaf_from_pos_map(key=key)

        r_key = None if r_index == -1 else key // self._num_ic * self._num_ic + r_index

        leaves = [cur_leaf, r_cur_leaf]

        self._client.add_read_path(label=self._name, leaves=leaves)
        result = self._client.execute()
        path = result.require(self._name)

        read_value = self._retrieve_data_block(key=key, value=value, new_leaf=new_leaf, path=path)

        self._update_stash_leaf(key=r_key, new_leaf=r_new_leaf)

        self._tmp_leaves = leaves

        return read_value

    @override
    def _eviction_with_update_stash(self, key: int, value: Any, execute: bool) -> None:
        found = False

        for data in self._stash:
            if data.key == key:
                data.value = value
                found = True

        if not found:
            raise KeyError(f"Key {key} not found.")

        assert self._tmp_leaves is not None
        evicted_path = self._evict_stash(leaves=self._tmp_leaves)

        self._client.add_write_path(label=self._name, data=evicted_path)
        if execute:
            self._client.execute()

        self._tmp_leaves = None
