"""Freecursive ORAM.

Like DAOram, the position map is compressed into a chain of position-map orams with PRF-derived
leaves, but counter resets are triggered probabilistically (``reset_method="prob"``) or on overflow
(``reset_method="hard"``). Call ``init_server_storage`` once, then use ``operate_on_key``.
"""

import math
import random
import secrets
from dataclasses import replace
from functools import cached_property
from typing import Any, NamedTuple, override

from oblivlib.dependency import UNSET, BinaryTree, Blake2Prf, Data, DataMap, Helper, PathData, ServerStorage
from oblivlib.dependency.config import FreecursiveOramConfig
from oblivlib.oram.tree_base_oram import TreeBaseOram


class ResetEntry(NamedTuple):
    """One counter's reset: (key, cur_leaf, new_leaf). key/new_leaf are None for a padding slot (a
    leaf past the end of the data, where only a random cur_leaf is read to hide the boundary)."""

    key: int | None
    cur_leaf: int
    new_leaf: int | None


ResetChunk = list[ResetEntry]  # a (1- or 2-long) group of reset entries read together
ResetLeaves = list[ResetChunk]  # the full reset plan for a block: all of its chunks


class ProcessedData(NamedTuple):
    """Counter-update result: (cur_leaf, new_leaf, reset_leaves). cur_leaf/new_leaf are None when the
    update triggered a reset, in which case the work to do is carried in reset_leaves instead."""

    cur_leaf: int | None
    new_leaf: int | None
    reset_leaves: ResetLeaves | None


class UnpackedReset(NamedTuple):
    """A reset chunk flattened for the two-path access: the (<=2) entries' keys/leaves plus the
    leaves list to read this round."""

    ck_a: int | None
    cl_a: int
    nl_a: int | None
    ck_b: int | None
    cl_b: int | None
    nl_b: int | None
    leaves: list[int]


class FreecursiveOram(TreeBaseOram[FreecursiveOramConfig]):
    def __init__(
        self,
        config: FreecursiveOramConfig,
        *,
        _is_pos_map: bool = False,
        _last_oram_data: int | None = None,
        _last_oram_level: int | None = None,
    ):
        # _is_pos_map marks an internal position-map child oram (shares the parent's client); the
        # _last_oram_* args carry the dimensions of the oram directly above it.
        if not _is_pos_map:
            if config.client is None:
                raise ValueError("Client is required for main ORAM.")
            if config.num_data <= config.on_chip_size:
                raise ValueError(
                    f"num_data ({config.num_data}) must be greater than on_chip_size ({config.on_chip_size})."
                )

        super().__init__(config)

        # Runtime state tracking the oram above this one; reassigned in _compress_pos_map.
        self._last_oram_data = _last_oram_data
        self._last_oram_level = _last_oram_level

        # Derived: default reset probability is 1/num_ic when the config leaves it unset.
        self._reset_prob = 1 / config.num_ic if config.reset_prob is None else config.reset_prob

        # Leaves held between a read-without-eviction and its later eviction. _tmp_leaf for the
        # simple case; _tmp_leaves / _tmp_reset_leaves when a reset spans multiple paths.
        self._tmp_leaf: int | None = None
        self._tmp_leaves: list[int] | None = None
        self._tmp_reset_leaves: ResetLeaves | None = None

        self._prf = Blake2Prf(key=config.prf_key)

        self._on_chip_storage: list[bytes] = []
        self._pos_maps: list[FreecursiveOram] = []

        self._init_pos_map()

    # Scheme-specific construction parameters — read-only views onto the frozen config.
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
    def _on_chip_size(self) -> int:
        return self._config.on_chip_size

    @property
    def _reset_method(self) -> str:
        return self._config.reset_method

    @cached_property
    def _count_length(self) -> int:
        """Bit length of the counter value a position-map block stores."""
        return self._num_ic * self._ic_length + self._gc_length

    @cached_property
    def _num_oram_pos_map(self) -> int:
        """Number of position-map orams needed; the last is kept on chip, hence the -1."""
        return math.ceil(math.log(self._num_data / self._on_chip_size, self._num_ic)) - 1

    @cached_property
    def _pos_map_oram_dummy_size(self) -> int:
        """Byte size of the dummy value stored in position maps."""
        return math.ceil(self._count_length / 8)

    @staticmethod
    def _unpack_reset_leaves(cur_reset_leaves: ResetChunk) -> UnpackedReset:
        """Unpack a 1- or 2-tuple reset chunk into (ck_a, cl_a, nl_a, ck_b, cl_b, nl_b, leaves)."""
        ck_a, cl_a, nl_a = cur_reset_leaves[0]
        if len(cur_reset_leaves) == 2:
            ck_b, cl_b, nl_b = cur_reset_leaves[1]
            leaves = [cl_a, cl_b]
        else:
            ck_b, cl_b, nl_b = None, None, None
            leaves = [cl_a]
        return UnpackedReset(ck_a, cl_a, nl_a, ck_b, cl_b, nl_b, leaves)

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
            Helper.binary_str_to_bytes(
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
            Helper.binary_str_to_bytes(
                bin(key)[2:].zfill(self._last_oram_level - 1)
                + bin(gc)[2:].zfill(self._gc_length)
                + bin(ic)[2:].zfill(self._ic_length)
            ),
            pow(2, self._last_oram_level - 1),
        )

    @override
    def _init_pos_map(self) -> None:
        self._pos_map = {i: self._get_leaf_from_prf(key=i, gc=0, ic=0) for i in range(self._num_data)}

    def _compress_pos_map(self) -> ServerStorage:
        """Compress the flat position map into a chain of position-map orams; returns server storage."""
        self._pos_map = {}

        server_storage: ServerStorage = {}

        # All counters (gc and ic) start at zero.
        value = Helper.binary_str_to_bytes("0" * self._count_length)

        last_oram_data = self._num_data
        last_oram_level = self._level

        for i in range(self._num_oram_pos_map):
            pos_map_size = math.ceil(last_oram_data / self._num_ic)

            pos_map_filename = (
                f"{self._filename}_pos_map_{self._num_oram_pos_map - i - 1}.bin" if self._filename else None
            )

            # The label this level's tree is stored under; also the child oram's name so the
            # child can drive its own server I/O on the shared client.
            pos_map_name = f"{self._name}_pos_map_{self._num_oram_pos_map - i - 1}"

            cur_pos_map_oram = FreecursiveOram(
                replace(
                    self._config,
                    prf_key=self._prf.key,
                    num_data=pos_map_size,
                    name=pos_map_name,
                    data_size=self._pos_map_oram_dummy_size,
                    reset_prob=self._reset_prob,
                    filename=pos_map_filename,
                ),
                _is_pos_map=True,
                _last_oram_data=last_oram_data,
                _last_oram_level=last_oram_level,
            )

            tree = BinaryTree(
                filename=pos_map_filename,
                num_data=pos_map_size,
                data_size=cur_pos_map_oram._dumped_data_size,
                bucket_size=self._bucket_size,
                disk_size=cur_pos_map_oram._disk_size,
                encryption=self._encryptor is not None,
            )

            for key, leaf in cur_pos_map_oram._pos_map.items():
                tree.fill_data_to_storage_leaf(data=Data(key=key, leaf=leaf, value=value))

            if self._encryptor:
                tree.storage.encrypt(encryptor=self._encryptor)

            last_oram_data = pos_map_size
            last_oram_level = cur_pos_map_oram._level

            server_storage[pos_map_name] = tree

            cur_pos_map_oram._pos_map = {}
            self._pos_maps.append(cur_pos_map_oram)

        self._last_oram_data = last_oram_data
        self._last_oram_level = last_oram_level

        self._on_chip_storage = [value for _ in range(math.ceil(last_oram_data / self._num_ic) + 1)]

        self._pos_maps.reverse()

        return server_storage

    @override
    def init_server_storage(self, data_map: DataMap | None = None) -> None:
        storage = self._init_storage_on_pos_map(data_map=data_map)

        pos_map_storage_dict = self._compress_pos_map()
        pos_map_storage_dict[self._name] = storage

        self._client.init_storage(storage=pos_map_storage_dict)

    def _update_stash_leaf(self, key: int | None, new_leaf: int | None) -> None:
        if key is None:
            return

        for data in self._stash:
            if data.key == key:
                data.leaf = new_leaf
                return

        raise KeyError(f"Key {key} not found.")

    def _update_stash_leaves(
        self, key_a: int | None, key_b: int | None, n_leaf_a: int | None, n_leaf_b: int | None, to_index: int
    ) -> None:
        """Remap key_a and key_b in the stash; to_index bounds the search to pre-existing blocks."""
        for data in self._stash[:to_index]:
            if key_a is not None and data.key == key_a:
                data.leaf = n_leaf_a
                key_a = None
            elif key_b is not None and data.key == key_b:
                data.leaf = n_leaf_b
                key_b = None

        if key_a is not None:
            raise KeyError(f"Key {key_a} not found.")
        if key_b is not None:
            raise KeyError(f"Key {key_b} not found.")

    def _update_block_leaves(
        self, key_a: int | None, key_b: int | None, n_leaf_a: int | None, n_leaf_b: int | None, path: PathData
    ) -> None:
        """Pull the path into the stash and remap key_a and key_b (a null op that hides the access)."""
        to_index = len(self._stash)

        decrypted = self._decrypt_path_data(path=path)

        for bucket in decrypted.values():
            for data in bucket:
                if data.key is None:
                    continue
                elif key_a is not None and data.key == key_a:
                    data.leaf = n_leaf_a
                    key_a = None
                elif key_b is not None and data.key == key_b:
                    data.leaf = n_leaf_b
                    key_b = None
                self._stash.append(data)

        if key_a is not None or key_b is not None:
            self._update_stash_leaves(key_a=key_a, key_b=key_b, n_leaf_a=n_leaf_a, n_leaf_b=n_leaf_b, to_index=to_index)

    def _get_reset_leaves(self, key: int, data: str, reset_size: int = 2) -> ResetLeaves:
        """On reset, return the (key, cur_leaf, new_leaf) chunks for every count in the block.

        :param data: The gc/ic counter string before resetting.
        :param reset_size: Number of leaves grouped per reset chunk.
        """
        assert self._last_oram_data is not None and self._last_oram_level is not None
        gc = int(data[: self._gc_length], 2)
        ic_str = data[self._gc_length :]
        ic = [int(ic_str[i * self._ic_length : (i + 1) * self._ic_length], 2) for i in range(self._num_ic)]

        reset_leaves: list[ResetEntry]
        # When the block straddles the end of the data, pad out-of-range slots with random leaves.
        if (key + 1) * self._num_ic > self._last_oram_data:
            reset_leaves = [
                ResetEntry(
                    key * self._num_ic + i,
                    self._get_previous_leaf_from_prf(key=key * self._num_ic + i, gc=gc, ic=ic[i]),
                    self._get_previous_leaf_from_prf(key=key * self._num_ic + i, gc=gc + 1, ic=0),
                )
                if key * self._num_ic + i < self._last_oram_data
                else ResetEntry(None, secrets.randbelow(pow(2, self._last_oram_level - 1)), None)
                for i in range(self._num_ic)
            ]
        else:
            reset_leaves = [
                ResetEntry(
                    key * self._num_ic + i,
                    self._get_previous_leaf_from_prf(key=key * self._num_ic + i, gc=gc, ic=ic[i]),
                    self._get_previous_leaf_from_prf(key=key * self._num_ic + i, gc=gc + 1, ic=0),
                )
                for i in range(self._num_ic)
            ]

        return [reset_leaves[i : i + reset_size] for i in range(0, len(reset_leaves), reset_size)]

    def _update_on_chip_data(self, key: int, offset: int) -> ProcessedData:
        """Advance the on-chip counter at offset; returns (cur_leaf, new_leaf, reset_leaves)."""
        data = Helper.bytes_to_binary_str(self._on_chip_storage[key]).zfill(self._count_length)

        gc = int(data[: self._gc_length], 2)

        ic_start = self._gc_length + offset * self._ic_length
        ic_end = self._gc_length + (offset + 1) * self._ic_length
        ic = int(data[ic_start:ic_end], 2)

        if self._reset_method == "prob":
            if random.random() <= self._reset_prob:
                reset_leaves = self._get_reset_leaves(key=key, data=data)
                self._on_chip_storage[key] = Helper.binary_str_to_bytes(
                    f"{bin(gc + 1)[2:].zfill(self._gc_length)}{'0' * self._ic_length * self._num_ic}"
                )
                return ProcessedData(None, None, reset_leaves)

            if ic + 1 >= pow(2, self._ic_length):
                raise ValueError("Overflow happened under probabilistic resets.")
            self._on_chip_storage[key] = Helper.binary_str_to_bytes(
                f"{data[:ic_start]}{bin(ic + 1)[2:].zfill(self._ic_length)}{data[ic_end:]}"
            )
            cur_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic)
            new_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic + 1)
            return ProcessedData(cur_leaf, new_leaf, None)

        elif self._reset_method == "hard":
            if ic + 1 >= pow(2, self._ic_length):
                reset_leaves = self._get_reset_leaves(key=key, data=data)
                self._on_chip_storage[key] = Helper.binary_str_to_bytes(
                    f"{bin(gc + 1)[2:].zfill(self._gc_length)}{'0' * self._ic_length * self._num_ic}"
                )
                return ProcessedData(None, None, reset_leaves)

            self._on_chip_storage[key] = Helper.binary_str_to_bytes(
                f"{data[:ic_start]}{bin(ic + 1)[2:].zfill(self._ic_length)}{data[ic_end:]}"
            )
            cur_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic)
            new_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic + 1)
            return ProcessedData(cur_leaf, new_leaf, None)

        else:
            raise ValueError(f"Unrecognized reset method {self._reset_method}.")

    def _update_data_prob_reset(self, key: int, data: Data, offset: int) -> ProcessedData:
        """Probabilistic-reset counter update on a stored Data block; returns (cur_leaf, new_leaf, reset_leaves)."""
        # A position-map block always carries its counter bytes as value.
        data.value = Helper.bytes_to_binary_str(data.value).zfill(self._count_length)

        gc = int(data.value[: self._gc_length], 2)

        ic_start = self._gc_length + offset * self._ic_length
        ic_end = self._gc_length + (offset + 1) * self._ic_length
        ic = int(data.value[ic_start:ic_end], 2)

        if random.random() <= self._reset_prob:
            reset_leaves = self._get_reset_leaves(key=key, data=data.value)
            data.value = Helper.binary_str_to_bytes(
                f"{bin(gc + 1)[2:].zfill(self._gc_length)}{'0' * self._ic_length * self._num_ic}"
            )
            return ProcessedData(None, None, reset_leaves)

        if ic + 1 >= pow(2, self._ic_length):
            raise ValueError("Overflow happened under probabilistic resets.")
        data.value = Helper.binary_str_to_bytes(
            f"{data.value[:ic_start]}{bin(ic + 1)[2:].zfill(self._ic_length)}{data.value[ic_end:]}"
        )
        cur_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic)
        new_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic + 1)
        return ProcessedData(cur_leaf, new_leaf, None)

    def _update_data_hard_reset(self, key: int, data: Data, offset: int) -> ProcessedData:
        """Hard-reset (on overflow) counter update on a stored Data block; returns (cur_leaf, new_leaf, reset_leaves)."""
        # A position-map block always carries its counter bytes as value.
        data.value = Helper.bytes_to_binary_str(data.value).zfill(self._count_length)

        gc = int(data.value[: self._gc_length], 2)

        ic_start = self._gc_length + offset * self._ic_length
        ic_end = self._gc_length + (offset + 1) * self._ic_length
        ic = int(data.value[ic_start:ic_end], 2)

        if ic + 1 >= pow(2, self._ic_length):
            reset_leaves = self._get_reset_leaves(key=key, data=data.value)
            data.value = Helper.binary_str_to_bytes(
                f"{bin(gc + 1)[2:].zfill(self._gc_length)}{'0' * self._ic_length * self._num_ic}"
            )
            return ProcessedData(None, None, reset_leaves)

        data.value = Helper.binary_str_to_bytes(
            f"{data.value[:ic_start]}{bin(ic + 1)[2:].zfill(self._ic_length)}{data.value[ic_end:]}"
        )
        cur_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic)
        new_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic + 1)
        return ProcessedData(cur_leaf, new_leaf, None)

    def _retrieve_pos_map_stash(self, key: int, offset: int, new_leaf: int, to_index: int) -> ProcessedData:
        """Find key in the stash, advance its counter, and remap it to new_leaf."""
        for data in self._stash[:to_index]:
            if data.key == key:
                next_cur_leaf, next_new_leaf, reset_leaves = (
                    self._update_data_prob_reset(key=key, data=data, offset=offset)
                    if self._reset_method == "prob"
                    else self._update_data_hard_reset(key=key, data=data, offset=offset)
                )
                data.leaf = new_leaf
                return ProcessedData(next_cur_leaf, next_new_leaf, reset_leaves)

        raise KeyError(f"Key {key} not found.")

    def _retrieve_pos_map_block(self, key: int, offset: int, new_leaf: int, path: PathData) -> ProcessedData:
        """Pull the path into the stash, advance key's counter, and remap it to new_leaf."""
        next_cur_leaf, next_new_leaf, reset_leaves = None, None, None
        to_index = len(self._stash)
        decrypted = self._decrypt_path_data(path=path)

        for bucket in decrypted.values():
            for data in bucket:
                if data.key is None:
                    continue
                elif data.key == key:
                    next_cur_leaf, next_new_leaf, reset_leaves = (
                        self._update_data_prob_reset(key=key, data=data, offset=offset)
                        if self._reset_method == "prob"
                        else self._update_data_hard_reset(key=key, data=data, offset=offset)
                    )
                    data.leaf = new_leaf
                self._stash.append(data)

        self._check_stash()

        if next_cur_leaf is None and reset_leaves is None:
            next_cur_leaf, next_new_leaf, reset_leaves = self._retrieve_pos_map_stash(
                key=key, offset=offset, new_leaf=new_leaf, to_index=to_index
            )

        return ProcessedData(next_cur_leaf, next_new_leaf, reset_leaves)

    def _access_pos_map_level(
        self, cur_key: int, cur_index: int, cur_leaf: int | None, new_leaf: int | None, reset_leaves: ResetLeaves | None
    ) -> ProcessedData:
        """Drive this position-map level's own server I/O and return ``(next_cur_leaf, next_new_leaf, reset_leaves)``.

        Reads/evicts its own paths on the shared client (under its own name). When the previous level
        produced ``reset_leaves`` they fan out into pairs of paths processed here; otherwise the single
        block for ``cur_key`` is read and its counter advanced.
        """
        next_cur_leaf, next_new_leaf = None, None

        if reset_leaves is not None:
            for cur_reset_leaves in reset_leaves:
                ck_a, _cl_a, nl_a, ck_b, _cl_b, nl_b, leaves = self._unpack_reset_leaves(cur_reset_leaves)

                self._client.add_read_path(label=self._name, leaves=leaves)
                result = self._client.execute()
                path = result.require(self._name)

                # Whichever reset key is the one we want next drives the next iteration.
                if ck_a == cur_key:
                    assert ck_a is not None and nl_a is not None
                    next_cur_leaf, next_new_leaf, reset_leaves = self._retrieve_pos_map_block(
                        key=ck_a, offset=cur_index, new_leaf=nl_a, path=path
                    )
                    self._update_stash_leaf(key=ck_b, new_leaf=nl_b)
                elif ck_b == cur_key:
                    assert ck_b is not None and nl_b is not None
                    next_cur_leaf, next_new_leaf, reset_leaves = self._retrieve_pos_map_block(
                        key=ck_b, offset=cur_index, new_leaf=nl_b, path=path
                    )
                    self._update_stash_leaf(key=ck_a, new_leaf=nl_a)
                else:
                    self._update_block_leaves(key_a=ck_a, key_b=ck_b, n_leaf_a=nl_a, n_leaf_b=nl_b, path=path)
                evicted_path = self._evict_stash(leaves=leaves)

                self._client.add_write_path(label=self._name, data=evicted_path)
                self._client.execute()

        else:
            assert cur_leaf is not None and new_leaf is not None
            self._client.add_read_path(label=self._name, leaves=[cur_leaf])
            result = self._client.execute()
            path = result.require(self._name)

            next_cur_leaf, next_new_leaf, reset_leaves = self._retrieve_pos_map_block(
                key=cur_key, offset=cur_index, new_leaf=new_leaf, path=path
            )

            evicted_path = self._evict_stash(leaves=[cur_leaf])

            self._client.add_write_path(label=self._name, data=evicted_path)
            self._client.execute()

        return ProcessedData(next_cur_leaf, next_new_leaf, reset_leaves)

    def _get_leaf_from_pos_map(self, key: int) -> ProcessedData:
        """Walk the position-map chain to find key's leaf and new leaf, handling resets along the way."""
        pos_map_keys = self._get_pos_map_keys(key=key)

        cur_leaf, new_leaf, reset_leaves = self._update_on_chip_data(key=pos_map_keys[0][0], offset=pos_map_keys[0][1])

        for pos_map_index, (cur_key, cur_index) in enumerate(pos_map_keys[1:]):
            # Each level owns its server I/O; the parent only threads the leaves along the chain.
            cur_leaf, new_leaf, reset_leaves = self._pos_maps[pos_map_index]._access_pos_map_level(
                cur_key=cur_key,
                cur_index=cur_index,
                cur_leaf=cur_leaf,
                new_leaf=new_leaf,
                reset_leaves=reset_leaves,
            )

        return ProcessedData(cur_leaf, new_leaf, reset_leaves)

    @override
    def operate_on_key(self, key: int, value: Any = UNSET) -> Any:
        cur_leaf, next_leaf, reset_leaves = self._get_leaf_from_pos_map(key=key)

        read_value = None

        if reset_leaves:
            for cur_reset_leaves in reset_leaves:
                ck_a, _cl_a, nl_a, ck_b, _cl_b, nl_b, leaves = self._unpack_reset_leaves(cur_reset_leaves)

                self._client.add_read_path(label=self._name, leaves=leaves)
                result = self._client.execute()
                path = result.require(self._name)

                if ck_a == key:
                    assert ck_a is not None and nl_a is not None
                    read_value = self._retrieve_data_block(key=ck_a, value=value, new_leaf=nl_a, path=path)
                    self._update_stash_leaf(key=ck_b, new_leaf=nl_b)

                elif ck_b == key:
                    assert ck_b is not None and nl_b is not None
                    read_value = self._retrieve_data_block(key=ck_b, value=value, new_leaf=nl_b, path=path)
                    self._update_stash_leaf(key=ck_a, new_leaf=nl_a)

                else:
                    self._update_block_leaves(key_a=ck_a, key_b=ck_b, n_leaf_a=nl_a, n_leaf_b=nl_b, path=path)

                evicted_path = self._evict_stash(leaves=leaves)

                self._client.add_write_path(label=self._name, data=evicted_path)
                self._client.execute()

            return read_value

        assert cur_leaf is not None and next_leaf is not None
        self._client.add_read_path(label=self._name, leaves=[cur_leaf])
        result = self._client.execute()
        path = result.require(self._name)

        read_value = self._retrieve_data_block(key=key, value=value, new_leaf=next_leaf, path=path)

        evicted_path = self._evict_stash(leaves=[cur_leaf])

        self._client.add_write_path(label=self._name, data=evicted_path)
        self._client.execute()

        return read_value

    @override
    def operate_on_key_without_eviction(self, key: int, value: Any = UNSET) -> Any:
        """Like operate_on_key but defers eviction to a later eviction_with_update_stash call.

        On a reset, the key's own path is read and its eviction deferred; remaining reset paths are
        stashed in _tmp_reset_leaves and finished during eviction.
        """
        cur_leaf, next_leaf, reset_leaves = self._get_leaf_from_pos_map(key=key)

        if reset_leaves:
            for index, cur_reset_leaves in enumerate(reset_leaves):
                ck_a, _cl_a, nl_a, ck_b, _cl_b, nl_b, leaves = self._unpack_reset_leaves(cur_reset_leaves)

                self._client.add_read_path(label=self._name, leaves=leaves)
                result = self._client.execute()
                path = result.require(self._name)

                if ck_a == key:
                    assert ck_a is not None and nl_a is not None
                    read_value = self._retrieve_data_block(key=ck_a, value=value, new_leaf=nl_a, path=path)
                    self._update_stash_leaf(key=ck_b, new_leaf=nl_b)

                    self._tmp_leaves = leaves

                    if index < len(reset_leaves) - 1:
                        self._tmp_reset_leaves = reset_leaves[index + 1 :]

                    return read_value

                elif ck_b == key:
                    assert ck_b is not None and nl_b is not None
                    read_value = self._retrieve_data_block(key=ck_b, value=value, new_leaf=nl_b, path=path)
                    self._update_stash_leaf(key=ck_a, new_leaf=nl_a)

                    self._tmp_leaves = leaves

                    if index < len(reset_leaves) - 1:
                        self._tmp_reset_leaves = reset_leaves[index + 1 :]

                    return read_value

                self._update_block_leaves(key_a=ck_a, key_b=ck_b, n_leaf_a=nl_a, n_leaf_b=nl_b, path=path)

                evicted_path = self._evict_stash(leaves=leaves)

                self._client.add_write_path(label=self._name, data=evicted_path)
                self._client.execute()

            raise KeyError(f"Key {key} not found in reset leaves.")

        assert cur_leaf is not None and next_leaf is not None
        self._client.add_read_path(label=self._name, leaves=[cur_leaf])
        result = self._client.execute()
        path = result.require(self._name)

        read_value = self._retrieve_data_block(key=key, value=value, new_leaf=next_leaf, path=path)

        self._tmp_leaf = cur_leaf

        return read_value

    @override
    def eviction_with_update_stash(self, key: int, value: Any, execute: bool = True) -> None:
        """Update key's block in the stash, then evict. If execute is False, queue the write for batching.

        With reset leaves outstanding, internal executes are always performed regardless of execute.
        """
        found = False

        for data in self._stash:
            if data.key == key:
                data.value = value
                found = True

        if not found:
            raise KeyError(f"Key {key} not found.")

        # Simple case: single deferred path. Reset case: finish the remaining reset chunks.
        if self._tmp_leaf is not None:
            evicted_path = self._evict_stash(leaves=[self._tmp_leaf])
            self._client.add_write_path(label=self._name, data=evicted_path)
            self._tmp_leaf = None
        else:
            assert self._tmp_leaves is not None
            evicted_path = self._evict_stash(leaves=self._tmp_leaves)
            self._client.add_write_path(label=self._name, data=evicted_path)
            # Must execute here due to internal dependencies with reset operations.
            self._client.execute()
            self._tmp_leaves = None
            if self._tmp_reset_leaves is not None:
                for cur_reset_leaves in self._tmp_reset_leaves:
                    ck_a, _cl_a, nl_a, ck_b, _cl_b, nl_b, leaves = self._unpack_reset_leaves(cur_reset_leaves)

                    self._client.add_read_path(label=self._name, leaves=leaves)
                    result = self._client.execute()
                    path = result.require(self._name)

                    self._update_block_leaves(key_a=ck_a, key_b=ck_b, n_leaf_a=nl_a, n_leaf_b=nl_b, path=path)

                    evicted_path = self._evict_stash(leaves=leaves)

                    self._client.add_write_path(label=self._name, data=evicted_path)
                self._tmp_reset_leaves = None

        if execute:
            self._client.execute()
