"""Recursive Path ORAM.

The large position map is compressed into a chain of smaller position-map orams (each storing
``compression_ratio`` leaves per block); only a small top-level map is kept on the client.
Call ``init_server_storage`` once, then use ``operate_on_key``.
"""

import math
import pickle
import secrets
from dataclasses import replace
from functools import cached_property
from typing import Any, override

from oblivlib.dependency import UNSET, BinaryTree, Data, DataMap, PathData, PosMap, ServerStorage
from oblivlib.dependency.config import RecursiveOramConfig
from oblivlib.oram.tree_base_oram import TreeBaseOram


class RecursivePathOram(TreeBaseOram[RecursiveOramConfig]):
    def __init__(self, config: RecursiveOramConfig, *, _is_pos_map: bool = False):
        # _is_pos_map marks an internal position-map child oram, which shares the parent's client.
        if not _is_pos_map:
            if config.client is None:
                raise ValueError("Client is required for main ORAM.")
            if config.num_data <= config.on_chip_mem:
                raise ValueError(
                    f"num_data ({config.num_data}) must be greater than on_chip_mem ({config.on_chip_mem})."
                )

        super().__init__(config)

        # The recursive position maps, ordered from smallest to the one above the data oram.
        self._pos_maps: list[RecursivePathOram] = []

        self._tmp_leaf: int | None = None

        self._init_pos_map()

    # Scheme-specific construction parameters — read-only views onto the frozen config.
    @property
    def _on_chip_mem(self) -> int:
        return self._config.on_chip_mem

    @property
    def _compression_ratio(self) -> int:
        return self._config.compression_ratio

    @cached_property
    def _num_oram_pos_map(self) -> int:
        return math.ceil(math.log(self._num_data / self._on_chip_mem, self._compression_ratio))

    @cached_property
    def _pos_map_oram_dummy_size(self) -> int:
        """Byte size of the dummy value stored in position maps."""
        return len(pickle.dumps([self._num_data - 1 - i for i in range(self._compression_ratio)]))

    def _get_pos_map_keys(self, key: int) -> list[tuple[int, int]]:
        """For each position map (outermost first), the (block key, offset within block) for this key."""
        pos_map_keys = []

        for _ in range(self._num_oram_pos_map):
            index = key % self._compression_ratio
            key = key // self._compression_ratio
            pos_map_keys.append((key, index))

        pos_map_keys.reverse()

        return pos_map_keys

    def _compress_pos_map(self) -> ServerStorage:
        """Compress the flat position map into a chain of position-map orams; returns server storage."""
        server_storage: ServerStorage = {}

        last_pos_map: PosMap = self._pos_map
        pos_map_size = self._num_data

        for i in range(self._num_oram_pos_map):
            last_pos_map_size = pos_map_size
            pos_map_size = math.ceil(pos_map_size / self._compression_ratio)

            pos_map_filename = (
                f"{self._filename}_pos_map_{self._num_oram_pos_map - i - 1}.bin" if self._filename else None
            )

            # The label this level's tree is stored under; also the child oram's name so the
            # child can drive its own server I/O on the shared client.
            pos_map_name = f"{self._name}_pos_map_{self._num_oram_pos_map - i - 1}"

            cur_pos_map_oram = RecursivePathOram(
                replace(
                    self._config,
                    num_data=pos_map_size,
                    name=pos_map_name,
                    data_size=self._pos_map_oram_dummy_size,
                    filename=pos_map_filename,
                ),
                _is_pos_map=True,
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
                # Pad with random leaves when pos_map_size is not a multiple of compression_ratio.
                value = [
                    last_pos_map[i] if i < last_pos_map_size else secrets.randbelow(pos_map_size)
                    for i in range(key * self._compression_ratio, (key + 1) * self._compression_ratio)
                ]
                tree.fill_data_to_storage_leaf(data=Data(key=key, leaf=leaf, value=value))

            if self._encryptor:
                tree.storage.encrypt(encryptor=self._encryptor)

            last_pos_map = cur_pos_map_oram._pos_map
            cur_pos_map_oram._pos_map = {}

            server_storage[pos_map_name] = tree
            self._pos_maps.append(cur_pos_map_oram)

        # Keep only the smallest map on chip.
        self._pos_map = last_pos_map

        self._pos_maps.reverse()

        return server_storage

    @override
    def init_server_storage(self, data_map: DataMap | None = None) -> None:
        storage: ServerStorage = {self._name: self._init_storage_on_pos_map(data_map=data_map)}

        pos_map_storage = self._compress_pos_map()
        storage.update(pos_map_storage)

        self._client.init_storage(storage=storage)

    def _retrieve_pos_map_stash(self, key: int, value: int, offset: int, new_leaf: int, to_index: int) -> int:
        """Find key in the stash, read the leaf at offset, write the new leaf there, and remap to new_leaf."""
        found = False
        read_value: int | None = None

        for data in self._stash[:to_index]:
            if data.key == key:
                # A position-map block's value is the list of child leaves.
                read_value = data.value[offset]
                data.value[offset] = value
                data.leaf = new_leaf
                found = True

        if not found:
            raise KeyError(f"Key {key} not found.")

        assert read_value is not None
        return read_value

    def _retrieve_pos_map_block(self, key: int, offset: int, new_leaf: int, value: int, path: PathData) -> Any:
        """Pull the path into the stash; read the leaf at offset for key, overwrite it with value, remap to new_leaf."""
        read_value = None
        to_index = len(self._stash)

        decrypted = self._decrypt_path_data(path=path)

        for bucket in decrypted.values():
            for data in bucket:
                if data.key is None:
                    continue
                elif data.key == key:
                    # A position-map block's value is the list of child leaves.
                    read_value = data.value[offset]
                    data.value[offset] = value
                    data.leaf = new_leaf

                self._stash.append(data)

        self._check_stash()

        if read_value is None:
            read_value = self._retrieve_pos_map_stash(
                key=key, value=value, offset=offset, new_leaf=new_leaf, to_index=to_index
            )

        return read_value

    def _access_pos_map_level(
        self, cur_key: int, cur_index: int, cur_leaf: int, new_cur_leaf: int, new_next_leaf: int
    ) -> int:
        """Drive this position-map level's own server I/O and return the leaf read at the offset.

        Reads its path on the shared client (under its own name), pulls the block into the stash
        while writing ``new_next_leaf`` at ``cur_index`` and remapping the block to ``new_cur_leaf``,
        then evicts and writes the path back.
        """
        self._client.add_read_path(label=self._name, leaves=[cur_leaf])
        result = self._client.execute()
        path_data = result.require(self._name)

        next_leaf = self._retrieve_pos_map_block(
            key=cur_key, offset=cur_index, path=path_data, new_leaf=new_cur_leaf, value=new_next_leaf
        )

        evicted_path = self._evict_stash(leaves=[cur_leaf])

        self._client.add_write_path(label=self._name, data=evicted_path)
        self._client.execute()

        return next_leaf

    def _get_leaf_from_pos_map(self, key: int) -> tuple[int, int]:
        """Walk the position-map chain to find key's current leaf and assign its new leaf."""
        cur_leaf: int | None = None
        new_cur_leaf: int | None = None

        for pos_map_index, (cur_key, cur_index) in enumerate(self._get_pos_map_keys(key=key)):
            if pos_map_index == 0:
                cur_leaf = self._pos_map[cur_key]
                new_cur_leaf = self._pos_maps[pos_map_index]._get_new_leaf()
                self._pos_map[cur_key] = new_cur_leaf

            # New leaf for the next level; sampled from the data oram on the last iteration.
            new_next_leaf = (
                self._pos_maps[pos_map_index + 1]._get_new_leaf()
                if pos_map_index < self._num_oram_pos_map - 1
                else self._get_new_leaf()
            )

            assert cur_leaf is not None and new_cur_leaf is not None

            # Each level owns its server I/O; the parent only threads the leaf along the chain.
            next_leaf = self._pos_maps[pos_map_index]._access_pos_map_level(
                cur_key=cur_key,
                cur_index=cur_index,
                cur_leaf=cur_leaf,
                new_cur_leaf=new_cur_leaf,
                new_next_leaf=new_next_leaf,
            )

            cur_leaf, new_cur_leaf = next_leaf, new_next_leaf

        assert cur_leaf is not None and new_cur_leaf is not None
        return cur_leaf, new_cur_leaf

    @override
    def operate_on_key(self, key: int, value: Any = UNSET) -> Any:
        leaf, new_leaf = self._get_leaf_from_pos_map(key=key)

        self._client.add_read_path(label=self._name, leaves=[leaf])
        result = self._client.execute()
        path_data = result.require(self._name)

        read_value = self._retrieve_data_block(key=key, path=path_data, value=value, new_leaf=new_leaf)

        evicted_path = self._evict_stash(leaves=[leaf])

        self._client.add_write_path(label=self._name, data=evicted_path)
        self._client.execute()

        return read_value

    @override
    def operate_on_key_without_eviction(self, key: int, value: Any = UNSET) -> Any:
        leaf, new_leaf = self._get_leaf_from_pos_map(key=key)

        self._client.add_read_path(label=self._name, leaves=[leaf])
        result = self._client.execute()
        path_data = result.require(self._name)

        read_value = self._retrieve_data_block(key=key, path=path_data, value=value, new_leaf=new_leaf)

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
