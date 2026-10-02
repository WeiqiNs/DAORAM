"""Shared core of the tree-structured schemes: config accessors, leaf math, stash, eviction, path crypto."""

import os
import secrets

from oblivlib.dependency.binary_tree import BinaryTree
from oblivlib.dependency.codec import BlockCodec, DefaultCodec
from oblivlib.dependency.config import OramConfig
from oblivlib.dependency.crypto import Encryptor
from oblivlib.dependency.helper import Data, Helper
from oblivlib.dependency.interact_server import InteractServer
from oblivlib.dependency.types import Block, PathData


class TreeStorageBase[ConfigT: OramConfig]:
    def __init__(self, config: ConfigT):
        self._config: ConfigT = config

        self._level: int = BinaryTree.compute_level(config.num_data)
        self._leaf_range: int = 1 << (self._level - 1)
        self._stash_capacity: int = config.stash_scale * max(1, self._level - 1)

        self._stash: list = []

        self._dumped_data_size: int | None = None
        if config.encryptor or config.filename:
            self._dumped_data_size = len(
                Data(key=config.num_data - 1, leaf=config.num_data - 1, value=os.urandom(config.data_size)).dump()
            )

    @property
    def _name(self) -> str:
        return self._config.name

    @property
    def _filename(self) -> str | None:
        return self._config.filename

    @property
    def _num_data(self) -> int:
        return self._config.num_data

    @property
    def _data_size(self) -> int:
        return self._config.data_size

    @property
    def _bucket_size(self) -> int:
        return self._config.bucket_size

    @property
    def _encryptor(self) -> Encryptor | None:
        return self._config.encryptor

    @property
    def _client(self) -> InteractServer:
        assert self._config.client is not None
        return self._config.client

    @property
    def _disk_size(self) -> int | None:
        """Bytes per file row: one ciphertext per bucket when encrypted, else one block slot."""
        if not self._filename:
            return None
        block_size = self._codec.block_size
        if self._encryptor:
            return self._encryptor.ciphertext_length(self._bucket_size * block_size)
        return block_size

    @property
    def stash(self) -> list:
        return self._stash

    @stash.setter
    def stash(self, value: list):
        self._stash = value

    @property
    def stash_size(self) -> int:
        return len(self._stash)

    def _get_new_leaf(self) -> int:
        return secrets.randbelow(self._leaf_range)

    def _check_stash(self) -> None:
        if len(self._stash) > self._stash_capacity:
            raise MemoryError("Stash overflow!")

    def _evict_stash(self, leaves: list[int]) -> PathData:
        """Evict stash blocks onto the given paths; blocks that don't fit stay in the stash."""
        path = BinaryTree.get_mul_path_dict(level=self._level, indices=leaves)
        remaining = []
        for data in self._stash:
            if not BinaryTree.fill_data_to_path(
                data=data, path=path, leaves=leaves, level=self._level, bucket_size=self._bucket_size
            ):
                remaining.append(data)
        self._stash = remaining
        return self._encrypt_path_data(path=path)

    @property
    def _codec(self) -> BlockCodec:
        assert self._dumped_data_size is not None
        return DefaultCodec(self._dumped_data_size)

    def _encrypt_path_data(self, path: PathData) -> PathData:
        """Encrypt each bucket into a single ciphertext blob, padding to bucket_size with dummies."""
        if not self._encryptor:
            return path

        codec = self._codec
        dummy = codec.dummy_block()
        return {
            idx: [
                Helper.encrypt_bucket(
                    self._encryptor,
                    [codec.dump_block(data) for data in bucket if isinstance(data, Data)],
                    dummy,
                    self._bucket_size,
                )
            ]
            for idx, bucket in path.items()
        }

    def _decrypt_path_data(self, path: PathData) -> dict[int, list[Data]]:
        """Decrypt each bucket's blob and drop dummy blocks; returns plaintext Data per bucket."""
        encryptor = self._encryptor
        if not encryptor:
            return {idx: [data for data in bucket if isinstance(data, Data)] for idx, bucket in path.items()}

        codec = self._codec
        block_size = codec.block_size

        def _dec_bucket(bucket: list[Block]) -> list[Data]:
            blob = bucket[0]
            assert isinstance(blob, bytes)
            blocks = Helper.decrypt_bucket(encryptor, blob, block_size)
            return [data for block in blocks if (data := codec.load_block(block)).is_real()]

        return {idx: _dec_bucket(bucket) for idx, bucket in path.items()}
