"""Shared storage/crypto/leaf-math base for the tree-structured schemes.

`TreeBaseOram` (ORAM) and `OstBaseOmap` (ODS OMAP) both lay a complete binary tree over
`InteractServer` storage. They share their entire construction core — the frozen config and its
read-only field accessors, the integer level/leaf-range/stash-size math, the padded-block/disk sizing,
random-leaf sampling, and per-bucket path encryption/decryption — and differ only in what they layer
on top (a position map + eviction for ORAM; a root/local + tree traversal for the ODS). That common
core lives here so it is defined once.

Private members use single underscores (not name-mangled ``__``) so subclasses can access them.
"""

import os
import secrets

from oblivlib.dependency.codec import BlockCodec, DefaultCodec
from oblivlib.dependency.config import OramConfig
from oblivlib.dependency.crypto import Encryptor
from oblivlib.dependency.helper import Data, Helper
from oblivlib.dependency.interact_server import InteractServer
from oblivlib.dependency.types import Block, PathData


class TreeStorageBase[ConfigT: OramConfig]:
    def __init__(self, config: ConfigT):
        self._config: ConfigT = config

        self._level: int = (config.num_data - 1).bit_length() + 1
        self._leaf_range: int = pow(2, self._level - 1)
        self._stash_size: int = config.stash_scale * (self._level - 1) if self._level > 1 else config.stash_scale

        self._stash: list = []

        self._dumped_data_size: int | None = None
        self._disk_size: int | None = None

        if config.encryptor or config.filename:
            self._dumped_data_size = len(
                Data(key=config.num_data - 1, leaf=config.num_data - 1, value=os.urandom(config.data_size)).dump()
            )

        if config.filename:
            if config.encryptor:
                # A row stores one ciphertext covering the whole (bucket_size-block) bucket.
                assert self._dumped_data_size is not None
                self._disk_size = config.encryptor.ciphertext_length(config.bucket_size * self._dumped_data_size)
            else:
                self._disk_size = self._dumped_data_size

    # Construction parameters — read-only views onto the frozen config (see OramConfig).
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
    def _stash_scale(self) -> int:
        return self._config.stash_scale

    @property
    def _encryptor(self) -> Encryptor | None:
        return self._config.encryptor

    @property
    def _client(self) -> InteractServer:
        # A main scheme always has a client (validated at construction); position-map children never
        # reach this accessor since the parent drives their I/O.
        assert self._config.client is not None
        return self._config.client

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
            # An encrypted bucket is a single ciphertext blob (bytes) at index 0.
            blob = bucket[0]
            assert isinstance(blob, bytes)
            blocks = Helper.decrypt_bucket(encryptor, blob, block_size)
            return [data for block in blocks if (data := codec.load_block(block)).is_real()]

        return {idx: _dec_bucket(bucket) for idx, bucket in path.items()}
