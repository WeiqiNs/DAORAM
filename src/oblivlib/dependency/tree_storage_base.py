import os
import secrets
from functools import cached_property

from oblivlib.dependency.binary_tree import BinaryTree
from oblivlib.dependency.codec import BlockCodec, DefaultCodec
from oblivlib.dependency.config import OramConfig
from oblivlib.dependency.crypto import Encryptor
from oblivlib.dependency.errors import MissingClientError, StashOverflowError
from oblivlib.dependency.heap_index import compute_level, empty_path, fill_data_to_path
from oblivlib.dependency.interact_server import InteractServer
from oblivlib.dependency.types import Block, Data, PathData


class TreeStorageBase[ConfigT: OramConfig]:
    def __init__(self, config: ConfigT):
        self._config: ConfigT = config

        self._level: int = compute_level(config.num_data)
        self._leaf_range: int = 1 << (self._level - 1)
        self._stash_capacity: int = config.stash_scale * max(1, self._level - 1)

        self._stash: list = []

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
        if self._config.client is None:
            raise MissingClientError(f"{type(self).__name__} {self._name!r} has no client; pass client= in its config")
        return self._config.client

    @cached_property
    def _dumped_data_size(self) -> int:
        return len(Data(key=self._num_data - 1, leaf=self._num_data - 1, value=os.urandom(self._data_size)).dump())

    @property
    def stash_size(self) -> int:
        return len(self._stash)

    def _get_new_leaf(self) -> int:
        return secrets.randbelow(self._leaf_range)

    def _check_stash(self) -> None:
        if len(self._stash) > self._stash_capacity:
            raise StashOverflowError(
                f"{type(self).__name__} {self._name!r}: stash holds {len(self._stash)} blocks, "
                + f"capacity {self._stash_capacity}"
            )

    def _evict_stash(self, leaves: list[int]) -> PathData:
        path = empty_path(level=self._level, leaves=leaves)
        remaining = []
        for data in self._stash:
            if not fill_data_to_path(data, path, leaves=leaves, level=self._level, bucket_size=self._bucket_size):
                remaining.append(data)
        self._stash = remaining
        return self._encrypt_path_data(path=path)

    @property
    def _codec(self) -> BlockCodec:
        return DefaultCodec(self._dumped_data_size)

    def _build_tree(self, blocks: list[Data]) -> BinaryTree:
        tree = BinaryTree(
            num_data=self._num_data,
            bucket_size=self._bucket_size,
            codec=self._codec,
            encryptor=self._encryptor,
            filename=self._filename,
        )
        for block in blocks:
            if not tree.fill_data_to_storage_leaf(data=block):
                self._stash.append(block)
        self._check_stash()

        if self._encryptor:
            tree.storage.seal(encryptor=self._encryptor)

        return tree

    def _encrypt_path_data(self, path: PathData) -> PathData:
        encryptor = self._encryptor
        if not encryptor:
            return path

        codec = self._codec
        return {
            idx: [codec.seal_bucket(encryptor, [data for data in bucket if isinstance(data, Data)], self._bucket_size)]
            for idx, bucket in path.items()
        }

    def _decrypt_path_data(self, path: PathData) -> dict[int, list[Data]]:
        encryptor = self._encryptor
        if not encryptor:
            return {
                idx: [data for data in bucket if isinstance(data, Data) and data.is_real()]
                for idx, bucket in path.items()
            }

        codec = self._codec

        def _open(bucket: list[Block]) -> list[Data]:
            blob = bucket[0]
            assert isinstance(blob, bytes)
            return codec.open_bucket(encryptor, blob)

        return {idx: _open(bucket) for idx, bucket in path.items()}
