from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import cast, override

from oblivlib.dependency.crypto import Encryptor
from oblivlib.dependency.types import Data, FieldTuplePickle


class BlockCodec(ABC):
    def __init__(self, block_size: int):
        self._block_size = block_size

    @property
    def block_size(self) -> int:
        return self._block_size

    def dummy_block(self) -> bytes:
        return Data().dump_pad(self._block_size)

    def seal_bucket(self, encryptor: Encryptor, blocks: Sequence[Data], bucket_size: int) -> bytes:
        if len(blocks) > bucket_size:
            raise ValueError(f"{type(self).__name__}: {len(blocks)} blocks exceed the bucket capacity {bucket_size}.")
        payload = b"".join(self.dump_block(data) for data in blocks)
        return encryptor.enc(plaintext=payload + self.dummy_block() * (bucket_size - len(blocks)))

    def open_bucket(self, encryptor: Encryptor, blob: bytes) -> list[Data]:
        plaintext = encryptor.dec(ciphertext=blob)
        size = self._block_size
        blocks = (self.load_block(plaintext[i : i + size]) for i in range(0, len(plaintext), size))
        return [data for data in blocks if data.is_real()]

    @abstractmethod
    def dump_block(self, data: Data) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def load_block(self, payload: bytes) -> Data:
        raise NotImplementedError


class DefaultCodec(BlockCodec):
    @override
    def dump_block(self, data: Data) -> bytes:
        return data.dump_pad(self._block_size)

    @override
    def load_block(self, payload: bytes) -> Data:
        return Data.load(payload)


class NodeCodec(BlockCodec):
    def __init__(self, block_size: int, value_cls: type[FieldTuplePickle]):
        super().__init__(block_size)
        self._value_cls = value_cls

    @override
    def dump_block(self, data: Data) -> bytes:
        return Data(data.key, data.leaf, cast(FieldTuplePickle, data.value).dump()).dump_pad(self._block_size)

    @override
    def load_block(self, payload: bytes) -> Data:
        data = Data.load(payload)
        if data.is_real():
            data.value = self._value_cls.load(data=cast(bytes, data.value))
        return data
