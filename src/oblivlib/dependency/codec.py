from abc import ABC, abstractmethod
from typing import Any, override

import msgpack

from oblivlib.dependency.types import Data, FieldTuple


def packed_size(obj: Any) -> int:
    return len(msgpack.packb(obj))


class BlockCodec(ABC):
    """Turns a block into the msgpack-ready ``[key, leaf, value]`` a bucket packs, and back.
    ``max_block_bytes`` bounds one packed block; sealed rows are sized from it."""

    def __init__(self, max_block_bytes: int):
        self._max_block_bytes = max_block_bytes

    @property
    def max_block_bytes(self) -> int:
        return self._max_block_bytes

    @abstractmethod
    def pack(self, data: Data) -> list[Any]:
        raise NotImplementedError

    @abstractmethod
    def unpack(self, fields: list[Any]) -> Data:
        raise NotImplementedError


class DefaultCodec(BlockCodec):
    @override
    def pack(self, data: Data) -> list[Any]:
        return data.to_fields()

    @override
    def unpack(self, fields: list[Any]) -> Data:
        return Data.from_fields(fields)


class NodeCodec(BlockCodec):
    def __init__(self, max_block_bytes: int, value_cls: type[FieldTuple]):
        super().__init__(max_block_bytes)
        self._value_cls = value_cls

    @override
    def pack(self, data: Data) -> list[Any]:
        return [data.key, data.leaf, data.value.to_fields()]

    @override
    def unpack(self, fields: list[Any]) -> Data:
        key, leaf, value = fields
        return Data(key=key, leaf=leaf, value=self._value_cls.from_fields(value))
