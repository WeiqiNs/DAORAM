"""Per-block serialization for path encryption: one ``Data`` to/from a fixed-width bucket slot."""

from abc import ABC, abstractmethod
from typing import Protocol, Self, cast, override

from oblivlib.dependency.helper import Data


class BlockCodec(ABC):
    def __init__(self, block_size: int):
        self._block_size = block_size

    @property
    def block_size(self) -> int:
        return self._block_size

    def dummy_block(self) -> bytes:
        return Data().dump_pad(self._block_size)

    @abstractmethod
    def dump_block(self, data: Data) -> bytes:
        """Serialize to exactly ``block_size`` bytes; must not mutate ``data``."""
        raise NotImplementedError

    @abstractmethod
    def load_block(self, payload: bytes) -> Data:
        """Inverse of ``dump_block``; a dummy payload loads as a dummy ``Data``."""
        raise NotImplementedError


class DefaultCodec(BlockCodec):
    @override
    def dump_block(self, data: Data) -> bytes:
        return data.dump_pad(self._block_size)

    @override
    def load_block(self, payload: bytes) -> Data:
        return Data.load_unpad(payload)


class NodeValue(Protocol):
    def dump(self) -> bytes: ...

    @classmethod
    def from_pickle(cls, data: bytes) -> Self: ...


class NodeCodec(BlockCodec):
    """Codec for ODS blocks whose value is a node object (``AVLData`` / ``BPlusData``) pickled on its own."""

    def __init__(self, block_size: int, value_cls: type[NodeValue]):
        super().__init__(block_size)
        self._value_cls = value_cls

    @override
    def dump_block(self, data: Data) -> bytes:
        return Data(data.key, data.leaf, cast(NodeValue, data.value).dump()).dump_pad(self._block_size)

    @override
    def load_block(self, payload: bytes) -> Data:
        data = Data.load_unpad(payload)
        if data.is_real():
            data.value = self._value_cls.from_pickle(data=cast(bytes, data.value))
        return data
