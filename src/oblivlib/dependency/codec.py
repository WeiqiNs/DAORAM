"""Per-block serialization codecs for path encryption.

Bucket sealing (one ciphertext per bucket) lives in ``Helper.encrypt/decrypt_bucket``; a
``BlockCodec`` handles only the per-block step. It's the seam ``_encrypt_path_data`` /
``_decrypt_path_data`` drive via ``self._codec``, so a scheme storing non-trivial values (AVL/B+
nodes) supplies a codec instead of reimplementing path encryption.

A codec's bytes must match what the init seal (``Storage.encrypt``) writes, or ``decrypt_bucket``
splits at the wrong boundaries — see ARCHITECTURE.md §9.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Self, cast, override

from oblivlib.dependency.helper import Data


class BlockCodec(ABC):
    """Serializes one ``Data`` block to/from the fixed-width payload a bucket ciphertext packs."""

    @property
    @abstractmethod
    def block_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def dump_block(self, data: Data) -> bytes:
        """Serialize to exactly ``block_size`` bytes; must not mutate ``data``."""
        raise NotImplementedError

    @abstractmethod
    def load_block(self, payload: bytes) -> Data:
        """Inverse of ``dump_block``; a dummy payload loads as a dummy ``Data`` (key ``None``)."""
        raise NotImplementedError

    @abstractmethod
    def dummy_block(self) -> bytes:
        raise NotImplementedError


class DefaultCodec(BlockCodec):
    """Plain ORAM / ODS codec: a block is its own padded pickle and the value is stored verbatim."""

    def __init__(self, block_size: int):
        self._block_size = block_size

    @property
    @override
    def block_size(self) -> int:
        return self._block_size

    @override
    def dump_block(self, data: Data) -> bytes:
        return data.dump_pad(self._block_size)

    @override
    def load_block(self, payload: bytes) -> Data:
        return Data.load_unpad(payload)

    @override
    def dummy_block(self) -> bytes:
        return Data().dump_pad(self._block_size)


class NodeValue(Protocol):
    """A node payload (AVLData / BPlusData) that serializes to/from its own pickle bytes."""

    def dump(self) -> bytes: ...

    @classmethod
    def from_pickle(cls, data: bytes) -> Self: ...


class NodeCodec(BlockCodec):
    """ODS-node codec for the AVL and B+ maps: the block's value is a node object (``AVLData`` /
    ``BPlusData``) stored as its own pickle bytes."""

    def __init__(self, block_size: int, value_cls: type[NodeValue]):
        self._block_size = block_size
        self._value_cls = value_cls

    @property
    @override
    def block_size(self) -> int:
        return self._block_size

    @override
    def dump_block(self, data: Data) -> bytes:
        return Data(data.key, data.leaf, cast(NodeValue, data.value).dump()).dump_pad(self._block_size)

    @override
    def load_block(self, payload: bytes) -> Data:
        data = Data.load_unpad(payload)
        if data.is_real():
            data.value = self._value_cls.from_pickle(data=cast(bytes, data.value))
        return data

    @override
    def dummy_block(self) -> bytes:
        return Data().dump_pad(self._block_size)
