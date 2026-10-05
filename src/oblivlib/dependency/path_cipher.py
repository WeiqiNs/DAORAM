from abc import ABC, abstractmethod
from typing import override

import msgpack

from oblivlib.dependency.codec import BlockCodec
from oblivlib.dependency.crypto import Encryptor
from oblivlib.dependency.errors import RowSizeError
from oblivlib.dependency.types import Bucket, PathData, PathRows

_LENGTH_BYTES = 4
_MAX_ARRAY_HEADER_BYTES = 5


class PathCipher(ABC):
    """Turns a scheme's buckets into server rows and back. A bucket packs as a msgpack array of its
    blocks; an absent row (``b""``) opens as an empty bucket. ``row_bytes`` is the fixed row length, or
    ``None`` when rows vary in length."""

    def __init__(self, codec: BlockCodec):
        self._codec = codec

    @property
    @abstractmethod
    def row_bytes(self) -> int | None: ...

    @abstractmethod
    def seal_bucket(self, blocks: Bucket) -> bytes: ...

    @abstractmethod
    def _open_row(self, row: bytes) -> bytes: ...

    def _pack(self, blocks: Bucket) -> bytes:
        return msgpack.packb([self._codec.pack(data) for data in blocks])

    def open_bucket(self, row: bytes) -> Bucket:
        if row == b"":
            return []
        return [self._codec.unpack(fields) for fields in msgpack.unpackb(self._open_row(row))]

    def seal_path(self, path: PathData) -> PathRows:
        return {index: self.seal_bucket(bucket) for index, bucket in path.items()}

    def open_path(self, rows: PathRows) -> PathData:
        return {index: self.open_bucket(row) for index, row in rows.items()}


class PlainPathCipher(PathCipher):
    """Rows are the packed buckets themselves, of varying length: for debugging, memory servers only."""

    @property
    @override
    def row_bytes(self) -> None:
        return None

    @override
    def seal_bucket(self, blocks: Bucket) -> bytes:
        return self._pack(blocks)

    @override
    def _open_row(self, row: bytes) -> bytes:
        return row


class SealedPathCipher(PathCipher):
    """Each row is ``enc(u32_be(len) || packed bucket || zero padding)``, padded to the capacity of a
    full bucket of widest blocks, so every row has one length whatever it holds."""

    def __init__(self, codec: BlockCodec, encryptor: Encryptor, bucket_size: int):
        super().__init__(codec)
        self._encryptor = encryptor
        self._bucket_size = bucket_size
        self._capacity = _MAX_ARRAY_HEADER_BYTES + bucket_size * codec.max_block_bytes
        self._row_bytes = encryptor.ciphertext_length(_LENGTH_BYTES + self._capacity)

    @property
    @override
    def row_bytes(self) -> int:
        return self._row_bytes

    @override
    def seal_bucket(self, blocks: Bucket) -> bytes:
        if len(blocks) > self._bucket_size:
            raise RowSizeError(
                f"{type(self).__name__}: {len(blocks)} blocks exceed the bucket size {self._bucket_size}."
            )
        packed = self._pack(blocks)
        if len(packed) > self._capacity:
            raise RowSizeError(
                f"{type(self).__name__}: a {len(packed)}-byte bucket exceeds the {self._capacity}-byte row."
            )
        padding = bytes(self._capacity - len(packed))
        return self._encryptor.enc(len(packed).to_bytes(_LENGTH_BYTES, "big") + packed + padding)

    @override
    def _open_row(self, row: bytes) -> bytes:
        plaintext = self._encryptor.dec(row)
        length = int.from_bytes(plaintext[:_LENGTH_BYTES], "big")
        return plaintext[_LENGTH_BYTES : _LENGTH_BYTES + length]


def make_path_cipher(codec: BlockCodec, encryptor: Encryptor | None, bucket_size: int) -> PathCipher:
    if encryptor is None:
        return PlainPathCipher(codec)
    return SealedPathCipher(codec, encryptor, bucket_size)
