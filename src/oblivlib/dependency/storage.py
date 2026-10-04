from abc import ABC, abstractmethod
from typing import BinaryIO, override

from oblivlib.dependency.codec import BlockCodec
from oblivlib.dependency.crypto import Encryptor
from oblivlib.dependency.types import Bucket, Data


def _blocks_of(bucket: Bucket) -> list[Data]:
    blocks = [data for data in bucket if isinstance(data, Data)]
    if len(blocks) != len(bucket):
        raise TypeError("Storage: a plaintext bucket must hold only Data blocks, not sealed blobs.")
    return blocks


class _Backend(ABC):
    def __init__(self, bucket_size: int, codec: BlockCodec):
        self._bucket_size = bucket_size
        self._codec = codec

    @abstractmethod
    def read_row(self, index: int) -> Bucket: ...

    @abstractmethod
    def write_row(self, index: int, data: Bucket) -> None: ...

    @abstractmethod
    def seal(self, encryptor: Encryptor) -> None: ...

    @abstractmethod
    def resize(self, size: int) -> None: ...

    def close(self) -> None:  # noqa: B027
        ...


class _MemoryBackend(_Backend):
    def __init__(self, size: int, bucket_size: int, codec: BlockCodec):
        super().__init__(bucket_size=bucket_size, codec=codec)
        self._rows: list[Bucket] = [[] for _ in range(size)]

    @override
    def read_row(self, index: int) -> Bucket:
        return self._rows[index]

    @override
    def write_row(self, index: int, data: Bucket) -> None:
        self._rows[index] = data

    @override
    def seal(self, encryptor: Encryptor) -> None:
        for i, row in enumerate(self._rows):
            self._rows[i] = [self._codec.seal_bucket(encryptor, _blocks_of(row), self._bucket_size)]

    @override
    def resize(self, size: int) -> None:
        del self._rows[size:]
        self._rows.extend([] for _ in range(size - len(self._rows)))


class _FileBackend(_Backend):
    def __init__(self, filename: str, size: int, bucket_size: int, codec: BlockCodec, row_bytes: int):
        super().__init__(bucket_size=bucket_size, codec=codec)
        self._size = size
        self._sealed = False
        self._row_bytes = row_bytes
        self._file: BinaryIO = open(filename, "w+b")
        self._file.truncate(size * row_bytes)

    def _write_payload(self, index: int, payload: bytes) -> None:
        if len(payload) > self._row_bytes:
            raise ValueError(
                f"{type(self).__name__}: row {index} payload is {len(payload)} bytes, row holds {self._row_bytes}."
            )
        self._file.seek(index * self._row_bytes)
        self._file.write(payload + b"\x00" * (self._row_bytes - len(payload)))

    @override
    def read_row(self, index: int) -> Bucket:
        self._file.seek(index * self._row_bytes)
        row = self._file.read(self._row_bytes)

        if self._sealed:
            return [row] if row.strip(b"\x00") else []

        slot = self._codec.block_size
        zero = b"\x00" * slot
        slots = (row[i * slot : (i + 1) * slot] for i in range(self._bucket_size))
        return [self._codec.load_block(payload) for payload in slots if payload != zero]

    @override
    def write_row(self, index: int, data: Bucket) -> None:
        if self._sealed:
            blob = data[0] if data else b""
            assert isinstance(blob, bytes)
            self._write_payload(index, blob)
            return

        if len(data) > self._bucket_size:
            raise ValueError(
                f"{type(self).__name__}: row {index} got {len(data)} blocks, bucket holds {self._bucket_size}."
            )
        self._write_payload(index, b"".join(self._codec.dump_block(block) for block in _blocks_of(data)))

    @override
    def seal(self, encryptor: Encryptor) -> None:
        for index in range(self._size):
            blob = self._codec.seal_bucket(encryptor, _blocks_of(self.read_row(index)), self._bucket_size)
            self._write_payload(index, blob)
        self._sealed = True

    @override
    def resize(self, size: int) -> None:
        self._file.truncate(size * self._row_bytes)
        self._size = size

    @override
    def close(self) -> None:
        self._file.close()


class Storage:
    def __init__(
        self,
        size: int,
        bucket_size: int,
        codec: BlockCodec,
        encryptor: Encryptor | None = None,
        filename: str | None = None,
    ) -> None:
        self._backend: _Backend
        if filename is None:
            self._backend = _MemoryBackend(size=size, bucket_size=bucket_size, codec=codec)
            return

        plaintext_row = bucket_size * codec.block_size
        row_bytes = encryptor.ciphertext_length(plaintext_row) if encryptor is not None else plaintext_row
        self._backend = _FileBackend(
            filename=filename, size=size, bucket_size=bucket_size, codec=codec, row_bytes=row_bytes
        )

    def __getitem__(self, index: int) -> Bucket:
        return self._backend.read_row(index=index)

    def __setitem__(self, index: int, data: Bucket) -> None:
        self._backend.write_row(index=index, data=data)

    def seal(self, encryptor: Encryptor) -> None:
        self._backend.seal(encryptor=encryptor)

    def resize(self, size: int) -> None:
        self._backend.resize(size=size)

    def close(self) -> None:
        self._backend.close()
