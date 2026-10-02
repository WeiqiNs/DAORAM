"""Bucket storage for tree nodes: in-memory or a pre-allocated file, sealed one ciphertext per bucket."""

import os
from abc import ABC, abstractmethod
from typing import BinaryIO, cast, override

from oblivlib.dependency.crypto import Encryptor
from oblivlib.dependency.helper import Data, Helper
from oblivlib.dependency.types import Bucket


class _Backend(ABC):
    """A row-addressable store of buckets; each row holds up to bucket_size blocks.

    Subclasses differ only in where bytes live (memory vs. disk); the read/write/encrypt/decrypt
    contract is identical.
    """

    def __init__(self, bucket_size: int, encryption: bool, data_size: int | None):
        self._bucket_size = bucket_size
        self._encryption = encryption
        self._data_size = data_size

    @abstractmethod
    def read_row(self, index: int) -> Bucket: ...

    @abstractmethod
    def write_row(self, index: int, data: Bucket) -> None: ...

    @abstractmethod
    def encrypt(self, encryptor: Encryptor) -> None:
        """Seal every bucket into one ciphertext, padding to bucket_size with dummies."""

    @abstractmethod
    def decrypt(self, encryptor: Encryptor) -> None:
        """Reverse ``encrypt``, restoring plaintext blocks (dummies kept)."""

    def close(self) -> None:  # noqa: B027
        ...

    def _bucket_to_blocks(self, bucket: list[Data]) -> list[bytes]:
        assert self._data_size is not None
        return [data.dump_pad(self._data_size) for data in bucket]

    @property
    def _dummy_block(self) -> bytes:
        assert self._data_size is not None
        return Data().dump_pad(self._data_size)


class _MemoryBackend(_Backend):
    """Buckets held as an in-memory list. A row is a list of Data while plaintext, or a single
    [ciphertext] element once encrypted."""

    def __init__(self, size: int, bucket_size: int, encryption: bool, data_size: int | None):
        super().__init__(bucket_size=bucket_size, encryption=encryption, data_size=data_size)
        self._internal_data: list[list] = [[] for _ in range(size)]

    @override
    def read_row(self, index: int) -> Bucket:
        return self._internal_data[index]

    @override
    def write_row(self, index: int, data: Bucket) -> None:
        self._internal_data[index] = data

    @override
    def encrypt(self, encryptor: Encryptor) -> None:
        dummy = self._dummy_block
        for i, bucket in enumerate(self._internal_data):
            blob = Helper.encrypt_bucket(encryptor, self._bucket_to_blocks(bucket), dummy, self._bucket_size)
            self._internal_data[i] = [blob]

    @override
    def decrypt(self, encryptor: Encryptor) -> None:
        assert self._data_size is not None
        for i, bucket in enumerate(self._internal_data):
            blocks = Helper.decrypt_bucket(encryptor, bucket[0], self._data_size)
            self._internal_data[i] = [Data.load_unpad(block) for block in blocks]


class _FileBackend(_Backend):
    """Buckets persisted to a pre-allocated backing file.

    Plaintext (or encrypted-but-not-yet-sealed) rows hold bucket_size fixed-width block slots;
    `encrypt()` seals each row into a single ciphertext blob and flips `_sealed`, after which a row
    holds one blob. `_row_bytes` is sized for the larger (sealed) form so the file never grows.
    """

    def __init__(
        self,
        filename: str,
        size: int,
        bucket_size: int,
        encryption: bool,
        data_size: int | None,
        disk_size: int,
    ):
        super().__init__(bucket_size=bucket_size, encryption=encryption, data_size=data_size)
        self._size = size
        self._sealed = False
        self._slot_size = data_size if data_size is not None else disk_size
        self._row_bytes = disk_size if encryption else bucket_size * disk_size

        total_bytes = size * self._row_bytes
        if not os.path.exists(filename) or os.path.getsize(filename) != total_bytes:
            with open(filename, "wb") as file:
                if total_bytes > 0:
                    file.seek(total_bytes - 1)
                    file.write(b"\x00")

        self._file: BinaryIO | None = open(filename, "r+b")

    def _row_offset(self, index: int) -> int:
        return index * self._row_bytes

    @override
    def read_row(self, index: int) -> Bucket:
        assert self._file is not None
        self._file.seek(self._row_offset(index))
        row = self._file.read(self._row_bytes)

        if self._encryption and self._sealed:
            return [row] if row.strip(b"\x00") else []

        zero = b"\x00" * self._slot_size
        slots = (row[i * self._slot_size : (i + 1) * self._slot_size] for i in range(self._bucket_size))
        return [Data.load_unpad(data=slot) for slot in slots if slot != zero]

    @override
    def write_row(self, index: int, data: Bucket) -> None:
        payload: bytes
        if self._encryption and self._sealed:
            blob = data[0] if data else b""
            assert isinstance(blob, bytes)
            payload = blob
        else:
            blocks = [elem.dump_pad(self._slot_size) if isinstance(elem, Data) else elem for elem in data]
            payload = b"".join(blocks)
        assert self._file is not None
        self._file.seek(self._row_offset(index))
        self._file.write(payload + b"\x00" * (self._row_bytes - len(payload)))

    @override
    def encrypt(self, encryptor: Encryptor) -> None:
        dummy = self._dummy_block
        assert self._file is not None
        for index in range(self._size):
            row = cast(list[Data], self.read_row(index))
            blocks = self._bucket_to_blocks(row)
            blob = Helper.encrypt_bucket(encryptor, blocks, dummy, self._bucket_size)
            self._file.seek(self._row_offset(index))
            self._file.write(blob + b"\x00" * (self._row_bytes - len(blob)))
        self._sealed = True

    @override
    def decrypt(self, encryptor: Encryptor) -> None:
        self._sealed = False
        assert self._file is not None
        assert self._data_size is not None
        for index in range(self._size):
            self._file.seek(self._row_offset(index))
            blocks = Helper.decrypt_bucket(encryptor, self._file.read(self._row_bytes), self._data_size)
            payload = b"".join(blocks)
            self._file.seek(self._row_offset(index))
            self._file.write(payload + b"\x00" * (self._row_bytes - len(payload)))

    @override
    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class Storage:
    """An n x m grid of fixed-size buckets (n = tree nodes, m = bucket_size), read/written one row at a
    time. File-backed and pre-allocated when given a filename, else in-memory; both go through
    interchangeable backends, so this class is just their shared front door."""

    def __init__(
        self,
        size: int,
        bucket_size: int,
        encryption: bool,
        filename: str | None = None,
        data_size: int | None = None,
        disk_size: int | None = None,
    ) -> None:
        if encryption and data_size is None:
            raise ValueError("Data size is required to be provided for encryption.")
        if filename is not None and disk_size is None:
            raise ValueError("Disk size is required to be provided when storing to a file.")

        self._backend: _Backend
        if filename is not None:
            assert disk_size is not None
            self._backend = _FileBackend(
                filename=filename,
                size=size,
                bucket_size=bucket_size,
                encryption=encryption,
                data_size=data_size,
                disk_size=disk_size,
            )
        else:
            self._backend = _MemoryBackend(
                size=size, bucket_size=bucket_size, encryption=encryption, data_size=data_size
            )

    def __getitem__(self, index: int) -> Bucket:
        return self._backend.read_row(index=index)

    def __setitem__(self, index: int, data: Bucket) -> None:
        self._backend.write_row(index=index, data=data)

    def encrypt(self, encryptor: Encryptor) -> None:
        """Seal every bucket into one ciphertext blob (pad to bucket_size with dummies first)."""
        self._backend.encrypt(encryptor=encryptor)

    def decrypt(self, encryptor: Encryptor) -> None:
        """Reverse ``encrypt``, restoring plaintext blocks (mainly for testing/one-time use)."""
        self._backend.decrypt(encryptor=encryptor)

    def close(self) -> None:
        self._backend.close()
