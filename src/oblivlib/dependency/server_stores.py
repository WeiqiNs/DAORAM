import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import override

from oblivlib.dependency.errors import ProtocolError, RowSizeError, ScaleDownError
from oblivlib.dependency.heap_index import tree_size
from oblivlib.dependency.types import ListOp, ListPopBack, ListPushFront, ListWrite


class _Region(ABC):
    @abstractmethod
    def read(self, offset: int, length: int) -> bytes: ...

    @abstractmethod
    def write(self, offset: int, data: bytes) -> None: ...

    @abstractmethod
    def truncate(self, length: int) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class _MemoryRegion(_Region):
    def __init__(self, length: int):
        self._buffer = bytearray(length)

    @override
    def read(self, offset: int, length: int) -> bytes:
        return bytes(self._buffer[offset : offset + length])

    @override
    def write(self, offset: int, data: bytes) -> None:
        self._buffer[offset : offset + len(data)] = data

    @override
    def truncate(self, length: int) -> None:
        del self._buffer[length:]
        self._buffer.extend(bytes(length - len(self._buffer)))

    @override
    def close(self) -> None:
        self._buffer = bytearray()


class _FileRegion(_Region):
    def __init__(self, path: Path):
        self._fd = os.open(path, os.O_RDWR)

    @override
    def read(self, offset: int, length: int) -> bytes:
        return os.pread(self._fd, length, offset)

    @override
    def write(self, offset: int, data: bytes) -> None:
        os.pwrite(self._fd, data, offset)

    @override
    def truncate(self, length: int) -> None:
        os.ftruncate(self._fd, length)

    @override
    def close(self) -> None:
        os.close(self._fd)


class TreeStore(ABC):
    """A server-side tree: ``2**level - 1`` rows in heap order, where ``b""`` is an absent row."""

    def __init__(self, level: int):
        self._level = level

    @property
    def level(self) -> int:
        return self._level

    @property
    def size(self) -> int:
        return tree_size(self._level)

    @abstractmethod
    def check_row(self, row: bytes) -> None: ...

    @abstractmethod
    def read_rows(self, indices: list[int]) -> list[bytes]: ...

    @abstractmethod
    def write_rows(self, indices: list[int], rows: list[bytes]) -> None: ...

    @abstractmethod
    def _row_present(self, index: int) -> bool: ...

    @abstractmethod
    def _set_size(self, size: int) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    def write_range(self, start: int, rows: list[bytes]) -> None:
        self.write_rows(list(range(start, start + len(rows))), rows)

    def resize(self, level: int) -> None:
        if level < 1:
            raise ScaleDownError(f"{type(self).__name__}: cannot resize below level 1, got {level}.")
        size = tree_size(level)
        if any(self._row_present(index) for index in range(size, self.size)):
            raise ScaleDownError(f"{type(self).__name__}: rows past level {level} are still present.")
        self._set_size(size)
        self._level = level


class FixedRowStore(TreeStore):
    """Rows of exactly ``row_bytes`` (sealed buckets) in one region; an all-zero row is absent."""

    def __init__(self, level: int, row_bytes: int, region: _Region):
        super().__init__(level)
        self._row_bytes = row_bytes
        self._zero_row = bytes(row_bytes)
        self._region = region

    @classmethod
    def in_memory(cls, level: int, row_bytes: int) -> "FixedRowStore":
        return cls(level, row_bytes, _MemoryRegion(tree_size(level) * row_bytes))

    @classmethod
    def create_file(cls, path: Path, level: int, row_bytes: int) -> "FixedRowStore":
        os.close(os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600))
        store = cls(level, row_bytes, _FileRegion(path))
        store._set_size(tree_size(level))
        return store

    @classmethod
    def open_file(cls, path: Path, level: int, row_bytes: int) -> "FixedRowStore":
        if path.stat().st_size != tree_size(level) * row_bytes:
            raise RowSizeError(
                f"{cls.__name__}: {path.name} is {path.stat().st_size} bytes, "
                + f"a level-{level} tree of {row_bytes}-byte rows is {tree_size(level) * row_bytes}."
            )
        return cls(level, row_bytes, _FileRegion(path))

    @override
    def check_row(self, row: bytes) -> None:
        if len(row) not in (0, self._row_bytes):
            raise RowSizeError(f"{type(self).__name__}: a row is {len(row)} bytes, rows hold {self._row_bytes}.")

    def _read_row(self, index: int) -> bytes:
        row = self._region.read(index * self._row_bytes, self._row_bytes)
        return b"" if row == self._zero_row else row

    @override
    def read_rows(self, indices: list[int]) -> list[bytes]:
        return [self._read_row(index) for index in indices]

    @override
    def write_rows(self, indices: list[int], rows: list[bytes]) -> None:
        for index, row in zip(indices, rows, strict=True):
            self._region.write(index * self._row_bytes, row or self._zero_row)

    @override
    def _row_present(self, index: int) -> bool:
        return self._read_row(index) != b""

    @override
    def _set_size(self, size: int) -> None:
        self._region.truncate(size * self._row_bytes)

    @override
    def close(self) -> None:
        self._region.close()


class VariableRowStore(TreeStore):
    """Rows of any length (plaintext debugging trees), memory only."""

    def __init__(self, level: int):
        super().__init__(level)
        self._rows: list[bytes] = [b""] * tree_size(level)

    @override
    def check_row(self, row: bytes) -> None:
        pass

    @override
    def read_rows(self, indices: list[int]) -> list[bytes]:
        return [self._rows[index] for index in indices]

    @override
    def write_rows(self, indices: list[int], rows: list[bytes]) -> None:
        for index, row in zip(indices, rows, strict=True):
            self._rows[index] = row

    @override
    def _row_present(self, index: int) -> bool:
        return self._rows[index] != b""

    @override
    def _set_size(self, size: int) -> None:
        del self._rows[size:]
        self._rows.extend([b""] * (size - len(self._rows)))

    @override
    def close(self) -> None:
        self._rows = []


class ListStore:
    def __init__(self) -> None:
        self._values: list[bytes] = []

    def __len__(self) -> int:
        return len(self._values)

    def apply(self, op: ListOp) -> None:
        match op:
            case ListWrite(index=index, value=value):
                self._values[index] = value
            case ListPushFront(value=value):
                self._values.insert(0, value)
            case ListPopBack():
                self._values.pop()

    def read(self, indices: list[int] | None) -> list[bytes]:
        if indices is None:
            return list(self._values)
        return [self._values[index] for index in indices]


def simulate_list_length(length: int, ops: list[ListOp]) -> int:
    for op in ops:
        match op:
            case ListWrite(index=index):
                if index >= length:
                    raise ProtocolError(f"ListWrite index {index} is out of range for a list of {length}.")
            case ListPushFront():
                length += 1
            case ListPopBack():
                if length == 0:
                    raise ProtocolError("ListPopBack on an empty list.")
                length -= 1
    return length
