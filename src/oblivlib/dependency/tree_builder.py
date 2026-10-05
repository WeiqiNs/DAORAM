import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import override

from oblivlib.dependency.heap_index import leaf_index, path_to_root, tree_size
from oblivlib.dependency.path_cipher import PathCipher
from oblivlib.dependency.types import Bucket, Data


@dataclass(frozen=True)
class MemoryImage:
    level: int
    row_bytes: int | None
    rows: list[bytes]


@dataclass(frozen=True)
class FileImage:
    """A built tree on disk in the server's format: ``2**level - 1`` sealed rows of ``row_bytes`` each,
    in heap order from the root."""

    level: int
    row_bytes: int
    path: Path


TreeImage = MemoryImage | FileImage


@dataclass(frozen=True)
class BuildResult:
    image: TreeImage
    overflow: list[Data]


class _Buckets(ABC):
    @abstractmethod
    def read(self, index: int) -> Bucket: ...

    @abstractmethod
    def write(self, index: int, bucket: Bucket) -> None: ...

    @abstractmethod
    def finish(self) -> TreeImage: ...


class _MemoryBuckets(_Buckets):
    def __init__(self, level: int, cipher: PathCipher):
        self._level = level
        self._cipher = cipher
        self._buckets: list[Bucket] = [[] for _ in range(tree_size(level))]

    @override
    def read(self, index: int) -> Bucket:
        return self._buckets[index]

    @override
    def write(self, index: int, bucket: Bucket) -> None:
        self._buckets[index] = bucket

    @override
    def finish(self) -> MemoryImage:
        rows = [self._cipher.seal_bucket(bucket) for bucket in self._buckets]
        return MemoryImage(level=self._level, row_bytes=self._cipher.row_bytes, rows=rows)


class _FileBuckets(_Buckets):
    def __init__(self, level: int, cipher: PathCipher, path: Path):
        if cipher.row_bytes is None:
            raise ValueError("build_tree: a build file needs fixed-size (encrypted) rows.")
        self._level = level
        self._cipher = cipher
        self._row_bytes = cipher.row_bytes
        self._path = path
        self._fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        for index in range(tree_size(level)):
            self.write(index, [])

    @override
    def read(self, index: int) -> Bucket:
        return self._cipher.open_bucket(os.pread(self._fd, self._row_bytes, index * self._row_bytes))

    @override
    def write(self, index: int, bucket: Bucket) -> None:
        os.pwrite(self._fd, self._cipher.seal_bucket(bucket), index * self._row_bytes)

    @override
    def finish(self) -> FileImage:
        os.close(self._fd)
        return FileImage(level=self._level, row_bytes=self._row_bytes, path=self._path)


def build_tree(
    blocks: Iterable[Data], *, level: int, bucket_size: int, cipher: PathCipher, build_file: Path | None
) -> BuildResult:
    """Place each block in the deepest non-full bucket on its leaf's path and seal every bucket into a
    row, in memory or, with ``build_file``, in that file (sealed throughout, never plaintext on disk).
    Blocks whose whole path is full come back as ``overflow``."""
    buckets = _MemoryBuckets(level, cipher) if build_file is None else _FileBuckets(level, cipher, build_file)
    overflow: list[Data] = []
    for block in blocks:
        for index in path_to_root(leaf_index(block.require_leaf(), level)):
            bucket = buckets.read(index)
            if len(bucket) < bucket_size:
                bucket.append(block)
                buckets.write(index, bucket)
                break
        else:
            overflow.append(block)
    return BuildResult(image=buckets.finish(), overflow=overflow)
