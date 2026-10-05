from collections.abc import Iterable
from typing import Self

from oblivlib.dependency.client import Client
from oblivlib.dependency.heap_index import compute_level, path_indices
from oblivlib.dependency.path_cipher import PathCipher
from oblivlib.dependency.tree_builder import build_tree
from oblivlib.dependency.types import Data, PathData


class FlexibleBinaryTree:
    """A resizable tree hosted under ``label``, whose every node is present (a bucket, possibly empty) or
    absent (an empty row). Leaves are int labels at the current ``level``; resizing moves no data."""

    def __init__(self, *, client: Client, label: str, cipher: PathCipher):
        self._client = client
        self._label = label
        self._cipher = cipher

    @classmethod
    def create(
        cls,
        *,
        client: Client,
        label: str,
        num_data: int,
        bucket_size: int,
        cipher: PathCipher,
        blocks: Iterable[Data] = (),
    ) -> tuple[Self, list[Data]]:
        """Host a tree whose nodes all start present, holding ``blocks`` at the deepest free bucket on
        each one's path; returns the tree and the blocks that did not fit."""
        result = build_tree(
            blocks, level=compute_level(num_data), bucket_size=bucket_size, cipher=cipher, build_file=None
        )
        client.host_tree(label, result.image)
        return cls(client=client, label=label, cipher=cipher), result.overflow

    @property
    def level(self) -> int:
        return self._client.level_of(self._label)

    def read_path(self, leaves: list[int]) -> PathData:
        """The present nodes on the paths to ``leaves``, root first."""
        self._client.add_read_path(self._label, leaves)
        rows = self._client.execute().require(self._label)
        return {index: self._cipher.open_bucket(row) for index, row in rows.items() if row != b""}

    def write_path(self, leaves: list[int], data: PathData) -> None:
        """Write each bucket in ``data`` (making it present) and clear every other node on the paths."""
        nodes = path_indices(leaves, self.level)
        off_path = sorted(set(data) - set(nodes))
        if off_path:
            raise ValueError(f"{type(self).__name__}: indices {off_path} are not on the paths of leaves {leaves}.")
        rows = {index: self._cipher.seal_bucket(data[index]) if index in data else b"" for index in nodes}
        self._client.add_write_path(self._label, rows)
        self._client.execute()

    def scale_up(self) -> None:
        self._client.resize(self._label, level=self.level + 1)

    def scale_down(self) -> None:
        """Drop the bottom layer; raises ``ScaleDownError`` at level 1 or while a bottom node is present."""
        self._client.resize(self._label, level=self.level - 1)
