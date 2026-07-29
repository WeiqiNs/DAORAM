"""This module implements a flexible binary tree whose leaves may live at different levels.

Unlike `BinaryTree`, where every leaf sits at the deepest level and is named by a single int, a
`FlexibleBinaryTree` leaf is labelled by a ``(leaf, level)`` tuple: ``leaf`` is the 0-based position
within ``level`` and ``level`` is the level (1 = root level) the leaf currently lives at. The storage
index of such a leaf is ``2 ** (level - 1) - 1 + leaf`` (`get_actual_leaf_index`). The tree can grow
(`scale_up`) or shrink (`scale_down`) by one level while preserving the data already stored; the
level component of a label is what lets a block be re-located across those resizes.
"""

from typing import cast

from oblivlib.dependency.helper import Data
from oblivlib.dependency.storage import Storage
from oblivlib.dependency.types import Block, Buckets


class FlexibleBinaryTree:
    def __init__(
        self,
        num_data: int,
        bucket_size: int,
        filename: str | None = None,
        data_size: int | None = None,
        disk_size: int | None = None,
        encryption: bool = False,
    ) -> None:
        self._num_data = num_data
        self._bucket_size = bucket_size
        self._filename = filename
        self._data_size = data_size
        self._disk_size = disk_size
        self._encryption = encryption

        self._level = (num_data - 1).bit_length() + 1
        self._size = (1 << self._level) - 1
        # Storage index of leaf 0 (= the number of internal nodes); leaves are start_leaf..size-1.
        self._start_leaf = (1 << (self._level - 1)) - 1

        self._storage = Storage(
            size=self._size,
            filename=filename,
            data_size=data_size,
            disk_size=disk_size,
            bucket_size=bucket_size,
            encryption=encryption,
        )

    @property
    def size(self) -> int:
        return self._size

    @property
    def level(self) -> int:
        return self._level

    @property
    def start_leaf(self) -> int:
        return self._start_leaf

    @property
    def storage(self) -> Storage:
        return self._storage

    @staticmethod
    def get_parent_index(index: int) -> int:
        return (index - 1) // 2

    @staticmethod
    def get_child_index(index: int) -> int:
        return index * 2 + 1

    @staticmethod
    def get_path_indices(index: int) -> list[int]:
        """Storage indices from the given node up to (and including) the root."""
        path = []
        while index >= 0:
            path.append(index)
            index = FlexibleBinaryTree.get_parent_index(index)

        return path

    @staticmethod
    def get_mul_path_indices(indices: list[int]) -> list[int]:
        """Union of the node-to-root paths, deduplicated and sorted deepest-first."""
        path = set()

        for index in indices:
            path.update(FlexibleBinaryTree.get_path_indices(index=index))

        return sorted(path, reverse=True)

    @staticmethod
    def get_mul_path_dict(level: int, indices: list[int]) -> dict[int, list]:
        start_leaf = (1 << (level - 1)) - 1
        indices = [index + start_leaf for index in indices]
        return {index: [] for index in FlexibleBinaryTree.get_mul_path_indices(indices=indices)}

    @staticmethod
    def fill_buckets_with_dummy_data(buckets: Buckets, bucket_size: int) -> None:
        for bucket in buckets:
            while len(bucket) < bucket_size:
                bucket.append(Data())

    @staticmethod
    def get_cross_index(leaf_a: int, leaf_b: int) -> int:
        """Lowest common ancestor of two *storage indices* (not leaf labels), found by walking both up
        parent-by-parent. Callers holding leaf labels convert via ``get_actual_leaf_index`` first."""
        while leaf_a != leaf_b:
            leaf_a = (leaf_a - 1) // 2
            leaf_b = (leaf_b - 1) // 2

        return leaf_a

    @staticmethod
    def get_cross_index_level(leaf_a: int, leaf_b: int) -> int:
        """Given two node storage indices, return the tree depth at which their paths cross."""
        index = FlexibleBinaryTree.get_cross_index(leaf_a=leaf_a, leaf_b=leaf_b)
        return (index + 1).bit_length() - 1

    @staticmethod
    def fill_data_to_path(data: Data, path: Buckets, leaf: tuple[int, int], bucket_size: int, level: int) -> bool:
        """Place data in the lowest legal bucket of a single leaf-to-root path; return False if full.

        The block's leaf and the path's leaf may live at different levels, so both are brought to a
        common level before computing where their paths cross.
        """
        # The block's leaf is a (leaf, level) label, but data.leaf is typed int | None on the shared
        # Data class, so reinterpret it (via object) as the tuple.
        data_leaf = cast(tuple[int, int], cast(object, data.leaf))
        leaf_a_index = (1 << (data_leaf[1] - 1)) - 1 + data_leaf[0]
        leaf_b_index = (1 << (leaf[1] - 1)) - 1 + leaf[0]
        leaf_a_index, leaf_b_index = FlexibleBinaryTree.adjust_to_same_level(
            leaf_a_index, leaf_b_index, data_leaf[1], leaf[1]
        )

        # Deepest legal bucket: the crossing depth, capped at the path's own depth.
        index = FlexibleBinaryTree.get_cross_index_level(leaf_a=leaf_a_index, leaf_b=leaf_b_index)
        index = min(index, level - 1)

        for path_index in range(index, -1, -1):
            if len(path[path_index]) < bucket_size:
                path[path_index].append(data)
                return True

        return False

    @staticmethod
    def fill_data_to_path_dict(data: Data, path: dict, leaf: tuple[int, int], bucket_size: int) -> bool:
        """Place data in the lowest legal bucket of a path dict (storage_index -> bucket); return False if full."""
        index = FlexibleBinaryTree.get_cross_index(
            leaf_a=FlexibleBinaryTree.get_actual_leaf_index(cast(tuple[int, int], cast(object, data.leaf))),
            leaf_b=FlexibleBinaryTree.get_actual_leaf_index(leaf),
        )

        while index >= 0:
            if len(path[index]) < bucket_size:
                path[index].append(data)
                return True
            else:
                index = FlexibleBinaryTree.get_parent_index(index=index)

        return False

    @staticmethod
    def fill_data_to_mul_path(data: Data, path: dict, leaves: list[int], bucket_size: int) -> bool:
        """Place data in the lowest legal bucket of a multi-path dict; return False if full.

        ``leaves`` are storage indices; the block may sit no deeper than the deepest point at which
        its own leaf crosses any of them.
        """
        # Here the block's leaf is already a storage index (an int), as are the target leaves.
        data_leaf = cast(int, data.leaf)
        max_index = max(FlexibleBinaryTree.get_cross_index(leaf_a=data_leaf, leaf_b=leaf) for leaf in leaves)

        while max_index >= 0:
            if len(path[max_index]) < bucket_size:
                path[max_index].append(data)
                return True
            else:
                max_index = FlexibleBinaryTree.get_parent_index(index=max_index)

        return False

    @staticmethod
    def adjust_to_same_level(
        leaf_a_index: int, leaf_b_index: int, leaf_a_level: int, leaf_b_level: int
    ) -> tuple[int, int]:
        """Bring two node indices (at the given levels) to the same level by walking the deeper one up."""
        gap = leaf_a_level - leaf_b_level
        if gap > 0:
            for _ in range(gap):
                leaf_a_index = FlexibleBinaryTree.get_parent_index(leaf_a_index)
        else:
            for _ in range(-gap):
                leaf_b_index = FlexibleBinaryTree.get_parent_index(leaf_b_index)
        return leaf_a_index, leaf_b_index

    @staticmethod
    def adjust_to_lowest_level(actual_index: int, old_level: int, level: int) -> int:
        """Move a node index from old_level to the tree's deepest level (descend via left children)."""
        gap = level - old_level
        if gap > 0:
            for _ in range(gap):
                actual_index = FlexibleBinaryTree.get_child_index(index=actual_index)
        else:
            for _ in range(-gap):
                actual_index = FlexibleBinaryTree.get_parent_index(index=actual_index)
        return actual_index

    @staticmethod
    def get_actual_leaf_index(leaf_info: tuple[int, int]) -> int:
        """Convert a (leaf, level) label to its storage index: 2 ** (level - 1) - 1 + leaf."""
        leaf, level = leaf_info
        return (1 << (level - 1)) - 1 + leaf

    def scale_up(self) -> None:
        """Grow the tree by one level, copying every existing bucket to the same storage index.

        The old tree is the top of the new one: a node keeps its storage index, so the copy is a
        straight index-for-index transfer and only the new bottom level starts empty.
        """
        new_level = self._level + 1
        new_size = (1 << new_level) - 1
        new_start_leaf = (1 << (new_level - 1)) - 1

        new_storage = Storage(
            size=new_size,
            filename=self._filename,
            data_size=self._data_size,
            disk_size=self._disk_size,
            bucket_size=self._bucket_size,
            encryption=self._encryption,
        )

        for i in range(self._size):
            new_storage[i] = self._storage[i]

        self._level = new_level
        self._size = new_size
        self._start_leaf = new_start_leaf
        self._storage = new_storage

    def scale_down(self) -> bool:
        """Shrink the tree by one level; return False (without changing anything) if data would be lost.

        Feasible only when every leaf path can hold its blocks after losing the bottom level (checked
        by `_can_scale_down`). When feasible, the bottom-level blocks are first pushed up into the
        retained nodes (`_remove_bottom_layer`), then the retained prefix of storage is copied over.
        """
        new_level = self._level - 1
        new_size = (1 << new_level) - 1
        new_start_leaf = (1 << (new_level - 1)) - 1

        if not self._can_scale_down():
            return False

        new_storage = Storage(
            size=new_size,
            filename=self._filename,
            data_size=self._data_size,
            disk_size=self._disk_size,
            bucket_size=self._bucket_size,
            encryption=self._encryption,
        )

        self._remove_bottom_layer()
        for i in range(new_size):
            new_storage[i] = self._storage[i]

        self._level = new_level
        self._size = new_size
        self._start_leaf = new_start_leaf
        self._storage = new_storage

        return True

    def _can_scale_down(self) -> bool:
        """Return True if every leaf path has room for its blocks once the bottom level is dropped."""
        num_leaves = 1 << (self._level - 1)

        for leaf_index in range(num_leaves):
            path_indices = self.get_path_indices(self._start_leaf + leaf_index)
            path_block_counts = sum(len(self._storage[bucket_index]) for bucket_index in path_indices)

            # After dropping the leaf bucket, the remaining (level - 2) buckets must hold what's left.
            if (path_block_counts - self._bucket_size) > (self._level - 2) * self._bucket_size:
                return False

        return True

    def _upthrust_data(self, index: int) -> bool:
        """Push a leaf bucket's blocks up into the first non-full ancestor; return True if any remain."""
        path_index = self.get_parent_index(self._start_leaf + index)
        target = self._storage[self._start_leaf + index]
        while target and path_index > 0:
            if len(self._storage[path_index]) < self._bucket_size:
                self._storage[path_index].append(target.pop())
            else:
                path_index = self.get_parent_index(index=path_index)
        return bool(target)

    def _remove_bottom_layer(self) -> None:
        """Push every bottom-level leaf bucket up into the nodes that survive a scale down."""
        for leaf in range(1 << (self._level - 1)):
            self._upthrust_data(leaf)

    def fill_data_to_storage_leaf(self, data: Data) -> bool:
        """Place data in the lowest non-full bucket on its leaf path; return False if the path is full."""
        # The block's leaf is a (leaf, level) label, but data.leaf is typed int | None on the shared
        # Data class, so reinterpret it (via object) as the tuple.
        for path_index in self.get_leaf_path(leaf=cast(tuple[int, int], cast(object, data.leaf))):
            if len(self._storage[path_index]) < self._bucket_size:
                self._storage[path_index] = self._storage[path_index] + [data]
                return True

        return False

    def get_leaf_path(self, leaf: tuple[int, int]) -> list[int]:
        """Storage indices from a (leaf, level) label — descended to the tree's deepest level — up to the root."""
        actual_index = FlexibleBinaryTree.get_actual_leaf_index(leaf)
        actual_index = self.adjust_to_lowest_level(actual_index, leaf[1], self._level)
        return self.get_path_indices(index=actual_index)

    def get_mul_leaf_path(self, leaves: list[tuple[int, int]]) -> list[int]:
        actual_indices = [FlexibleBinaryTree.get_actual_leaf_index(leaf) for leaf in leaves]
        actual_indices = [
            self.adjust_to_lowest_level(index, leaf[1], self._level)
            for index, leaf in zip(actual_indices, leaves, strict=True)
        ]
        return self.get_mul_path_indices(indices=actual_indices)

    def get_leaf_block(self, leaf: tuple[int, int], index: int) -> int:
        """Storage index of the bucket at depth ``index`` (0 = root) on the given leaf's path."""
        return self.get_leaf_path(leaf=leaf)[-index - 1]

    def read_path(self, leaf: tuple[int, int] | list[tuple[int, int]]) -> Buckets:
        """Read the buckets along the path(s) to one leaf label or a list of them (leaf(s) to root)."""
        if isinstance(leaf, tuple):
            path_to_read = self.get_leaf_path(leaf=leaf)
        else:
            path_to_read = self.get_mul_leaf_path(leaves=leaf)

        return [self._storage[data_index] for data_index in path_to_read]

    def write_path(self, leaf: tuple[int, int] | list[tuple[int, int]], data: Buckets) -> None:
        """Write buckets back along the path(s) to one leaf label or a list of them."""
        if isinstance(leaf, tuple):
            path_to_write = self.get_leaf_path(leaf=leaf)
        else:
            path_to_write = self.get_mul_leaf_path(leaves=leaf)
            if len(data) != len(path_to_write):
                raise ValueError("Wrong number of buckets on a path.")

        for i, path_index in enumerate(path_to_write):
            self._storage[path_index] = data[i]

    def read_block(self, leaf: tuple[int, int], bucket_id: int, block_id: int) -> Block:
        return self._storage[self.get_leaf_block(leaf=leaf, index=bucket_id)][block_id]

    def write_block(self, leaf: tuple[int, int], bucket_id: int, block_id: int, data: Block) -> None:
        index_to_write = self.get_leaf_block(leaf=leaf, index=bucket_id)
        bucket_data = self._storage[index_to_write]
        bucket_data[block_id] = data
        self._storage[index_to_write] = bucket_data
