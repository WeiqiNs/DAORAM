"""A complete binary tree laid over ``Storage`` (heap-indexed), with the path/bucket/block accessors
and the O(1) crossing-index math that tree-ORAM eviction relies on."""

from collections import defaultdict

from oblivlib.dependency.helper import Data
from oblivlib.dependency.storage import Storage
from oblivlib.dependency.types import Block, BlockData, BlockKey, BucketData, BucketKey, Buckets, PathData


class BinaryTree:
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

        self._level = self.compute_level(num_data)
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
    def compute_level(num_data: int) -> int:
        """Smallest level whose leaf count 2**(level-1) is >= num_data."""
        return (num_data - 1).bit_length() + 1

    @staticmethod
    def get_parent_index(index: int) -> int:
        return (index - 1) // 2

    @staticmethod
    def get_path_indices(index: int) -> list[int]:
        """Storage indices from the given node up to (and including) the root."""
        path = []
        while index >= 0:
            path.append(index)
            index = BinaryTree.get_parent_index(index)

        return path

    @staticmethod
    def get_mul_path_indices(indices: list[int]) -> list[int]:
        """Union of the node-to-root paths, deduplicated and sorted deepest-first."""
        path = set()

        for index in indices:
            path.update(BinaryTree.get_path_indices(index=index))

        return sorted(path, reverse=True)

    @staticmethod
    def get_mul_path_dict(level: int, indices: list[int]) -> PathData:
        start_leaf = (1 << (level - 1)) - 1
        indices = [index + start_leaf for index in indices]
        return {index: [] for index in BinaryTree.get_mul_path_indices(indices=indices)}

    @staticmethod
    def get_cross_index(leaf_a: int, leaf_b: int, level: int) -> int:
        """Given two leaf labels, return the storage index of their lowest common ancestor.

        Both leaves sit at the deepest level, so as 1-based heap indices they share a binary prefix.
        The highest set bit of their XOR is the deepest level at which they diverge, so shifting both
        past it yields the common ancestor in O(1) -- vs. walking up parent-by-parent in O(depth),
        which is the inner loop of stash eviction.
        """
        start_leaf = (1 << (level - 1)) - 1
        a = leaf_a + start_leaf + 1
        b = leaf_b + start_leaf + 1
        return (a >> (a ^ b).bit_length()) - 1

    @staticmethod
    def get_cross_index_level(leaf_a: int, leaf_b: int, level: int) -> int:
        """Given two leaf labels, return the tree depth at which their paths cross."""
        index = BinaryTree.get_cross_index(leaf_a=leaf_a, leaf_b=leaf_b, level=level)
        return (index + 1).bit_length() - 1

    @staticmethod
    def fill_data_to_path(data: Data, path: PathData, leaves: list[int], level: int, bucket_size: int) -> bool:
        """Place data in the lowest bucket of the PathData dict with room; return False if none has room."""
        data_leaf = data.require_leaf()
        # Deepest bucket on its own path that the path set covers: the lowest crossing point between the
        # block's leaf and any target leaf.
        max_index = max(BinaryTree.get_cross_index(leaf_a=data_leaf, leaf_b=leaf, level=level) for leaf in leaves)

        while max_index >= 0:
            if len(path[max_index]) < bucket_size:
                path[max_index].append(data)
                return True
            max_index = BinaryTree.get_parent_index(index=max_index)

        return False

    def fill_data_to_storage_leaf(self, data: Data) -> bool:
        """Place data in the lowest non-full bucket on its leaf path; return False if the path is full."""
        for path_index in self.get_leaf_path(leaf=data.require_leaf()):
            bucket = self._storage[path_index]
            if len(bucket) < self._bucket_size:
                bucket.append(data)
                self._storage[path_index] = bucket
                return True

        return False

    def get_leaf_path(self, leaf: int) -> list[int]:
        return self.get_path_indices(index=leaf + self._start_leaf)

    def get_mul_leaf_path(self, leaves: list[int]) -> list[int]:
        return self.get_mul_path_indices(indices=[leaf + self._start_leaf for leaf in leaves])

    def get_leaf_block(self, leaf: int, index: int) -> int:
        """Storage index of the bucket at depth ``index`` (0 = root) on the given leaf's path."""
        return self.get_leaf_path(leaf=leaf)[-index - 1]

    def read_path(self, leaves: list[int]) -> PathData:
        indices = self.get_mul_leaf_path(leaves)
        return {idx: self._storage[idx] for idx in indices}

    def write_path(self, data: PathData) -> None:
        for idx, bucket in data.items():
            self._storage[idx] = bucket

    def extract_path(self, leaf: int, data: PathData) -> Buckets:
        """A single leaf's path (leaf→root) pulled from a PathData dict; raises if any bucket is missing."""
        indices = self.get_leaf_path(leaf)
        missing = [idx for idx in indices if idx not in data]
        if missing:
            raise KeyError(f"Path for leaf {leaf} not fully contained. Missing indices: {missing}")
        return [data[idx] for idx in indices]

    def read_bucket(self, keys: list[BucketKey]) -> BucketData:
        return {
            BucketKey(leaf, bucket_id): self._storage[self.get_leaf_block(leaf, bucket_id)] for leaf, bucket_id in keys
        }

    def write_bucket(self, data: BucketData) -> None:
        for (leaf, bucket_id), bucket in data.items():
            self._storage[self.get_leaf_block(leaf, bucket_id)] = bucket

    def read_block(self, keys: list[BlockKey]) -> BlockData:
        return {
            BlockKey(leaf, bucket_id, block_id): self._storage[self.get_leaf_block(leaf, bucket_id)][block_id]
            for leaf, bucket_id, block_id in keys
        }

    def write_block(self, data: BlockData) -> None:
        by_bucket: dict[BucketKey, list[tuple[int, Block]]] = defaultdict(list)
        for (leaf, bucket_id, block_id), block in data.items():
            by_bucket[BucketKey(leaf, bucket_id)].append((block_id, block))

        # Read each bucket, modify it, then write it back explicitly so file-based storage works.
        for (leaf, bucket_id), blocks in by_bucket.items():
            storage_idx = self.get_leaf_block(leaf, bucket_id)
            bucket = self._storage[storage_idx]
            for block_id, block in blocks:
                bucket[block_id] = block
            self._storage[storage_idx] = bucket
