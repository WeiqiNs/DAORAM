"""This module implements a simple binary tree data structure that fits our need."""
import math
from collections import defaultdict
from typing import Dict, List, Tuple

from daoram.dependency.storage import Storage, Data
from daoram.dependency.types import BucketKey, BlockKey, PathData, BucketData, BlockData, Block, Buckets


class BinaryTree:
    def __init__(
            self,
            num_data: int,
            bucket_size: int,
            filename: str = None,
            data_size: int = None,
            disk_size: int = None,
            encryption: bool = False
    ) -> None:
        """
        Initializes the binary tree based on input parameters.

        :param num_data: The number of data points the tree should hold.
        :param bucket_size: Size of each node/bucket in the tree.
        :param filename: Optional backing file to persist the tree.
        :param data_size: Plaintext byte size of each data point (padding header is handled by Storage).
        :param disk_size: Plaintext byte length of each on-disk block; Storage adds the padding header internally.
        :param encryption: Deprecated alias for use_encryption; kept for compatibility.
        """
        # Store the number of data point and bucket size.
        self._num_data = num_data
        self._bucket_size = bucket_size

        # Compute the level of the binary tree based on the number of data we plan to store.
        self._level = int(math.ceil(math.log(num_data, 2))) + 1
        # Compute the size of the tree, which is the length of the storage list.
        self._size = pow(2, self._level) - 1
        # Compute the last index before actual leaves.
        self._start_leaf = pow(2, self._level - 1) - 1

        # Use the input buckets to initialize storage or create an empty list.
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
        """Returns the size of the binary tree."""
        return self._size

    @property
    def level(self) -> int:
        """Returns the level of the binary tree."""
        return self._level

    @property
    def start_leaf(self) -> int:
        """Returns the index of storage corresponding to leaf 0 in the binary tree."""
        return self._start_leaf

    @property
    def storage(self) -> Storage:
        """Return the current storage."""
        return self._storage

    @staticmethod
    def get_parent_index(index: int) -> int:
        """
        Given index of a node, find where its parent node is stored in the list.

        :param index: The index of a node.
        :return: The index of input node's parent node.
        """
        return int(math.ceil(index / 2)) - 1

    @staticmethod
    def get_path_indices(index: int) -> List[int]:
        """
        Given an index of a node, get the index of the path from itself to the root node.

        :param index: The index of a node.
        :return: A list of index from the input node to the root.
        """
        # Return a path as a list.
        path = []

        # Go through all possible parent indices.
        while index >= 0:
            # Do append first to include the input index as well.
            path.append(index)
            index = BinaryTree.get_parent_index(index)

        return path

    @staticmethod
    def get_mul_path_indices(indices: List[int]) -> List[int]:
        """
        Given a list of indices of nodes, get the index of the paths from themselves to the root node.

        :param indices: A list of indices of nodes.
        :return: A list of index from the input nodes to the root.
        """
        # Paths would be a set to exclude duplicates.
        path = set()

        # Add all path indices to the set.
        for index in indices:
            path.update(BinaryTree.get_path_indices(index=index))

        return sorted(list(path), reverse=True)

    @staticmethod
    def get_mul_path_dict(level: int, indices: List[int]) -> PathData:
        """
        Given a list of indices of nodes, get a dictionary whose keys are index to all buckets belong to these paths.

        :param level: The level of the tree.
        :param indices: A list of indices of nodes.
        :return: PathData dict where keys are indices from the input nodes to the root and each value is an empty list.
        """
        # Get the start leaf of the tree.
        indices = [index + pow(2, level - 1) - 1 for index in indices]
        # Return a dictionary where each index appears once with an empty list.
        return {index: [] for index in BinaryTree.get_mul_path_indices(indices=indices)}

    @staticmethod
    def get_cross_index(leaf_one: int, leaf_two: int, level: int) -> int:
        """
        Given two leaf labels, find the lowest index where their paths are crossed.

        :param leaf_one: The index of a node.
        :param leaf_two: The index of a node.
        :param level: The level of the entire tree.
        :return: The index of the lowest node where input nodes' paths are crossed.
        """
        start_leaf = pow(2, level - 1) - 1
        # Compute leaf index to its index in the storage list.
        leaf_one += start_leaf
        leaf_two += start_leaf

        # Tracing back to their list until they are the same node.
        while leaf_one != leaf_two:
            leaf_one = math.ceil(leaf_one / 2) - 1
            leaf_two = math.ceil(leaf_two / 2) - 1

        return leaf_one

    @staticmethod
    def get_cross_index_level(leaf_one: int, leaf_two: int, level: int) -> int:
        """
        Given two leaf labels, find the depth in the tree where their paths are crossed.

        :param leaf_one: The index of a node.
        :param leaf_two: The index of a node.
        :param level: The level of the entire tree.
        :return: The depth in the tree where input nodes' paths are crossed.
        """
        index = BinaryTree.get_cross_index(leaf_one=leaf_one, leaf_two=leaf_two, level=level)
        return int(math.ceil(math.log(index + 2, 2))) - 1

    @staticmethod
    def fill_data_to_path(data: Data, path: PathData, leaves: List[int], level: int, bucket_size: int) -> bool:
        """
        Fill data to the lowest possible bucket in a PathData dict.

        :param data: The data to be filled in.
        :param path: PathData dict (storage index -> bucket).
        :param leaves: List of leaves representing the paths in the dict.
        :param level: The level of the entire tree.
        :param bucket_size: The size of each bucket.
        :return: True if data was filled, False if no space available.
        """
        # Get the lowest crossed index for each leaf path.
        indices = [BinaryTree.get_cross_index(leaf_one=data.leaf, leaf_two=leaf, level=level) for leaf in leaves]
        max_index = max(indices)

        # Go backwards from bottom to up.
        while max_index >= 0:
            if len(path[max_index]) < bucket_size:
                path[max_index].append(data)
                return True
            else:
                max_index = BinaryTree.get_parent_index(index=max_index)

        return False

    def fill_data_to_storage_leaf(self, data: Data) -> bool:
        """Based on the input data, fill it to the proper path at the lowest leaf node."""
        # Go from leaf to node; check whether the current bucket is full.
        for path_index in self.get_leaf_path(leaf=data.leaf):
            if len(self._storage[path_index]) < self._bucket_size:
                self._storage[path_index] = self._storage[path_index] + [data]
                # If the data is inserted, return True.
                return True

        # Otherwise return False.
        return False

    def get_leaf_path(self, leaf: int) -> List[int]:
        """
        Given a leaf label, get the index of the path from itself to the root node.

        Assume that leaf ranges from 0 to 2 ** level - 1.
        :param leaf: The label of a leaf.
        :return: A list of index from the leaf node to the root.
        """
        return self.get_path_indices(index=leaf + self._start_leaf)

    def get_mul_leaf_path(self, leaves: List[int]) -> List[int]:
        """
        Given a list of leaf labels, get the index of the mul leaf path.

        :param leaves: A list of leaf labels.
        :return: A list of indices from multiple leaf nodes to the root.
        """
        return self.get_mul_path_indices(indices=[leaf + self._start_leaf for leaf in leaves])

    def get_leaf_block(self, leaf: int, index: int) -> int:
        """
        Given a leaf label, and an index, get the index of the block on that path.

        :param leaf: The label of a leaf.
        :param index: The index of the block of interest.
        :return: An index indicating the location of the block in the tree.
        """
        return self.get_leaf_path(leaf=leaf)[-index - 1]

    def read_path(self, leaves: List[int]) -> PathData:
        """
        Read unique buckets along the path(s) to the given leaves.

        :param leaves: A list of leaf labels.
        :return: PathData mapping storage_index -> bucket (deduplicated).
        """
        indices = self.get_mul_leaf_path(leaves)
        return {idx: self._storage[idx] for idx in indices}

    def write_path(self, data: PathData) -> None:
        """
        Write buckets to storage from a PathData dict.

        :param data: PathData mapping storage_index -> bucket.
        """
        for idx, bucket in data.items():
            self._storage[idx] = bucket

    def extract_path(self, leaf: int, data: PathData) -> Buckets:
        """
        Extract a single leaf's path from a PathData dict.

        :param leaf: The leaf label to extract.
        :param data: PathData mapping storage_index -> bucket.
        :return: List of buckets from leaf to root.
        :raises KeyError: If the path is not fully contained in data.
        """
        indices = self.get_leaf_path(leaf)
        missing = [idx for idx in indices if idx not in data]
        # Check if the data contains the desired path.
        if missing:
            raise KeyError(f"Path for leaf {leaf} not fully contained. Missing indices: {missing}")
        # Return data on that path.
        return [data[idx] for idx in indices]

    def read_bucket(self, keys: List[BucketKey]) -> BucketData:
        """
        Read buckets at the specified locations.

        :param keys: A list of (leaf, bucket_id) tuples specifying which buckets to read.
        :return: BucketData mapping (leaf, bucket_id) -> bucket.
        """
        return {
            BucketKey(leaf, bucket_id): self._storage[self.get_leaf_block(leaf, bucket_id)]
            for leaf, bucket_id in keys
        }

    def write_bucket(self, data: BucketData) -> None:
        """
        Write buckets to storage from a BucketData dict.

        :param data: BucketData mapping (leaf, bucket_id) -> bucket.
        """
        for (leaf, bucket_id), bucket in data.items():
            self._storage[self.get_leaf_block(leaf, bucket_id)] = bucket

    def read_block(self, keys: List[BlockKey]) -> BlockData:
        """
        Read blocks at the specified locations.

        :param keys: A list of (leaf, bucket_id, block_id) tuples specifying which blocks to read.
        :return: BlockData mapping (leaf, bucket_id, block_id) -> block.
        """
        return {
            BlockKey(leaf, bucket_id, block_id): self._storage[self.get_leaf_block(leaf, bucket_id)][block_id]
            for leaf, bucket_id, block_id in keys
        }

    def write_block(self, data: BlockData) -> None:
        """
        Write blocks to storage from a BlockData dict.

        :param data: BlockData mapping (leaf, bucket_id, block_id) -> block.
        """
        # Group by (leaf, bucket_id) to minimize storage accesses.
        by_bucket: Dict[BucketKey, List[Tuple[int, Block]]] = defaultdict(list)
        for (leaf, bucket_id, block_id), block in data.items():
            by_bucket[BucketKey(leaf, bucket_id)].append((block_id, block))

        # Unpack the input data dictionary.
        # Note: We read the bucket, modify it, then write back explicitly to make sure file-based storage works.
        for (leaf, bucket_id), blocks in by_bucket.items():
            storage_idx = self.get_leaf_block(leaf, bucket_id)
            # Get the bucket.
            bucket = self._storage[storage_idx]
            for block_id, block in blocks:
                bucket[block_id] = block
            # Write the whole bucket back.
            self._storage[storage_idx] = bucket
