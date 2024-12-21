"""This module implements a simple binary tree data structure that fits our need."""
import math

from typing import Dict, List, Union

# Hard code indices to access data.
KEY = 0
LEAF = 1
VALUE = 2
# We also assume if metadata exists, it's the first value in the data block.
META = 0
# We simply use a list to store data or when encrypted, use some bytes.
Data = Union[list, bytes]
# Buckets is a list of lists of Data.
Bucket = List[Data]
Buckets = List[Bucket]


class BinaryTree:
    def __init__(self, num_data: int, bucket_size: int, storage: Buckets = None) -> None:
        """
        Initializes the binary tree based on input parameters.

        :param num_data: Number of data points the tree should hold.
        :param bucket_size: Size of each node/bucket in the tree.
        :param storage: Buckets contains data or by default, emtpy.
        """
        # Store the number of data point and bucket size.
        self.__num_data = num_data
        self.__bucket_size = bucket_size

        # Compute the level of the binary tree based on number of data we plan to store.
        self.__level = int(math.ceil(math.log(num_data, 2))) + 1
        # Compute the size of the tree, which is the length of the storage list.
        self.__size = pow(2, self.__level) - 1
        # Compute the last index before actual leaves.
        self.__start_leaf = pow(2, self.__level - 1) - 1

        # Use the input buckets to initialize storage or create empty list.
        self.__storage = [[] for _ in range(self.__size)] if not storage else storage

    @property
    def size(self) -> int:
        """Returns the size of the binary tree."""
        return self.__size

    @property
    def level(self) -> int:
        """Returns the level of the binary tree."""
        return self.__level

    @property
    def start_leaf(self) -> int:
        """Returns the index of storage corresponding to leaf 0 in the binary tree."""
        return self.__start_leaf

    @property
    def storage(self) -> Buckets:
        """Return the current storage."""
        return self.__storage

    @storage.setter
    def storage(self, storage: Buckets):
        """Set the current storage to provided buckets."""
        self.__storage = storage

    @staticmethod
    def get_parent_index(index: int) -> int:
        """
        Given index of a node, find where its parent node is stored in the list.

        :param index: the index of a node.
        :return: the index of input node's parent node.
        """
        return int(math.ceil(index / 2)) - 1

    @staticmethod
    def get_path_indices(index: int) -> List[int]:
        """
        Given an index of a node, get the index of the path from itself to the root node.

        :param index: the index of a node.
        :return: a list of index from the input node to the root.
        """
        # Return path as a list.
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

        :param indices: a list of indices of nodes.
        :return: a list of index from the input nodes to the root.
        """
        # Paths would be a set to exclude duplicates.
        path = set()

        # Add all path indices to the set.
        for index in indices:
            path.update(BinaryTree.get_path_indices(index=index))

        return sorted(list(path), reverse=True)

    @staticmethod
    def get_mul_path_dict(level: int, indices: List[int]) -> Dict[int, list]:
        """
        Given a list of indices of nodes, get a dictionary whose keys are index to all buckets belong to these paths.

        :param level: the level of the tree.
        :param indices: a list of indices of nodes.
        :return: a dictionary where keys are indices from the input nodes to the root and each value is an empty list.
        """
        # Get the start leaf of the tree.
        indices = [index + pow(2, level - 1) - 1 for index in indices]
        # Return a dictionary where each index appears once with an empty list.
        return {index: [] for index in BinaryTree.get_mul_path_indices(indices=indices)}

    @staticmethod
    def fill_buckets_with_dummy_data(buckets: Buckets, bucket_size: int) -> None:
        """
        Given a list of buckets, fill them with dummy data.

        Note that the default dummy size eventually yields an encryption of 96 bytes. When saving this, there's an
        overhead of 33 bytes introduced by pickle. Also, when set to 1 to 10 bytes, this takes 32 bytes.
        :param buckets: A list of buckets.
        :param bucket_size: Size of each bucket.
        :return: Modify the input list of lists in-place.
        """
        for bucket in buckets:
            while len(bucket) < bucket_size:
                bucket.append([None, None, None])

    @staticmethod
    def get_cross_index(leaf_one: int, leaf_two: int, level: int) -> int:
        """
        Given two leaf labels, find the lowest index where their paths are crossed.

        :param leaf_one: the index of a node.
        :param leaf_two: the index of a node.
        :param level: the level of the entire tree.
        :return: the index of the lowest node where input nodes' paths are crossed.
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

        :param leaf_one: the index of a node.
        :param leaf_two: the index of a node.
        :param level: the level of the entire tree.
        :return: the depth in the tree where input nodes' paths are crossed.
        """
        index = BinaryTree.get_cross_index(leaf_one=leaf_one, leaf_two=leaf_two, level=level)
        return int(math.ceil(math.log(index + 2, 2))) - 1

    @staticmethod
    def fill_data_to_path(data: Data, path: Buckets, leaf: int, level: int, bucket_size: int) -> bool:
        """
        Provide a list of buckets (path), fill the data to the lowest leaf node possible.

        :param data: the data to be filled in.
        :param path: the list of buckets to fill.
        :param leaf: the leaf of the list of buckets.
        :param level: the level of the entire tree.
        :param bucket_size: the size of each bucket.
        :return: Modify the input list of lists in-place, return True if data was filled.
        """
        # Get the lowest crossed level index of the path between the data and the current path.
        index = BinaryTree.get_cross_index_level(leaf_one=data[LEAF], leaf_two=leaf, level=level)

        # Go backwards from bottom to up.
        for path_index in range(index, -1, -1):
            # Check if the bucket is full.
            if len(path[path_index]) < bucket_size:
                # If the data is inserted, return True.
                path[path_index].append(data)
                return True

        # Otherwise return False.
        return False

    @staticmethod
    def fill_data_to_path_dict(data: Data, path: dict, leaf: int, level: int, bucket_size: int) -> bool:
        """
        Provide a dictionary of buckets, fill the data to the lowest leaf node possible.

        :param data: the data to be filled in.
        :param path: the dictionary of buckets (tree index are the keys).
        :param leaf: the leaf of the list of buckets.
        :param level: the level of the entire tree.
        :param bucket_size: the size of each bucket.
        :return: Modify the input dict in-place, return True if data was filled.
        """
        # Get the lowest crossed index of the path between the data and the current path.
        index = BinaryTree.get_cross_index(leaf_one=data[LEAF], leaf_two=leaf, level=level)

        # Go backwards from bottom to up.
        while index >= 0:
            # Check if the bucket is full.
            if len(path[index]) < bucket_size:
                # If the data is inserted, return True.
                path[index].append(data)
                return True
            else:
                index = BinaryTree.get_parent_index(index=index)

        # Otherwise return False.
        return False

    @staticmethod
    def fill_data_to_mul_path(data: Data, path: dict, leaves: List[int], level: int, bucket_size: int) -> bool:
        """
        Provide a list of buckets, fill the data to the lowest leaf node possible.

        :param data: the data to be filled in.
        :param path: the dictionary of buckets (tree index are the keys).
        :param leaves: a list of leaves where the data can be potentially stored to.
        :param level: the level of the entire tree.
        :param bucket_size: the size of each bucket.
        :return: Modify the input dict in-place, return True if data was filled.
        """
        # Get the lowest crossed index of the path between the data and the current path.
        indices = [BinaryTree.get_cross_index(leaf_one=data[LEAF], leaf_two=leaf, level=level) for leaf in leaves]
        max_index = max(indices)

        # Go backwards from bottom to up.
        while max_index >= 0:
            # Check if the bucket is full.
            if len(path[max_index]) < bucket_size:
                # If the data is inserted, return True.
                path[max_index].append(data)
                return True
            else:
                max_index = BinaryTree.get_parent_index(index=max_index)

        # Otherwise return False.
        return False

    def fill_storage_with_dummy_data(self) -> None:
        """Fill storage binary tree with dummy data."""
        self.fill_buckets_with_dummy_data(buckets=self.__storage, bucket_size=self.__bucket_size)

    def fill_data_to_storage_leaf(self, data: Data) -> bool:
        """Based on the input data, fill it to the proper path at the lowest leaf node."""
        # Go from leaf to node; check whether bucket is full.
        for path_index in self.get_leaf_path(leaf=data[LEAF]):
            if len(self.__storage[path_index]) < self.__bucket_size:
                self.__storage[path_index].append(data)
                # If the data is inserted, return True.
                return True

        # Otherwise return False.
        return False

    def get_leaf_path(self, leaf: int) -> List[int]:
        """
        Given a leaf label, get the index of the path from itself to the root node.

        Assume that leaf ranges from 0 to 2 ** level - 1.
        :param leaf: the label of a leaf.
        :return: a list of index from the leaf node to the root.
        """
        return self.get_path_indices(index=leaf + self.__start_leaf)

    def get_mul_leaf_path(self, leaves: List[int]) -> List[int]:
        """
        Given a list of leaf labels, get the index of the mul leaf path.

        :param leaves: a list of leaf labels.
        :return: a list of indices from multiple leaf nodes to the root.
        """
        return self.get_mul_path_indices(indices=[leaf + self.__start_leaf for leaf in leaves])

    def get_leaf_block(self, leaf: int, index: int) -> int:
        """
        Given a leaf label, and an index, get the index of the block on that path.

        :param leaf: the label of a leaf.
        :param index: the index of the block of interest.
        :return: an index indicating location of the block in the tree.
        """
        return self.get_leaf_path(leaf=leaf)[-index - 1]

    def read_path(self, leaf: Union[int, List[int]]) -> Buckets:
        """
        Given one leaf node or a list of nodes, grab all buckets of data along the path(s).

        :param leaf: the label of a leaf or a list of leaves.
        :return: a list of buckets of Data object, from leaf/leaves to root.
        """
        # If the leaf is an integer, we read one path.
        if isinstance(leaf, int):
            path_to_read = self.get_leaf_path(leaf=leaf)

        # If the leaf is a list, we read multiple paths.
        elif isinstance(leaf, list):
            path_to_read = self.get_mul_leaf_path(leaves=leaf)

        # Otherwise raise a type error.
        else:
            raise TypeError("Leaf must be an integer or list of integers.")

        # Read the desired values.
        return [self.__storage[data_index] for data_index in path_to_read]

    def write_path(self, leaf: Union[int, List[int]], data: Buckets) -> None:
        """
        Given one leaf node or a list of nodes, write provided data to the path(s).

        :param leaf: the label of a leaf or a list of leaves.
        :param data: a list of buckets of Data object.
        """
        # If the leaf is an integer, we read one path.
        if isinstance(leaf, int):
            # Get the path to write to.
            path_to_write = self.get_leaf_path(leaf=leaf)

            # Check if the provided data to write has the same length.
            if len(data) != len(path_to_write):
                raise ValueError("Wrong number of buckets on a path.")

        # If the leaf is a list, we read multiple paths.
        elif isinstance(leaf, list):
            # Get the path to write to.
            path_to_write = self.get_mul_leaf_path(leaves=leaf)

            # Check if the provided data to write has the same length.
            if len(data) != len(path_to_write):
                raise ValueError("Wrong number of buckets on a path.")

        # Otherwise raise a type error.
        else:
            raise TypeError("Leaf must be an integer or list of integers.")

        # Write the data.
        for i, path_index in enumerate(path_to_write):
            self.__storage[path_index] = data[i]

    def read_block(self, leaf: int, bucket_id: int, block_id: int) -> Data:
        """
        Given a leaf node, a bucket index, and a block index, grab data stored in that location.

        :param leaf: the label of a leaf.
        :param bucket_id: the index of the bucket of interest.
        :param block_id: the index of the block of interest.
        :return: one data value.
        """
        # Read the desired values.
        return self.__storage[self.get_leaf_block(leaf=leaf, index=bucket_id)][block_id]

    def read_block_meta(self, leaf: int, bucket_id: int, block_id: int) -> Data:
        """
        Given a leaf node, a bucket index, and a block index, grab metadata stored in that location.

        :param leaf: the label of a leaf.
        :param bucket_id: the index of the bucket of interest.
        :param block_id: the index of the block of interest.
        :return: one data value.
        """
        # Read the desired values.
        return self.read_block(leaf=leaf, bucket_id=bucket_id, block_id=block_id)[META]

    def write_block(self, leaf: int, bucket_id: int, block_id: int, data: Data) -> None:
        """
        Given a leaf node, a bucket index, and a block index, write data to that location.

        :param leaf: the label of a leaf.
        :param bucket_id: the index of the bucket of interest.
        :param block_id: the index of the block of interest.
        :param data: the data to write.
        """
        # Write the value to the target block.
        self.__storage[self.get_leaf_block(leaf=leaf, index=bucket_id)][block_id] = data
