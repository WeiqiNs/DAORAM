"""This module implements a flexible binary tree structure for use in ORAM systems."""
import math
import pickle
from typing import Any, Dict, List, Union, Tuple

from daoram.dependency.helper import Data
from daoram.dependency.storage import Storage
from daoram.dependency.types import Block, Buckets


class FlexibleBinaryTree:
    def __init__(
            self,
            num_data: int,
            bucket_size: int,
            filename: str = None,
            data_size: int = None,
            enc_key_size: int = None) -> None:
        """
        Initializes the binary tree based on input parameters.

        :param num_data: The Number of data points the tree should hold.
        :param bucket_size: Size of each node/bucket in the tree.
        :param data_size: Size of each data point in the tree.
        :param filename: The Name of the file the tree should hold.
        :param enc_key_size: The key size for the encryption, if set to 0 it means encryption will not be used.
        """
        # Store the number of data point and bucket size.
        self._num_data = num_data
        self._bucket_size = bucket_size
        self._enc_key_size = enc_key_size
        self._data_size = data_size
        self._filename = filename

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
            bucket_size=bucket_size,
            enc_key_size=enc_key_size
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
    def get_child_index(index: int) -> int:
        """
        Given index of a node, find where its left child node is stored in the list.

        :param index: The index of a node.
        :return: The index of input node's left child node.
        """
        return index * 2 + 1

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
            index = FlexibleBinaryTree.get_parent_index(index)

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
            path.update(FlexibleBinaryTree.get_path_indices(index=index))

        return sorted(list(path), reverse=True)

    @staticmethod
    def get_mul_path_dict(level: int, indices: List[int]) -> Dict[int, list]:
        """
        Given a list of indices of nodes, get a dictionary whose keys are index to all buckets belong to these paths.

        :param level: The level of the tree.
        :param indices: A list of indices of nodes.
        :return: A dictionary where keys are indices from the input nodes to the root and each value is an empty list.
        """
        # Get the start leaf of the tree.
        indices = [index + pow(2, level - 1) - 1 for index in indices]
        # Return a dictionary where each index appears once with an empty list.
        return {index: [] for index in FlexibleBinaryTree.get_mul_path_indices(indices=indices)}

    @staticmethod
    def fill_buckets_with_dummy_data(buckets: Buckets, bucket_size: int) -> None:
        """
        Given a list of buckets, fill them with dummy data.

        Note that the default dummy size eventually yields encryption of 96 bytes. When saving this, there's an
        overhead of 33 bytes introduced by pickle. Also, when set to 1 to 10 bytes, this takes 32 bytes.
        :param buckets: A list of buckets.
        :param bucket_size: Size of each bucket.
        :return: Modify the input list of lists in-place.
        """
        for bucket in buckets:
            while len(bucket) < bucket_size:
                bucket.append(Data())

    @staticmethod
    def get_cross_index(leaf_one: int, leaf_two: int) -> int:
        """
        Given two indices of leaf nodes, find the lowest index where their paths are crossed.

        :param leaf_one: The index of a node.
        :param leaf_two: The index of a node.
        :param level: The level of the entire tree.
        :return: The index of the lowest node where input nodes' paths are crossed.
        """
        # Tracing back to their list until they are the same node.
        while leaf_one != leaf_two:
            leaf_one = math.ceil(leaf_one / 2) - 1
            leaf_two = math.ceil(leaf_two / 2) - 1

        return leaf_one

    @staticmethod
    def get_cross_index_level(leaf_one: int, leaf_two: int) -> int:
        """
        Given two indices of leaf nodes, find the depth in the tree where their paths are crossed.

        :param leaf_one: The index of a node.
        :param leaf_two: The index of a node.
        :param level: The level of the entire tree.
        :return: The depth in the tree where input nodes' paths are crossed.
        """
        index = FlexibleBinaryTree.get_cross_index(leaf_one=leaf_one, leaf_two=leaf_two)
        return int(math.ceil(math.log(index + 2, 2))) - 1

    @staticmethod
    def fill_data_to_path(data: Data, path: Buckets, leaf: Tuple[int, int], bucket_size: int, level: int) -> bool:
        """
        Provide a list of buckets (path), fill the data to the lowest leaf node possible.

        :param data: The data to be filled in.
        :param path: The list of buckets to fill.
        :param leaf: The leaf of the list of buckets.
        :param level: The level of the entire tree.
        :param bucket_size: The size of each bucket.
        :return: Modify the input list of lists in-place, return True if data was filled.
        """
        leaf_one_index = pow(2, data.leaf[1] - 1) - 1 + data.leaf[0]
        leaf_two_index = pow(2, leaf[1] - 1) - 1 + leaf[0]
        # Adjust to the same depth
        leaf_one_index, leaf_two_index = FlexibleBinaryTree.adjust_to_same_level(leaf_one_index, leaf_two_index,
                                                                                 data.leaf[1], leaf[1])

        # Get the lowest crossed level index of the path between the data and the current path.
        index = FlexibleBinaryTree.get_cross_index_level(leaf_one=leaf_one_index, leaf_two=leaf_two_index)

        # limit the depth to the current path
        index = min(index, level - 1)

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
    def fill_data_to_path_dict(data: Data, path: dict, leaf: Tuple[int, int], bucket_size: int) -> bool:
        """
        Provide a dictionary of buckets, fill the data to the lowest leaf node possible.

        :param data: The data to be filled in.
        :param path: The dictionary of buckets (tree index are the keys).
        :param leaf: The leaf of the list of buckets.
        :param bucket_size: The size of each bucket.
        :return: Modify the input dict in-place, return True if data was filled.
        """
        # Get the lowest crossed index of the path between the data and the current path.
        index = FlexibleBinaryTree.get_cross_index(leaf_one=FlexibleBinaryTree.get_actual_leaf_index(data.leaf),
                                                   leaf_two=FlexibleBinaryTree.get_actual_leaf_index(leaf))

        # Go backwards from bottom to up.
        while index >= 0:
            # Check if the bucket is full.
            if len(path[index]) < bucket_size:
                # If the data is inserted, return True.
                path[index].append(data)
                return True
            else:
                index = FlexibleBinaryTree.get_parent_index(index=index)

        # Otherwise return False.
        return False

    @staticmethod
    def fill_data_to_mul_path(data: Data, path: dict, leaves: List[int], level: int, bucket_size: int) -> bool:
        """
        Provide a list of buckets, fill the data to the lowest leaf node possible.

        :param data: The data to be filled in.
        :param path: The dictionary of buckets (tree index are the keys).
        :param leaves: A list of leaves where the data can be potentially stored to.
        :param level: The level of the entire tree.
        :param bucket_size: The size of each bucket.
        :return: Modify the input dict in-place, return True if data was filled.
        """
        # Get the lowest crossed index of the path between the data and the current path.
        indices = [FlexibleBinaryTree.get_cross_index(leaf_one=data.leaf, leaf_two=leaf, level=level) for leaf in
                   leaves]
        max_index = max(indices)

        # Go backwards from bottom to up.
        while max_index >= 0:
            # Check if the bucket is full.
            if len(path[max_index]) < bucket_size:
                # If the data is inserted, return True.
                path[max_index].append(data)
                return True
            else:
                max_index = FlexibleBinaryTree.get_parent_index(index=max_index)

        # Otherwise return False.
        return False

    @staticmethod
    def adjust_to_same_level(leaf_one_index: int, leaf_two_index: int, leaf_one_level: int, leaf_two_level: int) -> \
            Tuple[int, int]:
        """Given two indices of node, adjust them to the same level.

        :param leaf_one_index: The index of a node.
        :param leaf_two_index: The index of a node.
        :param leaf_one_level: The level of the first node.
        :param leaf_two_level: The level of the second node.
        :return: A tuple of two indices adjusted to the same level.
        """
        gap = leaf_one_level - leaf_two_level
        if gap > 0:
            for _ in range(gap):
                leaf_one_index = FlexibleBinaryTree.get_parent_index(leaf_one_index)
        else:
            for _ in range(-gap):
                leaf_two_index = FlexibleBinaryTree.get_parent_index(leaf_two_index)
        return leaf_one_index, leaf_two_index

    @staticmethod
    def adjust_to_lowest_level(actual_index: int, old_level: int, level: int) -> int:
        """Given a indice of node, adjust them to the lowest level.

        :param actual_index: The index of a node.
        :param old_level: The old level of the node.
        :param level: The level of the entire tree.
        :return: The index adjusted to the lowest level.
        """
        gap = level - old_level
        if gap > 0:
            for _ in range(gap):
                actual_index = FlexibleBinaryTree.get_child_index(index=actual_index)
        else:
            for _ in range(-gap):
                actual_index = FlexibleBinaryTree.get_parent_index(index=actual_index)
        return actual_index

    @staticmethod
    def get_actual_leaf_index(leaf_info: Tuple[int, int]) -> int:
        """
        Convert leaf information to actual node index.
        
        :param leaf_info: Either an int (for backward compatibility) or tuple (leaf, height)
        :return: Actual node index in storage
        """
        if isinstance(leaf_info, tuple) and len(leaf_info) == 2:
            leaf, level = leaf_info
            return pow(2, level - 1) - 1 + leaf
        else:
            raise TypeError("Leaf must be an integer or tuple (leaf, height)")

    def scale_up(self) -> None:
        """
        Expand the tree to accommodate more data points.
        
        :param new_num_data: New number of data points the tree should hold
        """

        # Calculate new level and size
        new_level = self._level + 1
        new_size = pow(2, new_level) - 1
        new_start_leaf = pow(2, new_level - 1) - 1

        # Create new storage
        new_storage = Storage(
            size=new_size,
            filename=self._filename,
            data_size=self._data_size,
            bucket_size=self._bucket_size,
            enc_key_size=self._enc_key_size
        )

        # Copy existing data to new storage
        # The old tree structure is embedded in the new tree
        for i in range(self._size):
            new_storage[i] = self._storage[i]

        # Update tree properties
        self._level = new_level
        self._size = new_size
        self._start_leaf = new_start_leaf
        self._storage = new_storage

    def scale_down(self) -> bool:
        """
        Contract the tree to reduce size.
        
        :param new_num_data: New number of data points the tree should hold
        :return: True if contraction was successful, False if not enough space
        """
        # Calculate new level and sizeint(math.ceil(math.log(new_num_data, 2))) + 1
        new_level = self._level - 1
        new_size = pow(2, new_level) - 1
        new_start_leaf = pow(2, new_level - 1) - 1

        # Check if contraction is possible
        if not self._can_scale_down():
            return False

        # Create new storage
        new_storage = Storage(
            size=new_size,
            filename=self._filename,
            data_size=self._data_size,
            bucket_size=self._bucket_size,
            enc_key_size=self._enc_key_size
        )

        for i in range(new_size):
            new_storage[i] = self._storage[i]

        # Migrate data from old tree to new tree
        self._remove_bottom_layer()

        # Update tree properties
        self._level = new_level
        self._size = new_size
        self._start_leaf = new_start_leaf
        self._storage = new_storage

        return True

    def _can_scale_down(self) -> bool:
        """Check if the tree can be scaled down."""
        num_leaves = int(pow(2, self._level - 1))

        print("Checking scale down feasibility...num_leaves:", num_leaves)
        for leaf_index in range(num_leaves):
            # Get all bucket indices on the path
            path_indices = self.get_path_indices(self._start_leaf + leaf_index)
            path_block_counts = 0
            for bucket_index in path_indices:
                path_block_counts += len(self._storage[bucket_index])

            buckets_of_leaf = len(self._storage[self._start_leaf + leaf_index])
            if (path_block_counts - self._bucket_size) > (self._level - 2) * self._bucket_size:
                return False

        return True

    def _upthrust_date(self, indice: int) -> bool:
        """ Given an leaf label, upthrust the data to the root. """
        path_index = self.get_parent_index(self._start_leaf + indice)
        target = self.storage[self._start_leaf + indice]
        while target and path_index > 0:
            # Check if the bucket is full.
            if len(self._storage[path_index]) < self._bucket_size:
                # If the data is inserted, return True.
                self._storage[path_index].append(target.pop())
            else:
                path_index = self.get_parent_index(index=path_index)
        if target:
            return True
            # Otherwise return False.
        else:
            return False

    def _remove_bottom_layer(self) -> None:
        """
        Migrate data from old tree to new tree during scale down.
        
        :param new_storage: New storage object
        :param new_level: New tree level
        :param new_size: New tree size
        """
        # Upthrust all the data in the bottom layer.
        for leaf in range(pow(2, self._level - 1)):
            self._upthrust_date(leaf)

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

    def get_leaf_path(self, leaf: Tuple[int, int]) -> List[int]:
        """
        Given a leaf label, get the index of the path from itself to the root node.

        Assume that leaf ranges from 0 to 2 ** level - 1.
        :param leaf: The label of a leaf (int or tuple (leaf, height)).
        :return: A list of index from the leaf node to the root.
        """
        # Get the actual index of the leaf.
        actual_index = FlexibleBinaryTree.get_actual_leaf_index(leaf)
        # Adjust the actual index to the lowest level.
        actual_index = self.adjust_to_lowest_level(actual_index, leaf[1], self._level)
        return self.get_path_indices(index=actual_index)

    def get_mul_leaf_path(self, leaves: List[Tuple[int, int]]) -> List[int]:
        """
        Given a list of leaf labels, get the index of the mul leaf path.

        :param leaves: A list of leaf labels (int or tuple (leaf, height)).
        :return: A list of indices from multiple leaf nodes to the root.
        """
        actual_indices = [FlexibleBinaryTree.get_actual_leaf_index(leaf) for leaf in leaves]
        actual_indices = [self.adjust_to_lowest_level(index, leaf[1], self._level) for index, leaf in
                          zip(actual_indices, leaves)]
        return self.get_mul_path_indices(indices=actual_indices)

    def get_leaf_block(self, leaf: Tuple[int, int], index: int) -> int:
        """
        Given a leaf label, and an index, get the index of the block on that path.

        :param leaf: The label of a leaf (int or tuple (leaf, height)).
        :param index: The index of the block of interest.
        :return: An index indicating the location of the block in the tree.
        """
        return self.get_leaf_path(leaf=leaf)[-index - 1]

    def read_path(self, leaf: Union[Tuple[int, int], List[Tuple[int, int]]]) -> Buckets:
        """
        Given one leaf node or a list of nodes, grab all buckets of data along the path(s).

        :param leaf: The label of a leaf or a list of leaves (int or tuple (leaf, height)).
        :return: A list of buckets of Data objects, from leaf/leaves to root.
        """
        # If the leaf is an integer or tuple, we read one path.
        if isinstance(leaf, tuple):
            path_to_read = self.get_leaf_path(leaf=leaf)

        # If the leaf is a list, we read multiple paths.
        elif isinstance(leaf, list):
            path_to_read = self.get_mul_leaf_path(leaves=leaf)

            # Otherwise, raise a type error.
        else:
            raise TypeError("Leaf must be an integer, tuple (leaf, height), or list of these.")

        return [self._storage[data_index] for data_index in path_to_read]

    def write_path(self, leaf: Union[Tuple[int, int], List[Tuple[int, int]]], data: Buckets) -> None:
        """
        Given one leaf node or a list of nodes, write provided data to the path(s).

        :param leaf: The label of a leaf or a list of leaves (int or tuple (leaf, height)).
        :param data: A list of buckets of Data objects.
        """
        # If the leaf is an integer or tuple, we read one path.
        if isinstance(leaf, tuple):
            # Get the path to write to.
            path_to_write = self.get_leaf_path(leaf=leaf)

            # # Check if the provided data to write has the same length.
            # if len(data) != len(path_to_write):
            #     raise ValueError("Wrong number of buckets on a path.")

        # If the leaf is a list, we read multiple paths.
        elif isinstance(leaf, list):
            # Get the path to write to.
            path_to_write = self.get_mul_leaf_path(leaves=leaf)

            # Check if the provided data to write has the same length.
            if len(data) != len(path_to_write):
                raise ValueError("Wrong number of buckets on a path.")

        # Otherwise, raise a type error.
        else:
            raise TypeError("Leaf must be an integer, tuple (leaf, height), or list of these.")

        # Write the data.
        for i, path_index in enumerate(path_to_write):
            self._storage[path_index] = data[i]

    def read_block(self, leaf: Tuple[int, int], bucket_id: int, block_id: int) -> Block:
        """
        Given a leaf node, a bucket index, and a block index, grab data stored in that location.

        :param leaf: The label of a leaf (int or tuple (leaf, height)).
        :param bucket_id: The index of the bucket of interest.
        :param block_id: The index of the block of interest.
        :return: One data value.
        """
        # Read the desired values.
        return self._storage[self.get_leaf_block(leaf=leaf, index=bucket_id)][block_id]

    def write_block(self, leaf: Tuple[int, int], bucket_id: int, block_id: int, data: Block) -> None:
        """
        Given a leaf node, a bucket index, and a block index, write data to that location.

        :param leaf: The label of a leaf (int or tuple (leaf, height)).
        :param bucket_id: The index of the bucket of interest.
        :param block_id: The index of the block of interest.
        :param data: The data to write.
        """
        # Compute the index to write to.
        index_to_write = self.get_leaf_block(leaf=leaf, index=bucket_id)
        # Get the bucket data.
        bucket_data = self._storage[index_to_write]
        # Update bucket data.
        bucket_data[block_id] = data
        # Write it back.
        self._storage[index_to_write] = bucket_data


def __getstate__(self) -> Dict[str, Any]:
    """Custom serialization method"""
    return {
        # Basic parameters
        '_num_data': self._num_data,
        '_bucket_size': self._group_size,
        '_enc_key_size': self._enc_key_size,
        '_data_size': self._data_size,
        '_filename': self._filename,
        '_level': self._level,
        '_size': self._size,
        '_start_leaf': self._start_leaf,

        # Storage configuration
        '_storage_filename': self._storage._Storage__filename,
        '_storage_data_size': self._storage._Storage__data_size,
        '_storage_encryption': self._storage._Storage__encryption,
        '_storage_total_size': self._storage._Storage__total_size,

        # Storage data
        '_storage_data': self._serialize_storage_data()
    }


def __setstate__(self, state: Dict[str, Any]) -> None:
    """Custom deserialization method"""
    # Restore basic parameters
    self._num_data = state['_num_data']
    self._group_size = state['_bucket_size']
    self._enc_key_size = state['_enc_key_size']
    self._data_size = state['_data_size']
    self._filename = state['_filename']
    self._level = state['_level']
    self._size = state['_size']
    self._start_leaf = state['_start_leaf']

    # Re-create the Storage object
    self._storage = Storage(
        size=state['_size'],
        filename=state['_storage_filename'],
        data_size=state['_storage_data_size'],
        bucket_size=state['_bucket_size'],
        enc_key_size=state['_enc_key_size']
    )

    # Restore storage data
    self._deserialize_storage_data(state['_storage_data'])

    def _serialize_storage_data(self) -> bytes:
        """Serialize storage data"""
        if self._storage._Storage__filename is None:
            # In-memory storage: directly serialize the internal data
            return pickle.dumps(self._storage._Storage__internal_data)
        else:
            # File-based storage: read file content and serialize it
            self._storage._Storage__file.seek(0)
            file_data = self._storage._Storage__file.read()
            return pickle.dumps(file_data)

    def _deserialize_storage_data(self, data: bytes) -> None:
        """Deserialize storage data"""
        storage_data = pickle.loads(data)

        if self._storage._Storage__filename is None:
            # In-memory storage: directly restore the internal data
            self._storage._Storage__internal_data = storage_data
        else:
            # File-based storage: write to the file
            self._storage._Storage__file.seek(0)
            self._storage._Storage__file.write(storage_data)
