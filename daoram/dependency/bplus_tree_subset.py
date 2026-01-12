"""Defines a modified B+ tree for storing subsets of {0,1,...,n-1} with availability tracking."""
from __future__ import annotations

from typing import Any, List, Optional
from dataclasses import dataclass, astuple

from daoram.dependency import Data

import pickle
import secrets

@dataclass
class BPlusSubsetData:
    """
    Create the data structure to hold a data record that should be put into a complete binary tree.

    It has four fields: keys, values, min_val, max_val.
    - keys: list of keys stored in this node
    - values: for leaf nodes [available], for internal nodes [node.available, (child_id, child_leaf, child_available), ...]
    - min_val: minimum value in the range this node covers (for leaves)
    - max_val: maximum value in the range this node covers (for leaves)
    By default, (when used as a dummy), initialize the fields to None.
    """
    keys: Optional[List[Any]] = None
    values: Optional[List[Any]] = None
    min_val: Optional[int] = None
    max_val: Optional[int] = None

    @classmethod
    def from_pickle(cls, data: bytes) -> BPlusSubsetData:
        """Given some pickled data, convert it to a Data typed object"""
        return cls(*pickle.loads(data))

    def dump(self) -> bytes:
        """Dump the data structure to bytes."""
        return pickle.dumps(astuple(self))  # type: ignore

class SubsetBPlusNode:
    def __init__(self, is_leaf: bool = True, min_val: Optional[int] = None, max_val: Optional[int] = None):
        """
        Initialize a SubsetBPlusNode.

        :param is_leaf: True if this is a leaf node.
        :param min_val: Minimum value in the range this node is responsible for (for leaves).
        :param max_val: Maximum value in the range this node is responsible for (for leaves).
        """
        self.is_leaf = is_leaf
        self.keys: List[int] = []  # For leaves: list of existing keys; for internals: separator keys
        self.values: List[SubsetBPlusNode] = []  # For internals: child nodes
        self.min_val = min_val
        self.max_val = max_val
        self.child_availables: List[bool] = []  # For internals: availability for each child subtree
        self.available = True  # Cached availability for this subtree

    def update_available(self):
        """Update the available flag based on children or own keys."""
        if self.is_leaf:
            if self.min_val is not None and self.max_val is not None:
                total_possible = self.max_val - self.min_val + 1
                self.available = len(self.keys) < total_possible
            else:
                self.available = True  # If no range, assume available
        else:
            self.available = any(self.child_availables)

    def find_available(self) -> Optional[int]:
        """Find an available element in this subtree."""
        if self.is_leaf:
            if self.min_val is None or self.max_val is None:
                return None
            for i in range(self.min_val, self.max_val + 1):
                if i not in self.keys:
                    return i
            return None
        else:
            for i, child in enumerate(self.values):
                if self.child_availables[i]:
                    result = child.find_available()
                    if result is not None:
                        return result
            return None


class SubsetBPlusTree:
    def __init__(self, order: int, n: int):
        """
        Initialize the SubsetBPlusTree.

        :param order: The branching degree (minimum 3).
        :param n: The total number of elements {0,1,...,n-1}.
        """
        self.__order = order
        self.__mid = order // 2
        self.__n = n
        self.__root: Optional[SubsetBPlusNode] = None
        # For get_data_list: number of possible leaf labels
        self.__leaf_range: int = 1 << 20  # 2^20 possible leaf values

    def __find_leaf(self, key: int) -> Optional[SubsetBPlusNode]:
        """Find the leaf node that should contain the key."""
        if self.__root is None:
            return None
        cur_node = self.__root
        while not cur_node.is_leaf:
            for index, each_key in enumerate(cur_node.keys):
                if key < each_key:
                    cur_node = cur_node.values[index]
                    break
                elif index + 1 == len(cur_node.keys):
                    cur_node = cur_node.values[index + 1]
                    break
        return cur_node

    def __find_leaf_path(self, key: int) -> List[SubsetBPlusNode]:
        """Find the path to the leaf for the key."""
        if self.__root is None:
            return []
        path = [self.__root]
        cur_node = self.__root
        while not cur_node.is_leaf:
            for index, each_key in enumerate(cur_node.keys):
                if key < each_key:
                    cur_node = cur_node.values[index]
                    break
                elif index + 1 == len(cur_node.keys):
                    cur_node = cur_node.values[index + 1]
                    break
            path.append(cur_node)
        return path

    def __split_node(self, node: SubsetBPlusNode) -> SubsetBPlusNode:
        """Split a full node."""
        right_node = SubsetBPlusNode(is_leaf=node.is_leaf)
        if node.is_leaf:
            mid_key = node.keys[self.__mid]
            right_node.keys = node.keys[self.__mid:]
            node.keys = node.keys[:self.__mid]
            right_node.min_val = mid_key
            right_node.max_val = node.max_val
            node.max_val = mid_key - 1
        else:
            right_node.is_leaf = False
            right_node.keys = node.keys[self.__mid + 1:]
            right_node.values = node.values[self.__mid + 1:]
            right_node.child_availables = node.child_availables[self.__mid + 1:]
            node.keys = node.keys[:self.__mid]
            node.values = node.values[:self.__mid + 1]
            node.child_availables = node.child_availables[:self.__mid + 1]
        # Update availables
        node.update_available()
        right_node.update_available()
        return right_node

    def _insert_in_parent(self, child_node: SubsetBPlusNode, parent_node: SubsetBPlusNode, insert_key: int):
        """Insert a split child into parent."""
        right_node = self.__split_node(child_node)
        for index, each_key in enumerate(parent_node.keys):
            if insert_key < each_key:
                parent_node.keys.insert(index, insert_key)
                parent_node.values.insert(index + 1, right_node)
                parent_node.child_availables.insert(index + 1, right_node.available)
                break
        else:
            parent_node.keys.append(insert_key)
            parent_node.values.append(right_node)
            parent_node.child_availables.append(right_node.available)
        parent_node.update_available()

    def _create_parent(self, child_node: SubsetBPlusNode) -> SubsetBPlusNode:
        """Create a new parent for split root."""
        insert_key = child_node.keys[self.__mid]
        right_node = self.__split_node(child_node)
        parent_node = SubsetBPlusNode(is_leaf=False)
        parent_node.keys.append(insert_key)
        parent_node.values = [child_node, right_node]
        parent_node.child_availables = [child_node.available, right_node.available]
        parent_node.update_available()
        # Set ranges for children
        child_node.min_val = 0
        child_node.max_val = insert_key - 1
        right_node.min_val = insert_key
        right_node.max_val = self.__n - 1
        return parent_node

    def insert(self, key: int):
        """Insert a key into the tree."""
        if self.__root is None:
            self.__root = SubsetBPlusNode(is_leaf=True, min_val=0, max_val=self.__n - 1)
            self.__root.keys.append(key)
            self.__root.update_available()
            return
        path = self.__find_leaf_path(key)
        leaf = path[-1]
        if key in leaf.keys:
            return  # Already exists
        # Insert into leaf
        for index, each_key in enumerate(leaf.keys):
            if key < each_key:
                leaf.keys.insert(index, key)
                break
        else:
            leaf.keys.append(key)
        leaf.update_available()
        # Propagate updates up
        for node in reversed(path[:-1]):
            node.update_available()
        # Check for split
        index = len(path) - 1
        while index >= 0:
            if len(path[index].keys) >= self.__order:
                if index > 0:
                    insert_key = path[index].keys[self.__mid]
                    self._insert_in_parent(path[index], path[index - 1], insert_key)
                    index -= 1
                else:
                    self.__root = self._create_parent(path[index])
                    break
            else:
                break

    def delete(self, key: int):
        """Delete a key from the tree."""
        if self.__root is None:
            return
        path = self.__find_leaf_path(key)
        leaf = path[-1]
        if key not in leaf.keys:
            return
        leaf.keys.remove(key)
        leaf.update_available()
        # Propagate updates up through the path
        for i in range(len(path) - 2, -1, -1):
            parent = path[i]
            # Find which child this corresponds to
            for child_idx, child in enumerate(parent.values):
                if child is path[i + 1]:
                    parent.child_availables[child_idx] = path[i + 1].available
                    break
            parent.update_available()
        # For simplicity, no merging implemented

    def find_available(self) -> Optional[int]:
        """Find an available (not stored) element."""
        if self.__root is None:
            return 0 if self.__n > 0 else None
        return self.__root.find_available()

    def contains(self, key: int) -> bool:
        """Check if key is in the tree."""
        leaf = self.__find_leaf(key)
        return leaf is not None and key in leaf.keys
    
    def __get_new_leaf(self) -> int:
        """Get a random leaf label within the range."""
        return secrets.randbelow(self.__leaf_range)
    
    def get_data_list(self, root: SubsetBPlusNode, block_id: int = 0, encryption: bool = False) -> List[Data]:
        """From root, expand the AVL tree as a list of Data objects.

        :param root: An AVL tree root node.
        :param block_id: The starting block ID, by default 0.
        :param encryption: Indicate whether encryption is needed, i.e., whether the value should be bytes.
        :return: A list of Data objects.
        """
        # Update the root id and assign a new leaf.
        root.id = block_id
        root.leaf = self.__get_new_leaf()
        # Increment the id.
        block_id += 1

        # Create a stack to hold root and future nodes and an empty result list.
        stack = [root]
        result = []

        # While stack, keep popping element.
        while stack:
            # Get the current node.
            node = stack.pop()

            # If the node is not a leaf node, update its children ids and leaves.
            if not node.is_leaf:
                for child in node.values:
                    child.id = block_id
                    child.leaf = self.__get_new_leaf()
                    block_id += 1

                # Create the bplus data with node keys and updated child ids and leaves.
                # values = [node.available, (child.id, child.leaf, child.available), ...]
                bplus_data = BPlusSubsetData(
                    keys=node.keys, 
                    values=[node.available] + [(child.id, child.leaf, child.available) for child in node.values],
                    min_val=None,  # Internal nodes don't have range info
                    max_val=None
                )
                # Add all children nodes to the stack.
                stack.extend([child for child in node.values])

            # Otherwise, store the keys and values.
            else:
                bplus_data = BPlusSubsetData(
                    keys=node.keys, 
                    values=[node.available],
                    min_val=node.min_val,
                    max_val=node.max_val
                )

            # Append the Data to result.
            if encryption:
                result.append(Data(key=node.id, leaf=node.leaf, value=bplus_data.dump()))
            else:
                result.append(Data(key=node.id, leaf=node.leaf, value=bplus_data))

        # Return the list.
        return result