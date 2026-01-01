"""Defines the B+ tree; note that the minimum order is 3 and inserting repeated keys may cause unexpected behavior."""
from __future__ import annotations

import pickle
import secrets
from dataclasses import astuple, dataclass
from typing import Any, List, Optional, Tuple

from daoram.dependency import Data
from daoram.dependency.avl_tree import KV_PAIR


@dataclass
class BPlusData:
    """
    Create the data structure to hold a data record that should be put into a complete binary tree.

    It has three fields: key, leaf, and value, where key and value could be anything but leaf needs to be an integer.
    By default, (when used as a dummy), when initialize the fields to None.
    """
    keys: Optional[List[Any]] = None
    values: Optional[List[Any]] = None

    @classmethod
    def from_pickle(cls, data: bytes) -> BPlusData:
        """Given some pickled data, convert it to a Data typed object"""
        return cls(*pickle.loads(data))

    def dump(self) -> bytes:
        """Dump the data structure to bytes."""
        return pickle.dumps(astuple(self))  # type: ignore


class BPlusTreeNode:
    def __init__(self):
        """
        Initialize a BPlusTreeNode object, containing the following attributes:
            - keys: a list of key values.
            - values: a list of values corresponding to the keys or a list of BPlusTreeNodes when this isn't a leaf.
            - parent: the parent node of this node.
            - is_leaf: True if this is a leaf node (containing actual values not nodes).
        """
        self.id: Optional[int] = None
        self.leaf: Optional[int] = None
        self.keys: list = []
        self.values: list = []
        self.is_leaf: bool = True

    # Insert at the leaf
    def add_kv_pair(self, kv_pair: KV_PAIR):
        """
        Add a key-value pair to this node.

        :param kv_pair: A tuple containing two values (key, value).
        """
        # Unpack the key, value pair.
        key, value = kv_pair

        # If keys is not empty, we find the correct place to insert the input value.
        if self.keys:
            for index, each_key in enumerate(self.keys):
                # Keys go from smaller to larger; hence if a larger key is found, we insert a new value in front.
                if key < each_key:
                    self.keys = self.keys[:index] + [key] + self.keys[index:]
                    self.values = self.values[:index] + [value] + self.values[index:]
                    # Terminate the function after insertion.
                    return

        # If keys is empty or all values were smaller, we append the new input value.
        self.keys.append(key)
        self.values.append(value)


class BPlusTree:
    def __init__(self, order: int, leaf_range: int):
        """
        The B+ tree needs to store the required branching degree and possible leaf range to sample.

        :param order: The branching degree of the B+ tree, minimum should be 3.
        """
        self.__order = order
        self.__mid = order // 2
        self.__leaf_range = leaf_range

    def __get_new_leaf(self) -> int:
        """Get a random leaf label within the range."""
        return secrets.randbelow(self.__leaf_range)

    @staticmethod
    def __find_leaf(key: Any, root: BPlusTreeNode) -> BPlusTreeNode:
        """
        Given B+ tree root and a key of interest, find the leaf where the key is stored.

        :param key: The key to search for.
        :param root: The root node of the B+ tree.
        :return: The leaf storing the key of interest.
        """
        # Set the root to current node for traversal.
        cur_node = root

        # While not reaching a leaf node, keep searching.
        while not cur_node.is_leaf:
            for index, each_key in enumerate(cur_node.keys):
                # If key equals, it is on the right.
                if key == each_key:
                    cur_node = cur_node.values[index + 1]
                    break
                # If the key is smaller, it is on the left.
                elif key < each_key:
                    cur_node = cur_node.values[index]
                    break
                # If we reached the end, it is on the right.
                elif index + 1 == len(cur_node.keys):
                    cur_node = cur_node.values[index + 1]
                    break

        # Return the node that was found.
        return cur_node

    @staticmethod
    def __find_leaf_path(key: Any, root: BPlusTreeNode) -> List[BPlusTreeNode]:
        """
        Given B+ tree root and a key of interest, find the entire path to the leaf where the key is stored.

        :param key: The key to search for.
        :param root: The root node of the B+ tree.
        :return: A list of nodes for the entire path.
        """
        # Get the result node list and set the root to current node for traversal.
        result = [root]
        cur_node = root

        # While not reaching a leaf node, keep searching.
        while not cur_node.is_leaf:
            for index, each_key in enumerate(cur_node.keys):
                # If key equals, it is on the right.
                if key == each_key:
                    cur_node = cur_node.values[index + 1]
                    break
                # If the key is smaller, it is on the left.
                elif key < each_key:
                    cur_node = cur_node.values[index]
                    break
                # If we reached the end, it is on the right.
                elif index + 1 == len(cur_node.keys):
                    cur_node = cur_node.values[index + 1]
                    break

            # Append the current node.
            result.append(cur_node)

        # Return the node that was found.
        return result

    def __split_node(self, node: BPlusTreeNode) -> BPlusTreeNode:
        """
        Given a node that is full, split it depends on whether it is a leaf or not.

        Note that the node itself is modified in place and only one new node is created.
        :param node: The node whose number of keys is the same as the branching degree.
        :return: The split node, which contains the right half of the input node.
        """
        # We break from the middle and create left child node.
        right_node = BPlusTreeNode()

        # Depending on whether the child node is a leaf node, we break it differently.
        if node.is_leaf:
            # New leaf gets half of the old leaf.
            right_node.keys = node.keys[self.__mid:]
            right_node.values = node.values[self.__mid:]

            # The old leaf keeps on the first half.
            node.keys = node.keys[:self.__mid]
            node.values = node.values[:self.__mid]

        else:
            # Mark the node as a non-leaf node.
            right_node.is_leaf = False

            # New leaf gets half of the old leaf.
            right_node.keys = node.keys[self.__mid + 1:]
            right_node.values = node.values[self.__mid + 1:]

            # The old leaf keeps on the first half.
            node.keys = node.keys[:self.__mid]
            node.values = node.values[:self.__mid + 1]

        # Because the nodes are modified in place, we only need to return the right one.
        return right_node

    def _insert_in_parent(self, child_node: BPlusTreeNode, parent_node: BPlusTreeNode) -> None:
        """
        Insert the child node into the parent node.

        :param child_node: A B+ tree node whose number of keys is the same as the branching degree.
        :param parent_node: A B+ tree node containing the child node.
        """
        # Store the key to insert to parent.
        insert_key = child_node.keys[self.__mid]

        # Perform the node split.
        right_node = self.__split_node(node=child_node)

        # Now we perform the actual insertion to parent.
        for index, each_key in enumerate(parent_node.keys):
            if insert_key < each_key:
                parent_node.keys = parent_node.keys[:index] + [insert_key] + parent_node.keys[index:]
                parent_node.values = parent_node.values[:index + 1] + [right_node] + parent_node.values[index + 1:]
                return
            elif index + 1 == len(parent_node.keys):
                parent_node.keys.append(insert_key)
                parent_node.values.append(right_node)
                return

    def _create_parent(self, child_node: BPlusTreeNode) -> BPlusTreeNode:
        """
        When a node has no parent and split is required, create a new parent node for them.

        :param child_node: A B+ tree node whose number of keys is the same as the branching degree.
        :return: A B+ tree node containing the split left and right child nodes.
        """
        # Store the key to insert to parent.
        insert_key = child_node.keys[self.__mid]

        # Perform the node split.
        right_node = self.__split_node(node=child_node)

        # Create the parent node.
        parent_node = BPlusTreeNode()

        # This parent node is not a leaf.
        parent_node.is_leaf = False

        # Fill in the information and return the parent node.
        parent_node.keys.append(insert_key)
        parent_node.values = [child_node, right_node]

        return parent_node

    def insert(self, root: BPlusTreeNode, kv_pair: KV_PAIR) -> BPlusTreeNode:
        """
        Inserts a new node into the tree, which is represented by the root.

        :param root: The root node of the B+ tree.
        :param kv_pair: A tuple containing two values (key, value).
        :return: The updated B+ tree root node.
        """
        # Find which leaf to insert.
        leaves = self.__find_leaf_path(root=root, key=kv_pair[0])
        # Add kv pair to the leaf node, which has to be the last one.
        leaves[-1].add_kv_pair(kv_pair=kv_pair)

        # Set the index.
        index = len(leaves) - 1

        # Iterate through the leaves.
        while index >= 0:
            if len(leaves[index].keys) >= self.__order:
                # When insertion is needed, we first locate the parent.
                if index > 0:
                    # Perform the insertion.
                    self._insert_in_parent(child_node=leaves[index], parent_node=leaves[index - 1])
                    index -= 1
                # Or we need a new parent node.
                else:
                    return self._create_parent(child_node=leaves[index])

            # We may reach to a point earlier than the root to stop splitting.
            else:
                break

        return root

    def search(self, key: Any, root: BPlusTreeNode) -> Any:
        """
        Performs a search on the provided key and root node.

        :param key: The key to search for.
        :param root: The root node of the B+ tree.
        :return: The value corresponding to the provided search key.
        """
        # First get the node leaf.
        leaf = self.__find_leaf(root=root, key=key)

        # Search in the node leaf.
        for index, each_key in enumerate(leaf.keys):
            if key == each_key:
                return leaf.values[index]

        raise KeyError(f"The key {key} is not found.")

    def __find_leaf_with_indices(self, key: Any, root: BPlusTreeNode) -> Tuple[List[BPlusTreeNode], List[int]]:
        """
        Find the path to the leaf and track child indices at each level.

        :param key: The key to search for.
        :param root: The root node of the B+ tree.
        :return: A tuple of (path, indices) where path is list of nodes and indices tracks which child was taken.
        """
        path = [root]
        indices = []
        cur_node = root

        while not cur_node.is_leaf:
            child_index = len(cur_node.keys)  # Default to rightmost child.
            for index, each_key in enumerate(cur_node.keys):
                if key < each_key:
                    child_index = index
                    break
                elif key == each_key:
                    child_index = index + 1
                    break

            indices.append(child_index)
            cur_node = cur_node.values[child_index]
            path.append(cur_node)

        return path, indices

    def __get_min_keys(self, is_leaf: bool) -> int:
        """Get the minimum number of keys a node must have."""
        if is_leaf:
            return (self.__order - 1) // 2
        return self.__mid - 1 if self.__order > 3 else 1

    def __update_parent_key(self, parent: BPlusTreeNode, old_key: Any, new_key: Any) -> None:
        """Update a key in the parent node when the first key of a child changes."""
        for i, k in enumerate(parent.keys):
            if k == old_key:
                parent.keys[i] = new_key
                return

    def delete(self, root: BPlusTreeNode, key: Any) -> Optional[BPlusTreeNode]:
        """
        Deletes a key from the B+ tree using an iterative approach.

        :param root: The root node of the B+ tree.
        :param key: The key to delete.
        :return: The updated B+ tree root node, or None if tree becomes empty.
        """
        # Find the path to the leaf and the child indices taken at each level.
        path, indices = self.__find_leaf_with_indices(key, root)
        leaf = path[-1]

        # Find and remove the key from the leaf.
        key_index = -1
        for i, k in enumerate(leaf.keys):
            if k == key:
                key_index = i
                break

        if key_index == -1:
            raise KeyError(f"The key {key} is not found.")

        # Remove the key-value pair.
        deleted_key = leaf.keys.pop(key_index)
        leaf.values.pop(key_index)

        # If this is the root and it's a leaf, handle specially.
        if len(path) == 1:
            return root if leaf.keys else None

        # Track if we need to update a parent key (when first key of leaf changes).
        if key_index == 0 and leaf.keys:
            # The first key changed, may need to update ancestor.
            new_first_key = leaf.keys[0]
            for level in range(len(path) - 2, -1, -1):
                parent = path[level]
                for i, k in enumerate(parent.keys):
                    if k == deleted_key:
                        parent.keys[i] = new_first_key
                        break
                else:
                    continue
                break

        # Handle underflow starting from the leaf going up.
        min_keys_leaf = self.__get_min_keys(is_leaf=True)
        min_keys_internal = self.__get_min_keys(is_leaf=False)

        level = len(path) - 1

        while level > 0:
            node = path[level]
            parent = path[level - 1]
            child_idx = indices[level - 1]
            min_keys = min_keys_leaf if node.is_leaf else min_keys_internal

            # Check if node has underflow.
            if len(node.keys) >= min_keys:
                break

            # Try to borrow from left sibling.
            if child_idx > 0:
                left_sibling = parent.values[child_idx - 1]
                if len(left_sibling.keys) > min_keys:
                    # Borrow from left sibling.
                    if node.is_leaf:
                        # Move last key-value from left sibling to this node.
                        node.keys.insert(0, left_sibling.keys.pop())
                        node.values.insert(0, left_sibling.values.pop())
                        # Update parent key.
                        parent.keys[child_idx - 1] = node.keys[0]
                    else:
                        # For internal node: rotate through parent.
                        node.keys.insert(0, parent.keys[child_idx - 1])
                        node.values.insert(0, left_sibling.values.pop())
                        parent.keys[child_idx - 1] = left_sibling.keys.pop()
                    break

            # Try to borrow from right sibling.
            if child_idx < len(parent.values) - 1:
                right_sibling = parent.values[child_idx + 1]
                if len(right_sibling.keys) > min_keys:
                    # Borrow from right sibling.
                    if node.is_leaf:
                        # Move first key-value from right sibling to this node.
                        node.keys.append(right_sibling.keys.pop(0))
                        node.values.append(right_sibling.values.pop(0))
                        # Update parent key.
                        parent.keys[child_idx] = right_sibling.keys[0]
                    else:
                        # For internal node: rotate through parent.
                        node.keys.append(parent.keys[child_idx])
                        node.values.append(right_sibling.values.pop(0))
                        parent.keys[child_idx] = right_sibling.keys.pop(0)
                    break

            # Must merge - prefer merging with left sibling.
            if child_idx > 0:
                left_sibling = parent.values[child_idx - 1]
                # Merge node into left sibling.
                if node.is_leaf:
                    left_sibling.keys.extend(node.keys)
                    left_sibling.values.extend(node.values)
                else:
                    # For internal node, bring down parent key.
                    left_sibling.keys.append(parent.keys[child_idx - 1])
                    left_sibling.keys.extend(node.keys)
                    left_sibling.values.extend(node.values)
                # Remove the merged child and corresponding key from parent.
                parent.keys.pop(child_idx - 1)
                parent.values.pop(child_idx)
                # Update indices for next iteration.
                indices[level - 1] = child_idx - 1
            else:
                # Merge with right sibling.
                right_sibling = parent.values[child_idx + 1]
                if node.is_leaf:
                    node.keys.extend(right_sibling.keys)
                    node.values.extend(right_sibling.values)
                else:
                    # For internal node, bring down parent key.
                    node.keys.append(parent.keys[child_idx])
                    node.keys.extend(right_sibling.keys)
                    node.values.extend(right_sibling.values)
                # Remove the merged child and corresponding key from parent.
                parent.keys.pop(child_idx)
                parent.values.pop(child_idx + 1)

            # Move up to handle potential parent underflow.
            level -= 1

        # Check if root needs to shrink.
        if not root.keys and not root.is_leaf:
            return root.values[0] if root.values else None

        return root

    def get_data_list(self, root: BPlusTreeNode, block_id: int = 0, encryption: bool = False) -> List[Data]:
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
                bplus_data = BPlusData(keys=node.keys, values=[(child.id, child.leaf) for child in node.values])
                # Add all children nodes to the stack.
                stack.extend([child for child in node.values])

            # Otherwise, store the keys and values.
            else:
                bplus_data = BPlusData(keys=node.keys, values=node.values)

            # Append the Data to result.
            if encryption:
                result.append(Data(key=node.id, leaf=node.leaf, value=bplus_data.dump()))
            else:
                result.append(Data(key=node.id, leaf=node.leaf, value=bplus_data))

        # Return the list.
        return result
