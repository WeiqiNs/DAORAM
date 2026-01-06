"""Defines the AVL tree; note that inserting repeated keys may cause unexpected behavior."""
from __future__ import annotations

import pickle
import secrets
from dataclasses import astuple, dataclass
from typing import Any, List, Optional

from daoram.dependency.storage import Data
from daoram.dependency.types import KVPair


@dataclass
class AVLData:
    """
    Create the data structure to hold a data record that should be put into a complete binary tree.

    It has three fields: key, leaf, and value, where key and value could be anything, but the leaf needs to be an int.
    By default, (when used as a dummy), when initialize the fields to None.
    """
    value: Optional[Any] = None
    r_key: Optional[Any] = None
    r_leaf: Optional[int] = None
    r_height: int = 0
    l_key: Optional[Any] = None
    l_leaf: Optional[int] = None
    l_height: int = 0

    @classmethod
    def from_pickle(cls, data: bytes) -> AVLData:
        """Given some pickled data, convert it to an AVLData-typed object"""
        return cls(*pickle.loads(data))

    def dump(self) -> bytes:
        """Dump the data structure to bytes."""
        return pickle.dumps(astuple(self))  # type: ignore


class AVLTreeNode:
    def __init__(self, kv_pair: KVPair):
        """
        Given a key-value pair, create a new AVL tree node.

        Comparing to the standard tree node, we add two fields:
            - Value, which holds the value from the input key-value pair.
            - Path, which can store the random path the oram will store the node.
        :param kv_pair: A KVPair containing key and value.
        """
        self.key: Any = kv_pair.key
        self.leaf: Optional[int] = None
        self.value: Any = kv_pair.value
        self.height: int = 1
        self.left_node: Optional[AVLTreeNode] = None
        self.right_node: Optional[AVLTreeNode] = None


class AVLTree:
    """
    Defines the AVL Tree for our application.

    We use this class to initialize the OMAP storage, given a list of kv pairs.
    """

    def __init__(self, leaf_range: int):
        """The AVL tree only needs to store the possible leaf range to sample."""
        self._leaf_range = leaf_range

    def _get_new_leaf(self) -> int:
        """Get a random leaf label within the range."""
        return secrets.randbelow(self._leaf_range)

    @staticmethod
    def _get_height(node: Optional[AVLTreeNode]) -> int:
        """Get the height of the input node."""
        # If the node is empty, the height would be 0; otherwise return height.
        return node.height if node else 0

    @staticmethod
    def _get_balance(node: Optional[AVLTreeNode]) -> int:
        """Get balance of the input node."""
        # If the node is empty, the balance would be 0; otherwise compute the balance.
        return AVLTree._get_height(node.left_node) - AVLTree._get_height(node.right_node) if node else 0

    def _update_height(self, node: AVLTreeNode) -> None:
        """Update the height of the input node."""
        node.height = 1 + max(self._get_height(node.left_node), self._get_height(node.right_node))

    def _rotate_left(self, in_node: AVLTreeNode) -> AVLTreeNode:
        """
        Perform a left rotation at the provided input node.

        :param in_node: Some AVLTreeNode to rotate.
        :return: The parent node of the rotated node.
        """
        # Save the right child of the input node as the parent node.
        p_node = in_node.right_node
        # The input node will be the left child of the parent node; we store its left child to a tmp variable.
        tmp_node = p_node.left_node

        # Now we set the input node as the left child of the parent node.
        p_node.left_node = in_node
        # The left child of the parent node was on the right of the input node.
        in_node.right_node = tmp_node

        # Update the input node height.
        self._update_height(in_node)
        # update the new parent node height.
        self._update_height(p_node)

        # Return the new parent node.
        return p_node

    def _rotate_right(self, in_node: AVLTreeNode) -> AVLTreeNode:
        """
        Perform a left rotation at the provided input node.

        :param in_node: Some AVLTreeNode to rotate.
        :return: The parent node of the rotated node.
        """
        # Save the left child of the input node as the parent node.
        p_node = in_node.left_node
        # The input node will be the right child of the parent node; we store its right child to a tmp variable.
        tmp_node = p_node.right_node

        # Now we set the input node as the right child of the parent node.
        p_node.right_node = in_node
        # The right child of the parent node was on the left of the input node.
        in_node.left_node = tmp_node

        # Update the input node height.
        self._update_height(in_node)
        # update the new parent node height.
        self._update_height(p_node)

        # Return the new parent node.
        return p_node

    def _balance(self, node: AVLTreeNode) -> AVLTreeNode:
        """Re-balance a node if it is unbalanced."""
        # Update the height of the node.
        self._update_height(node)
        # Get the balance factor.
        balance = self._get_balance(node)

        # Left heavy subtree rotation.
        if balance > 1:
            # The left-right case.
            if self._get_balance(node.left_node) < 0:
                node.left_node = self._rotate_left(node.left_node)
            # The left-left case.
            return self._rotate_right(node)

        # Right heavy subtree rotation.
        if balance < -1:
            # The right-left case.
            if self._get_balance(node.right_node) > 0:
                node.right_node = self._rotate_right(node.right_node)
            # The right-right case.
            return self._rotate_left(node)

        return node

    def insert(self, root: Optional[AVLTreeNode], kv_pair: KVPair) -> AVLTreeNode:
        """
        Inserts a new node into the tree, which is represented by the root.

        :param root: The root node of the AVL tree.
        :param kv_pair: A KVPair containing key and value.
        :return: The updated AVL tree root node.
        """
        # If the tree is empty, the new node becomes the root.
        if not root:
            return AVLTreeNode(kv_pair)

        # Create a stack to hold all visited nodes and set root to node for readability.
        stack = []
        node = root

        # Traverse the tree to find the insertion point.
        while node:
            # Add visited node to stack.
            stack.append(node)
            # If the value is smaller, we go left.
            if kv_pair.key < node.key:
                if not node.left_node:
                    node.left_node = AVLTreeNode(kv_pair)
                    stack.append(node.left_node)
                    break
                node = node.left_node
            # Otherwise go right.
            else:
                if not node.right_node:
                    node.right_node = AVLTreeNode(kv_pair)
                    stack.append(node.right_node)
                    break
                node = node.right_node

        # Rebalance the tree from the insertion point up to the root.
        while stack:
            # Get the last node and update its height.
            node = stack.pop()
            # Balance the node at this position.
            balanced_node = self._balance(node)

            # If a parent exists, update which node the parent should point to.
            if stack:
                parent = stack[-1]
                if parent.left_node == node:
                    parent.left_node = balanced_node
                else:
                    parent.right_node = balanced_node
            else:
                return balanced_node

        # The code should not exist the while loop without returning.
        raise ValueError("The node was not successfully inserted.")

    def recursive_insert(self, root: Optional[AVLTreeNode], kv_pair: KVPair) -> AVLTreeNode:
        """
        Inserts a new node into the tree, which is represented by the root.

        We also provide the recursive algorithm to validate the correctness of the non-recursive approach.
        :param root: The root node of the AVL tree.
        :param kv_pair: A KVPair containing key and value.
        :return: The updated AVL tree root node.
        """
        # When we reach an empty root, create a new tree node to store the KV pair.
        if root is None:
            return AVLTreeNode(kv_pair)
        # If not an empty node, we compare the key.
        elif kv_pair.key < root.key:
            root.left_node = self.recursive_insert(root=root.left_node, kv_pair=kv_pair)
        else:
            root.right_node = self.recursive_insert(root=root.right_node, kv_pair=kv_pair)

        # Update the height of the parent node.
        root.height = 1 + max(self._get_height(node=root.left_node), self._get_height(node=root.right_node))

        # Get the balance factor.
        balance = self._get_balance(node=root)

        # Left heavy subtree rotation.
        if balance > 1:
            # The left-right case.
            if self._get_balance(node=root.left_node) < 0:
                root.left_node = self._rotate_left(root.left_node)
            # The left-left case.
            return self._rotate_right(in_node=root)

        # Right heavy subtree rotation.
        if balance < -1:
            # The right-left case.
            if self._get_balance(root.right_node) > 0:
                root.right_node = self._rotate_right(root.right_node)
            # The right-right case.
            return self._rotate_left(in_node=root)

        return root

    def delete(self, root: Optional[AVLTreeNode], key: Any) -> Optional[AVLTreeNode]:
        """
        Deletes a node with the given key from the tree using non-recursive method.

        Template for oblivious version: uses 'local' list to track path, clear phases,
        and in-order successor (go right, then keep left) for two-children case.

        :param root: The root node of the AVL tree.
        :param key: The key to delete.
        :return: The updated AVL tree root node, or None if tree becomes empty.
        """
        if not root:
            return None

        # Use a similar approach to store node read to local (as a template for the oblivious approach).
        local = []
        current = root

        # Locate the node to delete.
        while current is not None:
            local.append(current)
            if current.key == key:
                break
            current = current.left_node if key < current.key else current.right_node

        # Key not found, return tree unchanged.
        if current is None or local[-1].key != key:
            return root

        # Target node is the last in local.
        node = local[-1]
        node_index = len(local) - 1

        # Case 1: Node has no children (leaf node).
        if node.left_node is None and node.right_node is None:
            # If deleting the only node (root).
            if len(local) == 1:
                return None
            # Remove from parent.
            parent = local[node_index - 1]
            if parent.left_node == node:
                parent.left_node = None
            else:
                parent.right_node = None
            # Remove from local.
            local.pop()

        # Case 2: Node has one child.
        elif node.left_node is None or node.right_node is None:
            # Get the single child.
            child = node.left_node if node.left_node is not None else node.right_node
            # If deleting root.
            if len(local) == 1:
                return child
            # Replace node with child in parent.
            parent = local[node_index - 1]
            if parent.left_node == node:
                parent.left_node = child
            else:
                parent.right_node = child
            # Remove from local.
            local.pop()

        # Case 3: Node has two children; choose based on subtree height.
        else:
            # Use predecessor (left then all right) if left is taller, else successor (right then all left).
            use_predecessor = self._get_height(node.left_node) > self._get_height(node.right_node)

            # Go to the taller subtree.
            current = node.left_node if use_predecessor else node.right_node
            local.append(current)

            # Traverse in opposite direction to find replacement.
            next_node = current.right_node if use_predecessor else current.left_node
            while next_node is not None:
                current = next_node
                local.append(current)
                next_node = current.right_node if use_predecessor else current.left_node

            # Replacement node is the last in local.
            replacement_node = local[-1]
            replacement_index = len(local) - 1

            # Copy replacement's data to the node being deleted.
            node.key = replacement_node.key
            node.value = replacement_node.value

            # Get the child to replace with (opposite of traversal direction).
            child = replacement_node.left_node if use_predecessor else replacement_node.right_node

            # Update parent's pointer.
            parent = local[replacement_index - 1]
            if parent.left_node == replacement_node:
                parent.left_node = child
            else:
                parent.right_node = child

            # Remove replacement node from local.
            local.pop()

        # Rebalance from bottom to up.
        for i in range(len(local) - 1, -1, -1):
            curr = local[i]
            balanced = self._balance(curr)

            # Update parent's pointer.
            if i > 0:
                parent = local[i - 1]
                if parent.left_node == curr:
                    parent.left_node = balanced
                else:
                    parent.right_node = balanced
            else:
                root = balanced

        return root

    @staticmethod
    def search(key: Any, root: Optional[AVLTreeNode]) -> Any:
        """
        Performs a search on the provided key and root node.

        :param key: The key to search for.
        :param root: The root node of the AVL tree.
        :return: The value corresponding to the provided search key.
        """
        # While the root is not empty.
        while root:
            if key < root.key:
                root = root.left_node
            elif key > root.key:
                root = root.right_node
            else:
                return root.value

        # If never found, return None.
        return None

    def get_data_list(self, root: AVLTreeNode, encryption: bool = False) -> List[Data]:
        """From the root, expand the AVL tree as a list of Data objects.

        :param root: An AVL tree root node.
        :param encryption: Indicate whether encryption is needed, i.e., whether the value should be bytes.
        :return: A list of Data objects.
        """
        # Otherwise, sample a new leaf for the root and add it to the stack.
        root.leaf = self._get_new_leaf()
        stack = [root]

        # Create an empty list to hold the result.
        result = []

        # While stack, keep popping element.
        while stack:
            # Get the current node.
            node = stack.pop()
            # Create an AVL Data with the node value.
            avl_data = AVLData(value=node.value)

            # Add value from the left node.
            if node.left_node:
                # Sample a new leaf for the left node.
                node.left_node.leaf = self._get_new_leaf()
                avl_data.l_key = node.left_node.key
                avl_data.l_leaf = node.left_node.leaf
                avl_data.l_height = node.left_node.height
                # Append the left node to the stack.
                stack.append(node.left_node)

            # Add value from the right node.
            if node.right_node:
                # Sample a new leaf for the right node.
                node.right_node.leaf = self._get_new_leaf()
                avl_data.r_key = node.right_node.key
                avl_data.r_leaf = node.right_node.leaf
                avl_data.r_height = node.right_node.height
                # Append the right node to the stack.
                stack.append(node.right_node)

            # Append the Data to result.
            if encryption:
                result.append(Data(key=node.key, leaf=node.leaf, value=avl_data.dump()))
            else:
                result.append(Data(key=node.key, leaf=node.leaf, value=avl_data))

        # Return the list.
        return result
