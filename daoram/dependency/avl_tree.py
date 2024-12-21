"""Defines the AVL tree; note that inserting repeated keys may cause unexpected behavior."""

import secrets
from typing import Any, Optional, Tuple

# Set the values to extract information from KV pair.
K = 0
V = 1

# Set the values for left and right.
L = 0
R = 1

# Define the KV pair as a tuple contain two values.
KV_PAIR = Tuple[Any, Any]


class AVLTreeNode:
    def __init__(self, kv_pair: KV_PAIR):
        """
        Given a key-value pair, create a new AVL tree node.

        Comparing to the standard tree node, we add two fields:
            - value, which holds the value from the input key-value pair.
            - path, which can store the random path the oram will store the node.
        :param kv_pair: a tuple containing two values (key, value).
        """
        self.key: Any = kv_pair[K]
        self.value: Any = kv_pair[V]
        self.path: Optional[int] = None
        self.left_node: Optional[AVLTreeNode] = None
        self.right_node: Optional[AVLTreeNode] = None
        self.height: int = 1


class AVLTree:
    """
    Defines the AVL Tree for our application.

    We use this class to initialize the OMAP storage, given a list of kv pairs.
    """

    def __init__(self, leaf_range: int):
        """The AVL tree only needs to store the possible leaf range to sample."""
        self.__leaf_range = leaf_range

    def __get_new_leaf(self) -> int:
        """Get a random leaf label within the range."""
        return secrets.randbelow(self.__leaf_range)

    @staticmethod
    def __get_height(node: Optional[AVLTreeNode]) -> int:
        """Get height of the input node."""
        # If node is empty, the height would be 0; otherwise return height.
        return node.height if node else 0

    @staticmethod
    def __get_balance(node: Optional[AVLTreeNode]) -> int:
        """Get balance of the input node."""
        # If node is empty, the balance would be 0; otherwise compute the balance.
        return AVLTree.__get_height(node.left_node) - AVLTree.__get_height(node.right_node) if node else 0

    def __update_height(self, node: AVLTreeNode) -> None:
        """Update the height of the input node."""
        node.height = 1 + max(self.__get_height(node.left_node), self.__get_height(node.right_node))

    def __rotate_left(self, in_node: AVLTreeNode) -> AVLTreeNode:
        """
        Perform a left rotation at the provided input node.

        :param in_node: some AVLTreeNode to rotate.
        :return: the parent node of the rotated node.
        """
        # Save the right child of the input node as the parent node.
        p_node = in_node.right_node
        # The input node will be the left child of the parent node; we store its left child to a tmp variable.
        tmp_mode = p_node.left_node

        # Now we set the input node as the left child of the parent node.
        p_node.left_node = in_node
        # The left child of parent node was on the right of input node.
        in_node.right_node = tmp_mode

        # Update the input node height.
        self.__update_height(in_node)
        # update the new parent node height.
        self.__update_height(p_node)

        # Return the new parent node.
        return p_node

    def __rotate_right(self, in_node: AVLTreeNode) -> AVLTreeNode:
        """
        Perform a left rotation at the provided input node.

        :param in_node: some AVLTreeNode to rotate.
        :return: the parent node of the rotated node.
        """
        # Save the left child of the input node as the parent node.
        p_node = in_node.left_node
        # The input node will be the right child of the parent node; we store its right child to a tmp variable.
        tmp_node = p_node.right_node

        # Now we set the input node as the right child of the parent node.
        p_node.right_node = in_node
        # The right child of parent node was on the left of input node.
        in_node.left_node = tmp_node

        # Update the input node height.
        self.__update_height(in_node)
        # update the new parent node height.
        self.__update_height(p_node)

        # Return the new parent node.
        return p_node

    def __balance(self, node: AVLTreeNode) -> AVLTreeNode:
        """Re-balance a node if it is unbalanced."""
        # Update the height of the node.
        self.__update_height(node)
        # Get the balance factor.
        balance = self.__get_balance(node)

        # Left heavy subtree rotation.
        if balance > 1:
            # The left-right case.
            if self.__get_balance(node.left_node) < 0:
                node.left_node = self.__rotate_left(node.left_node)
            # The left-left case.
            return self.__rotate_right(node)

        # Right heavy subtree rotation.
        if balance < -1:
            # The right-left case.
            if self.__get_balance(node.right_node) > 0:
                node.right_node = self.__rotate_right(node.right_node)
            # The right-right case.
            return self.__rotate_left(node)

        return node

    def insert(self, root: Optional[AVLTreeNode], kv_pair: KV_PAIR) -> AVLTreeNode:
        """
        Inserts a new node into the tree, which is represented by the root.

        :param root: the root node of the AVL tree.
        :param kv_pair: a tuple containing two values (key, value).
        :return: the updated AVL tree root node.
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
            if kv_pair[K] < node.key:
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

        # Re-balance the tree from the insertion point up to the root.
        while stack:
            # Get the last node and update its height.
            node = stack.pop()
            self.__update_height(node)
            # Balance the node at this position.
            balanced_node = self.__balance(node)

            # If parent exists, update which node the parent should point to.
            if stack:
                parent = stack[-1]
                if parent.left_node == node:
                    parent.left_node = balanced_node
                else:
                    parent.right_node = balanced_node
            else:
                return balanced_node

    def recursive_insert(self, root: Optional[AVLTreeNode], kv_pair: KV_PAIR) -> AVLTreeNode:
        """
        Inserts a new node into the tree, which is represented by the root.

        We also provide the recursive algorithm to validate correctness of the non-recursive approach.
        :param root: the root node of the AVL tree.
        :param kv_pair: a tuple containing two values (key, value).
        :return: the updated AVL tree root node.
        """
        # When we reach an empty root, create a new tree node to store the KV pair.
        if root is None:
            return AVLTreeNode(kv_pair)
        # If not empty node, we compare the key.
        elif kv_pair[K] < root.key:
            root.left_node = self.recursive_insert(root=root.left_node, kv_pair=kv_pair)
        else:
            root.right_node = self.recursive_insert(root=root.right_node, kv_pair=kv_pair)

        # Update height of the parent node.
        root.height = 1 + max(self.__get_height(node=root.left_node), self.__get_height(node=root.right_node))

        # Get the balance factor.
        balance = self.__get_balance(node=root)

        # Left heavy subtree rotation.
        if balance > 1:
            # The left-right case.
            if self.__get_balance(node=root.left_node) < 0:
                root.left_node = self.__rotate_left(root.left_node)
            # The left-left case.
            return self.__rotate_right(in_node=root)

        # Right heavy subtree rotation.
        if balance < -1:
            # The right-left case.
            if self.__get_balance(root.right_node) > 0:
                root.right_node = self.__rotate_right(root.right_node)
            # The right-right case.
            return self.__rotate_left(in_node=root)

        return root

    @staticmethod
    def search(key: Any, root: Optional[AVLTreeNode]) -> Any:
        """
        Performs a search on the provided key and root node.

        :param key: the key to search for.
        :param root: the root node of the AVL tree.
        :return: the value corresponding to the provided search key.
        """
        # While root is not empty.
        while root:
            if key < root.key:
                root = root.left_node
            elif key > root.key:
                root = root.right_node
            else:
                return root.value

        # If never found, return None.
        return None

    def post_order(self, root: Optional[AVLTreeNode], pos_map: dict) -> None:
        """
        Expand out the AVL tree stored in some root to a dictionary; the dictionary is modified in-place.

        The dictionary is of the following format:
            key: [leaf, [value, [left_key, right_key], [left_path, right_path], [left_height, right_height]]]
        :param root: the root node of the AVL tree.
        :param pos_map: some dictionary to store the tree information.
        """
        # Once we hit an empty root, terminate the recursion.
        if root is None:
            return

        # Create default values for the children information.
        tmp_child_key = [None, None]
        tmp_child_path = [None, None]
        tmp_child_height = [0, 0]

        # Traverse the left children.
        if root.left_node:
            self.post_order(pos_map=pos_map, root=root.left_node)
            tmp_child_key[L] = root.left_node.key
            tmp_child_path[L] = root.left_node.path
            tmp_child_height[L] = root.left_node.height

        # Travers the right children.
        if root.right_node:
            self.post_order(pos_map=pos_map, root=root.right_node)
            tmp_child_key[R] = root.right_node.key
            tmp_child_path[R] = root.right_node.path
            tmp_child_height[R] = root.right_node.height

        # Store the path of the leaf.
        root.path = self.__get_new_leaf()

        # Update the position map with correct information.
        pos_map[root.key] = [root.path, [root.value, tmp_child_key, tmp_child_path, tmp_child_height]]
