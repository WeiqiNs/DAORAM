"""Defines the B+ tree; note that the minimum order is 3 and inserting repeated keys may cause unexpected behavior."""
import secrets
from typing import Any, Optional, Tuple

from daoram.dependency.avl_tree import K, KV_PAIR


class BPlusTreeNode:
    def __init__(self):
        """
        Initialize a BPlusTreeNode object, containing the following attributes:
            - keys: a list of key values.
            - values: a list of values corresponding to the keys or a list of BPlusTreeNodes when this isn't a leaf.
            - parent: the parent node of this node.
            - is_leaf: True if this is a leaf node (containing actual values not nodes).
        """
        self.keys: list = []
        self.values: list = []
        self.parent: Optional[BPlusTreeNode] = None
        self.is_leaf: bool = True

    # Insert at the leaf
    def add_kv_pair(self, kv_pair: KV_PAIR):
        """
        Add a key-value pair to this node.

        :param kv_pair: a tuple containing two values (key, value).
        """
        # Unpack the key, value pair.
        key, value = kv_pair

        # If keys is not empty, we find the correct place to insert the input value.
        if self.keys:
            for index, each_key in enumerate(self.keys):
                # Keys go from smaller to larger; hence if a larger key is found, we insert new value in front.
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

        :param order: the branching degree of the B+ tree, minimum should be 3.
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

        :param key: the key to search for.
        :param root: the root node of the B+ tree.
        :return: the leaf storing the key of interest.
        """
        # Set the root to current node and iterate through it.
        cur_node = root

        # While not reaching a leaf node, keep searching.
        while not cur_node.is_leaf:
            for index, each_key in enumerate(cur_node.keys):
                # If key equals, it is on the right.
                if key == each_key:
                    cur_node = cur_node.values[index + 1]
                    break
                # If key is smaller, it is on the left.
                elif key < each_key:
                    cur_node = cur_node.values[index]
                    break
                # If we reached the end, it is on the right.
                elif index + 1 == len(cur_node.keys):
                    cur_node = cur_node.values[index + 1]
                    break

        # Return the node that was found.
        return cur_node

    def insert(self, root: BPlusTreeNode, kv_pair: KV_PAIR) -> BPlusTreeNode:
        """
        Inserts a new node into the tree, which is represented by the root.

        :param root: the root node of the B+ tree.
        :param kv_pair: a tuple containing two values (key, value).
        :return: the updated B+ tree root node.
        """
        # Find which leaf to insert.
        insert_leaf = self.__find_leaf(root=root, key=kv_pair[K])
        # Add the new input kv pair to this leaf.
        insert_leaf.add_kv_pair(kv_pair=kv_pair)

        # If the current leaf holds too many data, we split it.
        if len(insert_leaf.keys) == self.__order:
            new_leaf = BPlusTreeNode()
            new_leaf.parent = insert_leaf.parent
            # New leaf gets half of the old leaf.
            new_leaf.keys = insert_leaf.keys[self.__mid:]
            new_leaf.values = insert_leaf.values[self.__mid:]
            # The old leaf keeps on the first half.
            insert_leaf.keys = insert_leaf.keys[:self.__mid]
            insert_leaf.values = insert_leaf.values[:self.__mid]
            # Insert new key to the parent.
            root = self.__insert_in_parent(key=new_leaf.keys[0], left_node=insert_leaf, right_node=new_leaf)

        # Since we want to return the root, we keep going back up until no parent.
        while root.parent is not None:
            root = root.parent

        return root

    def __insert_in_parent(self, key: Any, left_node: BPlusTreeNode, right_node: BPlusTreeNode) -> BPlusTreeNode:
        """
        Inserts a node to the parent node.

        :param key: the key to insert to the parent node.
        :param left_node: the left node of the new parent node.
        :param right_node: the right node of the new parent node.
        :return: the parent node.
        """
        # If the parent node is empty, we create a new node.
        if left_node.parent is None:
            parent_node = BPlusTreeNode()
            # This parent node is not a leaf.
            parent_node.is_leaf = False
            parent_node.keys.append(key)
            parent_node.values = [left_node, right_node]
            left_node.parent = parent_node
            right_node.parent = parent_node
            return parent_node
        else:
            # Set the parent node.
            parent_node = left_node.parent

        # If parent node is not empty, we find where to insert.
        for index, each_key in enumerate(parent_node.keys):
            if key < each_key:
                parent_node.keys = parent_node.keys[:index] + [key] + parent_node.keys[index:]
                parent_node.values = parent_node.values[:index + 1] + [right_node] + parent_node.values[index + 1:]
                break
            elif index + 1 == len(parent_node.keys):
                parent_node.keys.append(key)
                parent_node.values.append(right_node)
                break

        # After insertion, we need to again check whether further insert to parent is needed.
        if len(parent_node.keys) == self.__order:
            # Create a new leaf.
            new_leaf = BPlusTreeNode()
            new_leaf.is_leaf = False
            new_leaf.parent = parent_node.parent

            # New leaf gets half of the old leaf.
            key = parent_node.keys[self.__mid]
            new_leaf.keys = parent_node.keys[self.__mid + 1:]
            new_leaf.values = parent_node.values[self.__mid + 1:]

            # Change their parents.
            for node in new_leaf.values:
                node.parent = new_leaf

            # The old leaf keeps on the first half.
            parent_node.keys = parent_node.keys[:self.__mid]
            parent_node.values = parent_node.values[:self.__mid + 1]
            # Insert new key to the parent.
            parent_node = self.__insert_in_parent(key=key, left_node=parent_node, right_node=new_leaf)

        return parent_node

    def search(self, key: Any, root: BPlusTreeNode) -> Any:
        """
        Performs a search on the provided key and root node.

        :param key: the key to search for.
        :param root: the root node of the B+ tree.
        :return: the value corresponding to the provided search key.
        """
        # First get the node leaf.
        leaf = self.__find_leaf(root=root, key=key)

        # Search in the node leaf.
        for index, each_key in enumerate(leaf.keys):
            if key == each_key:
                return leaf.values[index]

        raise KeyError(f"The key {key} is not found.")

    def post_order(self, root: BPlusTreeNode, pos_map: dict, block_id: int = 0) -> Tuple[int, int]:
        """
        Expand out the B+ tree stored in some root to a dictionary; the dictionary is modified in-place.

        The dictionary is of the following format:
            key: [block_id, leaf, [keys], [values]]
        :param root: the root node of the B+ tree.
        :param pos_map: some dictionary to store the tree information.
        :param block_id: some integer to store the block id.
        :return: block_id and leaf that represents the root.
        """
        # If the root is a leaf, we just store its keys and values.
        if root.is_leaf:
            # Create a new block to hold the id, leaf, keys, and values.
            block = [self.__get_new_leaf(), [root.keys, root.values]]
            pos_map[block_id] = block
            # Update the block id and return the leaf.
            return block_id + 1, block[0]

        # If the root is not a leaf, we keep searching until we reach a leaf.
        else:
            # Create a tmp list holder.
            tmp_values = []
            # Since this is not a leaf node, the values stored here are nodes.
            for i in root.values:
                # Recursively get block id and leafs.
                block_id, leaf = self.post_order(i, pos_map, block_id)
                # The block id was updated before return, hence -1.
                tmp_values.append((block_id - 1, leaf))

            # Create a new block and update the dictionary.
            block = [self.__get_new_leaf(), [root.keys, tmp_values]]
            pos_map[block_id] = block

            # If the root has parent, we return the updated id for future use, otherwise don't update it.
            if root.parent:
                return block_id + 1, block[0]
            else:
                return block_id, block[0]

    @staticmethod
    def __get_idx_in_parent(node: BPlusTreeNode) -> Optional[int]:
        """
        Given a node, find its index stored in parent values.

        :param node: some b+ tree node object.
        :return: the index of the node in the parent node values field.
        """
        # If parent is empty just return None.
        if node.parent is None:
            return None
        return node.parent.values.index(node)

    def __get_left_separator_idx(self, node: BPlusTreeNode) -> Optional[int]:
        """
        Given a node, find the index of its left separator.

        :param node: some b+ tree node object.
        :return: the index of the left key separating this value.
        """
        # Get index of this node in its parent node.
        index = self.__get_idx_in_parent(node=node)

        # If index of itself is None, just return None.
        if index is None:
            return None
        # Check if the index on the left is smaller than 0.
        elif index - 1 < 0:
            return None
        else:
            return index - 1

    def __get_right_separator_idx(self, node: BPlusTreeNode) -> Optional[int]:
        """
        Given a node, find the index of its right separator.

        :param node: some b+ tree node object.
        :return: the index of the right key separating this value.
        """
        # Get index of this node in its parent node.
        index = self.__get_idx_in_parent(node=node)

        # If index of itself is None, just return None.
        if index is None:
            return None
        # Check if the index on the right exceeds number of keys.
        elif index >= len(node.parent.keys):
            return None
        else:
            return index

    def __get_left_sibling(self, node: BPlusTreeNode) -> Optional[BPlusTreeNode]:
        """
        Given a node, find its left sibling node.

        :param node: some b+ tree node object.
        :return: the left sibling node of the input node.
        """
        # Get index of this node in its parent node.
        index = self.__get_idx_in_parent(node=node)

        # If index of itself is None, just return None.
        if index is None:
            return None
        # Check if the left sibling is empty.
        elif index - 1 < 0:
            return None
        else:
            return node.parent.values[index - 1]

    def __get_right_sibling(self, node: BPlusTreeNode) -> Optional[BPlusTreeNode]:
        """
        Given a node, find its right sibling node.

        :param node: some b+ tree node object.
        :return: the right sibling node of the input node.
        """
        index = self.__get_idx_in_parent(node=node)

        # If index of itself is None, just return None.
        if index is None:
            return None
        # Check if the right sibling is empty.
        elif index + 1 >= len(node.parent.values):
            return None
        else:
            return node.parent.values[index + 1]

    # def __borrow_from_left(self, node: BPlusTreeNode) -> bool:
    #     left = self.__get_left_sibling(node=node)
    #
    #     # If no left sibling return False.
    #     if left is None:
    #         return False
    #     # If there's not enough values to borrow, return False.
    #     elif len(left.values) <= self.__order // 2:
    #         return False
    #     # If there's sufficient amount of data, borrow.
    #     else:
    #         node.values
    #
    #
    #
    # def __borrow_from_right(self, node: BPlusTreeNode) -> bool:
    #
    #
    #
    # def delete(self, root, key):
    #     """Public method to initiate deletion."""
    #     self._delete(root, key)
    #     # If the root node becomes empty, make its child the new root
    #     if len(root.keys) == 0 and not root.is_leaf:
    #         root = root.values[0]
    #         root.parent = None
    #
    #     return root
    #
    # def _delete(self, node, key):
    #     """Recursive deletion method."""
    #     if node.is_leaf:
    #         # Delete key from leaf node
    #         if key in node.keys:
    #             node.keys.remove(key)
    #         return
    #
    #     # Key might be in internal node
    #     i = 0
    #     while i < len(node.keys) and key > node.keys[i]:
    #         i += 1
    #
    #     if i < len(node.keys) and key == node.keys[i]:
    #         if node.values[i].is_leaf:
    #             # Key is in internal node but its child is leaf, swap with predecessor
    #             node.keys[i] = self._get_predecessor(node, i)
    #             self._delete(node.values[i], node.keys[i])
    #         else:
    #             # Key is in internal node and child is internal, swap with successor
    #             node.keys[i] = self._get_successor(node, i)
    #             self._delete(node.values[i + 1], node.keys[i])
    #     else:
    #         # Recursively delete from child node
    #         self._delete(node.values[i], key)
    #
    #     # After deletion, ensure the child is valid
    #     if len(node.values[i].keys) < self.__order - 1:
    #         self._fix_deficiency(node, i)
    #
    # def _get_predecessor(self, node, idx):
    #     """Find the predecessor key of a given node and index."""
    #     current = node.values[idx]
    #     while not current.is_leaf:
    #         current = current.values[-1]
    #     return current.keys[-1]
    #
    # def _get_successor(self, node, idx):
    #     """Find the successor key of a given node and index."""
    #     current = node.values[idx + 1]
    #     while not current.is_leaf:
    #         current = current.values[0]
    #     return current.keys[0]
    #
    # def _fix_deficiency(self, parent, idx):
    #     """Handle deficiencies in the values of a node."""
    #     if idx > 0 and len(parent.values[idx - 1].keys) >= self.__order:
    #         # Borrow from left sibling
    #         self._borrow_from_left(parent, idx)
    #     elif idx < len(parent.values) - 1 and len(parent.values[idx + 1].keys) >= self.__order:
    #         # Borrow from right sibling
    #         self._borrow_from_right(parent, idx)
    #     else:
    #         # Merge with sibling
    #         if idx > 0:
    #             self._merge_with_left(parent, idx)
    #         else:
    #             self._merge_with_right(parent, idx)
    #
    # def _borrow_from_left(self, parent, idx):
    #     """Borrow key from left sibling."""
    #     left_sibling = parent.values[idx - 1]
    #     child = parent.values[idx]
    #
    #     # Shift keys in child to the right
    #     child.keys.insert(0, parent.keys[idx - 1])
    #     parent.keys[idx - 1] = left_sibling.keys.pop()
    #
    #     if not child.is_leaf:def is_power_of_2(n):
    #     return (n > 0) and (n & (n - 1)) == 0
    #
    #
    # def run_exp(oram, file_name):
    #     oram.init_server_storage()
    #
    #     for i in range(2 ** 20 + 1):
    #         oram.operate_on_key(op="w", key=i, value=i)
    #
    #         if is_power_of_2(i):
    #             with open(file_name, "a") as file:
    #                 # Write the content to the file
    #                 file.write(f"({math.log2(i)}, {oram.max_stash})\n")
    #
    #
    # run_exp(FreecursiveOram(
    #     num_data=2 ** 24, data_size=4, client=InteractLocalServer(), use_encryption=False
    # ), file_name="daoram_stash.txt")
    #
    # run_exp(FreecursiveOram(
    #     num_data=2 ** 24, data_size=4, client=InteractLocalServer(), use_encryption=False, reset_method="hard"
    # ), file_name="freecursive_hard.txt")
    #
    # run_exp(DAOram(
    #     num_data=2 ** 24, data_size=4, client=InteractLocalServer(), use_encryption=False
    # ), file_name="da_oram.txt")
    #         child.values.insert(0, left_sibling.values.pop())
    #         child.values[0].parent = child  # Update parent
    #
    # def _borrow_from_right(self, parent, idx):
    #     """Borrow key from right sibling."""
    #     right_sibling = parent.values[idx + 1]
    #     child = parent.values[idx]
    #
    #     # Shift keys in child to the left
    #     child.keys.append(parent.keys[idx])
    #     parent.keys[idx] = right_sibling.keys.pop(0)
    #
    #     if not child.is_leaf:
    #         child.values.append(right_sibling.values.pop(0))
    #         child.values[-1].parent = child  # Update parent
    #
    # def _merge_with_left(self, parent, idx):
    #     """Merge child with left sibling."""
    #     left_sibling = parent.values[idx - 1]
    #     child = parent.values[idx]
    #
    #     left_sibling.keys.append(parent.keys.pop(idx - 1))
    #     left_sibling.keys.extend(child.keys)
    #
    #     if not child.is_leaf:
    #         left_sibling.values.extend(child.values)
    #         for c in child.values:
    #             c.parent = left_sibling  # Update parent
    #
    #     parent.values.pop(idx)
    #
    # def _merge_with_right(self, parent, idx):
    #     """Merge child with right sibling."""
    #     right_sibling = parent.values[idx + 1]
    #     child = parent.values[idx]
    #
    #     child.keys.append(parent.keys.pop(idx))
    #     child.keys.extend(right_sibling.keys)
    #
    #     if not child.is_leaf:
    #         child.values.extend(right_sibling.values)
    #         for c in right_sibling.values:
    #             c.parent = child  # Update parent
    #
    #     parent.values.pop(idx + 1)
