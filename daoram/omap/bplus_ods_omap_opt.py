from typing import Any

from daoram.dependency import InteractServer
from daoram.omap import BPlusOdsOmap


class BPlusOdsOmapOptimized(BPlusOdsOmap):
    def __init__(self,
                 order: int,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 client: InteractServer,
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True):
        """
        Initializes the OMAP based on the B+ tree ODS.

        :param order: The branching order of the B+ tree.
        :param num_data: The number of data points the oram should store.
        :param key_size: The number of bytes the random dummy key should have.
        :param data_size: The number of bytes the random dummy data should have.
        :param client: The instance we use to interact with server.
        :param filename: The filename to save the oram data to.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param aes_key: The key to use for the AES instance.
        :param num_key_bytes: The number of bytes the aes key should have.
        :param use_encryption: A boolean indicating whether to use encryption.
        """
        # Initialize the parent BaseOmap class.
        super().__init__(
            order=order,
            client=client,
            aes_key=aes_key,
            filename=filename,
            num_data=num_data,
            key_size=key_size,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            num_key_bytes=num_key_bytes,
            use_encryption=use_encryption
        )

    def _perform_one_insertion(self):
        """Perform a single insertion in local."""
        # Assume we have not performed any insertion.
        inserted = False
        # We start from the last node.
        index = len(self._local) - 1

        # While we have not inserted and the index is larger than 0.
        while not inserted and index >= 0:
            # We check if the current node has too many values.
            if len(self._local[index].value.keys) >= 2 * self._order:
                raise MemoryError("There is an overflow to the block size.")

            # If so, we perform the insertion.
            if len(self._local[index].value.keys) >= self._order:
                # When insertion is needed, we first locate the parent.
                if index > 0:
                    # Perform the insertion.
                    self._insert_in_parent(child_node=self._local[index], parent_node=self._local[index - 1])
                    index -= 1
                # Or we need a new parent node.
                else:
                    self._create_parent(child_node=self._local[index])

                # Set inserted to be True.
                inserted = True

            else:
                # Decrement the index in local.
                index -= 1


    def insert(self, key: Any, value: Any) -> None:
        """
        Given key-value pair, insert the pair to the tree.

        :param key: the search key of interest.
        :param value: the value to insert.
        """
        # If the current root is empty, we simply set root as this new block.
        if self.root is None:
            # Create a new bplus data block.
            data_block = self._get_bplus_data(keys=[key], values=[value])
            # Append data block to the stash.
            self._stash.append(data_block)
            self.root = (data_block.key, data_block.leaf)
            # Perform dummy evictions.
            self._perform_dummy_operation(num_round=self._max_height + 1)
            # Terminates the function.
            return

        # If the current root is not empty, we might have cache.
        self._stash += self._local
        self._local = []

        # Get all nodes we need to visit until finding the key.
        self._find_leaf_to_local(key=key)

        # Set the last node in local as leaf.
        leaf = self._local[-1]

        # Find the proper place to insert the leaf.
        for index, each_key in enumerate(leaf.value.values):
            if key < each_key:
                leaf.value.keys = leaf.value.keys[:index] + [key] + leaf.value.keys[index:]
                leaf.value.values = leaf.value.values[:index] + [value] + leaf.value.values[index:]
                break
            elif index + 1 == len(leaf.value.keys):
                leaf.value.keys.append(key)
                leaf.value.values.append(value)
                break

        # Save the length of the local.
        num_retrieved_nodes = len(self._local)

        # Assume we have not performed any insertion.
        self._perform_one_insertion()

        # Perform desired number of dummy evictions.
        self._perform_dummy_operation(num_round=self._max_height + 1 - num_retrieved_nodes)

    def search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value.

        The difference here is that fast search will return the node immediately without keeping it in local.
        If the input value is not None, the value corresponding to the search tree will be updated.
        :param key: The search key of interest.
        :param value: The value to update.
        :return: The (old) value corresponding to the search key.
        """
        # If the local cache is not empty, move it to stash and evict while searching.
        if self._local:
            self._perform_insertion()
            self._stash += self._local
            self._local = []

        # Set the value.
        value = super().fast_search(key=key, value=value)

        # Append one more dummy operation.
        self._perform_dummy_operation(num_round=1)

        return value