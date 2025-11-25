from typing import Any

from pyoblivlib.dependency import InteractServer
from pyoblivlib.omap import AVLOdsOmap


class AVLOdsOmapOptimized(AVLOdsOmap):
    def __init__(self,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 client: InteractServer,
                 name: str = "avl_opt",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True):
        """
        Initializes the OMAP based on the AVL tree ODS.

        :param num_data: The number of data points the oram should store.
        :param key_size: The number of bytes the random dummy key should have.
        :param data_size: The number of bytes the random dummy data should have.
        :param client: The instance we use to interact with server.
        :param name: The name of the protocol, this should be unique if multiple schemes are used together.
        :param filename: The filename to save the oram data to.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param aes_key: The key to use for the AES instance.
        :param num_key_bytes: The number of bytes the aes key should have.
        :param use_encryption: A boolean indicating whether to use encryption.
        """
        # Initialize the inherited parent class.
        super().__init__(
            name=name,
            client=client,
            aes_key=aes_key,
            num_data=num_data,
            key_size=key_size,
            filename=filename,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            num_key_bytes=num_key_bytes,
            use_encryption=use_encryption
        )

    def insert(self, key: Any, value: Any) -> None:
        """
        Given key-value pair, insert the pair to the tree.

        :param key: the search key of interest.
        :param value: the value to insert.
        """
        # Create a new data block that holds the data to insert to tree.
        data_block = self._get_avl_data(key=key, value=value)

        # If the current root is empty, we simply set root as this new block.
        if self.root is None:
            # Add the data to stash and update root.
            self._stash.append(data_block)
            self.root = (data_block.key, data_block.leaf)
            # Perform the desired number of dummy operations.
            self._perform_dummy_operation(num_round=2 * self._max_height + 1)
            return

        # If the current root is not empty, we might have cache.
        self._stash += self._local
        self._local = []

        # Get the node information from oram storage.
        self._move_node_to_local(key=self.root[0], leaf=self.root[1])

        # Keep adding node to local until we find a place to insert the new node.
        while True:
            # Save the last node data and its value.
            node = self._local[-1]

            # If a node key is smaller, we go right and check whether a child is already there.
            if node.key < key:
                # If the child is there, we keep grabbing the next node.
                if node.value.r_key is not None:
                    self._move_node_to_local(key=node.value.r_key, leaf=node.value.r_leaf)
                # Else we store link the parent with the new node, other information will be updated during balance.
                else:
                    node.value.r_key = data_block.key
                    break

            # If the key is not smaller, we go left and check the same as above.
            else:
                if node.value.l_key is not None:
                    self._move_node_to_local(key=node.value.l_key, leaf=node.value.l_leaf)
                else:
                    node.value.l_key = data_block.key
                    break

        # Append the newly inserted node to local as well and perform balance.
        self._local.append(data_block)
        self._balance_local()

        # Save the number of retrieved nodes, move the local nodes to stash and perform dummy evictions.
        self._perform_dummy_operation(num_round=self._max_height + 1 - len(self._local))

    def search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value.

        If input value is not None, the value corresponding to the search tree will be updated.
        :param key: the search key of interest.
        :param value: the updated value.
        :return: the (old) value corresponding to the search key.
        """
        # If the local cache is not empty, move it to stash and evict while searching.
        if self._local:
            self._stash += self._local
            self._local = []

        # Set the value.
        value = super().fast_search(key=key, value=value)

        return value
