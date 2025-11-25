"""
This module defines the path oram class.

Path oram has two public methods:
    - init_storage_on_pos_map: this should be called first after the class object is created. This method constructs the
        storage the server should hold for the client.
    - operate_on_key: after the server gets the created storage, the client can use this function to obliviously access
        data points stored in the storage.
"""
from typing import List

from pyoblivlib.dependency import BinaryTree, InteractServer, Data, Buckets
from pyoblivlib.oram.path_oram import PathOram


class MulPathOram(PathOram):
    def __init__(self,
                 num_data: int,
                 data_size: int,
                 client: InteractServer,
                 name: str = "mp",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True):
        """
        Defines the path oram, including its attributes and methods.

        :param num_data: The number of data points the oram should store.
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
        # Initialize the parent BaseOram class.
        super().__init__(
            name=name,
            client=client,
            aes_key=aes_key,
            num_data=num_data,
            filename=filename,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            num_key_bytes=num_key_bytes,
            use_encryption=use_encryption
        )

        # In path oram, we initialize the position map.
        self._init_pos_map()

    @property
    def stash(self) -> list:
        """Return the stash."""
        return self._stash

    @stash.setter
    def stash(self, value: list):
        """Update the stash with input."""
        self._stash = value

    def init_server_storage(self, data: List[Data] = None) -> None:
        """
        Initialize the server storage based on the data map for this oram.

        :param data: A BinaryTree object.
        """
        # Create the binary tree object.
        tree = BinaryTree(
            filename=self._filename,
            num_data=self._num_data,
            data_size=self._max_block_size,
            bucket_size=self._bucket_size,
            enc_key_size=self._num_key_bytes if self._use_encryption else None,
        )

        # Fill the data to the correct place.
        for each_data in data:
            tree.fill_data_to_storage_leaf(data=each_data)

        # Encrypt the tree storage if needed.
        if self._use_encryption:
            tree.storage.encrypt(aes=self._cipher)

        # Initialize the storage and send it to the server.
        self.client.init_query(storage={self._name: tree})

    def process_path_to_stash(self, path: Buckets):
        """Decrypt path and add real data to stash."""
        # Read all buckets in the decrypted path and add real data to stash.
        for bucket in self._decrypt_buckets(buckets=path):
            for data in bucket:
                # We add all real data to stash and sample a new leaf.
                if data.key is not None:
                    data.leaf = self._get_new_leaf()
                    self._stash.append(data)

    def retrieve_path(self, leaves: List[int]) -> None:
        """
        Retrieve the provided path(s).

        :param leaves: A list of integers representing path numbers.
        """
        # Check that input path are within the correct range.
        for leaf in leaves:
            if leaf >= self._leaf_range:
                raise ValueError(f"The input path number {leaf} is not within the correct range.")

        # We read the path from the server.
        self.process_path_to_stash(path=self.client.read_query(label=self._name, leaf=leaves))

    def prepare_evict_path(self, leaves: List[int]) -> Buckets:
        # Create a temporary stash.
        temp_stash = []

        # Create a dictionary contains locations for where all leaves' paths touch.
        path_dict = BinaryTree.get_mul_path_dict(level=self._level, indices=leaves)

        # Now we evict the stash by going through all real data in it.
        for data in self._stash:
            # Attempt to insert actual data to the path.
            inserted = BinaryTree.fill_data_to_mul_path(
                data=data, path=path_dict, leaves=leaves, level=self._level, bucket_size=self._bucket_size
            )
            # If we were not able to insert data, overflow happened, put the block to the temp stash.
            if not inserted:
                temp_stash.append(data)

        # Update the stash.
        self._stash = temp_stash

        # After we are done with all real data, convert the dict to list of lists.
        return self._encrypt_buckets([path_dict[key] for key in path_dict.keys()])

    def evict_path(self, leaves: List[int]) -> None:
        """
        Evict the stash to the input paths.

        :param leaves: A list of integers representing path numbers.
        """
        # Encrypt and write the path back.
        self.client.write_query(label=self._name, leaf=leaves, data=self.prepare_evict_path(leaves=leaves))
