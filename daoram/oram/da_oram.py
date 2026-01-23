"""
This module defines the DAOram (De-Amortized Oram) class.

DAOram has three public methods:
    - init_storage_on_pos_map: this should be called first after the class object is created. This method constructs the
        storage the server should hold for the client.
    - compress_pos_map: this should be called after the storage is initialized as this will destroy the initial position
        map and compress it to a list of oram.
    - operate_on_key: after the server gets the created storage, the client can use this function to obliviously access
        data points stored in the storage.

By default, we assume the position map oram stores 512-bit values. We set the optimized parameters and have a
compression ratio of 1/64.
"""

import math
import secrets
from functools import cached_property
from typing import Any, List, Optional, Tuple

from daoram.dependency import BinaryTree, Data, Encryptor, Helper, InteractServer, PathData, Blake2Prf, UNSET
from daoram.oram.tree_base_oram import TreeBaseOram

# Reset leaf is a tuple (index, cur_leaf, new_leaf). The new_leaf would be None when the index is -1.
RESET_LEAF = Tuple[int, int, Optional[int]]
# Set the type of processed data, which should be cur_leaf, new_leaf, reset_index, reset_cur_leaf, reset_new_leaf.
PROCESSED_DATA = Tuple[int, int, int, int, Optional[int]]


class DAOram(TreeBaseOram):
    def __init__(self,
                 num_data: int,
                 data_size: int,
                 client: Optional[InteractServer] = None,
                 name: str = "da",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 num_ic: int = 64,
                 ic_length: int = 6,
                 gc_length: int = 64,
                 on_chip_mem: int = 10,
                 last_oram_data: int = None,
                 last_oram_level: int = None,
                 is_pos_map: bool = False,
                 evict_path_obo: bool = False,
                 encryptor: Encryptor = None,
                 prf_key: bytes = None):
        """
        Initialize the average oram with the following parameters.

        :param num_data: The number of data points the oram should store.
        :param data_size: The number of bytes the random dummy data should have.
        :param client: The instance we use to interact with server; None for pos map oram.
        :param name: The name of the protocol, this should be unique if multiple schemes are used together.
        :param filename: The filename to save the oram data to.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param num_ic: Number of individual counts we store per block.
        :param ic_length: Length of the binary representing individual count.
        :param gc_length: Length of the binary representing group count.
        :param on_chip_mem: The number of data points the client can store.
        :param last_oram_data: The number of data points last oram stored (only useful for position map oram).
        :param last_oram_level: The level the last oram has (only useful for position map oram).
        :param is_pos_map: Flag indicating this is a position map ORAM (no client required).
        :param evict_path_obo: A boolean indicating whether to evict path one by one.
        :param encryptor: The encryptor to use for encryption.
        :param prf_key: The key to use for the PRF instance, by default it will be randomly sampled.
        """
        # Validate that main ORAMs have a client.
        if not is_pos_map and client is None:
            raise ValueError("Client is required for main ORAM.")

        # Initialize the parent BaseOram class.
        super().__init__(
            num_data=num_data,
            data_size=data_size,
            client=client,
            name=name,
            filename=filename,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            encryptor=encryptor,
        )

        # Add children class attributes.
        self._num_ic: int = num_ic
        self._ic_length: int = ic_length
        self._gc_length: int = gc_length
        self._on_chip_mem: int = on_chip_mem
        self._last_oram_data: int = last_oram_data
        self._last_oram_level: int = last_oram_level

        # This attribute is used to store some leaves temporarily for reading paths without evicting them immediately.
        self._tmp_leaves: Optional[List[int]] = None

        # Create the prf instance.
        self._prf: Blake2Prf = Blake2Prf(key=prf_key)

        # Need to store a list of oram, and the on-chip storage is empty by default.
        self._on_chip_storage: List[bytes] = []
        self._pos_maps: List[DAOram] = []

        # Note this is only for experimental purpose and should not be used as it is slow with no benefits.
        self._evict_path_obo = evict_path_obo

        # Initialize the position map upon creation.
        self._init_pos_map()

    @cached_property
    def _count_length(self) -> int:
        """Get the length of the binary representation what position map oram stores."""
        return self._gc_length + (self._ic_length + 1) * self._num_ic

    @cached_property
    def _num_oram_pos_map(self) -> int:
        """Get the number of oram pos maps needed; note that the last one will be stored on chip, hence -1."""
        return math.ceil(math.log(self._num_data / self._on_chip_mem, self._num_ic)) - 1

    @cached_property
    def _pos_map_oram_dummy_size(self) -> int:
        """Get the byte size of the random dummy data to store in position maps."""
        return math.ceil(self._count_length / 8)

    def _get_pos_map_keys(self, key: int) -> List[Tuple[int, int]]:
        """
        Given a key, find what key and offset we should use for each position map oram.

        :param key: Key to some data of interest.
        :return: A list of (key, offset) pairs.
        """
        # Create an empty list to hold the result.
        pos_map_keys = []

        # For each position map, compute which block the key should be in, and its index in value list.
        for i in range(self._num_oram_pos_map + 1):
            index = key % self._num_ic
            key = key // self._num_ic
            pos_map_keys.append((key, index))

        # Reverse the list so we can go backwards.
        pos_map_keys.reverse()

        # Return the result.
        return pos_map_keys

    def _get_leaf_from_prf(self, key: int, gc: int, ic: int) -> int:
        """
        Provide the key, gc, and ic, we compute what leaf they correspond to.

        :param key: Key to some data of interest (index of a position map).
        :param gc: Group count, in binary representation.
        :param ic: Individual count, in binary representation.
        :return: Leaf computed as PRF(KEY||GC||IC) mod 2^L.
        """
        return self._prf.digest_mod_n(Helper.binary_str_to_bytes(
            bin(key)[2:].zfill(self._level - 1) +  # key of the data
            bin(gc)[2:].zfill(self._gc_length) +  # GC of the data
            bin(ic)[2:].zfill(self._ic_length)  # IC of the data
        ), pow(2, self._level - 1))

    def _get_previous_leaf_from_prf(self, key: int, gc: int, ic: int) -> int:
        """
        Provide the key, gc, and ic, we compute what leaf they correspond to in the previous oram.

        :param key: Key to some data of interest (index of a position map).
        :param gc: Group count, in binary representation.
        :param ic: Individual count, in binary representation.
        :return: Leaf computed as PRF(KEY||LAST_GC||IC) mod 2^LAST_L.
        """

        return self._prf.digest_mod_n(Helper.binary_str_to_bytes(
            bin(key)[2:].zfill(self._last_oram_level - 1) +  # key of the data
            bin(gc)[2:].zfill(self._gc_length) +  # last GC of the data
            bin(ic)[2:].zfill(self._ic_length)  # IC of the data
        ), pow(2, self._last_oram_level - 1))

    def _init_pos_map(self) -> None:
        """Use PRF values to initialize the position map and override the base class method."""
        # For each label, its leaf is computed as AES(key||GC||IC).
        self._pos_map = {i: self._get_leaf_from_prf(key=i, gc=0, ic=0) for i in range(self._num_data)}

    def _compress_pos_map(self) -> dict:
        """Compress the large position map to a list of position map oram. """
        # We first delete the inited position map as we are going to compress it.
        self._pos_map = {}

        # Create the list of storage server needs.
        server_storage = {}

        # We always set gc and ic to all zeros.
        value = Helper.binary_str_to_bytes("0" * self._count_length)

        # Store the useful data about upper level oram.
        last_oram_data = self._num_data
        last_oram_level = self._level

        for i in range(self._num_oram_pos_map):
            # Compute how many blocks this position map needs to store.
            pos_map_size = math.ceil(last_oram_data / self._num_ic)

            # Compute the filename for this position map ORAM.
            pos_map_filename = f"{self._filename}_pos_map_{self._num_oram_pos_map - i - 1}.bin" \
                if self._filename else None

            # Each position map now is an oram (without its own client).
            cur_pos_map_oram = DAOram(
                encryptor=self._encryptor,
                prf_key=self._prf.key,
                num_ic=self._num_ic,
                num_data=pos_map_size,
                ic_length=self._ic_length,
                gc_length=self._gc_length,
                data_size=self._pos_map_oram_dummy_size,
                bucket_size=self._bucket_size,
                stash_scale=self._stash_scale,
                last_oram_data=last_oram_data,
                last_oram_level=last_oram_level,
                filename=pos_map_filename,
                is_pos_map=True,
            )

            # For the current position map, get its corresponding binary tree.
            tree = BinaryTree(
                filename=pos_map_filename,
                num_data=pos_map_size,
                data_size=cur_pos_map_oram._dumped_data_size,
                bucket_size=self._bucket_size,
                disk_size=cur_pos_map_oram._disk_size,
                encryption=True if self._encryptor else False,
            )

            # Since key is set to range from 0 to num_data - 1.
            for key, leaf in cur_pos_map_oram._pos_map.items():
                tree.fill_data_to_storage_leaf(Data(key=key, leaf=leaf, value=value))

            # Encryption and fill with dummy data if needed.
            if self._encryptor:
                tree.storage.encrypt(encryptor=self._encryptor)

            # Update the last map used and last oram level.
            last_oram_data = pos_map_size
            last_oram_level = cur_pos_map_oram._level

            # Save the binary tree.
            server_storage[f"{self._name}_pos_map_{self._num_oram_pos_map - i - 1}"] = tree

            # Clear the current pos map oram and save it.
            cur_pos_map_oram._pos_map = {}
            self._pos_maps.append(cur_pos_map_oram)

        # The "master level" oram stores information about the smallest position map oram.
        self._last_oram_data = last_oram_data
        self._last_oram_level = last_oram_level

        # Get the on chip storage.
        self._on_chip_storage = [value for _ in range(math.ceil(last_oram_data / self._num_ic) + 1)]

        # Reverse the pos map oram.
        self._pos_maps.reverse()

        return server_storage

    def init_server_storage(self, data_map: dict = None) -> None:
        """
        Initialize the server storage based on the data map for this oram.

        :param data_map: A dictionary storing {key: data}.
        """
        # Initialize the storage.
        storage = self._init_storage_on_pos_map(data_map=data_map)

        # Compress the position map.
        pos_map_storage_dict = self._compress_pos_map()

        # Add the oram storage to the dictionary.
        pos_map_storage_dict[self._name] = storage

        # Let the server hold these storages.
        self.client.init_storage(storage=pos_map_storage_dict)

    def _evict_stash_obo(self, leaves: List[int]) -> PathData:
        """
        Evict data blocks in the stash to multiple paths while maintaining correctness.

        Note that this function maybe removed, it is a less aggressive way to perform eviction.
        :param leaves: A list of leaf labels of the path we are evicting data to.
        :return: PathData dict mapping storage index to encrypted bucket.
        """
        # Create a temporary stash.
        temp_stash = []

        # Create a dictionary contains locations for where all leaves' paths touch.
        path = BinaryTree.get_mul_path_dict(level=self._level, indices=leaves)

        # How do we evict path by path.
        for leaf in leaves:
            for data in self._stash:
                # Attempt to insert actual data to a path.
                inserted = BinaryTree.fill_data_to_path(
                    data=data, path=path, leaves=[leaf], level=self._level, bucket_size=self._bucket_size
                )

                # If we were not able to insert data, overflow happened, put the block to the temp stash.
                if not inserted:
                    temp_stash.append(data)

            # Update the stash to contain only not inserted value.
            self._stash = temp_stash
            # Set temp to empty again.
            temp_stash = []

        return self._encrypt_path_data(path=path)

    def _update_stash_leaf(self, key: int, new_leaf: int) -> None:
        """
        Look in the stash to update the leaf of the data block with a key.

        :param key: The key of the data block of interest.
        :param new_leaf: If a new leaf value is provided, store the accessed data to that leaf.
        """
        # If the key is empty, we don't need to perform update; terminate the function.
        if key is None:
            return

        # Look through the stash.
        for data in self._stash:
            # When we find the key, exit the function.
            if data.key == key:
                data.leaf = new_leaf
                return

        # If the key was never found, raise an error, since the stash is always searched after a path.
        raise KeyError(f"Key {key} not found.")

    def _perform_reset_on_chip(self, key: int) -> RESET_LEAF:
        """
        Find if a reset can be performed by looking at the ic indicators.

        :param key: Key to the current data of interest.
        :return: If reset can be performed, find the leaf that needs the reset, its cur leaf and new leaf. If reset
        cannot be performed, return (-1, random_path, None).
        """
        # Get the data to update from on chip storage.
        data = Helper.bytes_to_binary_str(self._on_chip_storage[key]).zfill(self._count_length)

        # Get the indicator for whether ic is a backup.
        ic_indicators = data[-self._num_ic:]

        # Finding 1 means we need a reset.
        offset = ic_indicators.find("1")

        # However, if the key is outside the number of data, we turn offset off.
        if key * self._num_ic + offset >= self._last_oram_data:
            offset = -1

        # If not -1, means 1 was found. Then we read that count.
        if offset != -1:
            cur_leaf, new_leaf = self._update_data_on_chip(key=key, offset=offset)
        # Get a random leaf to read if no one exists.
        else:
            cur_leaf, new_leaf = secrets.randbelow(pow(2, self._last_oram_level - 1)), None

        return offset, cur_leaf, new_leaf

    def _perform_reset(self, key: int, data: Data) -> RESET_LEAF:
        """
        Find if a reset can be performed by looking at the ic indicators.

        :param key: Key to the current data of interest.
        :param data: A Data object.
        :return: If reset can be performed, find the leaf that needs the reset, its cur leaf and new leaf. If reset
        cannot be performed, return (-1, random_path, None).
        """
        # First convert bytes to binary string.
        data.value = Helper.bytes_to_binary_str(data.value).zfill(self._count_length)

        # Get the indicator for whether ic is a backup.
        ic_indicators = data.value[-self._num_ic:]

        # Finding 1 means we need a reset.
        offset = ic_indicators.find("1")

        # Convert the value back to bytes.
        data.value = Helper.binary_str_to_bytes(data.value)

        # However, if the key is outside the number of data, we turn offset off.
        if key * self._num_ic + offset >= self._last_oram_data:
            offset = -1

        # If not -1, means 1 was found. Then we read that count.
        if offset != -1:
            cur_leaf, new_leaf = self._update_data(key=key, data=data, offset=offset)
        # Get a random leaf to read if no one exists.
        else:
            cur_leaf, new_leaf = secrets.randbelow(pow(2, self._last_oram_level - 1)), None

        return offset, cur_leaf, new_leaf

    def _update_data_on_chip(self, key: int, offset: int) -> Tuple[int, int]:
        """
        Given a data block and offset, compute the current.

        :param key: Key to some data of interest.
        :param offset: The offset of where the value ic is stored at.
        :return: The computed current leaf and updated leaf.
        """
        # First convert bytes to binary string.
        data = Helper.bytes_to_binary_str(self._on_chip_storage[key]).zfill(self._count_length)

        # Get the group count.
        gc = int(data[:self._gc_length], 2)

        # Use index to grab the individual count.
        ic_start = self._gc_length + offset * self._ic_length
        ic_end = self._gc_length + (offset + 1) * self._ic_length

        # Convert the ic from binary representation to integer.
        ic = int(data[ic_start: ic_end], 2)

        # Get the ic start index.
        ic_ind_start = self._gc_length + self._num_ic * self._ic_length

        # We check if the backup bit is 1.
        if data[ic_ind_start + offset] == "1":
            # Set values for next ic and next gc, in this case, ic, and the one backup indicator need to be updated.
            next_ic = 0
            next_gc = gc
            gc = gc - 1
            # Update the ic backup indicator.
            data = f"{data[:ic_ind_start + offset]}{'0'}{data[ic_ind_start + offset + 1:]}"

        # When overflow happens.
        elif ic + 1 >= pow(2, self._ic_length):
            # Set values for next ic and next gc; in this case, ic, gc, and backup indicators all need to be updated.
            next_ic = 0
            next_gc = gc + 1
            # Update the group counter and ic backup indicator.
            data = (
                f"{bin(next_gc)[2:].zfill(self._gc_length)}"
                f"{data[self._gc_length:-self._num_ic]}"
                f"{'1' * offset + '0' + '1' * (self._num_ic - offset - 1)}"
            )

        # When overflow does not happen.
        else:
            # Set values for next ic and next gc; in this case, only ic needs to be updated.
            next_ic = ic + 1
            next_gc = gc

        # IC update always needs to happen no matter what.
        self._on_chip_storage[key] = Helper.binary_str_to_bytes(
            f"{data[:ic_start]}{bin(next_ic)[2:].zfill(self._ic_length)}{data[ic_end:]}"
        )

        # Compute the leaves.
        cur_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic)
        new_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=next_gc, ic=next_ic)

        # Return the current leaf and new leaf.
        return cur_leaf, new_leaf

    def _update_data(self, key: int, data: Data, offset: int) -> Tuple[int, int]:
        """
        Given a data block and offset, compute the current.

        :param key: Key to some data of interest.
        :param data: A Data object.
        :param offset: The offset of where the value ic is stored at.
        :return: The computed current leaf and updated leaf.
        """
        # First convert bytes to binary string.
        data.value = Helper.bytes_to_binary_str(data.value).zfill(self._count_length)

        # Get the group count.
        gc = int(data.value[:self._gc_length], 2)

        # Use index to grab the individual count.
        ic_start = self._gc_length + offset * self._ic_length
        ic_end = self._gc_length + (offset + 1) * self._ic_length

        # Convert the ic from binary representation to integer.
        ic = int(data.value[ic_start: ic_end], 2)

        # Get the ic start index.
        ic_ind_start = self._gc_length + self._num_ic * self._ic_length

        # We check if the backup bit is 1.
        if data.value[ic_ind_start + offset] == "1":
            # Set values for next ic and next gc, in this case, ic, and the one backup indicator need to be updated.
            next_ic = 0
            next_gc = gc
            gc = gc - 1
            # Update the ic backup indicator.
            data.value = f"{data.value[:ic_ind_start + offset]}{'0'}{data.value[ic_ind_start + offset + 1:]}"

        # When overflow happens.
        elif ic + 1 >= pow(2, self._ic_length):
            # Set values for next ic and next gc; in this case, ic, gc, and backup indicators all need to be updated.
            next_ic = 0
            next_gc = gc + 1
            # Update the group counter and ic backup indicator.
            data.value = (
                f"{bin(next_gc)[2:].zfill(self._gc_length)}"
                f"{data.value[self._gc_length:-self._num_ic]}"
                f"{'1' * offset + '0' + '1' * (self._num_ic - offset - 1)}"
            )

        # When overflow does not happen.
        else:
            # Set values for next ic and next gc; in this case, only ic needs to be updated.
            next_ic = ic + 1
            next_gc = gc

        # IC update always needs to happen no matter what.
        data.value = Helper.binary_str_to_bytes(
            f"{data.value[:ic_start]}{bin(next_ic)[2:].zfill(self._ic_length)}{data.value[ic_end:]}"
        )

        # Compute the leaves.
        cur_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic)
        new_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=next_gc, ic=next_ic)

        # Return the current leaf and new leaf.
        return cur_leaf, new_leaf

    def _retrieve_pos_map_stash_with_reset(
            self, key: int, offset: int, new_leaf: int, to_index: int, r_key: Optional[int], r_new_leaf: Optional[int]
    ) -> PROCESSED_DATA:
        """
        Given key and reset key, retrieve the blocks from stash and apply the operation on only the data of interest.

        Note that the reset key maybe None.
        :param key: Key to some data of interest.
        :param r_key: Key to the data we need to reset, which may be None.
        :param offset: The offset of where the value ic is stored at.
        :param new_leaf: The leaf where the data of interest should be stored to.
        :param r_new_leaf: The leaf where the reset data it should be stored to.
        :param to_index: Up to which index in the stash, we should be checking.
        :return: A tuple of five values
            - next_cur_leaf: the leaf of the data of interest in the next position map.
            - next_new_leaf: the new leaf of the data of interest in the next position map.
            - r_index: -1 if no reset, otherwise some value to indicate which value we are resetting.
            - r_next_cur_leaf: the leaf of the reset data in the next position map. (a random path if no reset)
            - r_next_new_leaf: the new leaf of the reset data in the next position map. (None if no reset)
        """
        # Declare all the return values.
        next_cur_leaf, next_new_leaf, r_index, r_next_cur_leaf, r_next_new_leaf = None, None, None, None, None

        # Read data in the stash.
        for data in self._stash[:to_index]:
            # If we find the data of interest, perform operation.
            if data.key == key:
                # Get where the accessed data should be.
                next_cur_leaf, next_new_leaf = self._update_data(key=key, data=data, offset=offset)
                # Check if the reset can be performed.
                r_index, r_next_cur_leaf, r_next_new_leaf = self._perform_reset(key=key, data=data)
                # This data block should be placed to where the new leaf is.
                data.leaf = new_leaf
                # Set key as None to indicate this has been found and changed.
                key = None
            elif data.key == r_key:
                # This data block should be placed to where the new leaf is.
                data.leaf = r_new_leaf
                # Set b_key as None to indicate this has been found and changed.
                r_key = None
            # When both are set to None, terminate the loop.
            if key is None and r_key is None:
                continue

        # If the key or b_key was never found, raise an error, since the stash is always searched after the path.
        if key is not None:
            raise KeyError(f"Key {key} not found.")
        if r_key is not None:
            raise KeyError(f"The backup key {r_key} not found.")

        return next_cur_leaf, next_new_leaf, r_index, r_next_cur_leaf, r_next_new_leaf

    def _retrieve_pos_map_block_with_reset(
            self, key: int, offset: int, new_leaf: int, r_key: Optional[int], r_new_leaf: Optional[int], path: PathData
    ) -> PROCESSED_DATA:
        """
        Given key and reset key, retrieve the blocks from stash and apply the operation on only the data of interest.

        Note that the reset key maybe None.
        :param key: Key to some data of interest.
        :param r_key: Key to the data we need to reset, which may be None.
        :param offset: The offset of where the value ic is stored at.
        :param new_leaf: The leaf where the data of interest should be stored to.
        :param r_new_leaf: The leaf where the reset data it should be stored to.
        :param path: PathData dict mapping storage index to bucket.
        :return: A tuple of five values
            - next_cur_leaf: the leaf of the data of interest in the next position map.
            - next_new_leaf: the new leaf of the data of interest in the next position map.
            - r_index: -1 if no reset, otherwise some value to indicate which value we are resetting.
            - r_next_cur_leaf: the leaf of the reset data in the next position map. (a random path if no reset)
            - r_next_new_leaf: the new leaf of the reset data in the next position map. (None if no reset)
        """
        # Assume that reset does not happen.
        next_cur_leaf, next_new_leaf, r_index, r_next_cur_leaf, r_next_new_leaf = None, None, None, None, None

        # Store the current stash length.
        to_index = len(self._stash)

        # Decrypt data blocks if we use encryption.
        path = self.decrypt_path_data(path=path)

        # Read all buckets in the path and add real data to stash.
        for bucket in path.values():
            for data in bucket:
                # If dummy data, we skip it.
                if data.key is None:
                    continue
                # If it's the data of interest, we both read and write it, and give it a new path.
                elif data.key == key:
                    # Get where the accessed data should be.
                    next_cur_leaf, next_new_leaf = self._update_data(key=key, data=data, offset=offset)
                    # Perform the reset on this data block.
                    r_index, r_next_cur_leaf, r_next_new_leaf = self._perform_reset(key=key, data=data)
                    # This data block should be placed to where the new leaf is.
                    data.leaf = new_leaf
                    key = None
                # If the backup key is None already, this will never be triggered.
                elif data.key == r_key:
                    data.leaf = r_new_leaf
                    r_key = None

                # And all real data to the stash.
                self._stash.append(data)

        # Check if the stash overflows.
        if len(self._stash) > self._stash_size:
            raise OverflowError(
                f"Stash overflow in {self._name}: size {len(self._stash)} exceeds max {self._stash_size}.")

        # If unable to make both key and r_key to be None, something should be wrong.
        if key is not None or r_key is not None:
            next_cur_leaf, next_new_leaf, r_index, r_next_cur_leaf, r_next_new_leaf = (
                self._retrieve_pos_map_stash_with_reset(
                    key=key, r_key=r_key, offset=offset, to_index=to_index, new_leaf=new_leaf, r_new_leaf=r_new_leaf
                )
            )

        # Return the desired values.
        return next_cur_leaf, next_new_leaf, r_index, r_next_cur_leaf, r_next_new_leaf

    def _get_leaf_from_pos_map(self, key: int) -> PROCESSED_DATA:
        """
        Provide a key to some data, iterate through all position map oram to find where it is stored.

        :param key: The key of the data block of interest.
        :return: Which path the data block is on and the new path it should be stored to.
        """
        # We get the position map keys.
        pos_map_keys = self._get_pos_map_keys(key=key)

        # Retrieve leaf from the on chip storage.
        cur_leaf, new_leaf = self._update_data_on_chip(key=pos_map_keys[0][0], offset=pos_map_keys[0][1])

        # Check if the reset can be performed.
        r_index, r_cur_leaf, r_new_leaf = self._perform_reset_on_chip(key=pos_map_keys[0][0])

        # Go through position maps to get the data of interest.
        for pos_map_index, (cur_key, cur_index) in enumerate(pos_map_keys[1:]):
            # If the reset index is -1, no reset needs to happen and r_key should be None.
            if r_index == -1:
                r_key = None
            # Otherwise, compute the key for the value we are resetting.
            else:
                r_key = cur_key // self._num_ic * self._num_ic + r_index

            # We always retrieve two paths.
            leaves = [cur_leaf, r_cur_leaf]

            # Interact with server to get a path.
            self.client.add_read_path(label=f"{self._name}_pos_map_{pos_map_index}", leaves=leaves)
            result = self.client.execute()
            path = result.results[f"{self._name}_pos_map_{pos_map_index}"]

            # Find what leaves for the next iteration.
            # noinspection PyProtectedMember
            next_cur_leaf, next_new_leaf, r_index, r_cur_leaf, r_new_leaf = (
                self._pos_maps[pos_map_index]._retrieve_pos_map_block_with_reset(
                    key=cur_key,
                    r_key=r_key,
                    offset=cur_index,
                    new_leaf=new_leaf,
                    r_new_leaf=r_new_leaf,
                    path=path
                )
            )

            # Evict to multiple paths and get what data to send to server.
            if self._evict_path_obo:
                evicted_path = self._pos_maps[pos_map_index]._evict_stash_obo(leaves=leaves)
            else:
                evicted_path = self._pos_maps[pos_map_index]._evict_stash(leaves=leaves)

            # Interact with server to write a path.
            self.client.add_write_path(label=f"{self._name}_pos_map_{pos_map_index}", data=evicted_path)
            self.client.execute()

            # Update the new leaf to current leaf.
            cur_leaf, new_leaf = next_cur_leaf, next_new_leaf

        return cur_leaf, new_leaf, r_index, r_cur_leaf, r_new_leaf

    def operate_on_key(self, key: int, value: Any = UNSET) -> Any:
        """
        Perform operation on a given key. Always returns the current value.
        If value is provided (not UNSET), writes the new value.

        :param key: The key of the data block of interest.
        :param value: If provided (not UNSET), write this value to the data block.
        :return: The current value of the data block (before write if writing).
        """
        # Get current data leaves and reset leaves from position map oram.
        cur_leaf, new_leaf, r_index, r_cur_leaf, r_new_leaf = self._get_leaf_from_pos_map(key=key)

        # If the reset index is -1, no reset needs to happen and r_key should be None.
        if r_index == -1:
            r_key = None
        # Otherwise, compute the key for the value we are resetting.
        else:
            r_key = key // self._num_ic * self._num_ic + r_index

        # We always retrieve two paths.
        leaves = [cur_leaf, r_cur_leaf]

        # Interact with server to get the paths.
        self.client.add_read_path(label=self._name, leaves=leaves)
        result = self.client.execute()
        path = result.results[self._name]

        # Read the main oram and give the data of interest a new leaf label.
        read_value = self._retrieve_data_block(key=key, value=value, new_leaf=new_leaf, path=path)

        # Before we evict, we also need to make sure the reset key is updated.
        self._update_stash_leaf(key=r_key, new_leaf=r_new_leaf)

        # Perform an eviction and get a new path.
        if self._evict_path_obo:
            evicted_path = self._evict_stash_obo(leaves=leaves)
        else:
            evicted_path = self._evict_stash(leaves=leaves)

        # Write the path back to the server.
        self.client.add_write_path(label=self._name, data=evicted_path)
        self.client.execute()

        return read_value

    def operate_on_key_without_eviction(self, key: int, value: Any = UNSET) -> Any:
        """
        Perform operation on a given key without writing the data added to the stash back to the server.

        :param key: The key of the data block of interest.
        :param value: If provided (not UNSET), write this value to the data block.
        :return: The current value of the data block (before write if writing).
        """
        # Get current data leaves and reset leaves from position map oram.
        cur_leaf, new_leaf, r_index, r_cur_leaf, r_new_leaf = self._get_leaf_from_pos_map(key=key)

        # If the reset index is -1, no reset needs to happen and r_key should be None.
        if r_index == -1:
            r_key = None
        # Otherwise, compute the key for the value we are resetting.
        else:
            r_key = key // self._num_ic * self._num_ic + r_index

        # We always retrieve two paths.
        leaves = [cur_leaf, r_cur_leaf]

        # Interact with server to get the paths.
        self.client.add_read_path(label=self._name, leaves=leaves)
        result = self.client.execute()
        path = result.results[self._name]

        # Read the main oram and give the data of interest a new leaf label.
        read_value = self._retrieve_data_block(key=key, value=value, new_leaf=new_leaf, path=path)

        # Before we evict, we also need to make sure the reset key is updated.
        self._update_stash_leaf(key=r_key, new_leaf=r_new_leaf)

        # Temporarily save the leaves for future eviction.
        self._tmp_leaves = leaves

        return read_value

    def eviction_with_update_stash(self, key: int, value: Any, execute: bool = True) -> None:
        """Update a data block stored in the stash and then perform eviction.

        :param key: The key of the data block of interest.
        :param value: The value to update the data block of interest.
        :param execute: If True, execute immediately. If False, queue write for batching.
        """
        # Set found the key to False.
        found = False

        # Read all buckets stored in the stash and find the desired data block of interest.
        for data in self._stash:
            # If we find the data of interest, update value and set found to True.
            if data.key == key:
                data.value = value
                found = True

        # If the data was never found, we raise an error.
        if not found:
            raise KeyError(f"Key {key} not found.")

        # Perform an eviction and get a new path.
        evicted_path = self._evict_stash(leaves=self._tmp_leaves)

        # Write the path back to the server.
        self.client.add_write_path(label=self._name, data=evicted_path)
        if execute:
            self.client.execute()

        # Set temporary leaves to None.
        self._tmp_leaves = None
