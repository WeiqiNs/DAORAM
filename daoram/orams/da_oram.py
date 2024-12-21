"""
This module defines the DAOram (De-Amortized Oram) class.

DAOram has three public methods:
    - init_storage_on_pos_map: this should be called first after the class object is created. This method constructs the
        storage the server should hold for the client.
    - compress_pos_map: this should be called after the storage is initialized as this will destroy the initial position
        map and compress it to a list of orams.
    - operate_on_key: after the server get the created storage, the client can use this function to obliviously access
        data points stored in the storage.

By default, we assume the position map oram store 512 bit values. We set the optimized parameters and have a compression
ratio of 1/64.
"""

import math
import secrets
from typing import Any, List, Optional, Tuple

from daoram.dependency.binary_tree import BinaryTree, Buckets, KEY, LEAF, VALUE
from daoram.dependency.crypto import Prf
from daoram.dependency.helpers import Helper
from daoram.dependency.interact_server import InteractServer
from daoram.orams.tree_base_oram import TreeBaseOram

# Reset leaf is a tuple (index, cur_leaf, new_leaf). The new_leaf would be None, when the index is -1.
RESET_LEAF = Tuple[int, int, Optional[int]]
# Set the type of processed data, which should be cur_leaf, new_leaf, reset_index, reset_cur_leaf, reset_new_leaf.
PROCESSED_DATA = Tuple[int, int, int, int, Optional[int]]


class DAOram(TreeBaseOram):
    def __init__(self,
                 num_data: int,
                 data_size: int,
                 num_ic: int = 64,
                 ic_length: int = 6,
                 gc_length: int = 64,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 on_chip_mem: int = 10,
                 aes_key: bytes = None,
                 prf_key: bytes = None,
                 num_key_bytes: int = 16,
                 last_oram_data: int = None,
                 last_oram_level: int = None,
                 use_encryption: bool = True,
                 evict_path_obo: bool = False,
                 client: Optional[InteractServer] = None):
        """
        Initialize the average oram with the following parameters.

        :param num_data: the number of data points the oram should store.
        :param data_size: the number of bytes the random dummy data should have.
        :param num_ic: number of individual count we store per block.
        :param ic_length: length of the binary representing individual count.
        :param gc_length: length of the binary representing group count.
        :param bucket_size: the number of data each bucket should have.
        :param stash_scale: the scaling scale of the stash.
        :param on_chip_mem: the number of data points the client can store.
        :param aes_key: the key to use for the AES instance, by default it will be randomly sampled.
        :param prf_key: the key to use for the PRF instance, by default it will be randomly sampled.
        :param num_key_bytes: the number of bytes the aes key should have.
        :param last_oram_data: the number of data points last oram stored (only useful for position map oram).
        :param last_oram_level: the level last oram has (only useful for position map oram).
        :param use_encryption: a boolean indicating whether to use encryption.
        :param evict_path_obo: a boolean indicating whether to evict path one by one.
        :param client: the instance we use to interact with server; maybe None for pos map orams.
        """
        # Initialize the parent BaseOram class.
        super().__init__(
            client=client,
            aes_key=aes_key,
            num_data=num_data,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            num_key_bytes=num_key_bytes,
            use_encryption=use_encryption
        )

        # Add children class attributes.
        self.__num_ic: int = num_ic
        self.__ic_length: int = ic_length
        self.__gc_length: int = gc_length
        self.__on_chip_mem: int = on_chip_mem
        self.__last_oram_data: int = last_oram_data
        self.__last_oram_level: int = last_oram_level

        # This attribute is used to store some leaves temporarily for reading paths without evicting them immediately.
        self.__tmp_leaves: Optional[List[int]] = None

        # Create the prf instance.
        self.__prf: Prf = Prf(key=prf_key)

        # Need to store a list of oram and the on chip storage is empty by default.
        self.__on_chip_storage: List[bytes] = []
        self.__pos_maps: List[DAOram] = []

        # TODO: this could be removed in the future when experiments are done.
        self.__evict_path_obo = evict_path_obo

        # Initialize the position map upon creation.
        self._init_pos_map()

    @property
    def __count_length(self) -> int:
        """Get length of the binary representation what position map oram stores."""
        return self.__gc_length + (self.__ic_length + 1) * self.__num_ic

    @property
    def __num_oram_pos_map(self) -> int:
        """Get the number of oram pos maps needed; note that last one will be stored on chip, hence -1."""
        return math.ceil(math.log(self._num_data / self.__on_chip_mem, self.__num_ic)) - 1

    @property
    def __pos_map_oram_dummy_size(self) -> int:
        """Get the byte size of the random dummy data to store in position maps."""
        return math.ceil(self.__count_length / 8)

    def __get_pos_map_keys(self, key: int) -> List[Tuple[int, int]]:
        """
        Given a key, find what key and offset we should use for each position map oram.

        :param key: key to some data of interest.
        :return: a list of key, offset pairs.
        """
        # Create an empty list to hold result.
        pos_map_keys = []

        # For each position map, compute which block the key should be in, and it's index in value list.
        for i in range(self.__num_oram_pos_map + 1):
            index = key % self.__num_ic
            key = key // self.__num_ic
            pos_map_keys.insert(0, (key, index))

        # Return the result.
        return pos_map_keys

    def __get_leaf_from_prf(self, key: int, gc: int, ic: int) -> int:
        """
        Provide the key, gc, and ic, we compute what leaf they correspond to.

        :param key: key to some data of interest (index of position map).
        :param gc: group count, in binary representation.
        :param ic: individual count, in binary representation.
        :return: leaf computed as PRF(KEY||GC||IC) mod 2^L.
        """
        return self.__prf.digest_mod_n(Helper.binary_str_to_bytes(
            bin(key)[2:].zfill(self._level - 1) +  # key of the data
            bin(gc)[2:].zfill(self.__gc_length) +  # GC of the data
            bin(ic)[2:].zfill(self.__ic_length)  # IC of the data
        ), pow(2, self._level - 1))

    def __get_previous_leaf_from_prf(self, key: int, gc: int, ic: int) -> int:
        """
        Provide the key, gc, and ic, we compute what leaf they correspond to in the previous oram.

        :param key: key to some data of interest (index of position map).
        :param gc: group count, in binary representation.
        :param ic: individual count, in binary representation.
        :return: leaf computed as PRF(KEY||LAST_GC||IC) mod 2^LAST_L.
        """

        return self.__prf.digest_mod_n(Helper.binary_str_to_bytes(
            bin(key)[2:].zfill(self.__last_oram_level - 1) +  # key of the data
            bin(gc)[2:].zfill(self.__gc_length) +  # last GC of the data
            bin(ic)[2:].zfill(self.__ic_length)  # IC of the data
        ), pow(2, self.__last_oram_level - 1))

    def _init_pos_map(self) -> None:
        """Use PRF values to initialize the position map and override the base class method."""
        # For each label, its leaf is computed as AES(key||GC||IC).
        self._pos_map = {i: self.__get_leaf_from_prf(key=i, gc=0, ic=0) for i in range(self._num_data)}

    def __compress_pos_map(self) -> List[BinaryTree]:
        """Compress the large position map to a list of position map orams. """
        # We first delete the inited position map as we are going to compress it.
        self._pos_map = {}

        # Create the list of storage server needs.
        server_storage = []

        # We always set gc and ic to all zeros.
        value = Helper.binary_str_to_bytes("0" * self.__count_length)

        # Store the useful data about upper level oram.
        last_oram_data = self._num_data
        last_oram_level = self._level

        for i in range(self.__num_oram_pos_map):
            # Compute how many blocks this position map needs to store.
            pos_map_size = math.ceil(last_oram_data / self.__num_ic)

            # Each position map now is an oram. Note that here bucket size would be x.
            cur_pos_map_oram = DAOram(
                aes_key=self._cipher.key if self._cipher else None,
                prf_key=self.__prf.key,
                num_ic=self.__num_ic,
                num_data=pos_map_size,
                ic_length=self.__ic_length,
                gc_length=self.__gc_length,
                data_size=self.__pos_map_oram_dummy_size,
                bucket_size=self._bucket_size,
                stash_scale=self._stash_scale,
                last_oram_data=last_oram_data,
                last_oram_level=last_oram_level,
                num_key_bytes=self._num_key_bytes,
                use_encryption=self._use_encryption
            )

            # For current position map, get its corresponding binary tree.
            tree = BinaryTree(num_data=pos_map_size, bucket_size=self._bucket_size)

            # Since key is set to range from 0 to num_data - 1.
            for key, leaf in cur_pos_map_oram._pos_map.items():
                tree.fill_data_to_storage_leaf([key, leaf, value])

            # Before storing, fill tree with dummy data.
            tree.fill_storage_with_dummy_data()

            # Perform encryption if needed.
            tree.storage = cur_pos_map_oram._encrypt_buckets(buckets=tree.storage)

            # Update the last map used and last oram level.
            last_oram_data = pos_map_size
            last_oram_level = cur_pos_map_oram._level

            # Save the binary tree.
            server_storage.insert(0, tree)

            # Clear the current pos map oram and save it.
            cur_pos_map_oram.pos_map = {}
            self.__pos_maps.insert(0, cur_pos_map_oram)

        # The "master level" oram stores information about the smallest position map oram.
        self.__last_oram_data = last_oram_data
        self.__last_oram_level = last_oram_level

        # Get the on chip storage.
        self.__on_chip_storage = [value for _ in range(math.ceil(last_oram_data / self.__num_ic) + 1)]

        return server_storage

    def init_server_storage(self, data_map: dict = None) -> None:
        """
        Initialize the server storage based on the data map for this oram.

        :param data_map: a dictionary storing {key: data}.
        """
        # Initialize the storage.
        storage = self._init_storage_on_pos_map(data_map=data_map)

        # Compress the position map.
        pos_map_storage = self.__compress_pos_map()

        # Let server hold these storages.
        self.client.init_query(label="oram", storage=storage)
        self.client.init_query(label="pos_map", storage=pos_map_storage)

    def __evict_stash_to_mul(self, leaves: List[int]) -> Buckets:
        """
        Evict data blocks in the stash to multiple paths while maintaining correctness.

        :param leaves: a list of leaf labels of the path we are evicting data to.
        :return: The prepared path that should be written back to the storage.
        """
        # Create a temporary stash.
        temp_stash = []

        # Create a dictionary contains locations for where all leaves' paths touch.
        path_dict = BinaryTree.get_mul_path_dict(level=self._level, indices=leaves)

        # Now we evict the stash by going through all real data in it.
        for data in self._stash:
            # Attempt to insert actual data to path.
            inserted = BinaryTree.fill_data_to_mul_path(
                data=data, path=path_dict, leaves=leaves, level=self._level, bucket_size=self._bucket_size
            )
            # If we were not able to insert data, overflow happened, put the block to the temp stash.
            if not inserted:
                temp_stash.append(data)

        # After we are done with all real data, convert the dict to list of lists.
        path = [path_dict[key] for key in path_dict.keys()]

        # Then complete the path with dummy data.
        BinaryTree.fill_buckets_with_dummy_data(buckets=path, bucket_size=self._bucket_size)

        # Update the stash.
        self._stash = temp_stash

        # TODO: this could be deleted (or commented out), it's only here for experimental purposes.
        self.max_stash = self.get_stash_size if self.get_stash_size > self.max_stash else self.max_stash

        # Note that we return the path in the reversed order because we copy path from bottom up.
        return self._encrypt_buckets(buckets=path)

    def __evict_stash_to_mul_obo(self, leaves: List[int]) -> Buckets:
        """
        Evict data blocks in the stash to multiple paths while maintaining correctness.

        Note that this function maybe removed, it is a less aggressive way to perform eviction.
        :param leaves: a list of leaf labels of the path we are evicting data to.
        :return: The prepared path that should be written back to the storage.
        """
        # Create a temporary stash.
        temp_stash = []

        # Create a dictionary contains locations for where all leaves' paths touch.
        path_dict = BinaryTree.get_mul_path_dict(level=self._level, indices=leaves)

        # How do we evict path by path.
        for leaf in leaves:
            for data in self._stash:
                # Attempt to insert actual data to path.
                inserted = BinaryTree.fill_data_to_path_dict(
                    data=data, path=path_dict, leaf=leaf, level=self._level, bucket_size=self._bucket_size
                )

                # If we were not able to insert data, overflow happened, put the block to the temp stash.
                if not inserted:
                    temp_stash.append(data)

            # Update the stash to contain only not inserted value.
            self._stash = temp_stash
            # Set temp to empty again.
            temp_stash = []

        # After we are done with all real data, convert the dict to list of lists.
        path = [path_dict[key] for key in path_dict.keys()]

        # Then complete the path with dummy data.
        BinaryTree.fill_buckets_with_dummy_data(buckets=path, bucket_size=self._bucket_size)

        # TODO: this could be deleted (or commented out), it's only here for experimental purposes.
        self.max_stash = self.get_stash_size if self.get_stash_size > self.max_stash else self.max_stash

        # Note that we return the path in the reversed order because we copy path from bottom up.
        return self._encrypt_buckets(buckets=path)

    def __update_stash_leaf(self, key: int, new_leaf: int) -> None:
        """
        Look in the stash to update the leaf of the data block with key.

        :param key: the key of the data block of interest.
        :param new_leaf: If new leaf value is provided, store the accessed data to that leaf.
        """
        # If the key is empty, we don't need to perform update; terminate the function.
        if key is None:
            return

        # Look through the stash.
        for data in self._stash:
            # When we find the key, exit the function.
            if data[KEY] == key:
                data[LEAF] = new_leaf
                return

        # If the key was never found, raise an error, since the stash is always searched after path.
        raise KeyError(f"Key {key} not found.")

    def __perform_reset_on_chip(self, key: int) -> RESET_LEAF:
        """
        Find if a reset can be performed by looking at the ic indicators.

        :param key: key to the current data of interest.
        :return: if reset can be performed, find the leaf that needs the reset, its cur leaf and new leaf. If reset
        cannot be performed, return (-1, random_path, None).
        """
        # Get the data to update from on chip storage.
        data = Helper.bytes_to_binary_str(self.__on_chip_storage[key]).zfill(self.__count_length)

        # Get the indicator for whether ic is a backup.
        ic_indicators = data[-self.__num_ic:]

        # Finding 1 means we need a reset.
        offset = ic_indicators.find("1")

        # However if the key is outside the number of data, we turn offset off.
        if key * self.__num_ic + offset >= self.__last_oram_data:
            offset = -1

        # If not -1, means 1 was found. Then we read that count.
        if offset != -1:
            cur_leaf, new_leaf = self.__update_data_on_chip(key=key, offset=offset)
        # Just get a random leaf to read if no 1 exists.
        else:
            cur_leaf, new_leaf = secrets.randbelow(pow(2, self.__last_oram_level - 1)), None

        return offset, cur_leaf, new_leaf

    def __perform_reset(self, key: int, data: list) -> RESET_LEAF:
        """
        Find if a reset can be performed by looking at the ic indicators.

        :param key: key to the current data of interest.
        :param data: a list of lists, where each list contains key, index, and value.
        :return: if reset can be performed, find the leaf that needs the reset, its cur leaf and new leaf. If reset
        cannot be performed, return (-1, random_path, None).
        """
        # First convert bytes to binary string.
        data[VALUE] = Helper.bytes_to_binary_str(data[VALUE]).zfill(self.__count_length)

        # Get the indicator for whether ic is a backup.
        ic_indicators = data[VALUE][-self.__num_ic:]

        # Finding 1 means we need a reset.
        offset = ic_indicators.find("1")

        # Convert the value back to bytes.
        data[VALUE] = Helper.binary_str_to_bytes(data[VALUE])

        # However if the key is outside the number of data, we turn offset off.
        if key * self.__num_ic + offset >= self.__last_oram_data:
            offset = -1

        # If not -1, means 1 was found. Then we read that count.
        if offset != -1:
            cur_leaf, new_leaf = self.__update_data(key=key, data=data, offset=offset)
        # Just get a random leaf to read if no 1 exists.
        else:
            cur_leaf, new_leaf = secrets.randbelow(pow(2, self.__last_oram_level - 1)), None

        return offset, cur_leaf, new_leaf

    def __update_data_on_chip(self, key: int, offset: int) -> Tuple[int, int]:
        """
        Given a data block and offset, compute the current.

        :param key: key to some data of interest.
        :param offset: the offset of where the value ic is stored at.
        :return: the computed current leaf and updated leaf.
        """
        # First convert bytes to binary string.
        data = Helper.bytes_to_binary_str(self.__on_chip_storage[key]).zfill(self.__count_length)

        # Get the group count.
        gc = int(data[:self.__gc_length], 2)

        # Use index to grab the individual count.
        ic_start = self.__gc_length + offset * self.__ic_length
        ic_end = self.__gc_length + (offset + 1) * self.__ic_length

        # Convert the ic from binary representation to integer.
        ic = int(data[ic_start: ic_end], 2)

        # Get the ic start index.
        ic_ind_start = self.__gc_length + self.__num_ic * self.__ic_length

        # We check if the backup bit is 1.
        if data[ic_ind_start + offset] == "1":
            # Set values for next ic and next gc, in this case, ic and the one backup indicator need to be updated.
            next_ic = 0
            next_gc = gc
            gc = gc - 1
            # Update the ic backup indicator.
            data = f"{data[:ic_ind_start + offset]}{'0'}{data[ic_ind_start + offset + 1:]}"

        # When overflow happens.
        elif ic + 1 >= pow(2, self.__ic_length):
            # Set values for next ic and next gc; in this case ic, gc, and backup indicators all need to be updated.
            next_ic = 0
            next_gc = gc + 1
            # Update the group counter and ic backup indicator.
            data = (
                f"{bin(next_gc)[2:].zfill(self.__gc_length)}"
                f"{data[self.__gc_length:-self.__num_ic]}"
                f"{'1' * offset + '0' + '1' * (self.__num_ic - offset - 1)}"
            )

        # When overflow does not happen.
        else:
            # Set values for next ic and next gc; in this case, only ic needs to be updated.
            next_ic = ic + 1
            next_gc = gc

        # IC update always needs to happen no matter what.
        self.__on_chip_storage[key] = Helper.binary_str_to_bytes(
            f"{data[:ic_start]}{bin(next_ic)[2:].zfill(self.__ic_length)}{data[ic_end:]}"
        )

        # Compute the leaves.
        cur_leaf = self.__get_previous_leaf_from_prf(key=key * self.__num_ic + offset, gc=gc, ic=ic)
        new_leaf = self.__get_previous_leaf_from_prf(key=key * self.__num_ic + offset, gc=next_gc, ic=next_ic)

        # Return the current leaf and new leaf.
        return cur_leaf, new_leaf

    def __update_data(self, key: int, data: list, offset: int) -> Tuple[int, int]:
        """
        Given a data block and offset, compute the current.

        :param key: key to some data of interest.
        :param data: a list of lists, where each list contains key, index, and value.
        :param offset: the offset of where the value ic is stored at.
        :return: the computed current leaf and updated leaf.
        """
        # First convert bytes to binary string.
        data[VALUE] = Helper.bytes_to_binary_str(data[VALUE]).zfill(self.__count_length)

        # Get the group count.
        gc = int(data[VALUE][:self.__gc_length], 2)

        # Use index to grab the individual count.
        ic_start = self.__gc_length + offset * self.__ic_length
        ic_end = self.__gc_length + (offset + 1) * self.__ic_length

        # Convert the ic from binary representation to integer.
        ic = int(data[VALUE][ic_start: ic_end], 2)

        # Get the ic start index.
        ic_ind_start = self.__gc_length + self.__num_ic * self.__ic_length

        # We check if the backup bit is 1.
        if data[VALUE][ic_ind_start + offset] == "1":
            # Set values for next ic and next gc, in this case, ic and the one backup indicator need to be updated.
            next_ic = 0
            next_gc = gc
            gc = gc - 1
            # Update the ic backup indicator.
            data[VALUE] = f"{data[VALUE][:ic_ind_start + offset]}{'0'}{data[VALUE][ic_ind_start + offset + 1:]}"

        # When overflow happens.
        elif ic + 1 >= pow(2, self.__ic_length):
            # Set values for next ic and next gc; in this case ic, gc, and backup indicators all need to be updated.
            next_ic = 0
            next_gc = gc + 1
            # Update the group counter and ic backup indicator.
            data[VALUE] = (
                f"{bin(next_gc)[2:].zfill(self.__gc_length)}"
                f"{data[VALUE][self.__gc_length:-self.__num_ic]}"
                f"{'1' * offset + '0' + '1' * (self.__num_ic - offset - 1)}"
            )

        # When overflow does not happen.
        else:
            # Set values for next ic and next gc; in this case, only ic needs to be updated.
            next_ic = ic + 1
            next_gc = gc

        # IC update always needs to happen no matter what.
        data[VALUE] = Helper.binary_str_to_bytes(
            f"{data[VALUE][:ic_start]}{bin(next_ic)[2:].zfill(self.__ic_length)}{data[VALUE][ic_end:]}"
        )

        # Compute the leaves.
        cur_leaf = self.__get_previous_leaf_from_prf(key=key * self.__num_ic + offset, gc=gc, ic=ic)
        new_leaf = self.__get_previous_leaf_from_prf(key=key * self.__num_ic + offset, gc=next_gc, ic=next_ic)

        # Return the current leaf and new leaf.
        return cur_leaf, new_leaf

    def __retrieve_pos_map_stash_with_reset(
            self, key: int, offset: int, new_leaf: int, to_index: int, r_key: Optional[int], r_new_leaf: Optional[int]
    ) -> PROCESSED_DATA:
        """
        Given key and reset key, retrieve the blocks from stash and apply the operation on only the data of interest.

        Note that the reset key maybe None.
        :param key: key to some data of interest.
        :param r_key: key to the data we need to reset, which may be None.
        :param offset: the offset of where the value ic is stored at.
        :param new_leaf: the leaf where the data of interest should be stored to.
        :param r_new_leaf: the leaf where the reset data it should be stored to.
        :param to_index: up to which index in the stash we should be checking.
        :return: a tuple of five values
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
            if data[KEY] == key:
                # Get where the accessed data should be.
                next_cur_leaf, next_new_leaf = self.__update_data(key=key, data=data, offset=offset)
                # Check if the reset can be performed.
                r_index, r_next_cur_leaf, r_next_new_leaf = self.__perform_reset(key=key, data=data)
                # This data block should be placed to where new leaf is.
                data[LEAF] = new_leaf
                # Set key to None to indicate this has been found and changed.
                key = None
            elif data[KEY] == r_key:
                # This data block should be placed to where new leaf is.
                data[LEAF] = r_new_leaf
                # Set b_key to None to indicate this has been found and changed.
                r_key = None
            # When both are set to None, terminate the loop.
            if key is None and r_key is None:
                continue

        # If the key or b_key was never found, raise an error, since the stash is always searched after path.
        if key is not None:
            raise KeyError(f"Key {key} not found.")
        if r_key is not None:
            raise KeyError(f"The backup key {r_key} not found.")

        return next_cur_leaf, next_new_leaf, r_index, r_next_cur_leaf, r_next_new_leaf

    def __retrieve_pos_map_block_with_reset(
            self, key: int, offset: int, new_leaf: int, r_key: Optional[int], r_new_leaf: Optional[int], path: Buckets
    ) -> PROCESSED_DATA:
        """
        Given key and reset key, retrieve the blocks from stash and apply the operation on only the data of interest.

        Note that the reset key maybe None.
        :param key: key to some data of interest.
        :param r_key: key to the data we need to reset, which may be None.
        :param offset: the offset of where the value ic is stored at.
        :param new_leaf: the leaf where the data of interest should be stored to.
        :param r_new_leaf: the leaf where the reset data it should be stored to.
        :param path: a list of buckets of data.
        :return: a tuple of five values
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
        path = self._decrypt_buckets(buckets=path)

        # Read all buckets in the path and add real data to stash.
        for bucket in path:
            for data in bucket:
                # If dummy data, we skip it.
                if data[KEY] is None:
                    continue
                # If it's the data of interest, we both read and write it, and give it a new path.
                elif data[KEY] == key:
                    # Get where the accessed data should be.
                    next_cur_leaf, next_new_leaf = self.__update_data(key=key, data=data, offset=offset)
                    # Perform the reset on this data block.
                    r_index, r_next_cur_leaf, r_next_new_leaf = self.__perform_reset(key=key, data=data)
                    # This data block should be placed to where new leaf is.
                    data[LEAF] = new_leaf
                    key = None
                # If the backup key is None already, this will never be triggered.
                elif data[KEY] == r_key:
                    data[LEAF] = r_new_leaf
                    r_key = None

                # And all real data to the stash.
                self._stash.append(data)

        # Check if the stash overflows.
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")

        # If unable to make both key and r_key to be None, something should be wrong.
        if key is not None or r_key is not None:
            next_cur_leaf, next_new_leaf, r_index, r_next_cur_leaf, r_next_new_leaf = (
                self.__retrieve_pos_map_stash_with_reset(
                    key=key, r_key=r_key, offset=offset, to_index=to_index, new_leaf=new_leaf, r_new_leaf=r_new_leaf
                )
            )

        # Return the desired values.
        return next_cur_leaf, next_new_leaf, r_index, r_next_cur_leaf, r_next_new_leaf

    def __retrieve_data_stash(self, op: str, key: int, to_index: int, new_leaf: int, value: Any = None) -> Any:
        """
        Given a key and an operation, retrieve the block from stash and apply the operation to it.

        :param op: an operation, can be "r", "w" or "rw".
        :param key: the key of the data block of interest.
        :param to_index: up to which index we should be checking.
        :param value: If the operation is "write", this is the new value for data block.
        :param new_leaf: If new leaf value is provided, store the accessed data to that leaf.
        :return: The leaf of the data block we found, and a value if the operation is "read".
        """
        # Temp holder for the value to read.
        read_value = None

        # Read all buckets in the path and add real data to stash.
        for data in self._stash[:to_index]:
            # If we find the data of interest, perform operation, otherwise just skip over.
            if data[KEY] == key:
                if op == "r":
                    read_value = data[VALUE]
                elif op == "w":
                    data[VALUE] = value
                elif op == "rw":
                    read_value = data[VALUE]
                    data[VALUE] = value
                else:
                    raise ValueError("The provided operation is not valid.")
                # Get new path and update the position map.
                data[LEAF] = new_leaf
                # Set found to true.
                return read_value

        # If the key was never found, raise an error, since the stash is always searched after path.
        raise KeyError(f"Key {key} not found.")

    def __retrieve_data_block(self, op: str, key: int, new_leaf: int, path: Buckets, value: Any = None) -> Any:
        """
        Given a key and an operation, retrieve the block and apply the operation to it.

        :param op: an operation, can be "r", "w" or "rw".
        :param key: the key of the data block of interest.
        :param path: a list of buckets of data.
        :param value: If the operation is "write", this is the new value for data block.
        :param new_leaf: If new leaf value is provided, store the accessed data to that leaf.
        :return: The leaf of the data block we found, and a value if the operation is "read".
        """
        # Set a value for whether the key is found.
        found = False
        # Temp holder for the value to read.
        read_value = None
        # Store the current stash length.
        to_index = len(self._stash)

        # Decrypt the path if needed.
        path = self._decrypt_buckets(buckets=path)

        # Read all buckets in the path and add real data to stash.
        for bucket in path:
            for data in bucket:
                # If dummy data, we skip it.
                if data[KEY] is None:
                    continue
                # If it's the data of interest, we read/write it, and give it a new path.
                elif data[KEY] == key:
                    if op == "r":
                        read_value = data[VALUE]
                    elif op == "w":
                        data[VALUE] = value
                    elif op == "rw":
                        read_value = data[VALUE]
                        data[VALUE] = value
                    else:
                        raise ValueError("The provided operation is not valid.")
                    # Get new path and update the position map.
                    data[LEAF] = new_leaf
                    # Set found to True.
                    found = True
                # And all real data to the stash.
                self._stash.append(data)

        # Check if the stash overflows.
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")

        # If the value is not found, it might be in the stash.
        if not found:
            read_value = self.__retrieve_data_stash(op=op, key=key, to_index=to_index, value=value, new_leaf=new_leaf)

        return read_value

    def __get_leaf_from_pos_map(self, key: int) -> PROCESSED_DATA:
        """
        Provide a key to some data, iterate through all position map orams to find where it is stored.

        :param key: the key of the data block of interest.
        :return: which path the data block is on and the new path it should be stored to.
        """
        # We get the position map keys.
        pos_map_keys = self.__get_pos_map_keys(key=key)

        # Retrieve leaf from the on chip storage.
        cur_leaf, new_leaf = self.__update_data_on_chip(key=pos_map_keys[0][0], offset=pos_map_keys[0][1])

        # Check if the reset can be performed.
        r_index, r_cur_leaf, r_new_leaf = self.__perform_reset_on_chip(key=pos_map_keys[0][0])

        # Go through position maps to get the data of interest.
        for pos_map_index, (cur_key, cur_index) in enumerate(pos_map_keys[1:]):
            # If the reset index is -1, no reset needs to happen and r_key should be None.
            if r_index == -1:
                r_key = None
            # Otherwise compute the key for the value we are resetting.
            else:
                r_key = cur_key // self.__num_ic * self.__num_ic + r_index

            # We always retrieve two paths.
            leaves = [cur_leaf, r_cur_leaf]

            # Interact with server to get path.
            path = self.client.read_query(label="pos_map", leaf=leaves, index=pos_map_index)

            # Find what leaves for the next iteration.
            next_cur_leaf, next_new_leaf, r_index, r_cur_leaf, r_new_leaf = (
                self.__pos_maps[pos_map_index].__retrieve_pos_map_block_with_reset(
                    key=cur_key,
                    r_key=r_key,
                    offset=cur_index,
                    new_leaf=new_leaf,
                    r_new_leaf=r_new_leaf,
                    path=path
                )
            )

            # Evict to multiple paths and get what data to send to server.
            if self.__evict_path_obo:
                path = self.__pos_maps[pos_map_index].__evict_stash_to_mul_obo(leaves=leaves)
            else:
                path = self.__pos_maps[pos_map_index].__evict_stash_to_mul(leaves=leaves)

            # Interact with server to write path.
            self.client.write_query(label="pos_map", leaf=leaves, index=pos_map_index, data=path)

            # Update the new leaf to current leaf.
            cur_leaf, new_leaf = next_cur_leaf, next_new_leaf

        return cur_leaf, new_leaf, r_index, r_cur_leaf, r_new_leaf

    def operate_on_key(self, op: str, key: int, value: Any = None) -> Any:
        """
        Perform operation on a given key.

        :param op: an operation, can be "r", "w" or "rw".
        :param key: the key of the data block of interest.
        :param value: If the operation is "w", this is the new value for data block.
        :return: The leaf of the data block we found, and a value if the operation is "r" or "rw".
        """
        # Get current data leaves and reset leaves from position map orams.
        cur_leaf, new_leaf, r_index, r_cur_leaf, r_new_leaf = self.__get_leaf_from_pos_map(key=key)

        # If the reset index is -1, no reset needs to happen and r_key should be None.
        if r_index == -1:
            r_key = None
        # Otherwise compute the key for the value we are resetting.
        else:
            r_key = key // self.__num_ic * self.__num_ic + r_index

        # We always retrieve two paths.
        leaves = [cur_leaf, r_cur_leaf]

        # Interact with server to get path.
        path = self.client.read_query(label="oram", leaf=leaves)

        # Read the main oram and give the data of interest a new leaf label.
        value = self.__retrieve_data_block(op=op, key=key, value=value, new_leaf=new_leaf, path=path)

        # Before we evict, we also need to make sure the reset key is updated.
        self.__update_stash_leaf(key=r_key, new_leaf=r_new_leaf)

        # Perform an eviction and get a new path.
        if self.__evict_path_obo:
            path = self.__evict_stash_to_mul_obo(leaves=leaves)
        else:
            path = self.__evict_stash_to_mul(leaves=leaves)

        # Write the path back to the server.
        self.client.write_query(label="oram", leaf=leaves, data=path)

        return value

    def operate_on_key_without_eviction(self, op: str, key: int, value: Any = None) -> Any:
        """
        Perform operation on a given key without writing the data added to the stash back to the server.

        :param op: an operation, can be "r", "w" or "rw".
        :param key: the key of the data block of interest.
        :param value: If the operation is "w", this is the new value for data block.
        :return: The leaf of the data block we found, and a value if the operation is "r" or "rw".
        """
        # Get current data leaves and reset leaves from position map orams.
        cur_leaf, new_leaf, r_index, r_cur_leaf, r_new_leaf = self.__get_leaf_from_pos_map(key=key)

        # If the reset index is -1, no reset needs to happen and r_key should be None.
        if r_index == -1:
            r_key = None
        # Otherwise compute the key for the value we are resetting.
        else:
            r_key = key // self.__num_ic * self.__num_ic + r_index

        # We always retrieve two paths.
        leaves = [cur_leaf, r_cur_leaf]

        # Interact with server to get path.
        path = self.client.read_query(label="oram", leaf=leaves)

        # Read the main oram and give the data of interest a new leaf label.
        value = self.__retrieve_data_block(op=op, key=key, value=value, new_leaf=new_leaf, path=path)

        # Before we evict, we also need to make sure the reset key is updated.
        self.__update_stash_leaf(key=r_key, new_leaf=r_new_leaf)

        # Temporarily save the leaves for future eviction.
        self.__tmp_leaves = leaves

        return value

    def eviction_with_update_stash(self, key: int, value: Any) -> None:
        """Update a data block stored in the stash and then perform eviction.

        :param key: the key of the data block of interest.
        :param value: the value to update the data block of interest.
        """
        # Set found the key to False.
        found = False

        # Read all buckets stored in the stash and find the desired data block of interest.
        for data in self._stash:
            # If we find the data of interest, update value and set found to True.
            if data[KEY] == key:
                data[VALUE] = value
                found = True

        # If the data was never found, we raise an error.
        if not found:
            raise KeyError(f"Key {key} not found.")

        # Perform an eviction and get a new path.
        path = self.__evict_stash_to_mul(leaves=self.__tmp_leaves)

        # Write the path back to the server.
        self.client.write_query(label="oram", leaf=self.__tmp_leaves, data=path)

        # Set temporary leaves to None.
        self.__tmp_leaves = None
