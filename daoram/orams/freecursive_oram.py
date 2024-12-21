"""
This module defines the freecursive oram class.

Freecursive oram has three public methods:
    - init_storage_on_pos_map: this should be called first after the class object is created. This method constructs the
        storage the server should hold for the client.
    - compress_pos_map: this should be called after the storage is initialized as this will destroy the initial position
        map and compress it to a list of orams.
    - operate_on_key: after the server get the created storage, the client can use this function to obliviously access
        data points stored in the storage.

By default, we assume the position map oram store 512 bit values. We set the optimized parameters and have a compression
ratio of 1/48.
"""

import math
import random
import secrets
from typing import Any, List, Optional, Tuple

from daoram.dependency.binary_tree import BinaryTree, Buckets, KEY, LEAF, VALUE
from daoram.dependency.crypto import Prf
from daoram.dependency.helpers import Helper
from daoram.dependency.interact_server import InteractServer
from daoram.orams.tree_base_oram import TreeBaseOram

# Reset values is a list of the following [(key, cl, nl), (key, cl, nl)]; where only cl always has a value.
RESET_LEAVES = List[List[Tuple[Optional[int], int, Optional[int]]]]
# Set the type of processed data, which should be cur_leaf, new_leaf, and reset values if reset.
PROCESSED_DATA = Tuple[Optional[int], Optional[int], Optional[RESET_LEAVES]]


class FreecursiveOram(TreeBaseOram):
    def __init__(self,
                 num_data: int,
                 data_size: int,
                 num_ic: int = 48,
                 ic_length: int = 10,
                 gc_length: int = 32,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 aes_key: bytes = None,
                 prf_key: bytes = None,
                 on_chip_size: int = 10,
                 num_key_bytes: int = 16,
                 reset_method: str = "prob",
                 last_oram_data: int = None,
                 last_oram_level: int = None,
                 use_encryption: bool = True,
                 reset_prob: Optional[float] = None,
                 client: Optional[InteractServer] = None):
        """
        Initialize the freecursive oram with the following parameters.

        :param num_data: the number of data points the oram should store.
        :param data_size: the number of bytes the random dummy data should have.
        :param num_ic: number of individual count we store per block.
        :param ic_length: length of the binary representing individual count.
        :param gc_length: length of the binary representing group count.
        :param bucket_size: the number of data each bucket should have.
        :param stash_scale: the scaling scale of the stash.
        :param aes_key: the key to use for the AES instance, by default it will be randomly sampled.
        :param prf_key: the key to use for the PRF instance, by default it will be randomly sampled.
        :param on_chip_size: the number of data points the client can store.
        :param num_key_bytes: the number of bytes the aes key should have.
        :param reset_method: "prob" triggers reset with some probability and "hard" triggers reset when IC overflows.
        :param last_oram_data: the number of data points last oram stored (only useful for position map oram).
        :param last_oram_level: the level last oram has (only useful for position map oram).
        :param use_encryption: a boolean indicating whether to use encryption.
        :param reset_prob: the probability of reset to happen when reset method is "prob".
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
        self.__num_ic = num_ic
        self.__ic_length = ic_length
        self.__gc_length = gc_length
        self.__on_chip_size = on_chip_size
        self.__reset_method = reset_method
        self.__last_oram_data = last_oram_data
        self.__last_oram_level = last_oram_level
        self.__reset_prob = 1 / num_ic if reset_prob is None else reset_prob

        # This attribute is used to store a leaf temporarily for reading a path without evicting it immediately.
        self.__tmp_leaf: Optional[int] = None
        # We also need temp leaves in case of reset and evict multiple paths.
        self.__tmp_leaves: Optional[List[int]] = None
        # In case of reset, we also store leaves that have not been reset.
        self.__tmp_reset_leaves: Optional[RESET_LEAVES] = None

        # Create the prf instance.
        self.__prf = Prf(key=prf_key)

        # Need to store a list of oram and the on chip storage is empty by default.
        self.__on_chip_storage: List[bytes] = []
        self.__pos_maps: List[FreecursiveOram] = []

        # Initialize the position map upon creation.
        self._init_pos_map()

    @property
    def __count_length(self) -> int:
        """Get length of the binary representation what position map oram stores."""
        return self.__num_ic * self.__ic_length + self.__gc_length

    @property
    def __num_oram_pos_map(self) -> int:
        """Get the number of oram pos maps needed; note that last one will be stored on chip, hence -1."""
        return math.ceil(math.log(self._num_data / self.__on_chip_size, self.__num_ic)) - 1

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

            # Each position map now is an oram.
            cur_pos_map_oram = FreecursiveOram(
                aes_key=self._cipher.key if self._cipher else None,
                prf_key=self.__prf.key,
                num_ic=self.__num_ic,
                num_data=pos_map_size,
                ic_length=self.__ic_length,
                gc_length=self.__gc_length,
                data_size=self.__pos_map_oram_dummy_size,
                reset_prob=self.__reset_prob,
                bucket_size=self._bucket_size,
                stash_scale=self._stash_scale,
                reset_method=self.__reset_method,
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

            # Save the storage binary tree to server storage.
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

    def __evict_stash(self, leaf: int) -> Buckets:
        """
        Evict data blocks in the stash while maintaining correctness.

        :param leaf: the leaf label of the path we are evicting data to.
        :return: The leaf label and the path we should write there.
        """
        # Create a temporary stash.
        temp_stash = []

        # Create a placeholder for the new path.
        path = [[] for _ in range(self._level)]

        # Now we evict the stash by going through all real data in it.
        for data in self._stash:
            # Attempt to insert actual data to path.
            inserted = BinaryTree.fill_data_to_path(
                data=data, path=path, leaf=leaf, level=self._level, bucket_size=self._bucket_size
            )

            # If we were not able to insert data, overflow happened, put the block to the temp stash.
            if not inserted:
                temp_stash.append(data)

        # After we are done with all real data, complete the path with dummy data.
        BinaryTree.fill_buckets_with_dummy_data(buckets=path, bucket_size=self._bucket_size)

        # Update the stash.
        self._stash = temp_stash

        # Note that we return the path in the reversed order because we write path from bottom up.
        return self._encrypt_buckets(buckets=path[::-1])

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

    def __update_stash_leaves(
            self, key_one: int, key_two: int, n_leaf_one: int, n_leaf_two: int, to_index: int) -> None:
        """
        Perform a null operation for the purpose of updating leaves of a data block.

        :param key_one: the first key of interest.
        :param key_two: the second key of interest.
        :param n_leaf_one: the new leaf for the first key of interest.
        :param n_leaf_two: the new leaf for the second key of interest.
        :param to_index: check stash up until this index; so that the new added data won't be repeatedly checked.
        """
        # Read all buckets in the path and add real data to stash.
        for data in self._stash[:to_index]:
            # If we find key one, update it and set to None.
            if data[KEY] == key_one:
                data[LEAF] = n_leaf_one
                key_one = None
            # If we find key two, update it and set to None.
            elif data[KEY] == key_two:
                data[LEAF] = n_leaf_two
                key_two = None

        # When key one or key two is not None, raise a value.
        if key_one is not None:
            raise KeyError(f"Key {key_one} not found.")
        if key_two is not None:
            raise KeyError(f"Key {key_two} not found.")

    def __update_block_leaves(
            self, key_one: int, key_two: int, n_leaf_one: int, n_leaf_two: int, path: Buckets) -> None:
        """
        Perform a null operation for the purpose of updating leaves of a data block.

        :param key_one: the first key of interest.
        :param key_two: the second key of interest.
        :param n_leaf_one: the new leaf for the first key of interest.
        :param n_leaf_two: the new leaf for the second key of interest.
        :param path: the buckets of data that should contain the keys of interest.
        """
        # Store the current stash length.
        to_index = len(self._stash)

        # Decrypt data blocks if we use encryption.
        path = self._decrypt_buckets(buckets=path)

        # Read all buckets in the path and add real data to stash.
        for bucket in path:
            for data in bucket:
                # If key is None, we skip the bucket.
                if data[KEY] is None:
                    continue
                # If we find key one, update it and set to None.
                elif data[KEY] == key_one:
                    data[LEAF] = n_leaf_one
                    key_one = None
                # If we find key two, update it and set to None.
                elif data[KEY] == key_two:
                    data[LEAF] = n_leaf_two
                    key_two = None
                # Add to stash.
                self._stash.append(data)

        # When either key one or key two is not None, we search for the stash.
        if key_one is not None or key_two is not None:
            self.__update_stash_leaves(
                key_one=key_one,
                key_two=key_two,
                n_leaf_one=n_leaf_one,
                n_leaf_two=n_leaf_two,
                to_index=to_index
            )

    def __get_reset_leaves(self, key: int, data: str, reset_size: int = 2) -> RESET_LEAVES:
        """
        When reset happens, return list of current and next leaves.

        :param key: the key of the current data block where overflow happened.
        :param data: the gc and ic counts before resetting.
        :param reset_size: number of leaves we reset at once.
        :return: is a list of the following [(key, cl, nl), (key, cl, nl)]; where only cl always has a value.
        """
        # Get the group count.
        gc = int(data[:self.__gc_length], 2)
        # Get each individual count.
        ic = data[self.__gc_length:]
        ic = [int(ic[i * self.__ic_length: (i + 1) * self.__ic_length], 2) for i in range(self.__num_ic)]

        # If there's an overflow in the reset.
        if (key + 1) * self.__num_ic > self.__last_oram_data:
            reset_leaves = [
                (
                    key * self.__num_ic + i,
                    self.__get_previous_leaf_from_prf(key=key * self.__num_ic + i, gc=gc, ic=ic[i]),
                    self.__get_previous_leaf_from_prf(key=key * self.__num_ic + i, gc=gc + 1, ic=0)
                )
                # When the reset doesn't overflow, just randomly select a path.
                if key * self.__num_ic + i < self.__last_oram_data else
                (
                    None, secrets.randbelow(pow(2, self.__last_oram_level - 1)), None
                )
                for i in range(self.__num_ic)
            ]
        # If there's no overflow in the reset, compute everything as is.
        else:
            reset_leaves = [
                (
                    key * self.__num_ic + i,
                    self.__get_previous_leaf_from_prf(key=key * self.__num_ic + i, gc=gc, ic=ic[i]),
                    self.__get_previous_leaf_from_prf(key=key * self.__num_ic + i, gc=gc + 1, ic=0)
                )
                for i in range(self.__num_ic)
            ]

        # Break the list to chunks of reset_size.
        return [reset_leaves[i: i + reset_size] for i in range(0, len(reset_leaves), reset_size)]

    def __update_on_chip_data(self, key: int, offset: int) -> PROCESSED_DATA:
        """
        Given a data block and offset, compute the current.

        :param key: key to some data of interest.
        :param offset: the offset of where the value should be written to.
        :return: the computed current leaf and updated leaf, and reset leaves if reset happens.
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

        # Base on the reset method, perform the desired operation.
        if self.__reset_method == "prob":
            # Let reset happen with the desired probability.
            if random.random() <= self.__reset_prob:
                # We store the leaves that should be updated to reset_leaves.
                reset_leaves = self.__get_reset_leaves(key=key, data=data)
                # Now we set the new values for ic and gc.
                self.__on_chip_storage[key] = Helper.binary_str_to_bytes(
                    f"{bin(gc + 1)[2:].zfill(self.__gc_length)}{'0' * self.__ic_length * self.__num_ic}"
                )
                # In this case we don't need cl, nl; just the reset leaves.
                return None, None, reset_leaves

            # When reset does not happen.
            else:
                # We check whether the overflow happens; if so we throw an error.
                if ic + 1 >= pow(2, self.__ic_length):
                    raise ValueError("Overflow happened under probabilistic resets.")
                # We update the counts being stored.
                self.__on_chip_storage[key] = Helper.binary_str_to_bytes(
                    f"{data[:ic_start]}{bin(ic + 1)[2:].zfill(self.__ic_length)}{data[ic_end:]}"
                )
                # Compute the leaves.
                cur_leaf = self.__get_previous_leaf_from_prf(key=key * self.__num_ic + offset, gc=gc, ic=ic)
                new_leaf = self.__get_previous_leaf_from_prf(key=key * self.__num_ic + offset, gc=gc, ic=ic + 1)
                # In this case, we return the leaves and reset leaves are None.
                return cur_leaf, new_leaf, None

        elif self.__reset_method == "hard":
            # When overflow happens.
            if ic + 1 >= pow(2, self.__ic_length):
                # We store the leaves that should be updated to reset_leaves.
                reset_leaves = self.__get_reset_leaves(key=key, data=data)
                # Now we set the new values for ic and gc.
                self.__on_chip_storage[key] = Helper.binary_str_to_bytes(
                    f"{bin(gc + 1)[2:].zfill(self.__gc_length)}{'0' * self.__ic_length * self.__num_ic}"
                )
                # In this case we don't need cl, nl; just the reset leaves.
                return None, None, reset_leaves

            # When overflow does not happen.
            else:
                # We update the counts being stored.
                self.__on_chip_storage[key] = Helper.binary_str_to_bytes(
                    f"{data[:ic_start]}{bin(ic + 1)[2:].zfill(self.__ic_length)}{data[ic_end:]}"
                )
                # Compute the leaves.
                cur_leaf = self.__get_previous_leaf_from_prf(key=key * self.__num_ic + offset, gc=gc, ic=ic)
                new_leaf = self.__get_previous_leaf_from_prf(key=key * self.__num_ic + offset, gc=gc, ic=ic + 1)
                # In this case, we return the leaves and reset leaves are None.
                return cur_leaf, new_leaf, None

        else:
            raise ValueError(f"Unrecognized reset method {self.__reset_method}.")

    def __update_data_prob_reset(self, key: int, data: list, offset: int) -> PROCESSED_DATA:
        """
        Given a data block and offset, compute the current.

        :param key: key to some data of interest.
        :param data: a list contains key, index, and value.
        :param offset: the offset of where the value should be written to.
        :return: the computed current leaf and updated leaf, and reset leaves if reset happens.
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

        # When force_reset is set to True.
        if random.random() <= self.__reset_prob:
            # We store the leaves that should be updated to reset_leaves.
            reset_leaves = self.__get_reset_leaves(key=key, data=data[VALUE])
            # Now we set the new values for ic and gc.
            data[VALUE] = Helper.binary_str_to_bytes(
                f"{bin(gc + 1)[2:].zfill(self.__gc_length)}{'0' * self.__ic_length * self.__num_ic}"
            )
            # In this case we don't need cl, nl; just the reset leaves.
            return None, None, reset_leaves

        # When reset does not happen.
        else:
            # We check whether the overflow happens; if so we throw an error.
            if ic + 1 >= pow(2, self.__ic_length):
                raise ValueError("Overflow happened under probabilistic resets.")
            # We update the counts being stored.
            data[VALUE] = Helper.binary_str_to_bytes(
                f"{data[VALUE][:ic_start]}{bin(ic + 1)[2:].zfill(self.__ic_length)}{data[VALUE][ic_end:]}"
            )
            # Compute the leaves.
            cur_leaf = self.__get_previous_leaf_from_prf(key=key * self.__num_ic + offset, gc=gc, ic=ic)
            new_leaf = self.__get_previous_leaf_from_prf(key=key * self.__num_ic + offset, gc=gc, ic=ic + 1)
            # In this case, we return the leaves and reset leaves are None.
            return cur_leaf, new_leaf, None

    def __update_data_hard_reset(self, key: int, data: list, offset: int) -> PROCESSED_DATA:
        """
        Given a data block and offset, compute the current.

        :param key: key to some data of interest.
        :param data: a list contains key, index, and value.
        :param offset: the offset of where the value should be written to.
        :return: the computed current leaf and updated leaf, and reset leaves if reset happens.
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

        # When overflow happens.
        if ic + 1 >= pow(2, self.__ic_length):
            # We store the leaves that should be updated to reset_leaves.
            reset_leaves = self.__get_reset_leaves(key=key, data=data[VALUE])
            # Now we set the new values for ic and gc.
            data[VALUE] = Helper.binary_str_to_bytes(
                f"{bin(gc + 1)[2:].zfill(self.__gc_length)}{'0' * self.__ic_length * self.__num_ic}"
            )
            # In this case we don't need cl, nl; just the reset leaves.
            return None, None, reset_leaves

        # When overflow does not happen.
        else:
            # We update the counts being stored.
            data[VALUE] = Helper.binary_str_to_bytes(
                f"{data[VALUE][:ic_start]}{bin(ic + 1)[2:].zfill(self.__ic_length)}{data[VALUE][ic_end:]}"
            )
            # Compute the leaves.
            cur_leaf = self.__get_previous_leaf_from_prf(key=key * self.__num_ic + offset, gc=gc, ic=ic)
            new_leaf = self.__get_previous_leaf_from_prf(key=key * self.__num_ic + offset, gc=gc, ic=ic + 1)
            # In this case, we return the leaves and reset leaves are None.
            return cur_leaf, new_leaf, None

    def __retrieve_pos_map_stash(self, key: int, offset: int, new_leaf: int, to_index: int) -> PROCESSED_DATA:
        """
        Given a key and an operation, retrieve the block from stash and apply the operation to it.

        :param key: the key of the data block of interest.
        :param offset: the offset of where the value should be written to.
        :param to_index: up to which index in the stash we should be checking.
        :param new_leaf: if new leaf value is provided, store the accessed data to that leaf.
        :return: The leaf of the data block we found, and a value if the operation is "read".
        """
        # Read all buckets in the path and add real data to stash.
        for data in self._stash[:to_index]:
            # If we find the data of interest, perform operation, otherwise just skip over.
            if data[KEY] == key:
                next_cur_leaf, next_new_leaf, reset_leaves = self.__update_data_prob_reset(
                    key=key, data=data, offset=offset
                ) if self.__reset_method == "prob" else self.__update_data_prob_reset(
                    key=key, data=data, offset=offset
                )
                # This data block should be placed to where new leaf is.
                data[LEAF] = new_leaf
                # We can just break the loop as we found the target.
                return next_cur_leaf, next_new_leaf, reset_leaves

        # If the key was never found, raise an error, since the stash is always searched after path.
        raise KeyError(f"Key {key} not found.")

    def __retrieve_pos_map_block(self, key: int, offset: int, new_leaf: int, path: Buckets) -> PROCESSED_DATA:
        """
        Given a key and an operation, retrieve the block and apply the operation to it.

        :param key: a key to a data block.
        :param path: a list of buckets of data.
        :param offset: the offset of where the value should be written to.
        :param new_leaf: the leaf of where the block should be written to.
        :return: The leaf of the data block we found, and a value if the operation is "read".
        """
        # Assume that reset does not happen.
        next_cur_leaf, next_new_leaf, reset_leaves = None, None, None
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
                    next_cur_leaf, next_new_leaf, reset_leaves = self.__update_data_prob_reset(
                        key=key, data=data, offset=offset
                    ) if self.__reset_method == "prob" else self.__update_data_hard_reset(
                        key=key, data=data, offset=offset
                    )
                    # This data block should be placed to where new leaf is.
                    data[LEAF] = new_leaf
                # And all real data to the stash.
                self._stash.append(data)

        # Check if the stash overflows.
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")

        # If both next cur leaf and reset leaves are still None, we go search the stash.
        if next_cur_leaf is None and reset_leaves is None:
            next_cur_leaf, next_new_leaf, reset_leaves = self.__retrieve_pos_map_stash(
                key=key, offset=offset, new_leaf=new_leaf, to_index=to_index
            )

        # Return the desired values.
        return next_cur_leaf, next_new_leaf, reset_leaves

    def __retrieve_data_stash(self, op: str, key: int, to_index: int, new_leaf: int, value: Any = None) -> int:
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
        cur_leaf, new_leaf, reset_leaves = self.__update_on_chip_data(key=pos_map_keys[0][0], offset=pos_map_keys[0][1])

        # Declare useful variables.
        next_cur_leaf, next_new_leaf = None, None

        # We let the index start from -1, so that we read the on chip storage when it equals -1.
        for pos_map_index, (cur_key, cur_index) in enumerate(pos_map_keys[1:]):
            if reset_leaves is not None:
                # Iterate through all pairs of leaves.
                for cur_reset_leaves in reset_leaves:
                    # If the current reset leaves length is 2, we unpack both of them.
                    if len(cur_reset_leaves) == 2:
                        ck_one, cl_one, nl_one = cur_reset_leaves[0]
                        ck_two, cl_two, nl_two = cur_reset_leaves[1]
                        leaves = [cl_one, cl_two]
                    # Otherwise only unpack the first one, since the max length would be 2.
                    else:
                        ck_one, cl_one, nl_one = cur_reset_leaves[0]
                        ck_two, cl_two, nl_two = None, None, None
                        leaves = [cl_one]

                    # Read the leaves and get data.
                    path = self.client.read_query(label="pos_map", leaf=leaves, index=pos_map_index)

                    # Depends on which one is one we want next, compute next leaves.
                    if ck_one == cur_key:
                        next_cur_leaf, next_new_leaf, reset_leaves = (
                            self.__pos_maps[pos_map_index].__retrieve_pos_map_block(
                                key=ck_one, offset=cur_index, new_leaf=nl_one, path=path
                            )
                        )
                        self.__pos_maps[pos_map_index].__update_stash_leaf(key=ck_two, new_leaf=nl_two)
                    elif ck_two == cur_key:
                        next_cur_leaf, next_new_leaf, reset_leaves = (
                            self.__pos_maps[pos_map_index].__retrieve_pos_map_block(
                                key=ck_two, offset=cur_index, new_leaf=nl_two, path=path
                            )
                        )
                        self.__pos_maps[pos_map_index].__update_stash_leaf(key=ck_one, new_leaf=nl_one)
                    # Otherwise we just add them to stash and update leaves. (If key is not None.)
                    else:
                        self.__pos_maps[pos_map_index].__update_block_leaves(
                            key_one=ck_one, key_two=ck_two, n_leaf_one=nl_one, n_leaf_two=nl_two, path=path
                        )
                    # Evict the stash to current leaves.
                    path = self.__pos_maps[pos_map_index].__evict_stash_to_mul(leaves=leaves)

                    # Write to the server.
                    self.client.write_query(label="pos_map", leaf=leaves, index=pos_map_index, data=path)

            # Otherwise proceeds as normal.
            else:
                # Interact with server to get path.
                path = self.client.read_query(label="pos_map", leaf=cur_leaf, index=pos_map_index)

                # Find what leaves for the next iteration.
                next_cur_leaf, next_new_leaf, reset_leaves = (
                    self.__pos_maps[pos_map_index].__retrieve_pos_map_block(
                        key=cur_key, offset=cur_index, new_leaf=new_leaf, path=path
                    )
                )

                # Evict stash to current leaf.
                path = self.__pos_maps[pos_map_index].__evict_stash(leaf=cur_leaf)

                # Interact with server to store path.
                self.client.write_query(label="pos_map", leaf=cur_leaf, index=pos_map_index, data=path)

            # Update the next values to current.
            cur_leaf, new_leaf = next_cur_leaf, next_new_leaf

        return cur_leaf, new_leaf, reset_leaves

    def operate_on_key(self, op: str, key: int, value: Any = None) -> Any:
        """
        Perform operation on a given key.

        :param op: an operation, can be "r", "w" or "rw".
        :param key: the key of the data block of interest.
        :param value: If the operation is "w", this is the new value for data block.
        :return: The leaf of the data block we found, and a value if the operation is "r" or "rw".
        """
        # Find which path the data of interest lies on.
        cur_leaf, next_leaf, reset_leaves = self.__get_leaf_from_pos_map(key=key)

        # If reset happens at the last pos oram.
        if reset_leaves:
            # Iterate through all pairs of leaves.
            for cur_reset_leaves in reset_leaves:
                # If the current reset leaves length is 2, we unpack both of them.
                if len(cur_reset_leaves) == 2:
                    ck_one, cl_one, nl_one = cur_reset_leaves[0]
                    ck_two, cl_two, nl_two = cur_reset_leaves[1]
                    leaves = [cl_one, cl_two]
                # Otherwise only unpack the first one, since the max length would be 2.
                else:
                    ck_one, cl_one, nl_one = cur_reset_leaves[0]
                    ck_two, cl_two, nl_two = None, None, None
                    leaves = [cl_one]

                # Read the leaves and get data.
                path = self.client.read_query(label="oram", leaf=leaves)

                # Depends on which one is one we want next, compute next leaves.
                if ck_one == key:
                    value = self.__retrieve_data_block(
                        op=op, key=ck_one, value=value, new_leaf=nl_one, path=path
                    )
                    # Perform reset for the other value.
                    self.__update_stash_leaf(key=ck_two, new_leaf=nl_two)

                elif ck_two == key:
                    value = self.__retrieve_data_block(
                        op=op, key=ck_two, value=value, new_leaf=nl_two, path=path
                    )
                    # Perform reset for the other value.
                    self.__update_stash_leaf(key=ck_one, new_leaf=nl_one)

                # Otherwise we just add them to stash and update leaves. (If key is not None.)
                else:
                    self.__update_block_leaves(
                        key_one=ck_one, key_two=ck_two, n_leaf_one=nl_one, n_leaf_two=nl_two, path=path
                    )

                # Evict the stash to current leaves.
                path = self.__evict_stash_to_mul(leaves=leaves)

                # Interact with server to store path.
                self.client.write_query(label="oram", leaf=leaves, data=path)
        else:
            # Interact with server to get path.
            path = self.client.read_query(label="oram", leaf=cur_leaf)

            # Generate new leaf and read value from the path and show it.
            value = self.__retrieve_data_block(op=op, key=key, value=value, new_leaf=next_leaf, path=path)

            # Perform an eviction and get a new path.
            path = self.__evict_stash(leaf=cur_leaf)

            # Interact with server to store path.
            self.client.write_query(label="oram", leaf=cur_leaf, data=path)

        # TODO: this could be deleted (or commented out), it's only here for experimental purposes.
        self.max_stash = self.get_stash_size if self.get_stash_size > self.max_stash else self.max_stash

        return value

    def operate_on_key_without_eviction(self, op: str, key: int, value: Any = None) -> Any:
        """
        Perform operation on a given key without writing the data added to the stash back to the server.

        :param op: an operation, can be "r", "w" or "rw".
        :param key: the key of the data block of interest.
        :param value: If the operation is "w", this is the new value for data block.
        :return: The leaf of the data block we found, and a value if the operation is "r" or "rw".
        """
        # Find which path the data of interest lies on.
        cur_leaf, next_leaf, reset_leaves = self.__get_leaf_from_pos_map(key=key)

        # If reset happens at the last pos oram.
        if reset_leaves:
            # Iterate through all pairs of leaves.
            for index, cur_reset_leaves in enumerate(reset_leaves):
                # If the current reset leaves length is 2, we unpack both of them.
                if len(cur_reset_leaves) == 2:
                    ck_one, cl_one, nl_one = cur_reset_leaves[0]
                    ck_two, cl_two, nl_two = cur_reset_leaves[1]
                    leaves = [cl_one, cl_two]
                # Otherwise only unpack the first one, since the max length would be 2.
                else:
                    ck_one, cl_one, nl_one = cur_reset_leaves[0]
                    ck_two, cl_two, nl_two = None, None, None
                    leaves = [cl_one]

                # Read the leaves and get data.
                path = self.client.read_query(label="oram", leaf=leaves)

                # Depends on which one is one we want next, compute next leaves.
                if ck_one == key:
                    value = self.__retrieve_data_block(
                        op=op, key=ck_one, value=value, new_leaf=nl_one, path=path
                    )
                    # Perform reset for the other value.
                    self.__update_stash_leaf(key=ck_two, new_leaf=nl_two)

                    # Temporarily save the leaves and stash for future eviction.
                    self.__tmp_leaves = leaves

                    # If there are more values to reset, we temporarily store them.
                    if index < len(reset_leaves) - 1:
                        self.__tmp_reset_leaves = reset_leaves[index + 1:]

                    # Stop the reset process.
                    return value

                elif ck_two == key:
                    value = self.__retrieve_data_block(
                        op=op, key=ck_two, value=value, new_leaf=nl_two, path=path
                    )
                    # Perform reset for the other value.
                    self.__update_stash_leaf(key=ck_one, new_leaf=nl_one)

                    # Temporarily save the leaves and stash for future eviction.
                    self.__tmp_leaves = leaves

                    # If there are more values to reset, we temporarily store them.
                    if index < len(reset_leaves) - 1:
                        self.__tmp_reset_leaves = reset_leaves[index + 1:]

                    # Stop the reset process.
                    return value

                # Otherwise we just add them to stash and update leaves. (If key is not None.)
                else:
                    self.__update_block_leaves(
                        key_one=ck_one, key_two=ck_two, n_leaf_one=nl_one, n_leaf_two=nl_two, path=path
                    )

                    # In this case, we only evict data other than the one we are interested in.
                    path = self.__evict_stash_to_mul(leaves=leaves)

                    # Interact with server to store path.
                    self.client.write_query(label="oram", leaf=leaves, data=path)

        else:
            # Interact with server to get path.
            path = self.client.read_query(label="oram", leaf=cur_leaf)

            # Generate new leaf and read value from the path and show it.
            value = self.__retrieve_data_block(op=op, key=key, value=value, new_leaf=next_leaf, path=path)

            # Temporarily save the leaves for future eviction.
            self.__tmp_leaf = cur_leaf

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

        # Perform an eviction and get a new path depending on which temp leaf/leaves is not None.
        if self.__tmp_leaf is not None:
            # Perform an eviction and get a new path.
            path = self.__evict_stash(leaf=self.__tmp_leaf)
            # Interact with server to store path.
            self.client.write_query(label="oram", leaf=self.__tmp_leaf, data=path)
            # Set temporary leaf to None.
            self.__tmp_leaf = None
        else:
            # Perform an eviction and get a new path.
            path = self.__evict_stash_to_mul(leaves=self.__tmp_leaves)
            # Interact with server to store path.
            self.client.write_query(label="oram", leaf=self.__tmp_leaves, data=path)
            # Set temporary leaves to None.
            self.__tmp_leaves = None
            # Continue reset if there are values left.
            if self.__tmp_reset_leaves is not None:
                for cur_reset_leaves in self.__tmp_reset_leaves:
                    # If the current reset leaves length is 2, we unpack both of them.
                    if len(cur_reset_leaves) == 2:
                        ck_one, cl_one, nl_one = cur_reset_leaves[0]
                        ck_two, cl_two, nl_two = cur_reset_leaves[1]
                        leaves = [cl_one, cl_two]
                    # Otherwise only unpack the first one, since the max length would be 2.
                    else:
                        ck_one, cl_one, nl_one = cur_reset_leaves[0]
                        ck_two, cl_two, nl_two = None, None, None
                        leaves = [cl_one]

                    # Read the leaves and get data.
                    path = self.client.read_query(label="oram", leaf=leaves)

                    self.__update_block_leaves(
                        key_one=ck_one, key_two=ck_two, n_leaf_one=nl_one, n_leaf_two=nl_two, path=path
                    )

                    # In this case, we only evict data other than the one we are interested in.
                    path = self.__evict_stash_to_mul(leaves=leaves)

                    # Interact with server to store path.
                    self.client.write_query(label="oram", leaf=leaves, data=path)

            # Set temporary reset leaves to None.
            self.__tmp_reset_leaves = None
