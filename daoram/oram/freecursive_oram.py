"""
This module defines the freecursive oram class.

Freecursive oram has three public methods:
    - init_storage_on_pos_map: this should be called first after the class object is created. This method constructs the
        storage the server should hold for the client.
    - compress_pos_map: this should be called after the storage is initialized as this will destroy the initial position
        map and compress it to a list of oram.
    - operate_on_key: after the server gets the created storage, the client can use this function to obliviously access
        data points stored in the storage.

By default, we assume the position map oram stores 512-bit values. We set the optimized parameters and have a
compression ratio of 1/48.
"""

import math
import random
import secrets
from functools import cached_property
from typing import Any, List, Optional, Tuple

from daoram.dependency import BinaryTree, Data, Encryptor, Helper, InteractServer, PathData, Blake2Prf, UNSET
from daoram.oram.tree_base_oram import TreeBaseOram

# Reset values is a list of the following [(key, cl, nl), (key, cl, nl)]; where only cl always has a value.
RESET_LEAVES = List[List[Tuple[Optional[int], int, Optional[int]]]]
# Set the type of processed data, which should be cur_leaf, new_leaf, and reset values if reset.
PROCESSED_DATA = Tuple[Optional[int], Optional[int], Optional[RESET_LEAVES]]
# Unpacked reset leaves: (key_one, cur_leaf_one, new_leaf_one, key_two, cur_leaf_two, new_leaf_two, leaves).
UNPACKED_RESET = Tuple[Optional[int], int, Optional[int], Optional[int], Optional[int], Optional[int], List[int]]


class FreecursiveOram(TreeBaseOram):
    def __init__(self,
                 num_data: int,
                 data_size: int,
                 name: str = "fc",
                 num_ic: int = 48,
                 ic_length: int = 10,
                 gc_length: int = 32,
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 prf_key: bytes = None,
                 on_chip_size: int = 10,
                 is_pos_map: bool = False,
                 reset_method: str = "prob",
                 last_oram_data: int = None,
                 last_oram_level: int = None,
                 encryptor: Encryptor = None,
                 reset_prob: Optional[float] = None,
                 client: Optional[InteractServer] = None):
        """
        Initialize the freecursive oram with the following parameters.

        :param num_data: The number of data points the oram should store.
        :param data_size: The number of bytes the random dummy data should have.
        :param name: The name of the protocol, this should be unique if multiple schemes are used together.
        :param num_ic: Number of individual counts we store per block.
        :param ic_length: Length of the binary representing individual count.
        :param gc_length: Length of the binary representing group count.
        :param filename: The filename to save the oram data to.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param prf_key: The key to use for the PRF instance, by default it will be randomly sampled.
        :param on_chip_size: The number of data points the client can store.
        :param reset_method: "Prob" triggers reset with some probability and "hard" triggers reset when IC overflows.
        :param last_oram_data: The number of data points last oram stored (only useful for position map oram).
        :param last_oram_level: The level the last oram has (only useful for position map oram).
        :param reset_prob: The probability of reset to happen when reset method is "prob".
        :param is_pos_map: Flag indicating this is a position map ORAM (no client required).
        :param encryptor: The encryptor to use for encryption.
        :param client: The instance we use to interact with server; None for pos map oram.
        """
        # Validate that main ORAMs have a client.
        if not is_pos_map and client is None:
            raise ValueError("Client is required for main ORAM.")

        # Initialize the parent BaseOram class.
        super().__init__(
            name=name,
            client=client,
            filename=filename,
            num_data=num_data,
            data_size=data_size,
            encryptor=encryptor,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
        )

        # Add children class attributes.
        self._num_ic = num_ic
        self._ic_length = ic_length
        self._gc_length = gc_length
        self._on_chip_size = on_chip_size
        self._reset_method = reset_method
        self._last_oram_data = last_oram_data
        self._last_oram_level = last_oram_level
        self._reset_prob = 1 / num_ic if reset_prob is None else reset_prob

        # This attribute is used to store a leaf temporarily for reading a path without evicting it immediately.
        self._tmp_leaf: Optional[int] = None
        # We also need temp leaves in case of reset and evict multiple paths.
        self._tmp_leaves: Optional[List[int]] = None
        # In case of reset, we also store leaves that have not been reset.
        self._tmp_reset_leaves: Optional[RESET_LEAVES] = None

        # Create the prf instance.
        self._prf = Blake2Prf(key=prf_key)

        # Need to store a list of oram, and the on-chip storage is empty by default.
        self._on_chip_storage: List[bytes] = []
        self._pos_maps: List[FreecursiveOram] = []

        # Initialize the position map upon creation.
        self._init_pos_map()

    @cached_property
    def _count_length(self) -> int:
        """Get the length of the binary representation what position map oram stores."""
        return self._num_ic * self._ic_length + self._gc_length

    @cached_property
    def _num_oram_pos_map(self) -> int:
        """Get the number of oram pos maps needed; note that the last one will be stored on chip, hence -1."""
        return math.ceil(math.log(self._num_data / self._on_chip_size, self._num_ic)) - 1

    @cached_property
    def _pos_map_oram_dummy_size(self) -> int:
        """Get the byte size of the random dummy data to store in position maps."""
        return math.ceil(self._count_length / 8)

    @staticmethod
    def _unpack_reset_leaves(cur_reset_leaves: list) -> UNPACKED_RESET:
        """
        Unpack a reset leaves pair into individual components.

        :param cur_reset_leaves: A list of 1 or 2 tuples of (key, cur_leaf, new_leaf).
        :return: (ck_one, cl_one, nl_one, ck_two, cl_two, nl_two, leaves)
        """
        ck_one, cl_one, nl_one = cur_reset_leaves[0]
        if len(cur_reset_leaves) == 2:
            ck_two, cl_two, nl_two = cur_reset_leaves[1]
            leaves = [cl_one, cl_two]
        else:
            ck_two, cl_two, nl_two = None, None, None
            leaves = [cl_one]
        return ck_one, cl_one, nl_one, ck_two, cl_two, nl_two, leaves

    def _get_pos_map_keys(self, key: int) -> List[Tuple[int, int]]:
        """
        Given a key, find what key and offset we should use for each position map oram.

        :param key: Key to some data of interest.
        :return: A list of key offset pairs.
        """
        # Create an empty list to hold the result.
        pos_map_keys = []

        # For each position map, compute which block the key should be in, and its index in value list.
        for i in range(self._num_oram_pos_map + 1):
            index = key % self._num_ic
            key = key // self._num_ic
            pos_map_keys.append((key, index))

        # Reverse the pos map.
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
            cur_pos_map_oram = FreecursiveOram(
                encryptor=self._encryptor,
                prf_key=self._prf.key,
                num_ic=self._num_ic,
                num_data=pos_map_size,
                ic_length=self._ic_length,
                gc_length=self._gc_length,
                data_size=self._pos_map_oram_dummy_size,
                reset_prob=self._reset_prob,
                bucket_size=self._bucket_size,
                stash_scale=self._stash_scale,
                reset_method=self._reset_method,
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
                tree.fill_data_to_storage_leaf(data=Data(key=key, leaf=leaf, value=value))

            # Encryption and fill with dummy data if needed.
            if self._encryptor:
                tree.storage.encrypt(encryptor=self._encryptor)

            # Update the last map used and last oram level.
            last_oram_data = pos_map_size
            last_oram_level = cur_pos_map_oram._level

            # Save the storage binary tree to server storage.
            server_storage[f"{self._name}_pos_map_{self._num_oram_pos_map - i - 1}"] = tree

            # Clear the current pos map oram and save it.
            cur_pos_map_oram._pos_map = {}
            self._pos_maps.append(cur_pos_map_oram)

        # The "master level" oram stores information about the smallest position map oram.
        self._last_oram_data = last_oram_data
        self._last_oram_level = last_oram_level

        # Get the on chip storage.
        self._on_chip_storage = [value for _ in range(math.ceil(last_oram_data / self._num_ic) + 1)]

        # Reverse the position map.
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

        # If the key was never found, raise an error, since the stash is always searched after the path.
        raise KeyError(f"Key {key} not found.")

    def _update_stash_leaves(self, key_one: int, key_two: int, n_leaf_one: int, n_leaf_two: int, to_index: int) -> None:
        """
        Perform a null operation to update leaves of a data block.

        :param key_one: The first key of interest.
        :param key_two: The second key of interest.
        :param n_leaf_one: The new leaf for the first key of interest.
        :param n_leaf_two: The new leaf for the second key of interest.
        :param to_index: Check stash up until this index; so that the new added data won't be repeatedly checked.
        """
        # Read all buckets in the path and add real data to stash.
        for data in self._stash[:to_index]:
            # If we find key one, update it and set to None.
            if data.key == key_one:
                data.leaf = n_leaf_one
                key_one = None
            # If we find key two, update it and set to None.
            elif data.key == key_two:
                data.leaf = n_leaf_two
                key_two = None

        # When key one or key two is not None, raise a value.
        if key_one is not None:
            raise KeyError(f"Key {key_one} not found.")
        if key_two is not None:
            raise KeyError(f"Key {key_two} not found.")

    def _update_block_leaves(self, key_one: int, key_two: int, n_leaf_one: int, n_leaf_two: int,
                             path: PathData) -> None:
        """
        Perform a null operation to update leaves of a data block.

        :param key_one: The first key of interest.
        :param key_two: The second key of interest.
        :param n_leaf_one: The new leaf for the first key of interest.
        :param n_leaf_two: The new leaf for the second key of interest.
        :param path: PathData dict mapping storage index to bucket.
        """
        # Store the current stash length.
        to_index = len(self._stash)

        # Decrypt data blocks if we use encryption.
        path = self._decrypt_path_data(path=path)

        # Read all buckets in the path and add real data to stash.
        for bucket in path.values():
            for data in bucket:
                # If key is None, we skip the bucket.
                if data.key is None:
                    continue
                # If we find key one, update it and set to None.
                elif data.key == key_one:
                    data.leaf = n_leaf_one
                    key_one = None
                # If we find key two, update it and set to None.
                elif data.key == key_two:
                    data.leaf = n_leaf_two
                    key_two = None
                # Add to the stash.
                self._stash.append(data)

        # When either key one or key two is not None, we search for the stash.
        if key_one is not None or key_two is not None:
            self._update_stash_leaves(
                key_one=key_one,
                key_two=key_two,
                n_leaf_one=n_leaf_one,
                n_leaf_two=n_leaf_two,
                to_index=to_index
            )

    def _get_reset_leaves(self, key: int, data: str, reset_size: int = 2) -> RESET_LEAVES:
        """
        When reset happens, return the list of current and next leaves.

        :param key: The key of the current data block where overflow happened.
        :param data: The gc and ic count before resetting.
        :param reset_size: The number of leaves we reset at once.
        :return: A list of the following [(key, cl, nl), (key, cl, nl)]; where only cl always has a value.
        """
        # Get the group count.
        gc = int(data[:self._gc_length], 2)
        # Get each count.
        ic = data[self._gc_length:]
        ic = [int(ic[i * self._ic_length: (i + 1) * self._ic_length], 2) for i in range(self._num_ic)]

        # If there's an overflow in the reset.
        if (key + 1) * self._num_ic > self._last_oram_data:
            reset_leaves = [
                (
                    key * self._num_ic + i,
                    self._get_previous_leaf_from_prf(key=key * self._num_ic + i, gc=gc, ic=ic[i]),
                    self._get_previous_leaf_from_prf(key=key * self._num_ic + i, gc=gc + 1, ic=0)
                )
                # When the reset doesn't overflow, just randomly select a path.
                if key * self._num_ic + i < self._last_oram_data else
                (
                    None, secrets.randbelow(pow(2, self._last_oram_level - 1)), None
                )
                for i in range(self._num_ic)
            ]
        # If there's no overflow in the reset, compute everything as is.
        else:
            reset_leaves = [
                (
                    key * self._num_ic + i,
                    self._get_previous_leaf_from_prf(key=key * self._num_ic + i, gc=gc, ic=ic[i]),
                    self._get_previous_leaf_from_prf(key=key * self._num_ic + i, gc=gc + 1, ic=0)
                )
                for i in range(self._num_ic)
            ]

        # Break the list to chunks of reset_size.
        return [reset_leaves[i: i + reset_size] for i in range(0, len(reset_leaves), reset_size)]

    def _update_on_chip_data(self, key: int, offset: int) -> PROCESSED_DATA:
        """
        Given a data block and offset, compute the current.

        :param key: Key to some data of interest.
        :param offset: The offset of where the value should be written to.
        :return: The computed current leaf and updated leaf, and reset leaves if reset happens.
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

        # Base on the reset method, perform the desired operation.
        if self._reset_method == "prob":
            # Let reset happen with the desired probability.
            if random.random() <= self._reset_prob:
                # We store the leaves that should be updated to reset_leaves.
                reset_leaves = self._get_reset_leaves(key=key, data=data)
                # Now we set the new values for ic and gc.
                self._on_chip_storage[key] = Helper.binary_str_to_bytes(
                    f"{bin(gc + 1)[2:].zfill(self._gc_length)}{'0' * self._ic_length * self._num_ic}"
                )
                # In this case, we don't need cl, nl; just the reset leaves.
                return None, None, reset_leaves

            # When reset does not happen.
            else:
                # We check whether the overflow happens; if so, we throw an error.
                if ic + 1 >= pow(2, self._ic_length):
                    raise ValueError("Overflow happened under probabilistic resets.")
                # We update the counts being stored.
                self._on_chip_storage[key] = Helper.binary_str_to_bytes(
                    f"{data[:ic_start]}{bin(ic + 1)[2:].zfill(self._ic_length)}{data[ic_end:]}"
                )
                # Compute the leaves.
                cur_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic)
                new_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic + 1)
                # In this case, we return the leaves and reset leaves are None.
                return cur_leaf, new_leaf, None

        elif self._reset_method == "hard":
            # When overflow happens.
            if ic + 1 >= pow(2, self._ic_length):
                # We store the leaves that should be updated to reset_leaves.
                reset_leaves = self._get_reset_leaves(key=key, data=data)
                # Now we set the new values for ic and gc.
                self._on_chip_storage[key] = Helper.binary_str_to_bytes(
                    f"{bin(gc + 1)[2:].zfill(self._gc_length)}{'0' * self._ic_length * self._num_ic}"
                )
                # In this case, we don't need cl, nl; just the reset leaves.
                return None, None, reset_leaves

            # When overflow does not happen.
            else:
                # We update the counts being stored.
                self._on_chip_storage[key] = Helper.binary_str_to_bytes(
                    f"{data[:ic_start]}{bin(ic + 1)[2:].zfill(self._ic_length)}{data[ic_end:]}"
                )
                # Compute the leaves.
                cur_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic)
                new_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic + 1)
                # In this case, we return the leaves and reset leaves are None.
                return cur_leaf, new_leaf, None

        else:
            raise ValueError(f"Unrecognized reset method {self._reset_method}.")

    def _update_data_prob_reset(self, key: int, data: Data, offset: int) -> PROCESSED_DATA:
        """
        Given a data block and offset, compute the current.

        :param key: Key to some data of interest.
        :param data: A Data object.
        :param offset: The offset of where the value should be written to.
        :return: The computed current leaf and updated leaf, and reset leaves if reset happens.
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

        # When force_reset is set to True.
        if random.random() <= self._reset_prob:
            # We store the leaves that should be updated to reset_leaves.
            reset_leaves = self._get_reset_leaves(key=key, data=data.value)
            # Now we set the new values for ic and gc.
            data.value = Helper.binary_str_to_bytes(
                f"{bin(gc + 1)[2:].zfill(self._gc_length)}{'0' * self._ic_length * self._num_ic}"
            )
            # In this case, we don't need cl, nl; just the reset leaves.
            return None, None, reset_leaves

        # When reset does not happen.
        else:
            # We check whether the overflow happens; if so, we throw an error.
            if ic + 1 >= pow(2, self._ic_length):
                raise ValueError("Overflow happened under probabilistic resets.")
            # We update the counts being stored.
            data.value = Helper.binary_str_to_bytes(
                f"{data.value[:ic_start]}{bin(ic + 1)[2:].zfill(self._ic_length)}{data.value[ic_end:]}"
            )
            # Compute the leaves.
            cur_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic)
            new_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic + 1)
            # In this case, we return the leaves and reset leaves are None.
            return cur_leaf, new_leaf, None

    def _update_data_hard_reset(self, key: int, data: Data, offset: int) -> PROCESSED_DATA:
        """
        Given a data block and offset, compute the current.

        :param key: Key to some data of interest.
        :param data: A Data object.
        :param offset: The offset of where the value should be written to.
        :return: The computed current leaf and updated leaf, and reset leaves if reset happens.
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

        # When overflow happens.
        if ic + 1 >= pow(2, self._ic_length):
            # We store the leaves that should be updated to reset_leaves.
            reset_leaves = self._get_reset_leaves(key=key, data=data.value)
            # Now we set the new values for ic and gc.
            data.value = Helper.binary_str_to_bytes(
                f"{bin(gc + 1)[2:].zfill(self._gc_length)}{'0' * self._ic_length * self._num_ic}"
            )
            # In this case, we don't need cl, nl; just the reset leaves.
            return None, None, reset_leaves

        # When overflow does not happen.
        else:
            # We update the counts being stored.
            data.value = Helper.binary_str_to_bytes(
                f"{data.value[:ic_start]}{bin(ic + 1)[2:].zfill(self._ic_length)}{data.value[ic_end:]}"
            )
            # Compute the leaves.
            cur_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic)
            new_leaf = self._get_previous_leaf_from_prf(key=key * self._num_ic + offset, gc=gc, ic=ic + 1)
            # In this case, we return the leaves and reset leaves are None.
            return cur_leaf, new_leaf, None

    def _retrieve_pos_map_stash(self, key: int, offset: int, new_leaf: int, to_index: int) -> PROCESSED_DATA:
        """
        Given a key, retrieve the block from the stash and apply the operation to it.

        :param key: The key of the data block of interest.
        :param offset: The offset of where the value should be written to.
        :param to_index: Up to which index in the stash, we should be checking.
        :param new_leaf: If a new leaf value is provided, store the accessed data to that leaf.
        :return: The processed data tuple.
        """
        # Read all buckets in the path and add real data to stash.
        for data in self._stash[:to_index]:
            # If we find the data of interest, perform operation, otherwise skip over.
            if data.key == key:
                next_cur_leaf, next_new_leaf, reset_leaves = self._update_data_prob_reset(
                    key=key, data=data, offset=offset
                ) if self._reset_method == "prob" else self._update_data_hard_reset(
                    key=key, data=data, offset=offset
                )
                # This data block should be placed to where the new leaf is.
                data.leaf = new_leaf
                # We can just break the loop as we found the target.
                return next_cur_leaf, next_new_leaf, reset_leaves

        # If the key was never found, raise an error, since the stash is always searched after the path.
        raise KeyError(f"Key {key} not found.")

    def _retrieve_pos_map_block(self, key: int, offset: int, new_leaf: int, path: PathData) -> PROCESSED_DATA:
        """
        Given a key, retrieve the block and apply the operation to it.

        :param key: A key to a data block.
        :param path: PathData dict mapping storage index to bucket.
        :param offset: The offset of where the value should be written to.
        :param new_leaf: The leaf of where the block should be written to.
        :return: The processed data tuple.
        """
        # Assume that reset does not happen.
        next_cur_leaf, next_new_leaf, reset_leaves = None, None, None
        # Store the current stash length.
        to_index = len(self._stash)
        # Decrypt data blocks if we use encryption.
        path = self._decrypt_path_data(path=path)

        # Read all buckets in the path and add real data to stash.
        for bucket in path.values():
            for data in bucket:
                # If dummy data, we skip it.
                if data.key is None:
                    continue
                # If it's the data of interest, we both read and write it, and give it a new path.
                elif data.key == key:
                    next_cur_leaf, next_new_leaf, reset_leaves = self._update_data_prob_reset(
                        key=key, data=data, offset=offset
                    ) if self._reset_method == "prob" else self._update_data_hard_reset(
                        key=key, data=data, offset=offset
                    )
                    # This data block should be placed to where the new leaf is.
                    data.leaf = new_leaf
                # And all real data to the stash.
                self._stash.append(data)

        # Check if the stash overflows.
        if len(self._stash) > self._stash_size:
            raise MemoryError("Stash overflow!")

        # If both next cur leaf and reset leaves are still None, we go search the stash.
        if next_cur_leaf is None and reset_leaves is None:
            next_cur_leaf, next_new_leaf, reset_leaves = self._retrieve_pos_map_stash(
                key=key, offset=offset, new_leaf=new_leaf, to_index=to_index
            )

        # Return the desired values.
        return next_cur_leaf, next_new_leaf, reset_leaves

    def _get_leaf_from_pos_map(self, key: int) -> PROCESSED_DATA:
        """
        Provide a key to some data, iterate through all position map oram to find where it is stored.

        :param key: The key of the data block of interest.
        :return: Which path the data block is on and the new path it should be stored to.
        """
        # We get the position map keys.
        pos_map_keys = self._get_pos_map_keys(key=key)

        # Retrieve leaf from the on chip storage.
        cur_leaf, new_leaf, reset_leaves = self._update_on_chip_data(key=pos_map_keys[0][0], offset=pos_map_keys[0][1])

        # Declare useful variables.
        next_cur_leaf, next_new_leaf = None, None

        # We let the index start from -1, so that we read the on chip storage when it equals -1.
        for pos_map_index, (cur_key, cur_index) in enumerate(pos_map_keys[1:]):
            if reset_leaves is not None:
                # Iterate through all pairs of leaves.
                for cur_reset_leaves in reset_leaves:
                    ck_one, cl_one, nl_one, ck_two, cl_two, nl_two, leaves = (
                        self._unpack_reset_leaves(cur_reset_leaves)
                    )

                    # Read the leaves and get data.
                    self.client.add_read_path(label=f"{self._name}_pos_map_{pos_map_index}", leaves=leaves)
                    result = self.client.execute()
                    path = result.results[f"{self._name}_pos_map_{pos_map_index}"]

                    # Depends on which one is one we want next, compute next leaves.
                    if ck_one == cur_key:
                        next_cur_leaf, next_new_leaf, reset_leaves = (
                            self._pos_maps[pos_map_index]._retrieve_pos_map_block(
                                key=ck_one, offset=cur_index, new_leaf=nl_one, path=path
                            )
                        )
                        self._pos_maps[pos_map_index]._update_stash_leaf(key=ck_two, new_leaf=nl_two)
                    elif ck_two == cur_key:
                        next_cur_leaf, next_new_leaf, reset_leaves = (
                            self._pos_maps[pos_map_index]._retrieve_pos_map_block(
                                key=ck_two, offset=cur_index, new_leaf=nl_two, path=path
                            )
                        )
                        self._pos_maps[pos_map_index]._update_stash_leaf(key=ck_one, new_leaf=nl_one)
                    # Otherwise, we just add them to the stash and update leaves. (If key is not None.)
                    else:
                        self._pos_maps[pos_map_index]._update_block_leaves(
                            key_one=ck_one, key_two=ck_two, n_leaf_one=nl_one, n_leaf_two=nl_two, path=path
                        )
                    # Evict the stash to current leaves.
                    evicted_path = self._pos_maps[pos_map_index]._evict_stash(leaves=leaves)

                    # Write to the server.
                    self.client.add_write_path(label=f"{self._name}_pos_map_{pos_map_index}", data=evicted_path)
                    self.client.execute()

            # Otherwise proceeds as normal.
            else:
                # Interact with server to get a path.
                self.client.add_read_path(label=f"{self._name}_pos_map_{pos_map_index}", leaves=[cur_leaf])
                result = self.client.execute()
                path = result.results[f"{self._name}_pos_map_{pos_map_index}"]

                # Find what leaves for the next iteration.
                next_cur_leaf, next_new_leaf, reset_leaves = (
                    self._pos_maps[pos_map_index]._retrieve_pos_map_block(
                        key=cur_key, offset=cur_index, new_leaf=new_leaf, path=path
                    )
                )

                # Evict stash to current leaf.
                evicted_path = self._pos_maps[pos_map_index]._evict_stash(leaves=[cur_leaf])

                # Interact with server to store a path.
                self.client.add_write_path(label=f"{self._name}_pos_map_{pos_map_index}", data=evicted_path)
                self.client.execute()

            # Update the next values to current.
            cur_leaf, new_leaf = next_cur_leaf, next_new_leaf

        return cur_leaf, new_leaf, reset_leaves

    def operate_on_key(self, key: int, value: Any = UNSET) -> Any:
        """
        Perform operation on a given key. Always returns the current value.
        If value is provided (not UNSET), writes the new value.

        :param key: The key of the data block of interest.
        :param value: If provided (not UNSET), write this value to the data block.
        :return: The current value of the data block (before write if writing).
        """
        # Find which path the data of interest lies on.
        cur_leaf, next_leaf, reset_leaves = self._get_leaf_from_pos_map(key=key)

        # Initialize read_value to None; will be set when we find our key.
        read_value = None

        # If reset happens at the last pos oram.
        if reset_leaves:
            # Iterate through all pairs of leaves.
            for cur_reset_leaves in reset_leaves:
                ck_one, cl_one, nl_one, ck_two, cl_two, nl_two, leaves = (
                    self._unpack_reset_leaves(cur_reset_leaves)
                )

                # Read the leaves and get data.
                self.client.add_read_path(label=self._name, leaves=leaves)
                result = self.client.execute()
                path = result.results[self._name]

                # Depends on which one is one we want next, compute next leaves.
                if ck_one == key:
                    read_value = self._retrieve_data_block(
                        key=ck_one, value=value, new_leaf=nl_one, path=path
                    )
                    # Perform reset for the other value.
                    self._update_stash_leaf(key=ck_two, new_leaf=nl_two)

                elif ck_two == key:
                    read_value = self._retrieve_data_block(
                        key=ck_two, value=value, new_leaf=nl_two, path=path
                    )
                    # Perform reset for the other value.
                    self._update_stash_leaf(key=ck_one, new_leaf=nl_one)

                # Otherwise, we just add them to the stash and update leaves. (If key is not None.)
                else:
                    self._update_block_leaves(
                        key_one=ck_one, key_two=ck_two, n_leaf_one=nl_one, n_leaf_two=nl_two, path=path
                    )

                # Evict the stash to current leaves.
                evicted_path = self._evict_stash(leaves=leaves)

                # Interact with the server to store the path.
                self.client.add_write_path(label=self._name, data=evicted_path)
                self.client.execute()

            return read_value
        else:
            # Interact with server to get a path.
            self.client.add_read_path(label=self._name, leaves=[cur_leaf])
            result = self.client.execute()
            path = result.results[self._name]

            # Generate a new leaf and read value from the path and show it.
            read_value = self._retrieve_data_block(key=key, value=value, new_leaf=next_leaf, path=path)

            # Perform an eviction and get a new path.
            evicted_path = self._evict_stash(leaves=[cur_leaf])

            # Interact with the server to store the path.
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
        # Find which path the data of interest lies on.
        cur_leaf, next_leaf, reset_leaves = self._get_leaf_from_pos_map(key=key)

        # If reset happens at the last pos oram.
        if reset_leaves:
            # Iterate through all pairs of leaves.
            for index, cur_reset_leaves in enumerate(reset_leaves):
                ck_one, cl_one, nl_one, ck_two, cl_two, nl_two, leaves = (
                    self._unpack_reset_leaves(cur_reset_leaves)
                )

                # Read the leaves and get data.
                self.client.add_read_path(label=self._name, leaves=leaves)
                result = self.client.execute()
                path = result.results[self._name]

                # Depends on which one is one we want next, compute next leaves.
                if ck_one == key:
                    read_value = self._retrieve_data_block(
                        key=ck_one, value=value, new_leaf=nl_one, path=path
                    )
                    # Perform reset for the other value.
                    self._update_stash_leaf(key=ck_two, new_leaf=nl_two)

                    # Temporarily save the leaves and stash for future eviction.
                    self._tmp_leaves = leaves

                    # If there are more values to reset, we temporarily store them.
                    if index < len(reset_leaves) - 1:
                        self._tmp_reset_leaves = reset_leaves[index + 1:]

                    # Stop the reset process.
                    return read_value

                elif ck_two == key:
                    read_value = self._retrieve_data_block(
                        key=ck_two, value=value, new_leaf=nl_two, path=path
                    )
                    # Perform reset for the other value.
                    self._update_stash_leaf(key=ck_one, new_leaf=nl_one)

                    # Temporarily save the leaves and stash for future eviction.
                    self._tmp_leaves = leaves

                    # If there are more values to reset, we temporarily store them.
                    if index < len(reset_leaves) - 1:
                        self._tmp_reset_leaves = reset_leaves[index + 1:]

                    # Stop the reset process.
                    return read_value

                # Otherwise, we just add them to the stash and update leaves. (If key is not None.)
                else:
                    self._update_block_leaves(
                        key_one=ck_one, key_two=ck_two, n_leaf_one=nl_one, n_leaf_two=nl_two, path=path
                    )

                    # In this case, we only evict data other than the one we are interested in.
                    evicted_path = self._evict_stash(leaves=leaves)

                    # Interact with the server to store the path.
                    self.client.add_write_path(label=self._name, data=evicted_path)
                    self.client.execute()

            # If we get here, the key was not found in any of the reset leaves (should not happen).
            raise KeyError(f"Key {key} not found in reset leaves.")

        else:
            # Interact with server to get a path.
            self.client.add_read_path(label=self._name, leaves=[cur_leaf])
            result = self.client.execute()
            path = result.results[self._name]

            # Generate a new leaf and read value from the path and show it.
            read_value = self._retrieve_data_block(key=key, value=value, new_leaf=next_leaf, path=path)

            # Temporarily save the leaves for future eviction.
            self._tmp_leaf = cur_leaf

        return read_value

    def eviction_with_update_stash(self, key: int, value: Any) -> None:
        """Update a data block stored in the stash and then perform eviction.

        :param key: The key of the data block of interest.
        :param value: The value to update the data block of interest.
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

        # Perform an eviction and get a new path depending on which temp leaf/leaves is not None.
        if self._tmp_leaf is not None:
            # Perform an eviction and get a new path.
            evicted_path = self._evict_stash(leaves=[self._tmp_leaf])
            # Interact with server to store a path.
            self.client.add_write_path(label=self._name, data=evicted_path)
            self.client.execute()
            # Set temporary leaf to None.
            self._tmp_leaf = None
        else:
            # Perform an eviction and get a new path.
            evicted_path = self._evict_stash(leaves=self._tmp_leaves)
            # Interact with server to store a path.
            self.client.add_write_path(label=self._name, data=evicted_path)
            self.client.execute()
            # Set temporary leaves to None.
            self._tmp_leaves = None
            # Continue reset if there are values left.
            if self._tmp_reset_leaves is not None:
                for cur_reset_leaves in self._tmp_reset_leaves:
                    ck_one, cl_one, nl_one, ck_two, cl_two, nl_two, leaves = (
                        self._unpack_reset_leaves(cur_reset_leaves)
                    )

                    # Read the leaves and get data.
                    self.client.add_read_path(label=self._name, leaves=leaves)
                    result = self.client.execute()
                    path = result.results[self._name]

                    self._update_block_leaves(
                        key_one=ck_one, key_two=ck_two, n_leaf_one=nl_one, n_leaf_two=nl_two, path=path
                    )

                    # In this case, we only evict data other than the one we are interested in.
                    evicted_path = self._evict_stash(leaves=leaves)

                    # Interact with the server to store the path.
                    self.client.add_write_path(label=self._name, data=evicted_path)
                    self.client.execute()

            # Set temporary reset leaves to None.
            self._tmp_reset_leaves = None
