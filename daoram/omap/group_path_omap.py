"""Group-by-hash OMAP that maps group members to paths via PRF and batch-reads those paths.

Design notes:
- On init_server_storage(data): we partition input KV into `num_groups` using Helper.hash_data_to_map.
- We create a single BinaryTree storage that holds all blocks. Each KV is assigned a leaf via a PRF
  (with a small rehash loop if the chosen path is full during initialization).
- The client keeps a local mapping `group_map` from group index to the list of keys in that group so
  we can deterministically compute the list of leaves to batch-read for a group in one interaction.

This class provides `init_server_storage`, `search`, and `insert` (insert updates or adds a KV) with
batch reads/writes using `InteractServer.read_query` / `write_query` where `leaf` may be a list.
"""
import math

import random
from typing import Any, Dict, List, Optional, Tuple

from daoram.dependency import BinaryTree, Data, Helper, InteractServer, Prf, Aes


class GroupPathOmap:
    def __init__(self,
                 num_groups: int,
                 key_size: int,
                 data_size: int,
                 client: InteractServer,
                 bucket_size: int = 4,
                 stash_scale: int = 100,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True,
                 storage_label: str = "omap_group"):
        """Initialize the group-path OMAP.

        :param num_groups: number of groups to partition keys into, it can be only the exponent of 2, e.g., pow(2, 20)
        :param key_size: bytes for random dummy keys (used if encryption/padding relies on sizes).
        :param data_size: bytes for random dummy data used for padding/encryption sizing.
        :param client: InteractServer to talk to server.
        :param bucket_size: bucket size for the underlying binary tree.
        :param aes_key: optional AES key for encryption.
        :param num_key_bytes: AES block/key byte length.
        :param use_encryption: whether to encrypt stored blocks.
        :param storage_label: label used when calling client.read_query / write_query.
        """
        self._num_groups = num_groups
        self._operation_num = num_groups
        self._num_data = 0
        self._stash_scale = stash_scale
        self._key_size = key_size
        self._data_size = data_size
        self._bucket_size = bucket_size
        self._client = client
        self._use_encryption = use_encryption
        self._aes_key = aes_key
        self._num_key_bytes = num_key_bytes
        self._cipher = Aes(key=aes_key, key_byte_length=num_key_bytes) if use_encryption else None

        # PRFs: one for group hashing, one for leaf mapping
        self._group_prf = Prf()
        self._leaf_prf = Prf()
        self._stash: List[Data] = []

        # First compute how many are in the buckets, according to https://eprint.iacr.org/2021/1280.
        self.upper_bound = math.ceil(
            math.e ** (Helper.lambert_w(math.e ** -1 * (math.log(self._num_groups, 2) + 128 - 1)).real + 1)
        )

        # Storage label on server
        self._storage_label = storage_label

        # Local mapping: group index -> list of keys (kept client-side)
        self._group_map: Dict[int, List[Any]] = {}
        self._group_seed_map: Dict[int, List[Any]] = {}

        # Underlying BinaryTree storage (created at init_server_storage)
        self._tree: Optional[BinaryTree] = None

    def _compute_max_block_size(self) -> int:
        """Compute a conservative block size for storage padding similar to other OMAPs."""
        # Create a sample Data and measure its dump length.
        sample = Data(key=b"k" * self._key_size, leaf=0, value=b"v" * self._data_size)
        return len(sample.dump())

    def _encrypt_buckets(self, buckets: List[List[Data]]) -> List[List[bytes]]:
        """Encrypt and pad buckets for writing to the server."""
        if not self._use_encryption:
            return buckets  # type: ignore

        max_block = self._compute_max_block_size()

        enc_buckets: List[List[bytes]] = []
        for bucket in buckets:
            # Ensure each element's value is pickled bytes when necessary; Data.dump handles it
            enc_bucket = [self._cipher.enc(plaintext=Helper.pad_pickle(data=data.dump(), length=max_block))
                          for data in bucket]
            # pad with dummy encrypted blocks to bucket_size
            dummy_needed = self._bucket_size - len(enc_bucket)
            if dummy_needed > 0:
                enc_bucket.extend([self._cipher.enc(plaintext=Helper.pad_pickle(data=Data().dump(), length=max_block))
                                   for _ in range(dummy_needed)])
            enc_buckets.append(enc_bucket)

        return enc_buckets

    def _decrypt_buckets(self, buckets: List[List[bytes]]) -> List[List[Data]]:
        """Decrypt buckets read from server into Data objects (dropping dummies)."""
        if not self._use_encryption:
            return buckets  # type: ignore

        dec_buckets: List[List[Data]] = []
        for bucket in buckets:
            dec_bucket: List[Data] = []
            for blob in bucket:
                # decrypt and unpad then unpickle via Helper and Data.from_pickle
                dec = Data.from_pickle(Helper.unpad_pickle(data=self._cipher.dec(ciphertext=blob)))
                if dec.key is not None:
                    dec_bucket.append(dec)
            dec_buckets.append(dec_bucket)

        return dec_buckets

    def init_server_storage(self, data: Optional[List[Tuple[Any, Any]]] = None) -> None:
        """Partition input KV into groups, place all blocks into one BinaryTree storage, and upload it.

        Client stores the mapping group->keys locally to allow deterministic batch reads later.
        """
        if data is None:
            data = []

        # Partition into groups
        self._num_data = len(data)
        data_map = Helper.hash_data_to_map(prf=self._group_prf, data=data, map_size=self._num_groups)
        # Save group map locally (store only keys)
        self._group_map = {i: [kv[0] for kv in data_map[i]] for i in range(self._num_groups)}

        # each group corresponds to two integers, one implies the number of items within the group, the other records the access number of this group
        self._group_seed_map = {i: [0, len(data_map[i])] for i in range(self._num_groups)}

        # Initialize the BinaryTree storage
        max_block = self._compute_max_block_size()
        tree = BinaryTree(num_data=self._num_groups, bucket_size=self._bucket_size, data_size=max_block,
                          filename=None, enc_key_size=self._num_key_bytes if self._use_encryption else None)

        # Fill the tree with each KV mapped to a PRF-determined leaf. Use a small rehash loop upon collision.
        for group_index in range(self._num_groups):
            tmp = 0
            for kv in data_map[group_index]:
                key, value = kv
                seed = group_index.to_bytes(4, byteorder="big") + self._group_seed_map[group_index][0].to_bytes(2,
                                                                                                                byteorder="big") + tmp.to_bytes(
                    2, byteorder="big")
                tmp += 1
                leaf = Helper.hash_data_to_leaf(prf=self._leaf_prf, data=seed, map_size=self._num_groups)

                data_block = Data(key=key, leaf=leaf, value=value)
                inserted = tree.fill_data_to_storage_leaf(data=data_block)

                if not inserted:
                    self._stash.append(data_block)
            if len(data_map[group_index]) > self.upper_bound:
                raise MemoryError(f"Group {group_index} has more items ({len(data_map[group_index])}) than upper bound "
                                  f"({self.upper_bound}); increase the bound.")

        if len(self._stash) > self._stash_scale * int(math.log2(self._num_groups)):
            raise MemoryError(f"Stash size {len(self._stash)} exceeds allowed limit "
                              f"{self._stash_scale * int(math.log2(self._num_groups))}; increase the limit.")

        # Encrypt storage if needed
        if self._use_encryption:
            tree.storage.encrypt(aes=self._cipher)

        # Save tree locally and upload to server under configured label
        self._tree = tree
        self._client.init(storage={self._storage_label: tree})

        del self._group_map

    def _collect_group_leaves_retrieve(self, group_index: int) -> List[int]:
        """Deterministically compute the leaf indices for all keys in a group (unique, in stable order)."""
        group_seed = self._group_seed_map[group_index]

        # Compute the leaves for all keys in the group
        leaves: List[int] = []
        for item_index in range(self.upper_bound):
            seed = group_index.to_bytes(4, byteorder="big") + group_seed[0].to_bytes(2,byteorder="big") + item_index.to_bytes(2, byteorder="big")
            leaf = Helper.hash_data_to_leaf(prf=self._leaf_prf, data=seed, map_size=self._num_groups)
            leaves.append(leaf)

        # Remove duplicates while preserving order
        seen = set()
        uniq_leaves = []
        for l in leaves:
            if l not in seen:
                seen.add(l)
                uniq_leaves.append(l)
            else:
                while True:
                    t = random.randint(0, self._num_groups - 1)
                    if t not in seen:
                        seen.add(t)
                        uniq_leaves.append(t)
                        break

        return uniq_leaves

    def _collect_group_leaves_generate(self, group_index: int) -> List[int]:
        """Deterministically compute the leaf indices for all keys in a group (in stable order)."""
        group_seed = self._group_seed_map[group_index]

        # Compute the leaves for all keys in the group
        leaves: List[int] = []
        for item_index in range(self.upper_bound):
            seed = group_index.to_bytes(4, byteorder="big") + group_seed[0].to_bytes(2, byteorder="big") + item_index.to_bytes(2, byteorder="big")
            leaf = Helper.hash_data_to_leaf(prf=self._leaf_prf, data=seed, map_size=self._num_groups)
            leaves.append(leaf)

        return leaves

    def search(self, key: Any) -> Any:
        """Given a key, batch-download its group's paths and return the value for the key.

        This performs a single client.read_query with a list of leaves corresponding to the group's members.
        """
        group_index = Helper.hash_data_to_leaf(prf=self._group_prf, data=key, map_size=self._num_groups)

        retrieve_leaves = self._collect_group_leaves_retrieve(group_index=group_index)

        # Batch read
        raw_paths = self._client.read_query(label=self._storage_label, leaf=retrieve_leaves)
        # Decrypt/path -> list of buckets of Data
        paths = self._decrypt_buckets(buckets=raw_paths)

        # Flatten and find the desired key
        local_data = []
        value = None
        for data in self._stash:
            if Helper.hash_data_to_leaf(prf=self._group_prf, data=data.key, map_size=self._num_groups) == group_index:
                local_data.append(data)
                self._stash.remove(data)
            if data.key == key:
                value = data.value

        for bucket in paths:
            for data in bucket:
                if Helper.hash_data_to_leaf(prf=self._group_prf, data=data.key, map_size=self._num_groups) == group_index:
                    local_data.append(data)
                else:
                    self._stash.append(data)

                if data.key == key:
                    value = data.value

        self._group_seed_map[group_index][0] += 1

        generated_leaves = self._collect_group_leaves_generate(group_index=group_index)
        for i in range(len(local_data)):
            local_data[i].leaf = generated_leaves[i]
            self._stash.append(local_data[i])

        # evict stash to write paths back
        paths = self.evict_paths(retrieve_leaves=retrieve_leaves)
        self._client.write_query(label=self._storage_label, leaf=retrieve_leaves,
                                 data=self._encrypt_buckets(buckets=paths))

        return value

    def search_group(self, index: Any) -> Any:
        """Given a key, batch-download its group's paths and return the value for the key.

        This performs a single client.read_query with a list of leaves corresponding to the group's members.
        """
        group_index = index
        retrieve_leaves = self._collect_group_leaves_retrieve(group_index=group_index)


        # Batch read
        raw_paths = self._client.read_query(label=self._storage_label, leaf=retrieve_leaves)
        # Decrypt/path -> list of buckets of Data
        paths = self._decrypt_buckets(buckets=raw_paths)

        # Flatten and find the desired key
        local = []

        for data in self._stash:
            if Helper.hash_data_to_leaf(prf=self._group_prf, data=data.key, map_size=self._num_groups) == group_index:
                local.append(data)
                self._stash.remove(data)

        for bucket in paths:
            for data in bucket:
                if Helper.hash_data_to_leaf(prf=self._group_prf, data=data.key,
                                            map_size=self._num_groups) == group_index:
                    local.append(data)
                else:
                    self._stash.append(data)

        self._group_seed_map[group_index][0] += 1

        generated_leaves = self._collect_group_leaves_generate(group_index=group_index)
        for i in range(len(local)):
            local[i].leaf = generated_leaves[i]
            self._stash.append(local[i])

        # evict stash to write paths back
        paths = self.evict_paths(retrieve_leaves=retrieve_leaves)
        self._client.write_query(label=self._storage_label, leaf=retrieve_leaves,
                                 data=self._encrypt_buckets(buckets=paths))

        value_list = []
        for data in local:
            value_list.append((data.key, data.value))
        return value_list

    def insert(self, key: Any, value: Any) -> None:
        """Insert or update a key by batch reading its group, updating, and batch-writing modified paths back.

        This keeps the single-round batch read/write property: one read_query and one write_query.
        """
        group_index = Helper.hash_data_to_leaf(prf=self._group_prf, data=key, map_size=self._num_groups)
        group_seed = self._group_seed_map[group_index]
        leaves = [random.randint(0, self._num_groups - 1)]

        raw_paths = self._client.read_query(label=self._storage_label, leaf=leaves)
        paths = self._decrypt_buckets(buckets=raw_paths)

        for bucket in paths:
            for block in bucket:
                self._stash.append(block)

        seed = group_index.to_bytes(4, byteorder="big") + group_seed[0].to_bytes(2,
                                                                                 byteorder="big") + group_seed[1].to_bytes(
            2, byteorder="big")
        data = Data(key=key,
                    leaf=Helper.hash_data_to_leaf(prf=self._leaf_prf, data=seed,
                                                  map_size=self._num_groups), value=value)
        self._group_seed_map[group_index][1] += 1
        self._stash.append(data)

        if self._group_seed_map[group_index][1] > self.upper_bound:
            raise MemoryError(
                f"Group {group_index} has more items ({self._group_seed_map[group_index][1]}) than upper bound "
                f"({self.upper_bound}); increase the bound.")

        self._num_data += 1

        # evict stash to write paths back
        paths = self.evict_paths(retrieve_leaves=leaves)

        # Encrypt and write back the modified paths (the server expects buckets in the same order)
        enc = self._encrypt_buckets(buckets=paths)
        self._client.write_query(label=self._storage_label, leaf=leaves, data=enc)

    def evict_paths(self, retrieve_leaves: List[int]) -> List[List[Data]]:
        """
        Evict data blocks in the stash to multiple paths (corresponding to `retrieve_leaves`).

        This prepares the list of buckets (Data objects) to be written back to the server for the
        collection of leaves passed in. It mirrors the behavior of other multi-path eviction
        helpers in the repository (for example `_evict_stash_to_mul` in DA-ORAM).

        :param retrieve_leaves: list of leaf labels to evict to.
        :return: A list of buckets (list of lists of `Data`) ordered as the server expects for
                 multi-path writes (bottom-up ordering consistent with BinaryTree.get_mul_path_dict).
        """
        # Temporary stash for items that couldn't be placed back into the provided paths.
        temp_stash: List[Data] = []

        # Create a dict keyed by storage indices that covers all nodes on the multiple paths.
        path_dict = BinaryTree.get_mul_path_dict(level=self._tree.level, indices=retrieve_leaves)

        # Try to place every real data item from stash into one of the provided paths.
        for data in self._stash:
            inserted = BinaryTree.fill_data_to_mul_path(
                data=data,
                path=path_dict,
                leaves=retrieve_leaves,
                level=self._tree.level,
                bucket_size=self._bucket_size,
            )
            if not inserted:
                temp_stash.append(data)

        # Convert dict to list following the dict key order (which matches get_mul_path_indices ordering).
        path = [path_dict[key] for key in path_dict.keys()]

        # Update the stash to only those elements that could not be placed.
        self._stash = temp_stash

        # Return raw Data buckets; caller will encrypt if needed.
        return path
