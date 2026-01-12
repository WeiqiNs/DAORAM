"""Group-by-hash OMAP that maps group members to paths via PRF and batch-reads those paths.

Design notes:
- On init_server_storage(data): we partition input KV into `num_groups` using Helper.hash_data_to_map.
- We create a single BinaryTree storage that holds all blocks. Each KV is assigned a leaf via a PRF
  (with a small rehash loop if the chosen path is full during initialization).
- The client keeps a local mapping `group_seed_map` from group index to metadata so
  we can deterministically compute the list of leaves to batch-read for a group in one interaction.

This class provides `init_server_storage`, `search`, `search_group`, and `insert` with
batch reads/writes using the InteractServer batching API.
"""
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from daoram.dependency import BinaryTree, Blake2Prf, Data, Encryptor, Helper, InteractServer
from daoram.oram import TreeBaseOram, MulPathOram


class GroupOmap:
    def __init__(self,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 oram: TreeBaseOram,
                 client: InteractServer,
                 name: str = "group_omap",
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 encryptor: Encryptor = None):
        """Initialize the group-path OMAP.

        :param num_data: number of groups to partition keys into, should be a power of 2.
        :param key_size: bytes for random dummy keys (used for padding/encryption sizing).
        :param data_size: bytes for random dummy data used for padding/encryption sizing.
        :param client: InteractServer to talk to server.
        :param name: The name of the protocol, used as storage label on the server.
        :param bucket_size: bucket size for the underlying binary tree.
        :param stash_scale: scaling factor for stash size limit.
        :param encryptor: The encryptor to use for encryption.
        """
        self._oram = oram
        self._name = name
        self._num_groups = num_data
        self._operation_num = num_data
        self._num_data = 0
        self._stash_scale = stash_scale
        self._key_size = key_size
        self._data_size = data_size
        self._bucket_size = bucket_size
        self._client = client
        self._encryptor = encryptor

        # PRFs: one for group hashing, one for leaf mapping
        self._leaf_prf = Blake2Prf()
        self._group_prf = Blake2Prf()
        self._stash: List[Data] = []

        # First compute how many are in the buckets, according to https://eprint.iacr.org/2021/1280.
        self.upper_bound = math.ceil(
            math.e ** (Helper.lambert_w(math.e ** -1 * (math.log(self._num_groups, 2) + 128 - 1)).real + 1)
        )

        # Storage label on server
        self._name = name

        # Local mapping: group index -> [access_count, item_count]
        self._group_seed_map: Dict[int, List[int]] = {}

        # Underlying BinaryTree storage (created at init_server_storage)
        self._tree: Optional[BinaryTree] = None

    def _compute_max_block_size(self) -> int:
        """Compute a conservative block size for storage padding."""
        sample = Data(key=b"k" * self._key_size, leaf=0, value=b"v" * self._data_size)
        return len(sample.dump()) + Helper.LENGTH_HEADER_SIZE

    def _encrypt_path_data(self, path: Dict[int, List[Data]]) -> Dict[int, List[bytes]]:
        """Encrypt PathData for writing to the server."""
        if not self._encryptor:
            return path  # type: ignore

        max_block = self._compute_max_block_size()

        enc_path: Dict[int, List[bytes]] = {}
        for idx, bucket in path.items():
            enc_bucket = [
                self._encryptor.enc(plaintext=data.dump_pad(max_block))
                for data in bucket
            ]
            # Pad with dummy encrypted blocks to bucket_size
            dummy_needed = self._bucket_size - len(enc_bucket)
            if dummy_needed > 0:
                enc_bucket.extend([
                    self._encryptor.enc(plaintext=Data().dump_pad(max_block))
                    for _ in range(dummy_needed)
                ])
            enc_path[idx] = enc_bucket

        return enc_path

    def _decrypt_path_data(self, path: Dict[int, List[bytes]]) -> Dict[int, List[Data]]:
        """Decrypt PathData read from server into Data objects (dropping dummies)."""
        if not self._encryptor:
            return path  # type: ignore

        dec_path: Dict[int, List[Data]] = {}
        for idx, bucket in path.items():
            dec_bucket: List[Data] = []
            for blob in bucket:
                dec = Data.load_unpad(self._encryptor.dec(ciphertext=blob))
                if dec.key is not None:
                    dec_bucket.append(dec)
            dec_path[idx] = dec_bucket

        return dec_path

    def init_server_storage(self, data: Optional[List[Tuple[Any, Any]]] = None) -> None:
        """Partition input KV into groups, place all blocks into one BinaryTree storage, and upload it.

        Client stores the mapping group->metadata locally to allow deterministic batch reads later.
        """
        if data is None:
            data = []

        # Partition into groups
        self._num_data = len(data)
        data_map = Helper.hash_data_to_map(prf=self._group_prf, data=data, map_size=self._num_groups)

        # Each group corresponds to two integers: access_count and item_count
        self._group_seed_map = {i: [0, len(data_map[i])] for i in range(self._num_groups)}

        # Initialize the BinaryTree storage
        max_block = self._compute_max_block_size()
        tree = BinaryTree(
            num_data=self._num_groups,
            bucket_size=self._bucket_size,
            data_size=max_block,
            filename=None,
            encryption=True if self._encryptor else False
        )

        # Fill the tree with each KV mapped to a PRF-determined leaf.
        for group_index in range(self._num_groups):
            tmp = 0
            for kv in data_map[group_index]:
                key, value = kv
                seed = (group_index.to_bytes(4, byteorder="big") +
                        self._group_seed_map[group_index][0].to_bytes(2, byteorder="big") +
                        tmp.to_bytes(2, byteorder="big"))
                tmp += 1
                leaf = self._leaf_prf.digest_mod_n(message=seed, mod=self._num_groups)

                data_block = Data(key=key, leaf=leaf, value=value)
                inserted = tree.fill_data_to_storage_leaf(data=data_block)

                if not inserted:
                    self._stash.append(data_block)

            if len(data_map[group_index]) > self.upper_bound:
                raise MemoryError(
                    f"Group {group_index} has more items ({len(data_map[group_index])}) than upper bound "
                    f"({self.upper_bound}); increase the bound."
                )

        if len(self._stash) > self._stash_scale * int(math.log2(self._num_groups)):
            raise MemoryError(
                f"Stash size {len(self._stash)} exceeds allowed limit "
                f"{self._stash_scale * int(math.log2(self._num_groups))}; increase the limit."
            )

        # Encrypt storage if needed
        if self._encryptor:
            tree.storage.encrypt(encryptor=self._encryptor)

        # Save tree locally and upload to server
        self._tree = tree
        self._client.init_storage(storage={self._name: tree})

    def _collect_group_leaves_retrieve(self, group_index: int) -> List[int]:
        """Deterministically compute the leaf indices for all keys in a group (unique, in stable order)."""
        group_seed = self._group_seed_map[group_index]

        # Compute the leaves for all keys in the group
        leaves: List[int] = []
        for item_index in range(self.upper_bound):
            seed = (group_index.to_bytes(4, byteorder="big") +
                    group_seed[0].to_bytes(2, byteorder="big") +
                    item_index.to_bytes(2, byteorder="big"))
            leaf = self._leaf_prf.digest_mod_n(message=seed, mod=self._num_groups)
            leaves.append(leaf)

        # Remove duplicates while preserving order, adding random leaves for duplicates
        seen = set()
        uniq_leaves = []
        for leaf in leaves:
            if leaf not in seen:
                seen.add(leaf)
                uniq_leaves.append(leaf)
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
            seed = (group_index.to_bytes(4, byteorder="big") +
                    group_seed[0].to_bytes(2, byteorder="big") +
                    item_index.to_bytes(2, byteorder="big"))
            leaf = self._leaf_prf.digest_mod_n(message=seed, mod=self._num_groups)
            leaves.append(leaf)

        return leaves

    def search(self, key: Any) -> Any:
        """Given a key, batch-download its group's paths and return the value for the key."""
        group_index = self._group_prf.digest_mod_n(
            message=key if isinstance(key, bytes) else str(key).encode(),
            mod=self._num_groups
        )

        retrieve_leaves = self._collect_group_leaves_retrieve(group_index=group_index)

        # Batch read using the current API
        self._client.add_read_path(label=self._name, leaves=retrieve_leaves)
        result = self._client.execute()
        raw_paths = result.results[self._name]

        # Decrypt paths
        paths = self._decrypt_path_data(path=raw_paths)

        # Flatten and find the desired key
        local_data = []
        value = None

        # Check stash for items in this group
        for data in list(self._stash):
            data_group = self._group_prf.digest_mod_n(
                message=data.key if isinstance(data.key, bytes) else str(data.key).encode(),
                mod=self._num_groups
            )
            if data_group == group_index:
                local_data.append(data)
                self._stash.remove(data)
            if data.key == key:
                value = data.value

        # Check paths for items
        for bucket in paths.values():
            for data in bucket:
                data_group = self._group_prf.digest_mod_n(
                    message=data.key if isinstance(data.key, bytes) else str(data.key).encode(),
                    mod=self._num_groups
                )
                if data_group == group_index:
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

        # Evict stash to write paths back
        evicted_path = self._evict_paths(retrieve_leaves=retrieve_leaves)
        encrypted_path = self._encrypt_path_data(path=evicted_path)

        self._client.add_write_path(label=self._name, data=encrypted_path)
        self._client.execute()

        return value

    def search_group(self, index: int) -> List[Tuple[Any, Any]]:
        """Given a group index, batch-download the group's paths and return all key-value pairs."""
        group_index = index
        retrieve_leaves = self._collect_group_leaves_retrieve(group_index=group_index)

        # Batch read
        self._client.add_read_path(label=self._name, leaves=retrieve_leaves)
        result = self._client.execute()
        raw_paths = result.results[self._name]

        # Decrypt paths
        paths = self._decrypt_path_data(path=raw_paths)

        # Flatten and collect items in this group
        local = []

        for data in list(self._stash):
            data_group = self._group_prf.digest_mod_n(
                message=data.key if isinstance(data.key, bytes) else str(data.key).encode(),
                mod=self._num_groups
            )
            if data_group == group_index:
                local.append(data)
                self._stash.remove(data)

        for bucket in paths.values():
            for data in bucket:
                data_group = self._group_prf.digest_mod_n(
                    message=data.key if isinstance(data.key, bytes) else str(data.key).encode(),
                    mod=self._num_groups
                )
                if data_group == group_index:
                    local.append(data)
                else:
                    self._stash.append(data)

        self._group_seed_map[group_index][0] += 1

        generated_leaves = self._collect_group_leaves_generate(group_index=group_index)
        for i in range(len(local)):
            local[i].leaf = generated_leaves[i]
            self._stash.append(local[i])

        # Evict stash to write paths back
        evicted_path = self._evict_paths(retrieve_leaves=retrieve_leaves)
        encrypted_path = self._encrypt_path_data(path=evicted_path)

        self._client.add_write_path(label=self._name, data=encrypted_path)
        self._client.execute()

        return [(data.key, data.value) for data in local]

    def insert(self, key: Any, value: Any) -> None:
        """Insert or update a key by batch reading a random path, updating, and writing back."""
        group_index = self._group_prf.digest_mod_n(
            message=key if isinstance(key, bytes) else str(key).encode(),
            mod=self._num_groups
        )
        group_seed = self._group_seed_map[group_index]
        leaves = [random.randint(0, self._num_groups - 1)]

        self._client.add_read_path(label=self._name, leaves=leaves)
        result = self._client.execute()
        raw_paths = result.results[self._name]

        paths = self._decrypt_path_data(path=raw_paths)

        for bucket in paths.values():
            for block in bucket:
                self._stash.append(block)

        seed = (group_index.to_bytes(4, byteorder="big") +
                group_seed[0].to_bytes(2, byteorder="big") +
                group_seed[1].to_bytes(2, byteorder="big"))
        data = Data(
            key=key,
            leaf=self._leaf_prf.digest_mod_n(message=seed, mod=self._num_groups),
            value=value
        )
        self._group_seed_map[group_index][1] += 1
        self._stash.append(data)

        if self._group_seed_map[group_index][1] > self.upper_bound:
            raise MemoryError(
                f"Group {group_index} has more items ({self._group_seed_map[group_index][1]}) than upper bound "
                f"({self.upper_bound}); increase the bound."
            )

        self._num_data += 1

        # Evict stash to write paths back
        evicted_path = self._evict_paths(retrieve_leaves=leaves)
        encrypted_path = self._encrypt_path_data(path=evicted_path)

        self._client.add_write_path(label=self._name, data=encrypted_path)
        self._client.execute()

    def _evict_paths(self, retrieve_leaves: List[int]) -> Dict[int, List[Data]]:
        """Evict data blocks in the stash to multiple paths.

        :param retrieve_leaves: list of leaf labels to evict to.
        :return: PathData dict mapping storage index to bucket.
        """
        temp_stash: List[Data] = []

        # Create a dict keyed by storage indices that covers all nodes on the multiple paths
        path_dict = BinaryTree.get_mul_path_dict(level=self._tree.level, indices=retrieve_leaves)

        # Try to place every real data item from stash into one of the provided paths
        for data in self._stash:
            inserted = BinaryTree.fill_data_to_path(
                data=data,
                path=path_dict,
                leaves=retrieve_leaves,
                level=self._tree.level,
                bucket_size=self._bucket_size,
            )
            if not inserted:
                temp_stash.append(data)

        # Update the stash to only those elements that could not be placed
        self._stash = temp_stash

        return path_dict
