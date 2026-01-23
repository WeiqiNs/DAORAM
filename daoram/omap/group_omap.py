"""Group-by-hash OMAP with oblivious metadata storage.

Design:
- Upper ORAM: Stores (seed, count) for each hash bucket
- Lower ORAM (MulPathOram): Stores (key, value) pairs at PRF-computed paths
- Search: Computes paths from seed, fetches all items, reshuffles with new seed
"""
import math
import os
from typing import Any, List, Optional

from daoram.dependency import Blake2Prf, Data, Encryptor, Helper, InteractServer, PseudoRandomFunction, KVPair, UNSET
from daoram.oram import TreeBaseOram, MulPathOram


class GroupOmap:
    """Group-by-hash OMAP with oblivious metadata storage in upper ORAM."""

    # Seed size for PRF (matches Blake2Prf.KEY_SIZE)
    SEED_SIZE = 32

    def __init__(self,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 client: InteractServer,
                 upper_oram: TreeBaseOram,
                 name: str = "group_omap",
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 encryptor: Encryptor = None,
                 bucket_prf: PseudoRandomFunction = None,
                 leaf_prf: PseudoRandomFunction = None):
        """Initialize the group-path OMAP.

        :param num_data: Capacity for both ORAMs.
        :param key_size: Bytes for keys (used for sizing).
        :param data_size: Bytes for values (used for sizing).
        :param client: The instance we use to interact with server.
        :param upper_oram: ORAM for storing bucket metadata.
        :param name: The name of the protocol.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param encryptor: The encryptor to use for encryption.
        :param bucket_prf: PRF for bucket hashing. If None, a new Blake2Prf is created.
        :param leaf_prf: PRF for leaf path computation. If None, a new Blake2Prf is created.
        """
        self._name = name
        self._client = client
        self._num_data = num_data
        self._key_size = key_size
        self._data_size = data_size

        # Upper ORAM that stores
        self._upper_oram = upper_oram

        # PRFs: one for bucket hashing, one for leaf path computation
        self._bucket_prf = bucket_prf if bucket_prf is not None else Blake2Prf()
        self._leaf_prf = leaf_prf if leaf_prf is not None else Blake2Prf()

        # Compute upper bound on bucket size.
        self._group_size = math.ceil(
            math.e ** (Helper.lambert_w(math.e ** -1 * (math.log(self._num_data, 2) + 128 - 1)).real + 1)
        )
        self._group_byte_length = (self._group_size.bit_length() + 7) // 8

        # Store the lower ORAM label for use in execute results.
        self._lower_label = f"{name}_lower"

        # Create lower ORAM with same num_data, stash scaled by upper_bound
        self._lower_oram = MulPathOram(
            num_data=num_data,
            data_size=data_size,
            client=client,
            name=self._lower_label,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            encryptor=encryptor,
            stash_scale_multiplier=self._group_size
        )

    def _get_path_from_seed_and_index(self, index: int, seed: bytes) -> int:
        """Compute all paths from seed || 1 to seed || group_size.

        :param seed: The bucket's PRF seed.
        :return: The path numbers in lower ORAM.
        """
        message = seed + index.to_bytes(self._group_byte_length, byteorder="big")
        return self._leaf_prf.digest_mod_n(message=message, mod=self._num_data)

    def _get_all_path_from_seed(self, seed: bytes) -> List[int]:
        """Compute all paths from seed || 1 to seed || group_size.

        :param seed: The bucket's PRF seed.
        :return: The path numbers in lower ORAM.
        """
        return [self._get_path_from_seed_and_index(index=i, seed=seed) for i in range(1, self._group_size + 1)]

    def init_server_storage(self, data: Optional[List[KVPair]] = None) -> None:
        """
        Initialize the server storage for the input list of key-value pairs.

        :param data: A list of KVPair objects.
        """
        if data is None:
            data = []

        # Hash data into buckets.
        data_map = Helper.hash_data_to_map(prf=self._bucket_prf, data=data, map_size=self._num_data)

        upper_level_data = {}
        lower_data_map = {}
        lower_path_map = {}

        for bucket_id in range(self._num_data):
            bucket_items = data_map[bucket_id]
            seed = os.urandom(self.SEED_SIZE)
            upper_level_data[bucket_id] = (seed, len(bucket_items))

            for index, each_data in enumerate(bucket_items):
                path = self._get_path_from_seed_and_index(index=index + 1, seed=seed)
                lower_data_map[each_data.key] = each_data.value
                lower_path_map[each_data.key] = path

        # Initialize both ORAMs.
        # Pass empty dicts directly so MulPathOram doesn't create integer keys.
        self._upper_oram.init_server_storage(data_map=upper_level_data)
        self._lower_oram.init_server_storage(
            data_map=lower_data_map,
            path_map=lower_path_map
        )

    def search(self, key: Any, value: Any = UNSET) -> Any:
        """
        Given a search key, return its corresponding value.

        If value is provided (not UNSET), the value corresponding to the key will be updated.
        :param key: The search key of interest.
        :param value: If provided (not UNSET), write this value.
        :return: The (old) value corresponding to the search key.
        """
        # Hash key to bucket and read metadata from upper ORAM.
        bucket_id = Helper.hash_data_to_leaf(prf=self._bucket_prf, data=key, map_size=self._num_data)
        seed, count = self._upper_oram.operate_on_key_without_eviction(key=bucket_id)
        new_seed = os.urandom(self.SEED_SIZE)

        # Compute paths from old seed (for reading) and new seed (for writing).
        old_paths = self._get_all_path_from_seed(seed=seed)
        new_paths = self._get_all_path_from_seed(seed=new_seed)
        new_path_iter = iter(new_paths)

        # Read all paths (old and new) from lower ORAM.
        all_paths = list(set(old_paths + new_paths))
        self._lower_oram.queue_read(leaves=all_paths)
        result = self._client.execute()

        # Process path data directly - decrypt and iterate once.
        path_data = result.results[self._lower_label]
        path_data = self._lower_oram.decrypt_path_data(path=path_data)

        # Placeholder for the result.
        result_value = None

        # Go through the buckets.
        for bucket in path_data.values():
            for data in bucket:
                if data.key is None:
                    continue

                # Check if this key hashes to the current bucket.
                if Helper.hash_data_to_leaf(prf=self._bucket_prf, data=data.key, map_size=self._num_data) == bucket_id:
                    # Assign next available new path.
                    data.leaf = next(new_path_iter)
                    # Check if this is the key we're searching for.
                    if data.key == key:
                        result_value = data.value
                        if value is not UNSET:
                            data.value = value

                # Add to stash for eviction.
                self._lower_oram.stash.append(data)

        # Evict stash and write back to all paths.
        self._lower_oram.queue_write(leaves=all_paths)
        self._client.execute()

        # Update upper ORAM with new seed.
        self._upper_oram.eviction_with_update_stash(key=bucket_id, value=(new_seed, count))

        return result_value

    def insert(self, key: Any, value: Any) -> None:
        """
        Insert a key-value pair into the OMAP.

        :param key: The key to insert.
        :param value: The value to insert.
        """
        # Hash key to bucket and read metadata from upper ORAM.
        bucket_id = Helper.hash_data_to_leaf(prf=self._bucket_prf, data=key, map_size=self._num_data)
        seed, count = self._upper_oram.operate_on_key_without_eviction(key=bucket_id)

        # Check if bucket is full.
        if count >= self._group_size:
            raise OverflowError(f"Bucket {bucket_id} is full (count={count}, max={self._group_size}).")

        new_seed = os.urandom(self.SEED_SIZE)

        # Compute paths from old seed (for reading) and new seed (for writing).
        old_paths = self._get_all_path_from_seed(seed=seed)
        new_paths = self._get_all_path_from_seed(seed=new_seed)
        new_path_iter = iter(new_paths)

        # Read all paths (old and new) from lower ORAM.
        all_paths = list(set(old_paths + new_paths))
        self._lower_oram.queue_read(leaves=all_paths)
        result = self._client.execute()

        # Process path data directly - decrypt and iterate once.
        path_data = result.results[self._lower_label]
        path_data = self._lower_oram.decrypt_path_data(path=path_data)

        # Go through the buckets.
        for bucket in path_data.values():
            for data in bucket:
                if data.key is None:
                    continue

                # Check if this key hashes to the current bucket.
                if Helper.hash_data_to_leaf(prf=self._bucket_prf, data=data.key, map_size=self._num_data) == bucket_id:
                    # Assign next available new path.
                    data.leaf = next(new_path_iter)

                # Add to stash for eviction.
                self._lower_oram.stash.append(data)

        # Add the new key-value pair to stash with the next new path.
        self._lower_oram.stash.append(Data(key=key, leaf=next(new_path_iter), value=value))

        # Evict stash and write back to all paths.
        self._lower_oram.queue_write(leaves=all_paths)
        self._client.execute()

        # Update upper ORAM with new seed and incremented count.
        self._upper_oram.eviction_with_update_stash(key=bucket_id, value=(new_seed, count + 1))

    def fast_insert(self, key: Any, value: Any) -> None:
        """
        Fast insert a key-value pair without reshuffling existing items.

        Computes path from seed || (count + 1), reads only that path, adds new data, and evicts.
        More efficient than insert() but does not reshuffle existing bucket items.

        :param key: The key to insert.
        :param value: The value to insert.
        """
        # Hash key to bucket and read metadata from upper ORAM.
        bucket_id = Helper.hash_data_to_leaf(prf=self._bucket_prf, data=key, map_size=self._num_data)
        seed, count = self._upper_oram.operate_on_key_without_eviction(key=bucket_id)

        # Check if bucket is full.
        if count >= self._group_size:
            raise OverflowError(f"Bucket {bucket_id} is full (count={count}, max={self._group_size}).")

        # Compute path for the new item using seed || (count + 1).
        new_path = self._get_path_from_seed_and_index(index=count + 1, seed=seed)

        # Read the single path from lower ORAM.
        self._lower_oram.queue_read(leaves=[new_path])
        result = self._client.execute()

        # Process path data - decrypt and add to stash.
        path_data = result.results[self._lower_label]
        path_data = self._lower_oram.decrypt_path_data(path=path_data)

        for bucket in path_data.values():
            for data in bucket:
                if data.key is not None:
                    self._lower_oram.stash.append(data)

        # Add the new key-value pair to stash.
        self._lower_oram.stash.append(Data(key=key, leaf=new_path, value=value))

        # Evict stash and write back to the single path.
        self._lower_oram.queue_write(leaves=[new_path])
        self._client.execute()

        # Update upper ORAM with same seed but incremented count.
        self._upper_oram.eviction_with_update_stash(key=bucket_id, value=(seed, count + 1))
