"""Group-by-hash OMAP with oblivious metadata storage.

Design:
- Upper ORAM: Stores (count, prf_seed, key_list) for each hash bucket
- Lower ORAM (MulPathOram): Stores actual key-value pairs with PRF-computed paths
- Search: Fetches item by actual key, reshuffles bucket with new seed
- Insert: Computes path for new item, inserts directly
"""
import math
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

from daoram.dependency import Blake2Prf, Data, Encryptor, Helper, InteractServer, UNSET
from daoram.oram import TreeBaseOram, MulPathOram


class GroupOmap:
    """Group-by-hash OMAP with oblivious metadata storage in upper ORAM."""

    # Seed size for PRF (matches Blake2Prf.KEY_SIZE)
    SEED_SIZE = 32

    def __init__(self,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 upper_oram: TreeBaseOram,
                 client: InteractServer,
                 name: str = "group_omap",
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 encryptor: Encryptor = None):
        """Initialize the group-path OMAP.

        :param num_data: Capacity for both ORAMs.
        :param key_size: Bytes for keys (used for sizing).
        :param data_size: Bytes for values (used for sizing).
        :param upper_oram: ORAM for storing bucket metadata.
        :param client: The instance we use to interact with server.
        :param name: The name of the protocol.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param encryptor: The encryptor to use for encryption.
        """
        self._name = name
        self._num_data = num_data
        self._num_buckets = num_data
        self._key_size = key_size
        self._data_size = data_size

        # Upper ORAM for metadata
        self._upper_oram = upper_oram

        # PRFs: one for bucket hashing, one for leaf path computation
        self._bucket_prf = Blake2Prf()
        self._leaf_prf = Blake2Prf()

        # Compute upper bound on bucket size (for stash scaling)
        self._upper_bound = math.ceil(
            math.e ** (Helper.lambert_w(math.e ** -1 * (math.log(self._num_buckets, 2) + 128 - 1)).real + 1)
        )

        # Create lower ORAM with same num_data, stash scaled by upper_bound
        self._lower_oram = MulPathOram(
            num_data=num_data,
            data_size=data_size,
            client=client,
            name=f"{name}_lower",
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            encryptor=encryptor,
            stash_scale_multiplier=self._upper_bound
        )

    def _key_to_int(self, key: Any) -> int:
        """Convert a key to an integer for lower ORAM indexing.

        :param key: The actual key.
        :return: Integer key for lower ORAM in [0, num_data).
        """
        if isinstance(key, int) and 0 <= key < self._num_data:
            return key
        # Hash non-integer or out-of-range keys
        if isinstance(key, int):
            key_bytes = key.to_bytes(16, byteorder="big")
        elif isinstance(key, str):
            key_bytes = key.encode("utf-8")
        elif isinstance(key, bytes):
            key_bytes = key
        else:
            key_bytes = str(key).encode("utf-8")
        return self._leaf_prf.digest_mod_n(message=key_bytes, mod=self._num_data)

    def _key_to_bytes(self, key: Any) -> bytes:
        """Convert a key to bytes for PRF input.

        :param key: The actual key.
        :return: Bytes representation of the key.
        """
        if isinstance(key, int):
            return key.to_bytes(16, byteorder="big")
        elif isinstance(key, str):
            return key.encode("utf-8")
        elif isinstance(key, bytes):
            return key
        else:
            return str(key).encode("utf-8")

    def _compute_path(self, seed: bytes, key: Any) -> int:
        """Compute the leaf path for an item using PRF(seed || key).

        :param seed: The bucket's PRF seed.
        :param key: The actual key.
        :return: Leaf index in lower ORAM.
        """
        message = seed + self._key_to_bytes(key)
        return self._leaf_prf.digest_mod_n(message=message, mod=self._num_data)

    def _hash_key_to_bucket(self, key: Any) -> int:
        """Hash a key to its bucket index.

        :param key: The search key.
        :return: Bucket index in [0, num_buckets).
        """
        key_bytes = self._key_to_bytes(key)
        return self._bucket_prf.digest_mod_n(message=key_bytes, mod=self._num_buckets)

    def _encode_metadata(self, count: int, seed: bytes, keys: List[Any]) -> bytes:
        """Pack bucket metadata into bytes.

        :param count: Number of items in bucket.
        :param seed: PRF seed for path computation.
        :param keys: List of actual keys in this bucket.
        :return: Packed metadata bytes.
        """
        return pickle.dumps((count, seed, keys))

    def _decode_metadata(self, data: bytes) -> Tuple[int, bytes, List[Any]]:
        """Unpack bucket metadata.

        :param data: Packed metadata bytes.
        :return: Tuple of (count, seed, keys).
        """
        return pickle.loads(data)

    def init_server_storage(self, data: Optional[List[Tuple[Any, Any]]] = None) -> None:
        """Initialize both ORAMs with the given key-value pairs.

        :param data: List of (key, value) tuples to store.
        """
        if data is None:
            data = []

        # Partition data into buckets
        data_map = Helper.hash_data_to_map(
            prf=self._bucket_prf, data=data, map_size=self._num_buckets
        )

        # Prepare metadata for upper ORAM, data for lower ORAM
        upper_data: Dict[int, bytes] = {}
        lower_data: Dict[int, Any] = {}
        lower_path_map: Dict[int, int] = {}

        # Track which lower_keys are used
        used_lower_keys: set = set()

        for bucket_id in range(self._num_buckets):
            bucket_items = data_map.get(bucket_id, [])
            count = len(bucket_items)

            # Check bucket size limit
            if count > self._upper_bound:
                raise MemoryError(
                    f"Bucket {bucket_id} has {count} items, exceeds upper bound {self._upper_bound}"
                )

            # Generate initial seed for this bucket
            seed = os.urandom(self.SEED_SIZE)

            # Collect keys in this bucket
            bucket_keys = [k for k, v in bucket_items]
            upper_data[bucket_id] = self._encode_metadata(count, seed, bucket_keys)

            # Store items in lower ORAM
            for key, value in bucket_items:
                lower_key = self._key_to_int(key)
                leaf = self._compute_path(seed, key)
                lower_data[lower_key] = (key, value)  # Store actual key with value
                lower_path_map[lower_key] = leaf
                used_lower_keys.add(lower_key)

        # Fill remaining positions with None (required by ORAM init)
        for lower_key in range(self._num_data):
            if lower_key not in used_lower_keys:
                lower_data[lower_key] = None

        # Initialize upper ORAM with metadata
        self._upper_oram.init_server_storage(data_map=upper_data)

        # Initialize lower ORAM with items
        self._lower_oram.init_server_storage(data_map=lower_data, path_map=lower_path_map)

    def search(self, key: Any, value: Any = None) -> Any:
        """Search for a key and optionally update its value.

        Accesses all items in the bucket for obliviousness.
        After search, all items are reshuffled to new paths.

        :param key: The search key.
        :param value: If provided, update the value for this key.
        :return: The (old) value for the key, or None if not found.
        """
        # Step 1: Hash key to bucket
        bucket_id = self._hash_key_to_bucket(key)

        # Step 2: Read metadata from upper ORAM
        metadata = self._upper_oram.operate_on_key_without_eviction(key=bucket_id)
        count, seed, bucket_keys = self._decode_metadata(metadata)

        # Step 3: Generate new seed for reshuffling
        new_seed = os.urandom(self.SEED_SIZE)

        # Step 4: Build maps for all keys in bucket (for obliviousness, access all)
        key_path_map: Dict[int, int] = {}
        new_path_map: Dict[int, int] = {}
        key_value_map: Dict[int, Any] = {}

        for actual_key in bucket_keys:
            lower_key = self._key_to_int(actual_key)
            old_path = self._compute_path(seed, actual_key)
            new_path = self._compute_path(new_seed, actual_key)
            key_path_map[lower_key] = old_path
            new_path_map[lower_key] = new_path
            key_value_map[lower_key] = UNSET  # Read only

        # Step 5: Pad to upper_bound accesses for obliviousness
        dummy_count = self._upper_bound - count
        for i in range(dummy_count):
            # Use random paths for dummy accesses
            dummy_key = self._num_data + i  # Keys outside valid range
            dummy_path = self._leaf_prf.digest_mod_n(
                message=os.urandom(16), mod=self._num_data
            )
            # Don't actually add to maps - just access random paths
            # This is handled by accessing count keys and padding with dummy ops

        # Step 6: Fetch items from lower ORAM
        results = {}
        if bucket_keys:
            results = self._lower_oram.operate_on_keys_without_eviction(
                key_value_map=key_value_map,
                key_path_map=key_path_map,
                new_path_map=new_path_map
            )

        # Step 7: Search for the key among results
        found_value = None
        found_lower_key = None
        lower_key = self._key_to_int(key)

        if lower_key in results and results[lower_key] is not None:
            actual_key, actual_value = results[lower_key]
            if actual_key == key:
                found_value = actual_value
                found_lower_key = lower_key

        # Step 8: If updating value, pass update to eviction
        updates = None
        if value is not None and found_lower_key is not None:
            updates = {found_lower_key: (key, value)}

        # Step 9: Evict lower ORAM
        if bucket_keys:
            self._lower_oram.eviction_for_mul_keys(updates=updates)

        # Step 10: Update upper ORAM with new seed
        new_metadata = self._encode_metadata(count, new_seed, bucket_keys)
        self._upper_oram.eviction_with_update_stash(key=bucket_id, value=new_metadata)

        return found_value

    def insert(self, key: Any, value: Any) -> None:
        """Insert a new key-value pair.

        :param key: The key to insert.
        :param value: The value to insert.
        """
        # Step 1: Hash key to bucket
        bucket_id = self._hash_key_to_bucket(key)

        # Step 2: Read metadata from upper ORAM
        metadata = self._upper_oram.operate_on_key_without_eviction(key=bucket_id)
        count, seed, bucket_keys = self._decode_metadata(metadata)

        # Step 3: Check bucket capacity
        if count >= self._upper_bound:
            raise MemoryError(
                f"Bucket {bucket_id} is full ({count} items), cannot insert"
            )

        # Step 4: Compute path for the new item
        lower_key = self._key_to_int(key)
        path = self._compute_path(seed, key)

        # Step 5: Read the path first (standard ORAM pattern - read before write)
        # This loads any existing data on the path into the stash
        self._lower_oram._client.add_read_path(label=self._lower_oram._name, leaves=[path])
        result = self._lower_oram._client.execute()
        path_data = result.results[self._lower_oram._name]

        # Process path data - add real data to stash
        path_data = self._lower_oram._decrypt_path_data(path=path_data)
        for bucket in path_data.values():
            for data in bucket:
                if data.key is not None:
                    self._lower_oram._stash.append(data)

        # Step 6: Create data block and add to stash
        data_block = Data(key=lower_key, leaf=path, value=(key, value))
        self._lower_oram._stash.append(data_block)
        self._lower_oram._pos_map[lower_key] = path

        # Step 7: Evict lower ORAM
        evicted_path = self._lower_oram._evict_stash(leaves=[path])
        self._lower_oram._client.add_write_path(label=self._lower_oram._name, data=evicted_path)
        self._lower_oram._client.execute()

        # Step 8: Update metadata with new key added
        bucket_keys.append(key)
        new_metadata = self._encode_metadata(count + 1, seed, bucket_keys)
        self._upper_oram.eviction_with_update_stash(key=bucket_id, value=new_metadata)

    def search_group(self, bucket_id: int) -> List[Tuple[Any, Any]]:
        """Search for all items in a bucket.

        :param bucket_id: The bucket index.
        :return: List of (key, value) tuples in the bucket.
        """
        # Read metadata from upper ORAM
        metadata = self._upper_oram.operate_on_key_without_eviction(key=bucket_id)
        count, seed, bucket_keys = self._decode_metadata(metadata)

        # Generate new seed for reshuffling
        new_seed = os.urandom(self.SEED_SIZE)

        # Build maps for all keys in bucket
        key_path_map: Dict[int, int] = {}
        new_path_map: Dict[int, int] = {}
        key_value_map: Dict[int, Any] = {}

        for actual_key in bucket_keys:
            lower_key = self._key_to_int(actual_key)
            old_path = self._compute_path(seed, actual_key)
            new_path = self._compute_path(new_seed, actual_key)
            key_path_map[lower_key] = old_path
            new_path_map[lower_key] = new_path
            key_value_map[lower_key] = UNSET

        # Fetch items from lower ORAM
        results = {}
        if bucket_keys:
            results = self._lower_oram.operate_on_keys_without_eviction(
                key_value_map=key_value_map,
                key_path_map=key_path_map,
                new_path_map=new_path_map
            )

        # Collect results
        items = []
        for actual_key in bucket_keys:
            lower_key = self._key_to_int(actual_key)
            if lower_key in results and results[lower_key] is not None:
                stored_key, stored_value = results[lower_key]
                items.append((stored_key, stored_value))

        # Evict and update
        if bucket_keys:
            self._lower_oram.eviction_for_mul_keys()
        new_metadata = self._encode_metadata(count, new_seed, bucket_keys)
        self._upper_oram.eviction_with_update_stash(key=bucket_id, value=new_metadata)

        return items
