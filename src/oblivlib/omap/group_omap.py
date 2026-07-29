"""Group-by-hash OMAP with oblivious metadata storage.

An upper ORAM stores per-hash-bucket metadata (count, prf_seed, key_list); a lower ``MulPathOram``
stores the actual key-value pairs at PRF-computed paths. Search fetches the whole bucket and reshuffles
it under a new seed; insert computes the new item's path and writes it directly.
"""

import math
import os
import pickle
from typing import Any, override

from oblivlib.dependency import UNSET, Blake2Prf, Helper
from oblivlib.dependency.config import GroupOmapConfig, MulPathOramConfig
from oblivlib.omap.base_omap import BaseOmap
from oblivlib.oram import MulPathOram, TreeBaseOram


class GroupOmap(BaseOmap):
    """Group-by-hash OMAP with oblivious metadata storage in an upper ORAM."""

    # Seed size for PRF (matches Blake2Prf.KEY_SIZE).
    SEED_SIZE = 32

    def __init__(self, config: GroupOmapConfig, upper_oram: TreeBaseOram):
        # The frozen config is the single source of truth; the accessors below read straight from it.
        self._config: GroupOmapConfig = config

        # Number of hash buckets equals the data count (a derived alias, kept as a plain attr).
        self._num_buckets = config.num_data

        self._upper_oram = upper_oram

        # One PRF for bucket hashing, one for leaf path computation.
        self._bucket_prf = Blake2Prf()
        self._leaf_prf = Blake2Prf()

        # Worst-case items per hash bucket (used for stash scaling and metadata sizing).
        self._upper_bound = self._bucket_upper_bound(self._num_buckets)

        # The upper ORAM stores variable-length pickled bucket metadata. Fail clearly here if it is too
        # narrow for a full bucket, rather than cryptically later when an encrypted write overflows.
        required = self.upper_oram_data_size(config.num_data, config.key_size)
        if self._upper_oram._data_size < required:
            raise ValueError(
                f"GroupOmap upper ORAM data_size={self._upper_oram._data_size} is too small for bucket "
                + f"metadata; it must be at least {required}. Size it with "
                + "GroupOmap.upper_oram_data_size(num_data, key_size)."
            )

        # Lower ORAM: same num_data, stash scaled by the per-bucket upper bound.
        self._lower_oram = MulPathOram(
            MulPathOramConfig(
                num_data=config.num_data,
                data_size=config.data_size,
                client=config.client,
                name=f"{config.name}_lower",
                bucket_size=config.bucket_size,
                stash_scale=config.stash_scale,
                encryptor=config.encryptor,
                stash_scale_multiplier=self._upper_bound,
            )
        )

    @staticmethod
    def _bucket_upper_bound(num_data: int) -> int:
        """Worst-case number of items in one hash bucket (https://eprint.iacr.org/2021/1280)."""
        return math.ceil(math.e ** (Helper.lambert_w(math.e**-1 * (math.log(num_data, 2) + 128 - 1)).real + 1))

    @staticmethod
    def upper_oram_data_size(num_data: int, key_size: int) -> int:
        """Minimum ``data_size`` the upper ORAM needs to hold a full bucket's metadata. Metadata is
        ``pickle((count, seed, keys))`` with keys stored verbatim, so this assumes each key serializes to
        at most ``key_size`` bytes; construct the upper ORAM with at least this so a full bucket fits."""
        upper_bound = GroupOmap._bucket_upper_bound(num_data)
        worst_case = pickle.dumps((upper_bound, os.urandom(GroupOmap.SEED_SIZE), [os.urandom(key_size)] * upper_bound))
        return len(worst_case)

    # Construction parameters — read-only views onto the frozen config (see GroupOmapConfig).
    @property
    def _name(self) -> str:
        return self._config.name

    @property
    def _num_data(self) -> int:
        return self._config.num_data

    @property
    def _key_size(self) -> int:
        return self._config.key_size

    @property
    def _data_size(self) -> int:
        return self._config.data_size

    @property
    def _bucket_size(self) -> int:
        return self._config.bucket_size

    @property
    def _stash_scale(self) -> int:
        return self._config.stash_scale

    @property
    def _encryptor(self):
        return self._config.encryptor

    def _key_to_int(self, key: Any) -> int:
        """Map a key to its lower-ORAM index in [0, num_data): an in-range int is used as-is, else hashed."""
        if isinstance(key, int) and 0 <= key < self._num_data:
            return key
        return self._leaf_prf.digest_mod_n(message=self._key_to_bytes(key), mod=self._num_data)

    def _key_to_bytes(self, key: Any) -> bytes:
        """Bytes representation of a key, for PRF input."""
        if isinstance(key, int):
            return key.to_bytes(16, byteorder="big")
        elif isinstance(key, str):
            return key.encode("utf-8")
        elif isinstance(key, bytes):
            return key
        else:
            return str(key).encode("utf-8")

    def _compute_path(self, seed: bytes, key: Any) -> int:
        """Lower-ORAM leaf for an item, as PRF(seed || key)."""
        message = seed + self._key_to_bytes(key)
        return self._leaf_prf.digest_mod_n(message=message, mod=self._num_data)

    def _hash_key_to_bucket(self, key: Any) -> int:
        """Hash a key to its bucket index in [0, num_buckets)."""
        key_bytes = self._key_to_bytes(key)
        return self._bucket_prf.digest_mod_n(message=key_bytes, mod=self._num_buckets)

    def _encode_metadata(self, count: int, seed: bytes, keys: list[Any]) -> bytes:
        return pickle.dumps((count, seed, keys))

    def _decode_metadata(self, data: bytes) -> tuple[int, bytes, list[Any]]:
        return pickle.loads(data)

    @override
    def init_server_storage(self, data: list[tuple[Any, Any]] | None = None) -> None:
        """Initialize both ORAMs with the given key-value pairs."""
        if data is None:
            data = []

        data_map = Helper.hash_data_to_map(prf=self._bucket_prf, data=data, map_size=self._num_buckets)

        upper_data: dict[int, bytes] = {}
        lower_data: dict[int, Any] = {}
        lower_path_map: dict[int, int] = {}
        used_lower_keys: set = set()

        for bucket_id in range(self._num_buckets):
            bucket_items = data_map.get(bucket_id, [])
            count = len(bucket_items)

            if count > self._upper_bound:
                raise MemoryError(f"Bucket {bucket_id} has {count} items, exceeds upper bound {self._upper_bound}")

            seed = os.urandom(self.SEED_SIZE)

            bucket_keys = [k for k, _ in bucket_items]
            upper_data[bucket_id] = self._encode_metadata(count, seed, bucket_keys)

            for key, value in bucket_items:
                lower_key = self._key_to_int(key)
                leaf = self._compute_path(seed, key)
                lower_data[lower_key] = (key, value)  # store the actual key with the value
                lower_path_map[lower_key] = leaf
                used_lower_keys.add(lower_key)

        # Fill remaining positions with None (required by ORAM init).
        for lower_key in range(self._num_data):
            if lower_key not in used_lower_keys:
                lower_data[lower_key] = None

        self._upper_oram.init_server_storage(data_map=upper_data)
        self._lower_oram.init_server_storage(data_map=lower_data, path_map=lower_path_map)

    @override
    def search(self, key: Any, value: Any = None) -> Any:
        """Search for ``key``, optionally updating its value. Accesses every item in the bucket (for
        obliviousness) and reshuffles them all to new paths. Returns the old value, or None if absent."""
        bucket_id = self._hash_key_to_bucket(key)

        metadata = self._upper_oram.operate_on_key_without_eviction(key=bucket_id)
        count, seed, bucket_keys = self._decode_metadata(metadata)

        new_seed = os.urandom(self.SEED_SIZE)

        # Build maps over every key in the bucket (access all, for obliviousness).
        key_path_map: dict[int, int] = {}
        new_path_map: dict[int, int] = {}
        key_value_map: dict[int, Any] = {}

        for actual_key in bucket_keys:
            lower_key = self._key_to_int(actual_key)
            old_path = self._compute_path(seed, actual_key)
            new_path = self._compute_path(new_seed, actual_key)
            key_path_map[lower_key] = old_path
            new_path_map[lower_key] = new_path
            key_value_map[lower_key] = UNSET  # read only

        results = {}
        if bucket_keys:
            results = self._lower_oram.operate_on_keys_without_eviction(
                key_value_map=key_value_map, key_path_map=key_path_map, new_path_map=new_path_map
            )

        # Find the requested key among the results.
        found_value = None
        found_lower_key = None
        lower_key = self._key_to_int(key)

        if lower_key in results and results[lower_key] is not None:
            actual_key, actual_value = results[lower_key]
            if actual_key == key:
                found_value = actual_value
                found_lower_key = lower_key

        # On a value update, pass it through eviction.
        updates = None
        if value is not None and found_lower_key is not None:
            updates = {found_lower_key: (key, value)}

        if bucket_keys:
            self._lower_oram.eviction_for_mul_keys(updates=updates)

        # Update the upper ORAM with the new reshuffle seed.
        new_metadata = self._encode_metadata(count, new_seed, bucket_keys)
        self._upper_oram.eviction_with_update_stash(key=bucket_id, value=new_metadata)

        return found_value

    @override
    def insert(self, key: Any, value: Any) -> None:
        """Insert a new key-value pair."""
        bucket_id = self._hash_key_to_bucket(key)

        metadata = self._upper_oram.operate_on_key_without_eviction(key=bucket_id)
        count, seed, bucket_keys = self._decode_metadata(metadata)

        if count >= self._upper_bound:
            raise MemoryError(f"Bucket {bucket_id} is full ({count} items), cannot insert")

        # Update the key's lower-ORAM block in place and relocate it to its group path. Every lower_key
        # already has exactly one block (created at init -- a real value or a None placeholder), so we
        # UPDATE that block rather than appending a second one: appending would create a duplicate for the
        # same key, and a later read could pick the stale (None) copy and lose the value. operate_on_keys
        # reads the block from its current position-map path, writes the new value, and remaps it to the
        # group path -- keeping exactly one block per key.
        lower_key = self._key_to_int(key)
        new_path = self._compute_path(seed, key)
        self._lower_oram.operate_on_keys(
            key_value_map={lower_key: (key, value)},
            new_path_map={lower_key: new_path},
        )

        bucket_keys.append(key)
        new_metadata = self._encode_metadata(count + 1, seed, bucket_keys)
        self._upper_oram.eviction_with_update_stash(key=bucket_id, value=new_metadata)

    def search_group(self, bucket_id: int) -> list[tuple[Any, Any]]:
        """Return all (key, value) items in a bucket, reshuffling them to new paths."""
        metadata = self._upper_oram.operate_on_key_without_eviction(key=bucket_id)
        count, seed, bucket_keys = self._decode_metadata(metadata)

        new_seed = os.urandom(self.SEED_SIZE)

        key_path_map: dict[int, int] = {}
        new_path_map: dict[int, int] = {}
        key_value_map: dict[int, Any] = {}

        for actual_key in bucket_keys:
            lower_key = self._key_to_int(actual_key)
            old_path = self._compute_path(seed, actual_key)
            new_path = self._compute_path(new_seed, actual_key)
            key_path_map[lower_key] = old_path
            new_path_map[lower_key] = new_path
            key_value_map[lower_key] = UNSET

        results = {}
        if bucket_keys:
            results = self._lower_oram.operate_on_keys_without_eviction(
                key_value_map=key_value_map, key_path_map=key_path_map, new_path_map=new_path_map
            )

        items = []
        for actual_key in bucket_keys:
            lower_key = self._key_to_int(actual_key)
            if lower_key in results and results[lower_key] is not None:
                stored_key, stored_value = results[lower_key]
                items.append((stored_key, stored_value))

        if bucket_keys:
            self._lower_oram.eviction_for_mul_keys()
        new_metadata = self._encode_metadata(count, new_seed, bucket_keys)
        self._upper_oram.eviction_with_update_stash(key=bucket_id, value=new_metadata)

        return items
