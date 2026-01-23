"""Group-by-hash OMAP with oblivious metadata storage.

Design:
- Upper ORAM: Stores (seed, count) for each hash bucket
- Lower ORAM (MulPathOram): Stores (key, value) pairs at PRF-computed paths
- Search: Computes paths from seed, fetches all items, reshuffles with new seed
"""
import math
import os
from typing import Any, List, Optional

from daoram.dependency import Blake2Prf, Encryptor, Helper, InteractServer, PseudoRandomFunction, KVPair
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

        # Create lower ORAM with same num_data, stash scaled by upper_bound
        self._lower_oram = MulPathOram(
            num_data=num_data,
            data_size=data_size,
            client=client,
            name=f"{name}_lower",
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
        self._upper_oram.init_server_storage(data_map=upper_level_data)
        self._lower_oram.init_server_storage(data_map=lower_data_map, path_map=lower_path_map)

    def search(self, key: Any, value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value.

        If the input value is not None, the value corresponding to the search tree will be updated.
        :param key: The search key of interest.
        :param value: The updated value.
        :return: The (old) value corresponding to the search key.
        """
        # Hash key to bucket and read metadata from upper ORAM.
        bucket_id = self._bucket_prf.digest_mod_n(message=key, mod=self._num_data)
        seed, count = self._upper_oram.operate_on_key_without_eviction(key=bucket_id)
        new_seed = os.urandom(self.SEED_SIZE)

        # Compute all paths from seed (indices 1 to count).
        old_paths = [self._get_path_from_seed_and_index(index=i + 1, seed=seed) for i in range(count)]
        new_paths = [self._get_path_from_seed_and_index(index=i + 1, seed=new_seed) for i in range(count)]
