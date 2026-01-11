"""
This module defines an OMAP class that uses multi-path ORAM for batch retrieval.

The first layer is an ORAM that stores PRF seeds. When accessing a key, we retrieve
the PRF seed, use it to generate multiple path indices, then use MulPathOram to
retrieve all paths at once.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

from daoram.dependency import Blake2Prf, Helper, PseudoRandomFunction
from daoram.omap.oblivious_search_tree import ObliviousSearchTree
from daoram.oram.mul_path_oram import MulPathOram
from daoram.oram.tree_base_oram import TreeBaseOram


class MulPathOmap:
    def __init__(
        self,
        num_data: int,
        num_paths: int,
        ost: ObliviousSearchTree,
        seed_oram: TreeBaseOram,
        data_oram: MulPathOram
    ):
        """
        Initialize the multi-path OMAP construction.

        This OMAP uses a two-layer structure:
        - First layer (seed_oram): Stores PRF seeds for each logical bucket
        - Second layer (data_oram): MulPathOram that stores actual data across multiple paths

        When searching for a key:
        1. Hash the key to find which seed bucket it belongs to
        2. Retrieve the PRF seed from seed_oram
        3. Use the seed to generate `num_paths` path indices
        4. Use data_oram to batch retrieve all paths
        5. Search for the key in the retrieved data

        :param num_data: The number of data points the omap should store.
        :param num_paths: The number of paths to retrieve per operation.
        :param ost: Oblivious search tree for organizing data within paths.
        :param seed_oram: ORAM for storing PRF seeds (first layer).
        :param data_oram: MulPathOram for storing actual data (second layer).
        """
        self._num_data: int = num_data
        self._num_paths: int = num_paths
        self._ost: ObliviousSearchTree = ost
        self._seed_oram: TreeBaseOram = seed_oram
        self._data_oram: MulPathOram = data_oram

        # Update the OST height for multiple trees.
        self._ost.update_mul_tree_height(num_tree=num_data)

        # PRF for hashing input keys to seed bucket indices.
        self._key_prf: Blake2Prf = Blake2Prf()

    def _generate_path_keys(self, seed: bytes, num_paths: int) -> List[int]:
        """
        Use the PRF seed to generate a list of path keys.

        :param seed: The PRF seed retrieved from seed_oram.
        :param num_paths: Number of path keys to generate.
        :return: List of path keys for data_oram.
        """
        # Create a PRF from the seed.
        prf = Blake2Prf(key=seed)

        # Generate path keys using the PRF.
        path_keys = []
        for i in range(num_paths):
            # Hash the index to get a path key within data_oram's range.
            path_key = prf.hash_to_int(
                data=i.to_bytes(4, 'big'),
                output_range=self._data_oram._num_data
            )
            path_keys.append(path_key)

        return path_keys

    def init_server_storage(
        self,
        data: Optional[List[Tuple[Union[str, int, bytes], Any]]] = None
    ) -> None:
        """
        Initialize the server storage for the input list of key-value pairs.

        :param data: A list of key-value pairs to store.
        """
        if data is None:
            data = []

        # Hash data to buckets based on keys.
        data_map = Helper.hash_data_to_map(
            prf=self._key_prf,
            data=data,
            map_size=self._num_data
        )

        # Convert to list format for OST initialization.
        data_list = [data_map[key] for key in range(self._num_data)]

        # Initialize multiple trees in the OST.
        roots = self._ost.init_mul_tree_server_storage(data_list=data_list)

        # Generate PRF seeds for each bucket and store in seed_oram.
        # The seed_oram stores: key -> (seed, root)
        seed_map: Dict[int, Tuple[bytes, Any]] = {}
        for key in range(self._num_data):
            # Generate a random seed for this bucket.
            seed = Blake2Prf().key  # Use a new PRF's key as the seed
            seed_map[key] = (seed, roots[key])

        # Initialize seed_oram with the seeds and roots.
        self._seed_oram.init_server_storage(data_map=seed_map)

        # Initialize data_oram (empty initially, data added via insert).
        self._data_oram.init_server_storage()

    def search(self, key: Union[str, int, bytes], value: Any = None) -> Any:
        """
        Search for a key and optionally update its value.

        :param key: The search key.
        :param value: If provided, update the value for this key.
        :return: The current (or old, if updating) value for the key.
        """
        # Hash the key to find which seed bucket it belongs to.
        seed_key = Helper.hash_data_to_leaf(
            prf=self._key_prf,
            data=key,
            map_size=self._num_data
        )

        # Retrieve the seed and root from seed_oram.
        seed_root = self._seed_oram.operate_on_key_without_eviction(key=seed_key)
        seed, root = seed_root

        # Generate path keys using the seed.
        path_keys = self._generate_path_keys(seed=seed, num_paths=self._num_paths)

        # Batch retrieve all paths from data_oram.
        path_values = self._data_oram.operate_on_keys_without_eviction(keys=path_keys)

        # Set the OST root and search within the tree.
        self._ost.root = root
        result = self._ost.search(key=key, value=value)

        # Update the root in seed_oram and evict.
        new_seed_root = (seed, self._ost.root)
        self._seed_oram.eviction_with_update_stash(key=seed_key, value=new_seed_root)

        # Evict data_oram paths.
        self._data_oram.eviction_for_mul_keys()

        return result

    def insert(self, key: Union[str, int, bytes], value: Any) -> None:
        """
        Insert a key-value pair into the OMAP.

        :param key: The key to insert.
        :param value: The value to associate with the key.
        """
        # Hash the key to find which seed bucket it belongs to.
        seed_key = Helper.hash_data_to_leaf(
            prf=self._key_prf,
            data=key,
            map_size=self._num_data
        )

        # Retrieve the seed and root from seed_oram.
        seed_root = self._seed_oram.operate_on_key_without_eviction(key=seed_key)
        seed, root = seed_root

        # Generate path keys using the seed.
        path_keys = self._generate_path_keys(seed=seed, num_paths=self._num_paths)

        # Batch retrieve all paths from data_oram.
        self._data_oram.operate_on_keys_without_eviction(keys=path_keys)

        # Set the OST root and perform insert.
        self._ost.root = root
        self._ost.insert(key=key, value=value)

        # Update the root in seed_oram and evict.
        new_seed_root = (seed, self._ost.root)
        self._seed_oram.eviction_with_update_stash(key=seed_key, value=new_seed_root)

        # Evict data_oram paths.
        self._data_oram.eviction_for_mul_keys()

    def fast_search(self, key: Union[str, int, bytes], value: Any = None) -> Any:
        """
        Fast search for a key using the OST's fast_search method.

        :param key: The search key.
        :param value: If provided, update the value for this key.
        :return: The current (or old, if updating) value for the key.
        """
        # Hash the key to find which seed bucket it belongs to.
        seed_key = Helper.hash_data_to_leaf(
            prf=self._key_prf,
            data=key,
            map_size=self._num_data
        )

        # Retrieve the seed and root from seed_oram.
        seed_root = self._seed_oram.operate_on_key_without_eviction(key=seed_key)
        seed, root = seed_root

        # Generate path keys using the seed.
        path_keys = self._generate_path_keys(seed=seed, num_paths=self._num_paths)

        # Batch retrieve all paths from data_oram.
        self._data_oram.operate_on_keys_without_eviction(keys=path_keys)

        # Set the OST root and perform fast search.
        self._ost.root = root
        result = self._ost.fast_search(key=key, value=value)

        # Update the root in seed_oram and evict.
        new_seed_root = (seed, self._ost.root)
        self._seed_oram.eviction_with_update_stash(key=seed_key, value=new_seed_root)

        # Evict data_oram paths.
        self._data_oram.eviction_for_mul_keys()

        return result

    def delete(self, key: Union[str, int, bytes]) -> Any:
        """
        Delete a key from the OMAP.

        :param key: The key to delete.
        :return: The deleted value, or None if key not found.
        """
        # Hash the key to find which seed bucket it belongs to.
        seed_key = Helper.hash_data_to_leaf(
            prf=self._key_prf,
            data=key,
            map_size=self._num_data
        )

        # Retrieve the seed and root from seed_oram.
        seed_root = self._seed_oram.operate_on_key_without_eviction(key=seed_key)
        seed, root = seed_root

        # Generate path keys using the seed.
        path_keys = self._generate_path_keys(seed=seed, num_paths=self._num_paths)

        # Batch retrieve all paths from data_oram.
        self._data_oram.operate_on_keys_without_eviction(keys=path_keys)

        # Set the OST root and perform delete.
        self._ost.root = root
        result = self._ost.delete(key=key)

        # Update the root in seed_oram and evict.
        new_seed_root = (seed, self._ost.root)
        self._seed_oram.eviction_with_update_stash(key=seed_key, value=new_seed_root)

        # Evict data_oram paths.
        self._data_oram.eviction_for_mul_keys()

        return result
