import math

from daoram.omap.avl_omap_cache import AVLOmapCached


class TestAVLOmapCache:
    def test_int_key(self, num_data, client):
        # Create the omap instance.
        omap = AVLOmapCached(num_data=num_data, key_size=10, data_size=10, client=client)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(num_data):
            omap.insert(key=i, value=i)

        # Issue some search update queries.
        for i in range(num_data):
            omap.search(key=i, value=i * 2)

        # Issue some search queries.
        for i in range(num_data):
            assert omap.search(key=i) == i * 2

        # Issue some fast update queries.
        for i in range(num_data):
            omap.fast_search(key=i, value=i * 3)

        # Issue some fast search queries.
        for i in range(num_data):
            assert omap.fast_search(key=i) == i * 3

    def test_int_key_with_enc(self, num_data, client, encryptor):
        # Create the omap instance.
        omap = AVLOmapCached(
            num_data=num_data, key_size=10, data_size=10, client=client, encryptor=encryptor
        )

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(num_data // 10):
            omap.insert(key=i, value=i)

        # Issue some search queries.
        for i in range(num_data // 10):
            assert omap.search(key=i) == i

        # Issue some fast search queries.
        for i in range(num_data // 10):
            assert omap.fast_search(key=i) == i

    def test_int_key_with_enc_file(self, num_data, client, encryptor, test_file):
        # Create the omap instance.
        omap = AVLOmapCached(
            num_data=num_data,
            key_size=10,
            data_size=10,
            client=client,
            filename=str(test_file),
            encryptor=encryptor,
        )

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(num_data // 10):
            omap.insert(key=i, value=i)

        # Issue some search queries.
        for i in range(num_data // 10):
            assert omap.search(key=i) == i

        # Issue some fast search queries.
        for i in range(num_data // 10):
            assert omap.fast_search(key=i) == i

    def test_str_key(self, num_data, client):
        # Create the omap instance.
        omap = AVLOmapCached(num_data=num_data, key_size=10, data_size=10, client=client)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(num_data):
            omap.insert(key=f"{i}", value=f"{i}")

        # Issue some search queries.
        for i in range(num_data):
            assert omap.search(key=f"{i}") == f"{i}"

        # Issue some fast search queries.
        for i in range(num_data):
            assert omap.fast_search(key=f"{i}") == f"{i}"

    def test_data_init(self, num_data, client):
        # Set the init number of data.
        num_init = pow(2, 8)

        # Create the omap instance.
        omap = AVLOmapCached(num_data=num_data, key_size=10, data_size=10, client=client)

        # Initialize storage with a list of key-value pairs.
        omap.init_server_storage(data=[(f"{i}", f"{i}") for i in range(num_init)])

        # Issue some insert queries.
        for i in range(num_init, num_data):
            omap.insert(key=f"{i}", value=f"{i}")

        # Issue some search queries.
        for i in range(num_data):
            assert omap.search(key=f"{i}") == f"{i}"

        # Issue some fast search queries.
        for i in range(num_data):
            assert omap.fast_search(key=f"{i}") == f"{i}"

    def test_mul_data_init(self, num_data, client):
        # Set extra data to insert.
        extra = 3
        size_group = math.floor(math.log2(num_data))
        num_group = num_data // size_group

        # Set the init number of data.
        init_data = [[(j, j) for j in range(i * 2 * size_group, (i * 2 + 1) * size_group)] for i in range(num_group)]

        # Create the omap instance.
        omap = AVLOmapCached(num_data=num_data, key_size=10, data_size=10, client=client)

        # Initialize storage with lists of key-value pairs.
        roots = omap.init_mul_tree_server_storage(data_list=init_data)

        # Insert some data to each subgroup.
        for i, root in enumerate(roots):
            # Update root.
            omap.root = root
            # Insert some new data to this root.
            for j in range(extra):
                omap.insert(key=(i * 2 + 1) * size_group + j, value=(i * 2 + 1) * size_group + j)
            # Update the stored root.
            roots[i] = omap.root

        # For each root, search the key stored in this root.
        for i, root in enumerate(roots):
            # Update root.
            omap.root = root
            # Search for data in this root.
            for j in range(i * 2 * size_group, (i * 2 + 1) * size_group + extra):
                assert omap.search(key=j) == j
            # Fast search for data in this root.
            for j in range(i * 2 * size_group, (i * 2 + 1) * size_group + extra):
                assert omap.fast_search(key=j) == j

    def test_delete(self, num_data, client):
        # Create the omap instance.
        omap = AVLOmapCached(num_data=num_data, key_size=10, data_size=10, client=client)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Insert data.
        for i in range(num_data):
            omap.insert(key=i, value=i)

        # Delete half the keys and verify.
        for i in range(0, num_data, 2):
            deleted_value = omap.delete(key=i)
            assert deleted_value == i

        # Verify deleted keys return None, remaining keys still accessible.
        for i in range(num_data):
            if i % 2 == 0:
                assert omap.search(key=i) is None
            else:
                assert omap.search(key=i) == i

    def test_delete_all(self, num_data, client):
        # Create the omap instance.
        omap = AVLOmapCached(num_data=num_data, key_size=10, data_size=10, client=client)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Insert data.
        for i in range(num_data):
            omap.insert(key=i, value=i)

        # Delete all keys.
        for i in range(num_data):
            deleted_value = omap.delete(key=i)
            assert deleted_value == i

        # Verify tree is empty.
        assert omap.root is None

    def test_batch_search_basic(self, num_data, client):
        """Test basic batch_search functionality."""
        omap = AVLOmapCached(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Insert data.
        for i in range(num_data):
            omap.insert(key=i, value=i * 10)

        # Batch search for multiple keys.
        keys_to_search = [0, 5, 10, 15, 20]
        results = omap.batch_search(keys=keys_to_search)

        # Verify results.
        for key in keys_to_search:
            if key < num_data:
                assert results[key] == key * 10
            else:
                assert results[key] is None

    def test_batch_search_all_keys(self, num_data, client):
        """Test batch_search with all keys in the tree."""
        # Use smaller num_data for this test.
        test_size = min(32, num_data)
        omap = AVLOmapCached(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Insert data.
        for i in range(test_size):
            omap.insert(key=i, value=i * 2)

        # Batch search for all keys.
        keys_to_search = list(range(test_size))
        results = omap.batch_search(keys=keys_to_search)

        # Verify all results.
        for key in keys_to_search:
            assert results[key] == key * 2

    def test_batch_search_nonexistent_keys(self, num_data, client):
        """Test batch_search with keys that don't exist."""
        omap = AVLOmapCached(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Insert some data.
        for i in range(10):
            omap.insert(key=i, value=i)

        # Search for non-existent keys.
        keys_to_search = [100, 200, 300]
        results = omap.batch_search(keys=keys_to_search)

        # Verify all are None.
        for key in keys_to_search:
            assert results[key] is None

    def test_batch_search_empty_list(self, num_data, client):
        """Test batch_search with empty key list."""
        omap = AVLOmapCached(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Insert some data.
        for i in range(10):
            omap.insert(key=i, value=i)

        # Search with empty list.
        results = omap.batch_search(keys=[])

        # Verify empty result.
        assert results == {}

    def test_batch_search_with_encryption(self, num_data, client, encryptor):
        """Test batch_search with encryption enabled."""
        omap = AVLOmapCached(
            num_data=num_data, key_size=10, data_size=10, client=client, encryptor=encryptor
        )
        omap.init_server_storage()

        # Insert data.
        for i in range(20):
            omap.insert(key=i, value=i * 5)

        # Batch search.
        keys_to_search = [0, 5, 10, 15]
        results = omap.batch_search(keys=keys_to_search)

        # Verify results.
        for key in keys_to_search:
            assert results[key] == key * 5

    def test_batch_search_level_optimization(self, num_data, client):
        """Test that batch_search uses level-based optimization correctly."""
        # This test ensures the optimization kicks in:
        # - When searching for many keys (k), if level has fewer nodes, read all nodes
        # - Otherwise, read k paths
        test_size = 16
        omap = AVLOmapCached(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Insert data.
        for i in range(test_size):
            omap.insert(key=i, value=i)

        # Search for more keys than nodes at early levels (e.g., 8 keys)
        # At level 0: max 1 node, so read all (1 path)
        # At level 1: max 2 nodes, so read all (2 paths)
        # At level 2: max 4 nodes, so read all (4 paths)
        # At level 3: max 8 nodes = k, so read 8 paths
        keys_to_search = [0, 2, 4, 6, 8, 10, 12, 14]
        results = omap.batch_search(keys=keys_to_search)

        # Verify all results are correct.
        for key in keys_to_search:
            assert results[key] == key
