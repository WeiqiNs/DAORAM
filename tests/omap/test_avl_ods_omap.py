import math

from daoram.omap import AVLOmap
from daoram.omap.avl_omap_cache import AVLOmapOptimized


class TestAVLOdsOmap:
    def test_int_key(self, num_data, client):
        # Create the omap instance.
        omap = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)

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
        omap = AVLOmap(
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
        omap = AVLOmap(
            num_data=num_data,
            key_size=10,
            data_size=10,
            client=client,
            filename=str(test_file),
            encryptor=encryptor
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
        omap = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)

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
        omap = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)

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
        omap = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)

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


class TestAVLOdsOmapBatch:
    """Test cases for batch_search and batch_insert operations."""

    def test_batch_search_basic(self, num_data, client):
        """Test batch search returns correct values for multiple keys."""
        omap = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Insert data using single inserts
        for i in range(num_data):
            omap.insert(key=i, value=i * 10)

        # Batch search for multiple keys
        keys_to_search = [0, num_data // 4, num_data // 2, num_data - 1]
        results = omap.batch_search(keys=keys_to_search)

        # Verify results
        assert len(results) == len(keys_to_search)
        for i, key in enumerate(keys_to_search):
            assert results[i] == key * 10

    def test_batch_search_with_update(self, num_data, client):
        """Test batch search can update multiple values."""
        omap = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Insert data
        for i in range(num_data):
            omap.insert(key=i, value=i)

        # Batch search with updates
        keys = [0, 5, 10, 15]
        new_values = [100, 500, 1000, 1500]
        old_results = omap.batch_search(keys=keys, values=new_values)

        # Verify old values returned
        for i, key in enumerate(keys):
            assert old_results[i] == key

        # Verify new values are stored
        for i, key in enumerate(keys):
            assert omap.search(key=key) == new_values[i]

    def test_batch_search_not_found(self, num_data, client):
        """Test batch search returns None for keys not in tree."""
        omap = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Insert only even keys
        for i in range(0, num_data, 2):
            omap.insert(key=i, value=i)

        # Search for mix of existing and non-existing keys
        keys = [0, 1, 2, 3]  # 0, 2 exist; 1, 3 don't
        results = omap.batch_search(keys=keys)

        assert results[0] == 0  # exists
        assert results[1] is None  # doesn't exist
        assert results[2] == 2  # exists
        assert results[3] is None  # doesn't exist

    def test_batch_search_empty_tree(self, num_data, client):
        """Test batch search on empty tree returns all None."""
        omap = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        results = omap.batch_search(keys=[1, 2, 3])
        assert results == [None, None, None]

    def test_batch_insert_basic(self, num_data, client):
        """Test batch insert adds multiple keys correctly."""
        omap = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Batch insert multiple keys
        keys = list(range(0, min(20, num_data)))
        values = [k * 10 for k in keys]
        omap.batch_insert(keys=keys, values=values)

        # Verify all keys are searchable
        for key, val in zip(keys, values):
            assert omap.search(key=key) == val

    def test_batch_insert_reverse_order(self, num_data, client):
        """Test batch insert with keys in reverse order (tests sorting)."""
        omap = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Insert in reverse order - should still work due to sorting
        keys = list(range(min(20, num_data) - 1, -1, -1))
        values = [k * 5 for k in keys]
        omap.batch_insert(keys=keys, values=values)

        # Verify all keys are searchable
        for key, val in zip(keys, values):
            assert omap.search(key=key) == val

    def test_batch_insert_random_order(self, num_data, client):
        """Test batch insert with keys in random order."""
        import random
        omap = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Create shuffled keys
        keys = list(range(min(30, num_data)))
        random.seed(42)  # For reproducibility
        random.shuffle(keys)
        values = [k * 3 for k in keys]

        omap.batch_insert(keys=keys, values=values)

        # Verify all keys are searchable
        for key, val in zip(keys, values):
            assert omap.search(key=key) == val

    def test_batch_insert_then_batch_search(self, num_data, client):
        """Test batch insert followed by batch search."""
        omap = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Batch insert
        keys = list(range(min(25, num_data)))
        values = [k * 7 for k in keys]
        omap.batch_insert(keys=keys, values=values)

        # Batch search for subset
        search_keys = [0, 5, 10, 15, 20]
        search_keys = [k for k in search_keys if k < len(keys)]
        results = omap.batch_search(keys=search_keys)

        for i, key in enumerate(search_keys):
            assert results[i] == key * 7

    def test_batch_insert_empty(self, num_data, client):
        """Test batch insert with empty list does nothing."""
        omap = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Should not raise
        omap.batch_insert(keys=[], values=[])
        assert omap.root is None

    def test_batch_insert_single_key(self, num_data, client):
        """Test batch insert with single key works like regular insert."""
        omap = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        omap.batch_insert(keys=[42], values=["answer"])
        assert omap.search(key=42) == "answer"

    def test_mixed_batch_and_single_operations(self, num_data, client):
        """Test mixing batch and single insert/search operations."""
        omap = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Single inserts
        for i in range(5):
            omap.insert(key=i, value=f"single_{i}")

        # Batch insert
        omap.batch_insert(keys=[10, 11, 12], values=["batch_10", "batch_11", "batch_12"])

        # More single inserts
        for i in range(20, 23):
            omap.insert(key=i, value=f"single_{i}")

        # Batch search
        results = omap.batch_search(keys=[0, 10, 20])
        assert results[0] == "single_0"
        assert results[1] == "batch_10"
        assert results[2] == "single_20"

        # Single search
        assert omap.search(key=11) == "batch_11"
        assert omap.search(key=21) == "single_21"

    def test_batch_insert_triggers_rotations(self, num_data, client):
        """Test batch insert that requires AVL rotations."""
        omap = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Insert keys that will cause various rotation cases
        # Inserting 1,2,3,4,5,6,7 in order would cause right-heavy imbalance
        keys = [1, 2, 3, 4, 5, 6, 7]
        values = [v * 100 for v in keys]
        omap.batch_insert(keys=keys, values=values)

        # Verify all keys are still accessible after rotations
        for key, val in zip(keys, values):
            assert omap.search(key=key) == val

    def test_batch_insert_with_existing_tree(self, num_data, client):
        """Test batch insert into a tree that already has data."""
        omap = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # First, insert some data normally
        for i in range(0, 10, 2):  # Insert 0, 2, 4, 6, 8
            omap.insert(key=i, value=i)

        # Batch insert interleaved keys
        omap.batch_insert(keys=[1, 3, 5, 7, 9], values=[10, 30, 50, 70, 90])

        # Verify all keys
        for i in range(10):
            if i % 2 == 0:
                assert omap.search(key=i) == i
            else:
                assert omap.search(key=i) == i * 10


class TestAVLOdsOmapOptimized:
    def test_int_key(self, num_data, client):
        # Create the omap instance.
        omap = AVLOmapOptimized(num_data=num_data, key_size=10, data_size=10, client=client)

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
        omap = AVLOmapOptimized(
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
        omap = AVLOmapOptimized(
            num_data=num_data,
            key_size=10,
            data_size=10,
            client=client,
            filename=str(test_file),
            encryptor=encryptor
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
        omap = AVLOmapOptimized(num_data=num_data, key_size=10, data_size=10, client=client)

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
        omap = AVLOmapOptimized(num_data=num_data, key_size=10, data_size=10, client=client)

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
        omap = AVLOmapOptimized(num_data=num_data, key_size=10, data_size=10, client=client)

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
