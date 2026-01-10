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
