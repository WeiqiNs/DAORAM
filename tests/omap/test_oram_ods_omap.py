from daoram.dependency import InteractLocalServer
from daoram.omap import AVLOmap, BPlusOmap, OramSearchTreeOmap
from daoram.omap.avl_omap_cache import AVLOmapOptimized
from daoram.omap.bplus_omap_cache import BPlusOmapOptimized
from daoram.oram import DAOram

# Set a global parameter for the number of data the server should store.
NUM_DATA = pow(2, 10)


class TestOramOdsOmap:
    def test_daoram_avl_int_key(self):
        # Create a client object for shared usage.
        client = InteractLocalServer()

        # Create the ods object.
        ods = AVLOmap(num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramSearchTreeOmap(num_data=NUM_DATA, ost=ods, oram=oram)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(NUM_DATA):
            omap.insert(key=i, value=i)

        # Issue some update queries.
        for i in range(NUM_DATA):
            omap.search(key=i, value=i * 2)

        # Issue some search queries.
        for i in range(NUM_DATA):
            assert omap.search(key=i) == i * 2

    def test_daoram_avl_opt_int_key(self):
        # Create a client object for shared usage.
        client = InteractLocalServer()

        # Create the ods object.
        ods = AVLOmapOptimized(num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramSearchTreeOmap(num_data=NUM_DATA, ost=ods, oram=oram)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(NUM_DATA):
            omap.insert(key=i, value=i)

        # Issue some update queries.
        for i in range(NUM_DATA):
            omap.search(key=i, value=i * 2)

        # Issue some search queries.
        for i in range(NUM_DATA):
            assert omap.search(key=i) == i * 2

    def test_daoram_avl_str_key(self):
        # Create a client object for shared usage.
        client = InteractLocalServer()

        # Create the ods object.
        ods = AVLOmap(num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramSearchTreeOmap(num_data=NUM_DATA, ost=ods, oram=oram)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(NUM_DATA):
            omap.insert(key=f"{i}", value=f"{i}")

        # Issue some update queries.
        for i in range(NUM_DATA):
            omap.search(key=f"{i}", value=f"{i * 2}")

        # Issue some search queries.
        for i in range(NUM_DATA):
            assert omap.search(key=f"{i}") == f"{i * 2}"

    def test_daoram_avl_str_key_with_enc(self):
        # Create a client object for shared usage.
        client = InteractLocalServer()

        # Create the ods object.
        ods = AVLOmap(num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=True)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=True)

        # Create the omap object.
        omap = OramSearchTreeOmap(num_data=NUM_DATA, ost=ods, oram=oram)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(NUM_DATA // 10):
            omap.insert(key=f"{i}", value=f"{i}")

        # Issue some search queries.
        for i in range(NUM_DATA // 10):
            assert omap.search(key=f"{i}") == f"{i}"

    def test_daoram_avl_with_init_int(self):
        # Create a client object for shared usage.
        client = InteractLocalServer()

        # Create the ods object.
        ods = AVLOmap(num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramSearchTreeOmap(num_data=NUM_DATA, ost=ods, oram=oram)

        # Initialize the omap with some integer keys.
        omap.init_server_storage(data=[(i, i) for i in range(NUM_DATA // 4)])

        # Keep inserting more values.
        for i in range(NUM_DATA // 4, NUM_DATA):
            omap.insert(key=i, value=i)

        # Issue some search queries.
        for i in range(NUM_DATA):
            assert omap.search(key=i) == i

    def test_daoram_avl_with_init_str(self):
        # Create a client object for shared usage.
        client = InteractLocalServer()

        # Create the ods object.
        ods = AVLOmap(num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramSearchTreeOmap(num_data=NUM_DATA, ost=ods, oram=oram)

        # Initialize the omap with some integer keys.
        omap.init_server_storage(data=[(f"Key {i}", f"Value {i}") for i in range(NUM_DATA // 2)])

        # Keep inserting more values.
        for i in range(NUM_DATA // 2, NUM_DATA):
            omap.insert(key=f"Key {i}", value=f"Value {i}")

        # Issue some search queries.
        for i in range(NUM_DATA):
            assert omap.search(key=f"Key {i}") == f"Value {i}"

    def test_daoram_bplus_int_key(self):
        # Create a client object for shared usage.
        client = InteractLocalServer()

        # Create the ods object.
        ods = BPlusOmap(order=40, num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramSearchTreeOmap(num_data=NUM_DATA, ost=ods, oram=oram)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(NUM_DATA):
            omap.insert(key=i, value=i)

        # Issue some update queries.
        for i in range(NUM_DATA):
            omap.search(key=i, value=i * 2)

        # Issue some search queries.
        for i in range(NUM_DATA):
            assert omap.search(key=i) == i * 2

    def test_daoram_bplus_opt_int_key(self):
        # Create a client object for shared usage.
        client = InteractLocalServer()

        # Create the ods object.
        ods = BPlusOmapOptimized(
            order=40, num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False
        )

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramSearchTreeOmap(num_data=NUM_DATA, ost=ods, oram=oram)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(NUM_DATA):
            omap.insert(key=i, value=i)

        # Issue some update queries.
        for i in range(NUM_DATA):
            omap.search(key=i, value=i * 2)

        # Issue some search queries.
        for i in range(NUM_DATA):
            assert omap.search(key=i) == i * 2

    def test_daoram_bplus_int_key_with_enc(self):
        # Create a client object for shared usage.
        client = InteractLocalServer()

        # Create the ods object.
        ods = BPlusOmap(order=40, num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=True)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=True)

        # Create the omap object.
        omap = OramSearchTreeOmap(num_data=NUM_DATA, ost=ods, oram=oram)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(NUM_DATA // 10):
            omap.insert(key=i, value=i)

        # Issue some search queries.
        for i in range(NUM_DATA // 10):
            assert omap.search(key=i) == i

    def test_daoram_bplus_str_key(self):
        # Create a client object for shared usage.
        client = InteractLocalServer()

        # Create the ods object.
        ods = BPlusOmap(order=50, num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramSearchTreeOmap(num_data=NUM_DATA, ost=ods, oram=oram)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(NUM_DATA):
            omap.insert(key=f"{i}", value=f"{i}")

        # Issue some update queries.
        for i in range(NUM_DATA):
            omap.search(key=f"{i}", value=f"{i * 2}")

        # Issue some search queries.
        for i in range(NUM_DATA):
            assert omap.search(key=f"{i}") == f"{i * 2}"

    def test_daoram_bplus_with_init_int(self):
        # Create a client object for shared usage.
        client = InteractLocalServer()

        # Create the ods object.
        ods = BPlusOmap(order=60, num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramSearchTreeOmap(num_data=NUM_DATA, ost=ods, oram=oram)

        # Initialize the omap with some integer keys.
        omap.init_server_storage(data=[(i, i) for i in range(NUM_DATA // 4)])

        # Keep inserting more values.
        for i in range(NUM_DATA // 4, NUM_DATA):
            omap.insert(key=i, value=i)

        # Issue some search queries.
        for i in range(NUM_DATA):
            assert omap.search(key=i) == i

    def test_daoram_bplus_with_init_str(self):
        # Create a client object for shared usage.
        client = InteractLocalServer()

        # Create the ods object.
        ods = BPlusOmap(order=70, num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramSearchTreeOmap(num_data=NUM_DATA, ost=ods, oram=oram)

        # Initialize the omap with some integer keys.
        omap.init_server_storage(data=[(f"Key {i}", f"Value {i}") for i in range(NUM_DATA // 2)])

        # Keep inserting more values.
        for i in range(NUM_DATA // 2, NUM_DATA):
            omap.insert(key=f"Key {i}", value=f"Value {i}")

        # Issue some search queries.
        for i in range(NUM_DATA):
            assert omap.search(key=f"Key {i}") == f"Value {i}"

