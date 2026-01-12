from daoram.dependency import AesGcm
from daoram.omap import AVLOmap, BPlusOmap, OramOstOmap
from daoram.omap.avl_omap_cache import AVLOmapCached
from daoram.omap.bplus_omap_cache import BPlusOmapCached
from daoram.oram import DAOram


class TestOramOdsOmap:
    def test_daoram_avl_int_key(self, num_data, client):
        # Create the ods object.
        ods = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)

        # Create the oram object.
        oram = DAOram(num_data=num_data, data_size=10, client=client)

        # Create the omap object.
        omap = OramOstOmap(num_data=num_data, ost=ods, oram=oram)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(num_data):
            omap.insert(key=i, value=i)

        # Issue some update queries.
        for i in range(num_data):
            omap.search(key=i, value=i * 2)

        # Issue some search queries.
        for i in range(num_data):
            assert omap.search(key=i) == i * 2

    def test_daoram_avl_opt_int_key(self, num_data, client):
        # Create the ods object.
        ods = AVLOmapCached(num_data=num_data, key_size=10, data_size=10, client=client)

        # Create the oram object.
        oram = DAOram(num_data=num_data, data_size=10, client=client)

        # Create the omap object.
        omap = OramOstOmap(num_data=num_data, ost=ods, oram=oram)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(num_data):
            omap.insert(key=i, value=i)

        # Issue some update queries.
        for i in range(num_data):
            omap.search(key=i, value=i * 2)

        # Issue some search queries.
        for i in range(num_data):
            assert omap.search(key=i) == i * 2

    def test_daoram_avl_str_key(self, num_data, client):
        # Create the ods object.
        ods = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)

        # Create the oram object.
        oram = DAOram(num_data=num_data, data_size=10, client=client)

        # Create the omap object.
        omap = OramOstOmap(num_data=num_data, ost=ods, oram=oram)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(num_data):
            omap.insert(key=f"{i}", value=f"{i}")

        # Issue some update queries.
        for i in range(num_data):
            omap.search(key=f"{i}", value=f"{i * 2}")

        # Issue some search queries.
        for i in range(num_data):
            assert omap.search(key=f"{i}") == f"{i * 2}"

    def test_daoram_avl_str_key_with_enc(self, num_data, client, encryptor):
        # Create the ods object with encryption.
        ods = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client, encryptor=encryptor)

        # Create the oram object with encryption (use separate encryptor).
        oram_encryptor = AesGcm()
        oram = DAOram(num_data=num_data, data_size=10, client=client, encryptor=oram_encryptor)

        # Create the omap object.
        omap = OramOstOmap(num_data=num_data, ost=ods, oram=oram)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(num_data // 10):
            omap.insert(key=f"{i}", value=f"{i}")

        # Issue some search queries.
        for i in range(num_data // 10):
            assert omap.search(key=f"{i}") == f"{i}"

    def test_daoram_avl_with_init_int(self, num_data, client):
        # Create the ods object.
        ods = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)

        # Create the oram object.
        oram = DAOram(num_data=num_data, data_size=10, client=client)

        # Create the omap object.
        omap = OramOstOmap(num_data=num_data, ost=ods, oram=oram)

        # Initialize the omap with some integer keys.
        omap.init_server_storage(data=[(i, i) for i in range(num_data // 4)])

        # Keep inserting more values.
        for i in range(num_data // 4, num_data):
            omap.insert(key=i, value=i)

        # Issue some search queries.
        for i in range(num_data):
            assert omap.search(key=i) == i

    def test_daoram_avl_with_init_str(self, num_data, client):
        # Create the ods object.
        ods = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)

        # Create the oram object.
        oram = DAOram(num_data=num_data, data_size=10, client=client)

        # Create the omap object.
        omap = OramOstOmap(num_data=num_data, ost=ods, oram=oram)

        # Initialize the omap with some integer keys.
        omap.init_server_storage(data=[(f"Key {i}", f"Value {i}") for i in range(num_data // 2)])

        # Keep inserting more values.
        for i in range(num_data // 2, num_data):
            omap.insert(key=f"Key {i}", value=f"Value {i}")

        # Issue some search queries.
        for i in range(num_data):
            assert omap.search(key=f"Key {i}") == f"Value {i}"

    def test_daoram_bplus_int_key(self, num_data, client):
        # Create the ods object.
        ods = BPlusOmap(order=40, num_data=num_data, key_size=10, data_size=10, client=client)

        # Create the oram object.
        oram = DAOram(num_data=num_data, data_size=10, client=client)

        # Create the omap object.
        omap = OramOstOmap(num_data=num_data, ost=ods, oram=oram)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(num_data):
            omap.insert(key=i, value=i)

        # Issue some update queries.
        for i in range(num_data):
            omap.search(key=i, value=i * 2)

        # Issue some search queries.
        for i in range(num_data):
            assert omap.search(key=i) == i * 2

    def test_daoram_bplus_opt_int_key(self, num_data, client):
        # Create the ods object.
        ods = BPlusOmapCached(
            order=40, num_data=num_data, key_size=10, data_size=10, client=client
        )

        # Create the oram object.
        oram = DAOram(num_data=num_data, data_size=10, client=client)

        # Create the omap object.
        omap = OramOstOmap(num_data=num_data, ost=ods, oram=oram)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(num_data):
            omap.insert(key=i, value=i)

        # Issue some update queries.
        for i in range(num_data):
            omap.search(key=i, value=i * 2)

        # Issue some search queries.
        for i in range(num_data):
            assert omap.search(key=i) == i * 2

    def test_daoram_bplus_int_key_with_enc(self, num_data, client, encryptor):
        # Create the ods object with encryption.
        ods = BPlusOmap(order=40, num_data=num_data, key_size=10, data_size=10, client=client, encryptor=encryptor)

        # Create the oram object with encryption (use separate encryptor).
        oram_encryptor = AesGcm()
        oram = DAOram(num_data=num_data, data_size=10, client=client, encryptor=oram_encryptor)

        # Create the omap object.
        omap = OramOstOmap(num_data=num_data, ost=ods, oram=oram)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(num_data // 10):
            omap.insert(key=i, value=i)

        # Issue some search queries.
        for i in range(num_data // 10):
            assert omap.search(key=i) == i

    def test_daoram_bplus_str_key(self, num_data, client):
        # Create the ods object.
        ods = BPlusOmap(order=50, num_data=num_data, key_size=10, data_size=10, client=client)

        # Create the oram object.
        oram = DAOram(num_data=num_data, data_size=10, client=client)

        # Create the omap object.
        omap = OramOstOmap(num_data=num_data, ost=ods, oram=oram)

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(num_data):
            omap.insert(key=f"{i}", value=f"{i}")

        # Issue some update queries.
        for i in range(num_data):
            omap.search(key=f"{i}", value=f"{i * 2}")

        # Issue some search queries.
        for i in range(num_data):
            assert omap.search(key=f"{i}") == f"{i * 2}"

    def test_daoram_bplus_with_init_int(self, num_data, client):
        # Create the ods object.
        ods = BPlusOmap(order=60, num_data=num_data, key_size=10, data_size=10, client=client)

        # Create the oram object.
        oram = DAOram(num_data=num_data, data_size=10, client=client)

        # Create the omap object.
        omap = OramOstOmap(num_data=num_data, ost=ods, oram=oram)

        # Initialize the omap with some integer keys.
        omap.init_server_storage(data=[(i, i) for i in range(num_data // 4)])

        # Keep inserting more values.
        for i in range(num_data // 4, num_data):
            omap.insert(key=i, value=i)

        # Issue some search queries.
        for i in range(num_data):
            assert omap.search(key=i) == i

    def test_daoram_bplus_with_init_str(self, num_data, client):
        # Create the ods object.
        ods = BPlusOmap(order=70, num_data=num_data, key_size=10, data_size=10, client=client)

        # Create the oram object.
        oram = DAOram(num_data=num_data, data_size=10, client=client)

        # Create the omap object.
        omap = OramOstOmap(num_data=num_data, ost=ods, oram=oram)

        # Initialize the omap with some integer keys.
        omap.init_server_storage(data=[(f"Key {i}", f"Value {i}") for i in range(num_data // 2)])

        # Keep inserting more values.
        for i in range(num_data // 2, num_data):
            omap.insert(key=f"Key {i}", value=f"Value {i}")

        # Issue some search queries.
        for i in range(num_data):
            assert omap.search(key=f"Key {i}") == f"Value {i}"
