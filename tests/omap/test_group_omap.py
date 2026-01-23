from daoram.dependency import AesGcm, KVPair
from daoram.omap.group_omap import GroupOmap
from daoram.oram import DAOram


class TestGroupOmap:
    def test_insert_search_int_key(self, num_data, client):
        # Create the upper ORAM for bucket metadata.
        upper_oram = DAOram(num_data=num_data, data_size=64, client=client, name="upper")

        # Create the GroupOmap.
        omap = GroupOmap(
            num_data=num_data,
            key_size=10,
            data_size=10,
            client=client,
            upper_oram=upper_oram,
        )

        # Initialize empty storage.
        omap.init_server_storage()

        # Insert some values.
        for i in range(num_data // 10):
            omap.insert(key=i, value=i)

        # Search for inserted values.
        for i in range(num_data // 10):
            assert omap.search(key=i) == i

    def test_insert_search_str_key(self, num_data, client):
        # Create the upper ORAM for bucket metadata.
        upper_oram = DAOram(num_data=num_data, data_size=64, client=client, name="upper")

        # Create the GroupOmap.
        omap = GroupOmap(
            num_data=num_data,
            key_size=10,
            data_size=10,
            client=client,
            upper_oram=upper_oram,
        )

        # Initialize empty storage.
        omap.init_server_storage()

        # Insert some string key-value pairs.
        for i in range(num_data // 10):
            omap.insert(key=f"key_{i}", value=f"value_{i}")

        # Search for inserted values.
        for i in range(num_data // 10):
            assert omap.search(key=f"key_{i}") == f"value_{i}"

    def test_fast_insert_search(self, num_data, client):
        # Create the upper ORAM for bucket metadata.
        upper_oram = DAOram(num_data=num_data, data_size=64, client=client, name="upper")

        # Create the GroupOmap.
        omap = GroupOmap(
            num_data=num_data,
            key_size=10,
            data_size=10,
            client=client,
            upper_oram=upper_oram,
        )

        # Initialize empty storage.
        omap.init_server_storage()

        # Fast insert some values.
        for i in range(num_data // 10):
            omap.fast_insert(key=i, value=i)

        # Search for inserted values.
        for i in range(num_data // 10):
            assert omap.search(key=i) == i

    def test_search_update(self, num_data, client):
        # Create the upper ORAM for bucket metadata.
        upper_oram = DAOram(num_data=num_data, data_size=64, client=client, name="upper")

        # Create the GroupOmap.
        omap = GroupOmap(
            num_data=num_data,
            key_size=10,
            data_size=10,
            client=client,
            upper_oram=upper_oram,
        )

        # Initialize empty storage.
        omap.init_server_storage()

        # Insert some values.
        for i in range(num_data // 10):
            omap.insert(key=i, value=i)

        # Update values via search.
        for i in range(num_data // 10):
            old_value = omap.search(key=i, value=i * 2)
            assert old_value == i

        # Verify updates.
        for i in range(num_data // 10):
            assert omap.search(key=i) == i * 2

    def test_with_init_data(self, num_data, client):
        # Create the upper ORAM for bucket metadata.
        upper_oram = DAOram(num_data=num_data, data_size=64, client=client, name="upper")

        # Create the GroupOmap.
        omap = GroupOmap(
            num_data=num_data,
            key_size=10,
            data_size=10,
            client=client,
            upper_oram=upper_oram,
        )

        # Initialize with some data.
        init_count = num_data // 20
        omap.init_server_storage(data=[KVPair(key=i, value=i) for i in range(init_count)])

        # Search for initialized values.
        for i in range(init_count):
            assert omap.search(key=i) == i

        # Insert more values.
        for i in range(init_count, init_count * 2):
            omap.insert(key=i, value=i)

        # Search for all values.
        for i in range(init_count * 2):
            assert omap.search(key=i) == i

    def test_with_encryption(self, num_data, client, encryptor):
        # Create the upper ORAM with encryption.
        upper_encryptor = AesGcm()
        upper_oram = DAOram(
            num_data=num_data, data_size=64, client=client, name="upper", encryptor=upper_encryptor
        )

        # Create the GroupOmap with encryption.
        omap = GroupOmap(
            num_data=num_data,
            key_size=10,
            data_size=10,
            client=client,
            upper_oram=upper_oram,
            encryptor=encryptor,
        )

        # Initialize empty storage.
        omap.init_server_storage()

        # Insert some values.
        for i in range(num_data // 20):
            omap.insert(key=i, value=i)

        # Search for inserted values.
        for i in range(num_data // 20):
            assert omap.search(key=i) == i

    def test_mixed_insert_fast_insert(self, num_data, client):
        # Create the upper ORAM for bucket metadata.
        upper_oram = DAOram(num_data=num_data, data_size=64, client=client, name="upper")

        # Create the GroupOmap.
        omap = GroupOmap(
            num_data=num_data,
            key_size=10,
            data_size=10,
            client=client,
            upper_oram=upper_oram,
        )

        # Initialize empty storage.
        omap.init_server_storage()

        # Mix of insert and fast_insert.
        for i in range(num_data // 20):
            if i % 2 == 0:
                omap.insert(key=i, value=i)
            else:
                omap.fast_insert(key=i, value=i)

        # Search for all inserted values.
        for i in range(num_data // 20):
            assert omap.search(key=i) == i
