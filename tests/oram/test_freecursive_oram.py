import random

from daoram.dependency import AesGcm, InteractLocalServer
from daoram.oram import FreecursiveOram


class TestFreecursiveOram:
    def test_prob_without_init(self, num_data):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = FreecursiveOram(num_data=num_data, data_size=10, client=InteractLocalServer())

        # Initialize the server with storage.
        oram.init_server_storage()

        # We check that after compression, the position map is empty; the storage is in on chip mem.
        assert oram._pos_map == {}

        # Issue some queries for writing.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(num_data):
            assert oram.operate_on_key(key=i) == i

        # Issues some random queries.
        for _ in range(num_data * 5):
            # Get a random key to test.
            key = random.randint(0, num_data - 1)
            # Write a new value.
            oram.operate_on_key(key=key, value=key * 2)
            # Check if the new value is written properly.
            assert oram.operate_on_key(key=key) == key * 2

    def test_prob_with_init(self, num_data):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = FreecursiveOram(num_data=num_data, data_size=10, client=InteractLocalServer())

        # Initialize the server with storage.
        oram.init_server_storage({i: i * 2 for i in range(num_data)})

        # Check for whether all values are correctly written.
        for i in range(num_data):
            assert oram.operate_on_key(key=i) == i * 2

    def test_prob_with_enc(self, num_data):
        # Create the oram instance with encryption enabled.
        oram = FreecursiveOram(num_data=num_data, data_size=10, client=InteractLocalServer(), encryptor=AesGcm())

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some queries for writing.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(num_data):
            assert oram.operate_on_key(key=i) == i

    def test_prob_with_file(self, num_data, test_file):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = FreecursiveOram(
            num_data=num_data, data_size=10, client=InteractLocalServer(), filename=str(test_file)
        )

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some queries for writing.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(num_data):
            assert oram.operate_on_key(key=i) == i

    def test_prob_with_file_enc(self, num_data, test_file):
        # Create the oram instance with encryption enabled.
        oram = FreecursiveOram(
            num_data=num_data, data_size=10, client=InteractLocalServer(), filename=str(test_file), encryptor=AesGcm()
        )

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some queries for writing.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(num_data):
            assert oram.operate_on_key(key=i) == i

    def test_prob_operate_then_evict(self, num_data):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = FreecursiveOram(num_data=num_data, data_size=10, client=InteractLocalServer())

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some partial read queries and write at eviction time.
        for i in range(num_data):
            oram.operate_on_key_without_eviction(key=i)
            oram.eviction_with_update_stash(key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(num_data):
            assert oram.operate_on_key(key=i) == i

    def test_hard_without_init(self, num_data):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = FreecursiveOram(
            num_data=num_data, data_size=10, reset_method="hard", client=InteractLocalServer()
        )

        # Initialize the server with storage.
        oram.init_server_storage()

        # We check that after compression, the position map is empty; the storage is in on chip mem.
        assert oram._pos_map == {}

        # Issue some queries for writing.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(num_data):
            assert oram.operate_on_key(key=i) == i

        # Issues some random queries to trigger reset.
        for _ in range(num_data * 5):
            # Get a random key to test.
            value = random.randint(0, num_data - 1)
            # Write a new value.
            oram.operate_on_key(key=0, value=value)
            # Check if the new value is written properly.
            assert oram.operate_on_key(key=0) == value

    def test_hard_with_init(self, num_data):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = FreecursiveOram(
            num_data=num_data, data_size=10, reset_method="hard", client=InteractLocalServer()
        )

        # Initialize the server with storage.
        oram.init_server_storage({i: i * 2 for i in range(num_data)})

        # Check for whether all values are correctly written.
        for i in range(num_data):
            assert oram.operate_on_key(key=i) == i * 2

    def test_hard_with_enc(self, num_data):
        # Create the oram instance with encryption enabled.
        oram = FreecursiveOram(
            num_data=num_data, data_size=10, reset_method="hard", client=InteractLocalServer(), encryptor=AesGcm()
        )

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some queries for writing.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(num_data):
            assert oram.operate_on_key(key=i) == i

    def test_hard_with_file(self, num_data, test_file):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = FreecursiveOram(
            num_data=num_data, data_size=10, reset_method="hard", client=InteractLocalServer(), filename=str(test_file)
        )

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some queries for writing.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(num_data):
            assert oram.operate_on_key(key=i) == i

    def test_hard_with_file_enc(self, num_data, test_file):
        # Create the oram instance with encryption enabled.
        oram = FreecursiveOram(
            num_data=num_data,
            data_size=10,
            reset_method="hard",
            client=InteractLocalServer(),
            filename=str(test_file),
            encryptor=AesGcm()
        )

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some queries for writing.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(num_data):
            assert oram.operate_on_key(key=i) == i

    def test_hard_operate_then_evict(self, num_data):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = FreecursiveOram(
            num_data=num_data, data_size=10, reset_method="hard", client=InteractLocalServer()
        )

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some partial read queries and write at eviction time.
        for i in range(num_data):
            oram.operate_on_key_without_eviction(key=i)
            oram.eviction_with_update_stash(key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(num_data):
            assert oram.operate_on_key(key=i) == i

