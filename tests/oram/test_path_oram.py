import random

from daoram.dependency import AesGcm, InteractLocalServer
from daoram.oram import PathOram

# Set a global parameter for the number of data the server should store.
NUM_DATA = pow(2, 10)


class TestPathOram:
    def test_without_init(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = PathOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer())

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some queries for writing.
        for i in range(NUM_DATA):
            oram.operate_on_key(op="w", key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i) == i

        # Issues some random queries.
        for _ in range(NUM_DATA * 5):
            # Get a random key to test.
            key = random.randint(0, NUM_DATA - 1)
            # Write a new value.
            oram.operate_on_key(op="w", key=key, value=key * 2)
            # Check if the new value is written properly.
            assert oram.operate_on_key(op="r", key=key) == key * 2

    def test_with_init(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = PathOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer())

        # Initialize the server with storage.
        oram.init_server_storage(data_map={i: i * 2 for i in range(NUM_DATA)})

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i) == i * 2

    def test_with_enc(self):
        # Create the oram instance with encryption enabled.
        oram = PathOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), encryptor=AesGcm())

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some queries for writing.
        for i in range(NUM_DATA):
            oram.operate_on_key(op="w", key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i) == i

    def test_with_file(self, test_file):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = PathOram(
            num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), filename=str(test_file)
        )

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some queries for writing.
        for i in range(NUM_DATA):
            oram.operate_on_key(op="w", key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i) == i

    def test_with_file_enc(self, test_file):
        # Create the oram instance with encryption enabled.
        oram = PathOram(
            num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), filename=str(test_file), encryptor=AesGcm()
        )

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some queries for writing.
        for i in range(NUM_DATA):
            oram.operate_on_key(op="w", key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i) == i

    def test_operate_then_evict(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = PathOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer())

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some partial read queries and write at eviction time.
        for i in range(NUM_DATA):
            oram.operate_on_key_without_eviction(op="r", key=i)
            oram.eviction_with_update_stash(key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i) == i

