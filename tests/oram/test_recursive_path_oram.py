import glob
import os
import random

from daoram.dependency import InteractLocalServer
from daoram.oram import RecursivePathOram

# Set a global parameter for the number of data the server should store.
NUM_DATA = pow(2, 10)
TEST_FILE = "oram.bin"


def remove_file():
    """Helper function to remove files generated during testing."""
    for file in glob.glob("*.bin"):
        os.remove(file)


class TestRecursivePathOram:
    def test_without_init(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = RecursivePathOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), use_encryption=False)

        # Initialize the server with storage.
        oram.init_server_storage()

        # We check that after compression, the position map is smaller than the default size 10.
        assert len(oram._pos_map) <= 10

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
        oram = RecursivePathOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), use_encryption=False)

        # Initialize the server with storage.
        oram.init_server_storage(data_map={i: i * 2 for i in range(NUM_DATA)})

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i) == i * 2

    def test_with_enc(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = RecursivePathOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), use_encryption=True)

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some queries for writing.
        for i in range(NUM_DATA):
            oram.operate_on_key(op="w", key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i) == i

    def test_with_file_enc(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = RecursivePathOram(
            num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), filename=TEST_FILE, use_encryption=True
        )

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some queries for writing.
        for i in range(NUM_DATA):
            oram.operate_on_key(op="w", key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i) == i

        # Remove the testing file.
        remove_file()

    def test_operate_then_evict(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = RecursivePathOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), use_encryption=False)

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some partial read queries and write at eviction time.
        for i in range(NUM_DATA):
            oram.operate_on_key_without_eviction(op="r", key=i)
            oram.eviction_with_update_stash(key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i) == i

