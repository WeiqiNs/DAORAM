import glob
import os
import random
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from daoram.dependency import InteractLocalServer
from daoram.so.soram import Soram

# Set a global parameter for the number of data the server should store.
NUM_DATA = pow(2, 10)  # Using smaller size for Soram tests
CACHE_SIZE = 16  # Cache size for Soram
TEST_FILE = "soram.bin"


# Helper function to remove files generated during testing.
def remove_file():
    for file in glob.glob("*.bin"):
        os.remove(file)


class TestSoram:
    def setup_method(self):
        """Make sure the test file is removed before each test"""
        remove_file()

    def test_init_with_data(self):
        """Test Soram initialization with initial data"""
        # Create initial data map
        initial_data = {i: f"value_{i}" for i in range(NUM_DATA)}

        # Create the Soram instance
        soram = Soram(
            num_data=NUM_DATA,
            cache_size=CACHE_SIZE,
            data_size=10,
            client=InteractLocalServer(),
            use_encryption=False
        )

        # Initialize the server with initial data.
        soram.setup(data_map=initial_data)

        # Check that the extended size is correct
        assert soram._extended_size == NUM_DATA + 2 * CACHE_SIZE

        # Test reading initial data
        for i in range(min(10, NUM_DATA)):  # Test first 10 items
            value = soram.access(key=i, op='read')
            assert value == f"value_{i}"

    def test_read_write_operations(self):
        """Test basic read and write operations"""
        # Create the Soram instance
        soram = Soram(
            num_data=NUM_DATA,
            cache_size=CACHE_SIZE,
            data_size=10,
            client=InteractLocalServer(),
            use_encryption=False
        )

        # Initialize the server with empty storage.
        soram.setup()

        # Write some data
        for i in range(min(50, NUM_DATA)):  # Write first 50 items
            soram.access(key=i, op='write', value=f"value_{i}")

        # Read the data back
        for i in range(min(50, NUM_DATA)):  # Read first 50 items
            value = soram.access(key=i, op='read')
            assert value == f"value_{i}"

    def test_random_operations(self):
        """Test random read and write operations"""

        initial_data = {i: f"value_{i}" for i in range(NUM_DATA)}
        # Create the Soram instance
        soram = Soram(
            num_data=NUM_DATA,
            cache_size=CACHE_SIZE,
            data_size=10,
            client=InteractLocalServer(),
            use_encryption=False
        )

        # Initialize the server with empty storage.
        soram.setup(initial_data)

        # Initialize with some data
        for i in range(min(100, NUM_DATA)):
            soram.access(key=i, op='write', value=f"initial_value_{i}")

        # Perform random operations
        for _ in range(200):
            key = random.randint(0, min(99, NUM_DATA - 1))
            new_value = f"updated_value_{key}_{random.randint(1000, 9999)}"

            # Write new value
            soram.access(key=key, op='write', value=new_value)

            # Read it back
            value = soram.access(key=key, op='read')
            assert value == new_value

    def test_with_encryption(self):
        """Test Soram with encryption enabled"""
        # Create the Soram instance with encryption
        soram = Soram(
            num_data=NUM_DATA,
            cache_size=CACHE_SIZE,
            data_size=10,
            client=InteractLocalServer(),
            use_encryption=True
        )

        # Initialize the server with empty storage.
        soram.setup()

        # Write some data
        for i in range(min(20, NUM_DATA)):
            soram.access(key=i, op='write', value=f"encrypted_value_{i}")

        # Read the data back
        for i in range(min(20, NUM_DATA)):
            value = soram.access(key=i, op='read')
            assert value == f"encrypted_value_{i}"

    def test_cache_behavior(self):
        """Test the behavior of the O_W and O_R caches"""
        # Create the Soram instance with a small cache for easier testing
        soram = Soram(
            num_data=NUM_DATA,
            cache_size=5,  # Small cache size
            data_size=10,
            client=InteractLocalServer(),
            use_encryption=False
        )

        data_map = {}
        for i in range(100):
            data_map[i] = f"Data value {i}".encode()

        soram.setup(data_map=data_map)

        # Access other set of keys to fill the cache
        for i in range(10, 20):
            soram.access(key=i, op='read', value=f"Data value {i}".encode())

        # Access the same keys multiple times to test cache behavior
        for i in range(5):
            for j in range(9):
                # for i >= 1,find key = j in Ow and Or, _dummy_index should be updated
                value = soram.access(key=j, op='read')
                assert value == f"Data value {j}".encode()

        # Test that dummy index is being updated correctly
        assert soram._dummy_index == (4 * 9) % (2 * soram._cache_size)
