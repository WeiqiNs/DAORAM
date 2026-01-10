import glob
import math
import os

from daoram.dependency import InteractLocalServer
from daoram.omap.bplus_omap_cache import BPlusOmapOptimized

# Set a global parameter for the number of data the server should store.
NUM_DATA = pow(2, 10)
TEST_FILE = "oram.bin"


def remove_file():
    """Helper function to remove files generated during testing."""
    for file in glob.glob("*.bin"):
        os.remove(file)


class TestBPlusOmapCache:
    def test_int_key(self):
        # Create the omap instance.
        omap = BPlusOmapOptimized(
            order=5, num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=False
        )

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(NUM_DATA):
            omap.insert(key=i, value=i)

        # Issue some search update queries.
        for i in range(NUM_DATA):
            omap.search(key=i, value=i * 2)

        # Issue some search queries.
        for i in range(NUM_DATA):
            assert omap.search(key=i) == i * 2

        # Issue some fast update queries.
        for i in range(NUM_DATA):
            omap.fast_search(key=i, value=i * 3)

        # Issue some fast search queries.
        for i in range(NUM_DATA):
            assert omap.fast_search(key=i) == i * 3

    def test_str_key(self):
        # Create the omap instance.
        omap = BPlusOmapOptimized(
            order=10, num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=False
        )

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(NUM_DATA):
            omap.insert(key=f"{i}", value=f"{i}")

        # Issue some search queries.
        for i in range(NUM_DATA):
            assert omap.search(key=f"{i}") == f"{i}"

        # Issue some fast search queries.
        for i in range(NUM_DATA):
            assert omap.fast_search(key=f"{i}") == f"{i}"

    def test_str_key_with_enc(self):
        # Create the omap instance.
        omap = BPlusOmapOptimized(
            order=10, num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=True
        )

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(NUM_DATA // 10):
            omap.insert(key=f"{i}", value=f"{i}")

        # Issue some search queries.
        for i in range(NUM_DATA // 10):
            assert omap.search(key=f"{i}") == f"{i}"

        # Issue some fast search queries.
        for i in range(NUM_DATA // 10):
            assert omap.fast_search(key=f"{i}") == f"{i}"

    def test_str_key_with_enc_file(self):
        # Create the omap instance.
        omap = BPlusOmapOptimized(
            order=10,
            num_data=NUM_DATA,
            key_size=10,
            data_size=10,
            client=InteractLocalServer(),
            filename=TEST_FILE,
            use_encryption=True
        )

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(NUM_DATA // 10):
            omap.insert(key=f"{i}", value=f"{i}")

        # Issue some search queries.
        for i in range(NUM_DATA // 10):
            assert omap.search(key=f"{i}") == f"{i}"

        # Issue some fast search queries.
        for i in range(NUM_DATA // 10):
            assert omap.fast_search(key=f"{i}") == f"{i}"

        # Remove testing files.
        remove_file()

    def test_data_init(self):
        # Set the init number of data.
        num_init = pow(2, 8)

        # Create the omap instance.
        omap = BPlusOmapOptimized(
            order=20, num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=False
        )

        # Initialize storage with a list of key-value pairs.
        omap.init_server_storage(data=[(f"{i}", f"{i}") for i in range(num_init)])

        # Issue some insert queries.
        for i in range(num_init, NUM_DATA):
            omap.insert(key=f"{i}", value=f"{i}")

        # Issue some search queries.
        for i in range(NUM_DATA):
            assert omap.search(key=f"{i}") == f"{i}"

        # Issue some fast search queries.
        for i in range(NUM_DATA):
            assert omap.fast_search(key=f"{i}") == f"{i}"

    def test_mul_data_init(self):
        # Set extra data to insert.
        extra = 3
        size_group = math.floor(math.log2(NUM_DATA))
        num_group = NUM_DATA // size_group

        # Set the init number of data.
        init_data = [[(j, j) for j in range(i * 2 * size_group, (i * 2 + 1) * size_group)] for i in range(num_group)]

        # Create the omap instance.
        omap = BPlusOmapOptimized(
            order=30, num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=False
        )

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
