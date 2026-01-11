import glob
import math
import os

from daoram.dependency import InteractLocalServer
from daoram.omap import AVLOdsOmap, BPlusOdsOmap, OramTreeOdsOmap
from daoram.omap.avl_ods_omap_opt import AVLOdsOmapOptimized
from daoram.omap.bplus_ods_omap_opt import BPlusOdsOmapOptimized
from daoram.oram import DAOram

# Set a global parameter for the number of data the server should store.
NUM_DATA = pow(2, 10)
TEST_FILE = "oram.bin"


# Helper function to remove files generated during testing.
def remove_file():
    for file in glob.glob("*.bin"):
        os.remove(file)


class TestAVLOdsOmap:
    def test_int_key(self):
        # Create the omap instance.
        omap = AVLOdsOmap(
            num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=False
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

    def test_int_key_with_enc(self):
        # Create the omap instance.
        omap = AVLOdsOmap(
            num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=True
        )

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(NUM_DATA // 10):
            omap.insert(key=i, value=i)

        # Issue some search queries.
        for i in range(NUM_DATA // 10):
            assert omap.search(key=i) == i

        # Issue some fast search queries.
        for i in range(NUM_DATA // 10):
            assert omap.fast_search(key=i) == i

    def test_int_key_with_enc_file(self):
        # Create the omap instance.
        omap = AVLOdsOmap(
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
            omap.insert(key=i, value=i)

        # Issue some search queries.
        for i in range(NUM_DATA // 10):
            assert omap.search(key=i) == i

        # Issue some fast search queries.
        for i in range(NUM_DATA // 10):
            assert omap.fast_search(key=i) == i

        # Remove testing files.
        remove_file()

    def test_str_key(self):
        # Create the omap instance.
        omap = AVLOdsOmap(
            num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=False
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

    def test_data_init(self):
        # Set the init number of data.
        num_init = pow(2, 8)

        # Create the omap instance.
        omap = AVLOdsOmap(
            num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=False
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
        omap = AVLOdsOmap(
            num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=False
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


class TestAVLOdsOmapOptimized:
    def test_int_key(self):
        # Create the omap instance.
        omap = AVLOdsOmapOptimized(
            num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=False
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

    def test_int_key_with_enc(self):
        # Create the omap instance.
        omap = AVLOdsOmapOptimized(
            num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=True
        )

        # Initialize an empty storage.
        omap.init_server_storage()

        # Issue some insert queries.
        for i in range(NUM_DATA // 10):
            omap.insert(key=i, value=i)

        # Issue some search queries.
        for i in range(NUM_DATA // 10):
            assert omap.search(key=i) == i

        # Issue some fast search queries.
        for i in range(NUM_DATA // 10):
            assert omap.fast_search(key=i) == i

    def test_int_key_with_enc_file(self):
        # Create the omap instance.
        omap = AVLOdsOmapOptimized(
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
            omap.insert(key=i, value=i)

        # Issue some search queries.
        for i in range(NUM_DATA // 10):
            assert omap.search(key=i) == i

        # Issue some fast search queries.
        for i in range(NUM_DATA // 10):
            assert omap.fast_search(key=i) == i

        # Remove testing files.
        remove_file()

    def test_str_key(self):
        # Create the omap instance.
        omap = AVLOdsOmapOptimized(
            num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=False
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

    def test_data_init(self):
        # Set the init number of data.
        num_init = pow(2, 8)

        # Create the omap instance.
        omap = AVLOdsOmapOptimized(
            num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=False
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
        omap = AVLOdsOmapOptimized(
            num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=False
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


class TestBPlusOdsOmap:
    def test_int_key(self):
        # Create the omap instance.
        omap = BPlusOdsOmap(
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
        omap = BPlusOdsOmap(
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
        omap = BPlusOdsOmap(
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
        omap = BPlusOdsOmap(
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
        omap = BPlusOdsOmap(
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
        omap = BPlusOdsOmap(
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

    def test_delete_basic(self):
        """Test basic delete operations."""
        omap = BPlusOdsOmap(
            order=5, num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=False
        )
        omap.init_server_storage()

        # Insert and update like the original tests (required for proper tree structure)
        for i in range(NUM_DATA):
            omap.insert(key=i, value=i)
        for i in range(NUM_DATA):
            omap.search(key=i, value=i * 2)  # This updates and "fixes" the tree

        # Delete every other element in first 50
        for i in range(0, 50, 2):
            deleted_value = omap.delete(key=i)
            assert deleted_value == i * 2  # value was updated to i * 2

        # Verify deleted keys are gone and remaining keys are still there
        for i in range(50):
            result = omap.search(key=i)
            if i % 2 == 0:
                assert result is None  # Deleted
            else:
                assert result == i * 2  # Still exists with updated value

    def test_delete_with_underflow(self):
        """Test delete that triggers underflow handling."""
        omap = BPlusOdsOmap(
            order=3, num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=False
        )
        omap.init_server_storage()

        # Insert and update like original tests
        for i in range(NUM_DATA):
            omap.insert(key=i, value=i)
        for i in range(NUM_DATA):
            omap.search(key=i, value=i)

        # Delete elements to trigger underflow
        for i in range(10):
            omap.delete(key=i)

        # Remaining elements should still be searchable
        for i in range(10, 20):
            assert omap.search(key=i) == i

    def test_delete_all_elements(self):
        """Test deleting all elements from the tree."""
        omap = BPlusOdsOmap(
            order=4, num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=False
        )
        omap.init_server_storage()

        # Insert and update
        for i in range(NUM_DATA):
            omap.insert(key=i, value=i)
        for i in range(NUM_DATA):
            omap.search(key=i, value=i)

        # Delete first 30 elements
        for i in range(30):
            deleted = omap.delete(key=i)
            assert deleted == i

    def test_delete_nonexistent_key(self):
        """Test deleting a key that doesn't exist."""
        omap = BPlusOdsOmap(
            order=5, num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=False
        )
        omap.init_server_storage()

        # Insert and update
        for i in range(NUM_DATA):
            omap.insert(key=i, value=i)
        for i in range(NUM_DATA):
            omap.search(key=i, value=i)

        # Try to delete nonexistent key (larger than NUM_DATA)
        result = omap.delete(key=NUM_DATA + 100)
        assert result is None

        # Original keys should still exist
        for i in range(10):
            assert omap.search(key=i) == i

    def test_delete_and_reinsert(self):
        """Test deleting and then updating values with search."""
        omap = BPlusOdsOmap(
            order=5, num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=False
        )
        omap.init_server_storage()

        # Insert and update
        for i in range(NUM_DATA):
            omap.insert(key=i, value=i)
        for i in range(NUM_DATA):
            omap.search(key=i, value=i)

        # Delete first 20 elements
        for i in range(20):
            omap.delete(key=i)

        # Verify they're gone
        for i in range(20):
            assert omap.search(key=i) is None

        # Remaining elements should work
        for i in range(20, 30):
            assert omap.search(key=i) == i

    def test_delete_with_encryption(self):
        """Test delete with encryption enabled."""
        omap = BPlusOdsOmap(
            order=5, num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=True
        )
        omap.init_server_storage()

        # Insert and update
        for i in range(NUM_DATA // 10):
            omap.insert(key=i, value=i)
        for i in range(NUM_DATA // 10):
            omap.search(key=i, value=i)

        # Delete some elements
        for i in range(0, 20, 2):
            deleted = omap.delete(key=i)
            assert deleted == i

        # Verify remaining elements
        for i in range(1, 20, 2):
            assert omap.search(key=i) == i

    def test_delete_string_keys(self):
        """Test delete with string keys."""
        omap = BPlusOdsOmap(
            order=10, num_data=NUM_DATA, key_size=10, data_size=10, client=InteractLocalServer(), use_encryption=False
        )
        omap.init_server_storage()

        # Insert string key-value pairs (note: order=10 is used for string keys, key==value)
        for i in range(NUM_DATA):
            omap.insert(key=f"{i}", value=f"{i}")

        # Verify search works
        for i in range(30):
            assert omap.search(key=f"{i}") == f"{i}"

        # Delete some
        for i in range(0, 30, 3):
            deleted = omap.delete(key=f"{i}")
            assert deleted == f"{i}"

        # Verify
        for i in range(30):
            result = omap.search(key=f"{i}")
            if i % 3 == 0:
                assert result is None
            else:
                assert result == f"{i}"


class TestBPlusOdsOmapOptimized:
    def test_int_key(self):
        # Create the omap instance.
        omap = BPlusOdsOmapOptimized(
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
        omap = BPlusOdsOmapOptimized(
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
        omap = BPlusOdsOmapOptimized(
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
        omap = BPlusOdsOmapOptimized(
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
        omap = BPlusOdsOmapOptimized(
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
        omap = BPlusOdsOmapOptimized(
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


class TestOramOdsOmap:
    def test_daoram_avl_int_key(self):
        # Create a client object for shared usage.
        client = InteractLocalServer()

        # Create the ods object.
        ods = AVLOdsOmap(num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramTreeOdsOmap(num_data=NUM_DATA, ods=ods, oram=oram)

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
        ods = AVLOdsOmapOptimized(num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramTreeOdsOmap(num_data=NUM_DATA, ods=ods, oram=oram)

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
        ods = AVLOdsOmap(num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramTreeOdsOmap(num_data=NUM_DATA, ods=ods, oram=oram)

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
        ods = AVLOdsOmap(num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=True)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=True)

        # Create the omap object.
        omap = OramTreeOdsOmap(num_data=NUM_DATA, ods=ods, oram=oram)

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
        ods = AVLOdsOmap(num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramTreeOdsOmap(num_data=NUM_DATA, ods=ods, oram=oram)

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
        ods = AVLOdsOmap(num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramTreeOdsOmap(num_data=NUM_DATA, ods=ods, oram=oram)

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
        ods = BPlusOdsOmap(order=40, num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramTreeOdsOmap(num_data=NUM_DATA, ods=ods, oram=oram)

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
        ods = BPlusOdsOmapOptimized(
            order=40, num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False
        )

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramTreeOdsOmap(num_data=NUM_DATA, ods=ods, oram=oram)

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
        ods = BPlusOdsOmap(order=40, num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=True)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=True)

        # Create the omap object.
        omap = OramTreeOdsOmap(num_data=NUM_DATA, ods=ods, oram=oram)

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
        ods = BPlusOdsOmap(order=50, num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramTreeOdsOmap(num_data=NUM_DATA, ods=ods, oram=oram)

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
        ods = BPlusOdsOmap(order=60, num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramTreeOdsOmap(num_data=NUM_DATA, ods=ods, oram=oram)

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
        ods = BPlusOdsOmap(order=70, num_data=NUM_DATA, key_size=10, data_size=10, client=client, use_encryption=False)

        # Create the oram object.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=client, use_encryption=False)

        # Create the omap object.
        omap = OramTreeOdsOmap(num_data=NUM_DATA, ods=ods, oram=oram)

        # Initialize the omap with some integer keys.
        omap.init_server_storage(data=[(f"Key {i}", f"Value {i}") for i in range(NUM_DATA // 2)])

        # Keep inserting more values.
        for i in range(NUM_DATA // 2, NUM_DATA):
            omap.insert(key=f"Key {i}", value=f"Value {i}")

        # Issue some search queries.
        for i in range(NUM_DATA):
            assert omap.search(key=f"Key {i}") == f"Value {i}"
