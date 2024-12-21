import random

from daoram.dependency import InteractLocalServer
from daoram.orams import DAOram, FreecursiveOram, PathOram, RecursivePathOram

# Set a global parameter for number of data the server should store.
NUM_DATA = pow(2, 10)


class TestPathOram:
    def test_without_init(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = PathOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), use_encryption=False)

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some write queries.
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
        oram = PathOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), use_encryption=False)

        # Initialize the server with storage.
        oram.init_server_storage(data_map={i: i * 2 for i in range(NUM_DATA)})

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i) == i * 2

    def test_with_enc(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = PathOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), use_encryption=True)

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some write queries.
        for i in range(NUM_DATA):
            oram.operate_on_key(op="w", key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i) == i

    def test_operate_then_evict(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = PathOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), use_encryption=False)

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some partial read queries and write at eviction time.
        for i in range(NUM_DATA):
            oram.operate_on_key_without_eviction(op="r", key=i)
            oram.eviction_with_update_stash(key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i) == i


class TestRecursivePathOram:
    def test_without_init(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = RecursivePathOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), use_encryption=False)

        # Initialize the server with storage.
        oram.init_server_storage()

        # We check that after compression, the position map is smaller than the default size 10.
        assert len(oram._pos_map) <= 10

        # Issue some write queries.
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

        # Issue some write queries.
        for i in range(NUM_DATA):
            oram.operate_on_key(op="w", key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i) == i

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


class TestFreecursiveOram:
    def test_prob_without_init(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = FreecursiveOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), use_encryption=False)

        # Initialize the server with storage.
        oram.init_server_storage()

        # We check that after compression, the position map is emtpy; the storage is in on chip mem.
        assert oram._pos_map == {}

        # Issue some write queries.
        for i in range(NUM_DATA):
            oram.operate_on_key(op="w", key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i, value=None) == i

        # Issues some random queries.
        for _ in range(NUM_DATA * 5):
            # Get a random key to test.
            key = random.randint(0, NUM_DATA - 1)
            # Write a new value.
            oram.operate_on_key(op="w", key=key, value=key * 2)
            # Check if the new value is written properly.
            assert oram.operate_on_key(op="r", key=key, value=None) == key * 2

    def test_prob_with_init(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = FreecursiveOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), use_encryption=False)

        # Initialize the server with storage.
        oram.init_server_storage({i: i * 2 for i in range(NUM_DATA)})

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i, value=None) == i * 2

    def test_prob_with_enc(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = FreecursiveOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), use_encryption=True)

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some write queries.
        for i in range(NUM_DATA):
            oram.operate_on_key(op="w", key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i, value=None) == i

    def test_prob_operate_then_evict(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = FreecursiveOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), use_encryption=False)

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some partial read queries and write at eviction time.
        for i in range(NUM_DATA):
            oram.operate_on_key_without_eviction(op="r", key=i)
            oram.eviction_with_update_stash(key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i) == i

    def test_hard_without_init(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = FreecursiveOram(
            num_data=NUM_DATA, data_size=10, reset_method="hard", client=InteractLocalServer(), use_encryption=False
        )

        # Initialize the server with storage.
        oram.init_server_storage()

        # We check that after compression, the position map is emtpy; the storage is in on chip mem.
        assert oram._pos_map == {}

        # Issue some write queries.
        for i in range(NUM_DATA):
            oram.operate_on_key(op="w", key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i, value=None) == i

        # Issues some random queries to trigger reset.
        for _ in range(NUM_DATA * 5):
            # Get a random key to test.
            value = random.randint(0, NUM_DATA - 1)
            # Write a new value.
            oram.operate_on_key(op="w", key=0, value=value)
            # Check if the new value is written properly.
            assert oram.operate_on_key(op="r", key=0, value=None) == value

    def test_hard_with_init(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = FreecursiveOram(
            num_data=NUM_DATA, data_size=10, reset_method="hard", client=InteractLocalServer(), use_encryption=False
        )

        # Initialize the server with storage.
        oram.init_server_storage({i: i * 2 for i in range(NUM_DATA)})

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i, value=None) == i * 2

    def test_hard_with_enc(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = FreecursiveOram(
            num_data=NUM_DATA, data_size=10, reset_method="hard", client=InteractLocalServer(), use_encryption=True
        )

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some write queries.
        for i in range(NUM_DATA):
            oram.operate_on_key(op="w", key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i, value=None) == i

    def test_hard_operate_then_evict(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = FreecursiveOram(
            num_data=NUM_DATA, data_size=10, reset_method="hard", client=InteractLocalServer(), use_encryption=False
        )

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some partial read queries and write at eviction time.
        for i in range(NUM_DATA):
            oram.operate_on_key_without_eviction(op="r", key=i)
            oram.eviction_with_update_stash(key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i) == i


class TestDAOram:
    def test_without_init(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), use_encryption=False)

        # Initialize the server with storage.
        oram.init_server_storage()

        # We check that after compression, the position map is emtpy; the storage is in on chip mem.
        assert oram._pos_map == {}

        # Issue some write queries.
        for i in range(NUM_DATA):
            oram.operate_on_key(op="w", key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i, value=None) == i

        # Issues some random queries.
        for _ in range(NUM_DATA * 5):
            # Get a random key to test.
            key = random.randint(0, NUM_DATA - 1)
            # Write a new value.
            oram.operate_on_key(op="w", key=key, value=key * 2)
            # Check if the new value is written properly.
            assert oram.operate_on_key(op="r", key=key, value=None) == key * 2

    def test_with_init(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), use_encryption=False)

        # Initialize the server with storage.
        oram.init_server_storage({i: i * 2 for i in range(NUM_DATA)})

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i, value=None) == i * 2

    def test_with_enc(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), use_encryption=True)

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some write queries.
        for i in range(NUM_DATA):
            oram.operate_on_key(op="w", key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i, value=None) == i

    def test_operate_then_evict(self):
        # Create the oram instance; encryption turned off for testing efficiency.
        oram = DAOram(num_data=NUM_DATA, data_size=10, client=InteractLocalServer(), use_encryption=False)

        # Initialize the server with storage.
        oram.init_server_storage()

        # Issue some partial read queries and write at eviction time.
        for i in range(NUM_DATA):
            oram.operate_on_key_without_eviction(op="r", key=i)
            oram.eviction_with_update_stash(key=i, value=i)

        # Check for whether all values are correctly written.
        for i in range(NUM_DATA):
            assert oram.operate_on_key(op="r", key=i) == i
