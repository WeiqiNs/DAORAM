import random

from daoram.dependency import UNSET
from daoram.oram.mul_path_oram import MulPathOram


class TestMulPathOram:
    def test_batch_write_and_read(self, num_data, client):
        # Create the oram instance.
        oram = MulPathOram(num_data=num_data, data_size=10, client=client)

        # Initialize the server with storage.
        oram.init_server_storage()

        # Write values one at a time first to populate the oram.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Batch read multiple keys at once.
        keys_to_read = [0, 1, 2, 3, 4]
        results = oram.operate_on_keys(keys=keys_to_read)

        # Verify we got all values back.
        for key in keys_to_read:
            assert results[key] == key

    def test_batch_write(self, num_data, client):
        # Create the oram instance.
        oram = MulPathOram(num_data=num_data, data_size=10, client=client)

        # Initialize the server with storage.
        oram.init_server_storage()

        # Write initial values.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Batch update multiple keys at once.
        updates = {0: 100, 1: 101, 2: 102}
        old_values = oram.operate_on_keys(key_value_map=updates)

        # Verify we got the old values back.
        assert old_values[0] == 0
        assert old_values[1] == 1
        assert old_values[2] == 2

        # Verify new values are stored.
        for key, new_value in updates.items():
            assert oram.operate_on_key(key=key) == new_value

    def test_batch_read_only(self, num_data, client):
        # Create the oram instance.
        oram = MulPathOram(num_data=num_data, data_size=10, client=client)

        # Initialize the server with storage.
        oram.init_server_storage()

        # Write initial values.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i * 2)

        # Batch read using key_value_map with UNSET values.
        read_map = {0: UNSET, 5: UNSET, 10: UNSET}
        results = oram.operate_on_keys(key_value_map=read_map)

        # Verify values are unchanged (read-only).
        assert results[0] == 0
        assert results[5] == 10
        assert results[10] == 20

        # Verify values in oram are still the same.
        assert oram.operate_on_key(key=0) == 0
        assert oram.operate_on_key(key=5) == 10
        assert oram.operate_on_key(key=10) == 20

    def test_without_eviction(self, num_data, client):
        # Create the oram instance.
        oram = MulPathOram(num_data=num_data, data_size=10, client=client)

        # Initialize the server with storage.
        oram.init_server_storage()

        # Write initial values.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Read without eviction.
        keys_to_read = [0, 1, 2]
        results = oram.operate_on_keys_without_eviction(keys=keys_to_read)

        # Verify we got the values.
        assert results[0] == 0
        assert results[1] == 1
        assert results[2] == 2

        # Now evict with updates.
        oram.eviction_for_mul_keys(updates={0: 100, 1: 101})

        # Verify updated values.
        assert oram.operate_on_key(key=0) == 100
        assert oram.operate_on_key(key=1) == 101
        assert oram.operate_on_key(key=2) == 2  # Not updated

    def test_with_enc(self, num_data, client, encryptor):
        # Create the oram instance with encryption.
        oram = MulPathOram(num_data=num_data, data_size=10, client=client, encryptor=encryptor)

        # Initialize the server with storage.
        oram.init_server_storage()

        # Write initial values.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Batch read multiple keys.
        keys_to_read = [0, 1, 2, 3, 4]
        results = oram.operate_on_keys(keys=keys_to_read)

        # Verify values.
        for key in keys_to_read:
            assert results[key] == key

        # Batch write.
        updates = {5: 50, 6: 60}
        oram.operate_on_keys(key_value_map=updates)

        # Verify updates.
        assert oram.operate_on_key(key=5) == 50
        assert oram.operate_on_key(key=6) == 60

    def test_with_file(self, num_data, client, test_file):
        # Create the oram instance with file storage.
        oram = MulPathOram(num_data=num_data, data_size=10, client=client, filename=str(test_file))

        # Initialize the server with storage.
        oram.init_server_storage()

        # Write initial values.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Batch operations.
        results = oram.operate_on_keys(keys=[0, 1, 2])
        assert results[0] == 0
        assert results[1] == 1
        assert results[2] == 2

    def test_random_batch_operations(self, num_data, client):
        # Create the oram instance.
        oram = MulPathOram(num_data=num_data, data_size=10, client=client)

        # Initialize the server with storage.
        oram.init_server_storage()

        # Write initial values.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Perform random batch operations.
        for _ in range(10):
            # Pick random keys.
            batch_size = random.randint(2, min(10, num_data))
            keys = random.sample(range(num_data), batch_size)

            # Read all keys.
            results = oram.operate_on_keys(keys=keys)

            # Verify all keys have values.
            assert len(results) == batch_size
            for key in keys:
                assert key in results
