import random

from daoram.dependency import UNSET
from daoram.oram.mul_path_oram import MulPathOram


class TestMulPathOramIntegerKeys:
    """Tests for MulPathOram with default integer keys (0 to num_data-1)."""

    def test_batch_read(self, num_data, client):
        """Test batch read operations with UNSET values."""
        oram = MulPathOram(num_data=num_data, data_size=10, client=client, stash_scale_multiplier=3)
        oram.init_server_storage()

        # Write initial values.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i * 2)

        # Batch read using UNSET values.
        keys = [0, 5, 10]
        results = oram.operate_on_keys(key_value_map={k: UNSET for k in keys})

        # Verify values are correct.
        assert results[0] == 0
        assert results[5] == 10
        assert results[10] == 20

        # Verify values in ORAM unchanged.
        assert oram.operate_on_key(key=0) == 0
        assert oram.operate_on_key(key=5) == 10
        assert oram.operate_on_key(key=10) == 20

    def test_batch_write(self, num_data, client):
        """Test batch write operations and verify old values are returned."""
        oram = MulPathOram(num_data=num_data, data_size=10, client=client, stash_scale_multiplier=3)
        oram.init_server_storage()

        # Write initial values.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Batch update multiple keys.
        updates = {0: 100, 1: 101, 2: 102}
        old_values = oram.operate_on_keys(key_value_map=updates)

        # Verify old values returned.
        assert old_values[0] == 0
        assert old_values[1] == 1
        assert old_values[2] == 2

        # Verify new values stored.
        for key, new_value in updates.items():
            assert oram.operate_on_key(key=key) == new_value

    def test_batch_read_and_write(self, num_data, client):
        """Test combined batch read and write operations."""
        oram = MulPathOram(num_data=num_data, data_size=10, client=client, stash_scale_multiplier=5)
        oram.init_server_storage()

        # Write initial values.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Batch read multiple keys.
        keys = [0, 1, 2, 3, 4]
        results = oram.operate_on_keys(key_value_map={k: UNSET for k in keys})

        # Verify values.
        for key in keys:
            assert results[key] == key

    def test_without_eviction(self, num_data, client):
        """Test operate_on_keys_without_eviction followed by eviction_for_mul_keys."""
        oram = MulPathOram(num_data=num_data, data_size=10, client=client, stash_scale_multiplier=3)
        oram.init_server_storage()

        # Write initial values.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Read without eviction.
        keys = [0, 1, 2]
        results = oram.operate_on_keys_without_eviction(key_value_map={k: UNSET for k in keys})

        # Verify values.
        assert results[0] == 0
        assert results[1] == 1
        assert results[2] == 2

        # Evict with updates.
        oram.eviction_for_mul_keys(updates={0: 100, 1: 101})

        # Verify updated values.
        assert oram.operate_on_key(key=0) == 100
        assert oram.operate_on_key(key=1) == 101
        assert oram.operate_on_key(key=2) == 2  # Not updated

    def test_with_encryption(self, num_data, client, encryptor):
        """Test batch operations with encryption enabled."""
        oram = MulPathOram(
            num_data=num_data, data_size=10, client=client,
            stash_scale_multiplier=5, encryptor=encryptor
        )
        oram.init_server_storage()

        # Write initial values.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Batch read.
        keys = [0, 1, 2, 3, 4]
        results = oram.operate_on_keys(key_value_map={k: UNSET for k in keys})
        for key in keys:
            assert results[key] == key

        # Batch write.
        updates = {5: 50, 6: 60}
        oram.operate_on_keys(key_value_map=updates)

        # Verify updates.
        assert oram.operate_on_key(key=5) == 50
        assert oram.operate_on_key(key=6) == 60

    def test_with_file_storage(self, num_data, client, test_file):
        """Test batch operations with file-based storage."""
        oram = MulPathOram(
            num_data=num_data, data_size=10, client=client,
            filename=str(test_file), stash_scale_multiplier=3
        )
        oram.init_server_storage()

        # Write initial values.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Batch read.
        results = oram.operate_on_keys(key_value_map={0: UNSET, 1: UNSET, 2: UNSET})
        assert results[0] == 0
        assert results[1] == 1
        assert results[2] == 2

    def test_random_operations(self, num_data, client):
        """Test random batch operations to verify correctness under varied access patterns."""
        oram = MulPathOram(num_data=num_data, data_size=10, client=client, stash_scale_multiplier=10)
        oram.init_server_storage()

        # Write initial values.
        for i in range(num_data):
            oram.operate_on_key(key=i, value=i)

        # Perform random batch reads.
        for _ in range(10):
            batch_size = random.randint(2, min(10, num_data))
            keys = random.sample(range(num_data), batch_size)

            results = oram.operate_on_keys(key_value_map={k: UNSET for k in keys})

            assert len(results) == batch_size
            for key in keys:
                assert key in results


class TestMulPathOramNonIntegerKeys:
    """Tests for MulPathOram with non-integer keys (strings, tuples, etc.)."""

    def test_string_keys_read(self, num_data, client):
        """Test batch read with string keys."""
        oram = MulPathOram(num_data=num_data, data_size=10, client=client, stash_scale_multiplier=3)

        keys = ["apple", "banana", "cherry"]
        data_map = {key: f"value_{key}" for key in keys}
        path_map = {key: i % num_data for i, key in enumerate(keys)}

        oram.init_server_storage(data_map=data_map, path_map=path_map)

        # Batch read.
        results = oram.operate_on_keys(
            key_value_map={key: UNSET for key in keys},
            key_path_map=path_map
        )

        for key in keys:
            assert results[key] == f"value_{key}"

    def test_string_keys_write(self, num_data, client):
        """Test batch write with string keys and verify old values returned."""
        oram = MulPathOram(num_data=num_data, data_size=10, client=client, stash_scale_multiplier=3)

        keys = ["key1", "key2", "key3"]
        data_map = {key: f"initial_{key}" for key in keys}
        path_map = {key: i % num_data for i, key in enumerate(keys)}

        oram.init_server_storage(data_map=data_map, path_map=path_map)

        # Batch update.
        updates = {"key1": "updated_key1", "key2": "updated_key2"}
        old_values = oram.operate_on_keys(
            key_value_map=updates,
            key_path_map={k: path_map[k] for k in updates}
        )

        # Verify old values returned.
        assert old_values["key1"] == "initial_key1"
        assert old_values["key2"] == "initial_key2"

    def test_string_keys_without_eviction(self, num_data, client):
        """Test operate_on_keys_without_eviction with string keys."""
        oram = MulPathOram(num_data=num_data, data_size=10, client=client, stash_scale_multiplier=3)

        keys = ["foo", "bar", "baz"]
        data_map = {key: f"data_{key}" for key in keys}
        path_map = {key: i % num_data for i, key in enumerate(keys)}

        oram.init_server_storage(data_map=data_map, path_map=path_map)

        # Read without eviction.
        results = oram.operate_on_keys_without_eviction(
            key_value_map={key: UNSET for key in keys},
            key_path_map=path_map
        )

        for key in keys:
            assert results[key] == f"data_{key}"

        # Evict with update.
        oram.eviction_for_mul_keys(updates={"foo": "new_foo"})

    def test_tuple_keys(self, num_data, client):
        """Test batch operations with tuple keys."""
        oram = MulPathOram(num_data=num_data, data_size=10, client=client, stash_scale_multiplier=3)

        keys = [(0, "a"), (1, "b"), (2, "c")]
        data_map = {key: f"value_{key}" for key in keys}
        path_map = {key: i % num_data for i, key in enumerate(keys)}

        oram.init_server_storage(data_map=data_map, path_map=path_map)

        # Batch read.
        results = oram.operate_on_keys(
            key_value_map={key: UNSET for key in keys},
            key_path_map=path_map
        )

        for key in keys:
            assert results[key] == f"value_{key}"
