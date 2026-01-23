import pytest

from daoram.omap.avl_omap_batch import AVLOmapBatch


class TestAVLOmapBatch:
    def test_batch_insert_search(self, num_data, client):
        """Insert multiple keys via batch, search for all."""
        omap = AVLOmapBatch(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Batch insert
        keys = list(range(num_data // 4))
        values = [i * 10 for i in keys]
        omap.batch_insert(keys=keys, values=values)

        # Batch search (value=None means just search)
        results = omap.batch_search(kv_pairs=[(k, None) for k in keys])
        assert results == values

    def test_batch_search_update(self, num_data, client):
        """Search and update multiple keys via batch."""
        omap = AVLOmapBatch(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Insert data first
        keys = list(range(num_data // 4))
        values = [i * 10 for i in keys]
        omap.batch_insert(keys=keys, values=values)

        # Batch search with update
        new_values = [i * 100 for i in keys]
        old_values = omap.batch_search(kv_pairs=list(zip(keys, new_values)))
        assert old_values == values

        # Verify update worked
        results = omap.batch_search(kv_pairs=[(k, None) for k in keys])
        assert results == new_values

    def test_batch_delete(self, num_data, client):
        """Delete multiple keys via batch."""
        omap = AVLOmapBatch(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Insert data first
        keys = list(range(num_data // 4))
        values = [i * 10 for i in keys]
        omap.batch_insert(keys=keys, values=values)

        # Delete half the keys
        keys_to_delete = keys[::2]  # Every other key
        expected_deleted = [i * 10 for i in keys_to_delete]
        deleted_values = omap.batch_delete(keys=keys_to_delete)
        assert deleted_values == expected_deleted

        # Verify deleted keys are gone, remaining keys still accessible
        remaining_keys = keys[1::2]
        remaining_values = omap.batch_search(kv_pairs=[(k, None) for k in remaining_keys])
        assert remaining_values == [i * 10 for i in remaining_keys]

    def test_mixed_batch_single(self, num_data, client):
        """Mix batch and single operations."""
        omap = AVLOmapBatch(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Single insert
        for i in range(10):
            omap.insert(key=i, value=i * 10)

        # Batch insert more
        omap.batch_insert(keys=list(range(10, 20)), values=[i * 10 for i in range(10, 20)])

        # Single search
        for i in range(20):
            assert omap.search(key=i) == i * 10

        # Batch search
        results = omap.batch_search(kv_pairs=[(k, None) for k in range(20)])
        assert results == [i * 10 for i in range(20)]

        # Single delete
        omap.delete(key=5)
        assert omap.search(key=5) is None

        # Batch delete
        omap.batch_delete(keys=[10, 15])
        results = omap.batch_search(kv_pairs=[(10, None), (15, None)])
        assert results == [None, None]

    def test_batch_with_encryption(self, num_data, client, encryptor):
        """Batch operations with encryption."""
        omap = AVLOmapBatch(
            num_data=num_data, key_size=10, data_size=10, client=client, encryptor=encryptor
        )
        omap.init_server_storage()

        # Batch insert
        keys = list(range(num_data // 10))
        values = [f"value_{i}" for i in keys]
        omap.batch_insert(keys=keys, values=values)

        # Batch search
        results = omap.batch_search(kv_pairs=[(k, None) for k in keys])
        assert results == values

        # Batch search with update
        new_values = [f"new_value_{i}" for i in keys]
        old_values = omap.batch_search(kv_pairs=list(zip(keys, new_values)))
        assert old_values == values

        # Verify update
        results = omap.batch_search(kv_pairs=[(k, None) for k in keys])
        assert results == new_values

    def test_batch_empty_operations(self, num_data, client):
        """Test batch operations with empty lists."""
        omap = AVLOmapBatch(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Empty batch insert should not raise
        omap.batch_insert(keys=[])

        # Insert some data
        omap.batch_insert(keys=[1, 2, 3], values=[10, 20, 30])

        # Empty batch search should return empty list
        results = omap.batch_search(kv_pairs=[])
        assert results == []

        # Empty batch delete should return empty list
        deleted = omap.batch_delete(keys=[])
        assert deleted == []

    def test_batch_str_keys(self, num_data, client):
        """Test batch operations with string keys."""
        omap = AVLOmapBatch(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Batch insert with string keys
        keys = [f"key_{i}" for i in range(20)]
        values = [f"value_{i}" for i in range(20)]
        omap.batch_insert(keys=keys, values=values)

        # Batch search
        results = omap.batch_search(kv_pairs=[(k, None) for k in keys])
        assert results == values

        # Batch delete some keys
        keys_to_delete = keys[:10]
        omap.batch_delete(keys=keys_to_delete)

        # Verify deletion
        results = omap.batch_search(kv_pairs=[(k, None) for k in keys_to_delete])
        assert results == [None] * 10

        # Verify remaining keys
        results = omap.batch_search(kv_pairs=[(k, None) for k in keys[10:]])
        assert results == values[10:]

    def test_batch_single_key(self, num_data, client):
        """Test batch operations with single key (edge case)."""
        omap = AVLOmapBatch(num_data=num_data, key_size=10, data_size=10, client=client)
        omap.init_server_storage()

        # Single key batch insert
        omap.batch_insert(keys=[42], values=[420])

        # Single key batch search
        results = omap.batch_search(kv_pairs=[(42, None)])
        assert results == [420]

        # Single key batch delete
        deleted = omap.batch_delete(keys=[42])
        assert deleted == [420]

        # Verify deletion
        results = omap.batch_search(kv_pairs=[(42, None)])
        assert results == [None]
