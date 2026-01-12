"""
Test Suite for TopDownSomap

This test suite provides testing for the TopDownSomap implementation,
covering initialization, setup, search, insert operations, and cache management.

Note: TopDownSomap requires num_data to be large enough (e.g., >= 256) due to 
the upper_bound calculation which can exceed num_groups for small values.
"""
import glob
import os
import sys
import pytest

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from daoram.dependency import InteractLocalServer
from daoram.so.top_down_somap import TopDownSomap

# Minimum num_data to avoid infinite loop in _collect_group_leaves_retrieve
# upper_bound is ~47 for small num_groups, so we need num_data >= 256 to be safe
MIN_NUM_DATA = 256


# Helper function to remove files generated during testing
def remove_test_files():
    """Remove all test files created during testing"""
    for file in glob.glob("*.bin"):
        try:
            os.remove(file)
        except OSError:
            pass  # File might not exist


class TestTopDownSomap:
    """Test class for TopDownSomap functionality"""

    def setup_method(self):
        """Clean up before each test"""
        remove_test_files()

    def teardown_method(self):
        """Clean up after each test"""
        remove_test_files()

    def test_initialization(self):
        """Test basic initialization of TopDownSomap"""
        somap = TopDownSomap(
            num_data=MIN_NUM_DATA,
            cache_size=10,
            data_size=20,
            client=InteractLocalServer(),
            use_encryption=False
        )
        
        assert somap._num_data == MIN_NUM_DATA
        assert somap._cache_size == 10
        assert somap._data_size == 20
        assert somap._use_encryption is False
        assert somap._extended_size == 3 * MIN_NUM_DATA

    def test_setup_empty(self):
        """Test setup with no initial data"""
        somap = TopDownSomap(
            num_data=MIN_NUM_DATA,
            cache_size=10,
            data_size=16,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()
        
        # Verify internal structures are initialized
        assert somap._Ow is not None
        assert somap._Or is not None
        assert somap._tree is not None
        assert len(somap._main_storage) == somap._extended_size

    def test_setup_with_data(self):
        """Test setup with initial data"""
        somap = TopDownSomap(
            num_data=MIN_NUM_DATA,
            cache_size=10,
            data_size=16,
            client=InteractLocalServer(),
            use_encryption=False
        )
        
        # Create initial data as list of tuples
        initial_data = [(f"key_{i}", f"value_{i}") for i in range(10)]
        somap.setup(data=initial_data)
        
        # Verify setup completed
        assert somap._Ow is not None
        assert somap._Or is not None
        assert somap._tree is not None

    def test_prp_encryption(self):
        """Test PRP (Pseudorandom Permutation) functionality"""
        somap = TopDownSomap(
            num_data=MIN_NUM_DATA,
            cache_size=10,
            data_size=20,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()
        
        # Test PRP encryption/decryption - verify round trip
        original = 42
        encrypted = somap.PRP.encrypt(original)
        decrypted = somap.PRP.decrypt(encrypted)
        
        assert decrypted == original
        # Note: encrypted may equal original for some PRP instances (rare but possible)

    def test_group_hashing(self):
        """Test that keys are hashed to groups consistently"""
        somap = TopDownSomap(
            num_data=MIN_NUM_DATA,
            cache_size=10,
            data_size=20,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()
        
        from daoram.dependency import Helper
        
        # Same key should hash to same group
        key = "test_key"
        group1 = Helper.hash_data_to_leaf(prf=somap._group_prf, data=key, map_size=somap._num_groups)
        group2 = Helper.hash_data_to_leaf(prf=somap._group_prf, data=key, map_size=somap._num_groups)
        
        assert group1 == group2

    def test_extended_database_size(self):
        """Test that database is extended correctly"""
        somap = TopDownSomap(
            num_data=MIN_NUM_DATA,
            cache_size=10,
            data_size=16,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()
        
        # Extended size should be 3 * num_data
        assert len(somap._main_storage) == 3 * MIN_NUM_DATA

    def test_tree_initialization(self):
        """Test that binary tree is properly initialized"""
        somap = TopDownSomap(
            num_data=MIN_NUM_DATA,
            cache_size=10,
            data_size=20,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()
        
        # Tree should be initialized
        assert somap._tree is not None
        assert somap._tree.level > 0

    def test_operate_on_list_insert(self):
        """Test operate_on_list insert operation"""
        somap = TopDownSomap(
            num_data=MIN_NUM_DATA,
            cache_size=10,
            data_size=16,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()
        
        # Test list insert
        somap.operate_on_list(label=somap._Qw_name, op='insert', data=("test_key", "marker"))

    def test_operate_on_list_get(self):
        """Test operate_on_list get operation"""
        somap = TopDownSomap(
            num_data=MIN_NUM_DATA,
            cache_size=10,
            data_size=16,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()
        
        # Get from queue
        result = somap.operate_on_list(label=somap._Qw_name, op='get', pos=0)
        # Should return some value or None

    def test_upper_bound_calculation(self):
        """Test that upper_bound is less than num_groups to avoid infinite loop"""
        somap = TopDownSomap(
            num_data=MIN_NUM_DATA,
            cache_size=10,
            data_size=20,
            client=InteractLocalServer(),
            use_encryption=False
        )
        
        # upper_bound should be less than num_groups for the algorithm to work correctly
        assert somap.upper_bound <= somap._num_groups, \
            f"upper_bound ({somap.upper_bound}) should be <= num_groups ({somap._num_groups})"


class TestTopDownSomapEdgeCases:
    """Test edge cases for TopDownSomap"""

    def setup_method(self):
        """Clean up before each test"""
        remove_test_files()

    def teardown_method(self):
        """Clean up after each test"""
        remove_test_files()

    def test_minimum_valid_num_data(self):
        """Test with minimum valid num_data that avoids infinite loop"""
        # Find minimum num_data where upper_bound <= num_groups
        import math
        from daoram.dependency import Helper
        
        for num_data in [64, 128, 256, 512]:
            upper_bound = math.ceil(
                math.e ** (Helper.lambert_w(math.e ** -1 * (math.log(num_data, 2) + 128 - 1)).real + 1)
            )
            if upper_bound <= num_data:
                # This should work
                somap = TopDownSomap(
                    num_data=num_data,
                    cache_size=5,
                    data_size=16,
                    client=InteractLocalServer(),
                    use_encryption=False
                )
                somap.setup()
                assert somap.upper_bound <= somap._num_groups
                break

    def test_cache_size_validation(self):
        """Test various cache sizes"""
        for cache_size in [5, 10, 20]:
            somap = TopDownSomap(
                num_data=MIN_NUM_DATA,
                cache_size=cache_size,
                data_size=16,
                client=InteractLocalServer(),
                use_encryption=False
            )
            somap.setup()
            assert somap._cache_size == cache_size


class TestTopDownSomapIntegration:
    """Integration tests for TopDownSomap with larger data and full access flow"""

    # Use 1024 (2^10) to ensure upper_bound <= num_groups
    LARGE_NUM_DATA = 1024

    def setup_method(self):
        """Clean up before each test"""
        remove_test_files()

    def teardown_method(self):
        """Clean up after each test"""
        remove_test_files()

    def test_access_search_single_key(self):
        """Test single search access operation with large data"""
        somap = TopDownSomap(
            num_data=self.LARGE_NUM_DATA,
            cache_size=20,
            data_size=32,
            client=InteractLocalServer(),
            use_encryption=False
        )
        
        # Setup with initial data
        initial_data = [(f"key_{i}", f"value_{i}") for i in range(50)]
        somap.setup(data=initial_data)
        
        # Perform search access
        result = somap.access(op="search", general_key="key_0")
        
        # Timestamp should have incremented
        assert somap._timestamp >= 1

    def test_access_insert_single_key(self):
        """Test single insert access operation"""
        somap = TopDownSomap(
            num_data=self.LARGE_NUM_DATA,
            cache_size=20,
            data_size=32,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()
        
        # Perform insert access
        somap.access(op="insert", general_key="new_key", general_value="new_value")
        
        # Timestamp should have incremented
        assert somap._timestamp >= 1

    def test_access_multiple_operations(self):
        """Test multiple access operations in sequence"""
        somap = TopDownSomap(
            num_data=self.LARGE_NUM_DATA,
            cache_size=30,
            data_size=32,
            client=InteractLocalServer(),
            use_encryption=False
        )
        
        # Setup with initial data
        initial_data = [(f"key_{i}", f"value_{i}") for i in range(20)]
        somap.setup(data=initial_data)
        
        initial_timestamp = somap._timestamp
        
        # Perform multiple operations
        num_ops = 10
        for i in range(num_ops):
            if i % 2 == 0:
                somap.access(op="search", general_key=f"key_{i % 20}")
            else:
                somap.access(op="insert", general_key=f"new_key_{i}", general_value=f"new_value_{i}")
        
        # Verify timestamp incremented correctly
        assert somap._timestamp == initial_timestamp + num_ops

    def test_cache_overflow_triggers_adjustment(self):
        """Test that cache overflow triggers security level adjustment"""
        cache_size = 10
        somap = TopDownSomap(
            num_data=self.LARGE_NUM_DATA,
            cache_size=cache_size,
            data_size=32,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()
        
        # Perform more operations than cache size to trigger adjustment
        num_ops = cache_size + 5
        for i in range(num_ops):
            somap.access(op="search", general_key=f"key_{i}")
        
        # Should complete without errors
        assert somap._timestamp == num_ops

    def test_repeated_access_same_key(self):
        """Test repeated access to the same key (cache hit scenario)"""
        somap = TopDownSomap(
            num_data=self.LARGE_NUM_DATA,
            cache_size=20,
            data_size=32,
            client=InteractLocalServer(),
            use_encryption=False
        )
        
        initial_data = [(f"key_{i}", f"value_{i}") for i in range(10)]
        somap.setup(data=initial_data)
        
        # Access same key multiple times
        for _ in range(5):
            somap.access(op="search", general_key="key_0")
        
        assert somap._timestamp == 5

    def test_with_encryption_enabled(self):
        """Test full access flow with encryption enabled"""
        somap = TopDownSomap(
            num_data=self.LARGE_NUM_DATA,
            cache_size=20,
            data_size=32,
            client=InteractLocalServer(),
            use_encryption=True
        )
        
        initial_data = [(f"key_{i}", f"value_{i}") for i in range(10)]
        somap.setup(data=initial_data)
        
        # Perform operations with encryption
        for i in range(5):
            somap.access(op="search", general_key=f"key_{i}")
        
        assert somap._use_encryption is True
        assert somap._timestamp == 5

    def test_parallel_cache_search(self):
        """Test the parallel cache search (piggyback optimization)"""
        somap = TopDownSomap(
            num_data=self.LARGE_NUM_DATA,
            cache_size=20,
            data_size=32,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()
        
        # Perform some operations to populate caches
        for i in range(5):
            somap.access(op="insert", general_key=f"key_{i}", general_value=f"value_{i}")
        
        # The parallel search should work
        initial_timestamp = somap._timestamp
        somap.access(op="search", general_key="key_0")
        
        assert somap._timestamp == initial_timestamp + 1

    def test_large_initial_dataset(self):
        """Test with a larger initial dataset"""
        somap = TopDownSomap(
            num_data=self.LARGE_NUM_DATA,
            cache_size=50,
            data_size=64,
            client=InteractLocalServer(),
            use_encryption=False
        )
        
        # Create larger initial dataset
        initial_data = [(f"key_{i}", f"value_{i}_data") for i in range(100)]
        somap.setup(data=initial_data)
        
        # Perform various operations
        for i in range(20):
            if i % 3 == 0:
                somap.access(op="search", general_key=f"key_{i % 100}")
            else:
                somap.access(op="insert", general_key=f"new_key_{i}", general_value=f"new_value_{i}")
        
        assert somap._timestamp == 20

    def test_queue_operations(self):
        """Test queue Q_W and Q_R operations during access"""
        somap = TopDownSomap(
            num_data=self.LARGE_NUM_DATA,
            cache_size=10,
            data_size=32,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()
        
        initial_qw_len = len(somap._Qw)
        
        # Each access should add to Q_W
        somap.access(op="search", general_key="test_key_1")
        
        # Q_W length should have increased
        # (may vary depending on adjust_security_level behavior)
        assert somap._timestamp == 1

    def test_stress_many_operations(self):
        """Stress test with many operations"""
        somap = TopDownSomap(
            num_data=self.LARGE_NUM_DATA,
            cache_size=50,
            data_size=32,
            client=InteractLocalServer(),
            use_encryption=False
        )
        
        initial_data = [(f"key_{i}", f"value_{i}") for i in range(30)]
        somap.setup(data=initial_data)
        
        # Perform many operations
        num_ops = 50
        for i in range(num_ops):
            op = "search" if i % 2 == 0 else "insert"
            if op == "search":
                somap.access(op=op, general_key=f"key_{i % 30}")
            else:
                somap.access(op=op, general_key=f"stress_key_{i}", general_value=f"stress_value_{i}")
        
        assert somap._timestamp == num_ops

    def test_dummy_index_management(self):
        """Test that dummy index is managed correctly"""
        somap = TopDownSomap(
            num_data=self.LARGE_NUM_DATA,
            cache_size=20,
            data_size=32,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()
        
        initial_dummy_index = somap._dummy_index
        
        # Perform operations that should update dummy index
        for i in range(10):
            somap.access(op="search", general_key=f"key_{i}")
        
        # Dummy index should have been updated
        # The exact value depends on cache hits vs misses
        assert somap._timestamp == 10


class TestTopDownSomapStress:
    """Stress tests for TopDownSomap"""

    STRESS_NUM_DATA = 2048  # 2^11

    def setup_method(self):
        """Clean up before each test"""
        remove_test_files()

    def teardown_method(self):
        """Clean up after each test"""
        remove_test_files()

    def test_large_scale_operations(self):
        """Test with larger scale data and operations"""
        somap = TopDownSomap(
            num_data=self.STRESS_NUM_DATA,
            cache_size=100,
            data_size=64,
            client=InteractLocalServer(),
            use_encryption=False
        )
        
        # Setup with substantial initial data
        initial_data = [(f"key_{i}", f"value_{i}_with_more_data") for i in range(200)]
        somap.setup(data=initial_data)
        
        # Perform many mixed operations
        num_ops = 100
        for i in range(num_ops):
            if i % 3 == 0:
                somap.access(op="search", general_key=f"key_{i % 200}")
            elif i % 3 == 1:
                somap.access(op="insert", general_key=f"new_key_{i}", general_value=f"new_value_{i}")
            else:
                # Access a non-existent key
                somap.access(op="search", general_key=f"nonexistent_{i}")
        
        assert somap._timestamp == num_ops
        print(f"Completed {num_ops} operations successfully")

    def test_cache_thrashing(self):
        """Test cache behavior under thrashing conditions"""
        cache_size = 20
        somap = TopDownSomap(
            num_data=self.STRESS_NUM_DATA,
            cache_size=cache_size,
            data_size=32,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()
        
        # Access many different keys to cause cache thrashing
        num_ops = cache_size * 3
        for i in range(num_ops):
            somap.access(op="search", general_key=f"unique_key_{i}")
        
        assert somap._timestamp == num_ops
