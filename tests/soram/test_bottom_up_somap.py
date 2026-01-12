"""
Enhanced Test Suite for BottomUpSomap

This test suite provides comprehensive testing for the BottomUpSomap implementation,
covering all major functionality including initialization, operations, cache management,
edge cases, and performance characteristics.
"""
import glob
import os
import random
import sys
import time

import pytest

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from daoram.dependency import InteractLocalServer
from daoram.so.bottom_to_up_somap import BottomUpSomap

# Test Configuration
TEST_CONFIG = {
    "small": {"num_data": 50, "cache_size": 5, "data_size": 16},
    "medium": {"num_data": pow(2, 8), "cache_size": 10, "data_size": 20},
    "large": {"num_data": pow(2, 10), "cache_size": 20, "data_size": 32}
}


# Helper function to remove files generated during testing
def remove_test_files():
    """Remove all test files created during testing"""
    for file in glob.glob("*.bin"):
        try:
            os.remove(file)
        except OSError:
            pass  # File might not exist


class TestBottomUpSomap:
    """Comprehensive test class for BottomUpSomap functionality"""

    def setup_method(self):
        """Clean up before each test"""
        remove_test_files()

    def teardown_method(self):
        """Clean up after each test"""
        remove_test_files()

    def test_access_method_operations(self):
        """Test the unified access method for all operations"""
        somap = BottomUpSomap(
            num_data=100,
            cache_size=10,
            data_size=20,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()

        # Test insert operation
        result = somap.access(key=0, op="insert", value="test_value")
        assert result is None  # No previous value

        # Test read operation
        result = somap.access(key=0, op="read")
        assert result == "test_value"

        # Test update operation
        result = somap.access(key=0, op="update", value="updated_value")
        assert result == "test_value"  # Should return old value

        # Test read after update
        result = somap.access(key=0, op="read")
        assert result == "updated_value"

    def test_concurrent_operations_on_multiple_keys(self):
        """Test operations on multiple keys simultaneously"""
        somap = BottomUpSomap(
            num_data=100,
            cache_size=10,
            data_size=20,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()

        # Insert multiple keys
        for i in range(20):
            somap.access(key=i, op="insert", value=f"value_{i}")

        # Verify all inserted keys
        for i in range(20):
            result = somap.access(key=i, op="read")
            assert result == f"value_{i}"

        # Update multiple keys
        for i in range(20):
            somap.access(key=i, op="update", value=f"updated_{i}")

        # Verify all updates
        for i in range(20):
            result = somap.access(key=i, op="read")
            assert result == f"updated_{i}"

    def test_cache_management_and_security_adjustment(self):
        """Test cache overflow behavior and security level adjustment"""
        small_cache_size = 5
        somap = BottomUpSomap(
            num_data=50,
            cache_size=small_cache_size,
            data_size=20,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()

        # Insert more items than cache size to trigger security adjustment
        num_insertions = small_cache_size + 3
        for i in range(num_insertions):
            somap.access(key=i, op="insert", value=f"value_{i}")

        # Verify all items are still accessible
        for i in range(num_insertions):
            result = somap.access(key=i, op="read")
            assert result == f"value_{i}"

        # Verify cache size management
        assert somap._Qw_len == somap._cache_size

    def test_dynamic_cache_size_adjustment(self):
        """Test dynamic adjustment of cache size"""
        somap = BottomUpSomap(
            num_data=100,
            cache_size=10,
            data_size=20,
            client=InteractLocalServer(),
            use_encryption=False
        )
        data_map = {}
        for i in range(100):
            data_map[i] = i
        somap.setup(data_map=data_map)

        # Insert some data
        for i in range(15):
            somap.access(key=i, op="insert", value=f"value_{i}")

        # Adjust cache size to smaller value
        new_cache_size = 5
        somap.adjust_cache_size(new_cache_size)
        assert somap._cache_size == new_cache_size

        # Verify data is still accessible
        for i in range(15):
            result = somap.access(key=i, op="read")
            assert result == f"value_{i}"

        # Adjust cache size to larger value
        larger_cache_size = 15
        somap.adjust_cache_size(larger_cache_size)
        assert somap._cache_size == larger_cache_size

        # Insert more data and verify
        for i in range(15, 20):
            somap.access(key=i, op="insert", value=f"value_{i}")
            result = somap.access(key=i, op="read")
            assert result == f"value_{i}"

    def test_timestamp_management(self):
        """Test timestamp incrementation with operations"""
        somap = BottomUpSomap(
            num_data=100,
            cache_size=10,
            data_size=20,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()

        initial_timestamp = somap._timestamp

        # Perform operations and verify timestamp increments
        operations = 10
        for i in range(operations):
            somap.access(key=i, op="insert", value=f"value_{i}")
            assert somap._timestamp == initial_timestamp + i + 1

    def test_edge_cases_and_boundary_conditions(self):
        """Test edge cases and boundary conditions"""
        somap = BottomUpSomap(
            num_data=100,
            cache_size=10,
            data_size=20,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()

        # Test operations on key 0
        somap.access(key=0, op="insert", value="zero")
        assert somap.access(key=0, op="read") == "zero"

        # Test operations on maximum key
        max_key = 99  # num_data - 1
        somap.access(key=max_key, op="insert", value="max")
        assert somap.access(key=max_key, op="read") == "max"

        # Test operations on middle key
        middle_key = 50
        somap.access(key=middle_key, op="insert", value="middle")
        assert somap.access(key=middle_key, op="read") == "middle"

        # Test operations on non-existent keys
        result = somap.access(key=200, op="read")  # Key beyond range
        assert result is None

    def test_encryption_functionality(self):
        """Test BottomUpSomap with encryption enabled"""
        somap = BottomUpSomap(
            num_data=100,
            cache_size=10,
            data_size=20,
            client=InteractLocalServer(),
            use_encryption=True
        )
        somap.setup()

        # Perform operations with encryption
        for i in range(10):
            somap.access(key=i, op="insert", value=f"encrypted_value_{i}")

        # Verify operations work correctly with encryption
        for i in range(10):
            result = somap.access(key=i, op="read")
            assert result == f"encrypted_value_{i}"

    def test_performance_characteristics(self):
        """Test basic performance characteristics"""
        somap = BottomUpSomap(
            num_data=100,
            cache_size=10,
            data_size=20,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()

        # Time insertion operations
        start_time = time.time()
        for i in range(50):
            somap.access(key=i, op="insert", value=f"value_{i}")
        insert_time = time.time() - start_time

        # Time read operations
        start_time = time.time()
        for i in range(50):
            somap.access(key=i, op="read")
        read_time = time.time() - start_time

        # Basic performance assertions (these are just sanity checks)
        assert insert_time < 10.0  # Should complete in reasonable time
        assert read_time < 10.0  # Should complete in reasonable time

    def test_error_handling_and_robustness(self):
        """Test error handling and robustness"""
        somap = BottomUpSomap(
            num_data=100,
            cache_size=10,
            data_size=20,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()

        # Test operations with invalid operation types
        # Should handle gracefully (implementation dependent)
        try:
            somap.access(key=0, op="invalid_op", value="test")
            # If no exception, that's acceptable for this implementation
        except Exception as e:
            # If exception is raised, it should be a known type
            assert isinstance(e, (ValueError, TypeError))

        # Test operations with None values
        somap.access(key=1, op="insert", value=None)
        result = somap.access(key=1, op="read")
        assert result is None

    def test_consecutive_operations_same_key(self):
        """Test multiple consecutive operations on the same key"""
        somap = BottomUpSomap(
            num_data=100,
            cache_size=10,
            data_size=20,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()

        key = 42

        # Insert
        somap.access(key=key, op="insert", value="first_value")
        assert somap.access(key=key, op="read") == "first_value"

        # Update multiple times
        somap.access(key=key, op="update", value="second_value")
        assert somap.access(key=key, op="read") == "second_value"

        somap.access(key=key, op="update", value="third_value")
        assert somap.access(key=key, op="read") == "third_value"

        # Final update
        somap.access(key=key, op="update", value="final_value")
        assert somap.access(key=key, op="read") == "final_value"

    def test_randomized_operation_sequence(self):
        """Test with randomized operation sequences"""
        somap = BottomUpSomap(
            num_data=100,
            cache_size=10,
            data_size=20,
            client=InteractLocalServer(),
            use_encryption=False
        )

        data_map = {}
        for i in range(100):
            data_map[i] = i
        somap.setup(data_map=data_map)

        # Initialize with some data
        for i in range(20):
            somap.access(key=i, op="insert", value=f"initial_{i}")

        # Perform random operations
        operations = 100
        test_data = {i: f"initial_{i}" for i in range(20)}

        for _ in range(operations):
            key = random.randint(0, 19)
            operation = random.choice(["read", "write"])

            if operation == "read":
                result = somap.access(key=key, op="read")
                assert result == test_data[key]
            else:  # update
                new_value = f"updated_{key}_{random.randint(1000, 9999)}"
                old_value = somap.access(key=key, op="write", value=new_value)
                assert old_value == test_data[key]
                test_data[key] = new_value

    @pytest.mark.slow
    def test_large_scale_operations(self):
        """Test with larger scale operations (marked as slow)"""
        large_config = TEST_CONFIG["large"]
        somap = BottomUpSomap(
            num_data=large_config["num_data"],
            cache_size=large_config["cache_size"],
            data_size=large_config["data_size"],
            client=InteractLocalServer(),
            use_encryption=False
        )
        data_map = {}
        for i in range(large_config["num_data"]):
            data_map[i] = i
        somap.setup(data_map=data_map)

        # Insert large number of items
        for i in range(large_config["num_data"]):
            somap.access(key=i, op="insert", value=f"large_scale_value_{i}")

        # Verify all items
        for i in range(large_config["num_data"]):
            result = somap.access(key=i, op="read")
            assert result == f"large_scale_value_{i}"


def test_standalone_functionality():
    """Test standalone functionality without class structure"""
    remove_test_files()

    try:
        somap = BottomUpSomap(
            num_data=50,
            cache_size=5,
            data_size=16,
            client=InteractLocalServer(),
            use_encryption=False
        )
        somap.setup()

        # Basic operations
        somap.access(key=0, op="insert", value="standalone_test")
        result = somap.access(key=0, op="read")
        assert result == "standalone_test"

    finally:
        remove_test_files()


if __name__ == "__main__":
    # Run the tests if executed directly
    pytest.main([__file__, "-v"])
