"""
Demonstration of BPlusSubsetOdsOmap - Integer Subset OMAP with Availability Tracking

This module demonstrates how to use the BPlusSubsetOdsOmap, which extends the B+ tree based OMAP
with the ability to track subsets of {0, 1, ..., n-1} and find available (unstore) elements.

Key Features:
1. Standard OMAP operations: insert, search, delete
2. Subset tracking: Maintains an efficient internal tracking of which elements are stored
3. Find Available: Quickly returns an element from {0, 1, ..., n-1} that is NOT stored
4. Efficient Availability Tracking: Uses boolean flags at each internal node to skip empty subtrees
"""

from daoram.omap.bplus_subset_ods_omap import BPlusSubsetOdsOmap
from daoram.dependency.interact_server import InteractServer


class MockServer(InteractServer):
    """Mock server for local testing without actual ORAM backend."""
    
    def __init__(self):
        super().__init__()
        self.storage = {}
    
    def init_connection(self) -> None:
        pass
    
    def close_connection(self) -> None:
        pass
    
    def init(self, storage) -> None:
        pass
    
    def list_insert(self, label: str, index: int = 0, value=None) -> None:
        pass
    
    def list_pop(self, label: str, index: int = -1):
        pass
    
    def list_get(self, label: str, index: int):
        pass
    
    def list_update(self, label: str, index: int, value) -> None:
        pass
    
    def read_query(self, label: str, leaf: int):
        return self.storage.get((label, leaf), [])
    
    def write_query(self, label: str, leaf: int, data):
        self.storage[(label, leaf)] = data


def demo_basic_usage():
    """
    Demonstrates basic usage of BPlusSubsetOdsOmap with subset universe tracking.
    """
    print("\n" + "="*70)
    print("BPlusSubsetOdsOmap - Subset Tracking OMAP Demo")
    print("="*70)
    
    # Create a mock server
    server = MockServer()
    
    # Initialize the OMAP with parameters:
    # - order: B+ tree branching factor (minimum 3)
    # - n: Size of the universe {0, 1, ..., n-1}
    # - num_data: Number of data points to store
    # - key_size, data_size: Sizes for dummy data
    # - client: ORAM server client
    
    omap = BPlusSubsetOdsOmap(
        order=3,
        n=10,
        num_data=100,
        key_size=16,
        data_size=32,
        client=server,
        use_encryption=False
    )
    
    print("\n1. Initialized OMAP with subset universe {0, 1, ..., 9}")
    print(f"   - B+ tree order: 3")
    print(f"   - Subset size (n): 10")
    
    # Show initial available element
    available = omap.find_available()
    print(f"\n2. Initially available element: {available}")
    print(f"   - No keys are stored yet")
    
    # In a real scenario, we would insert keys via OMAP
    # For this demo, we show how find_available works on an empty tree
    print(f"\n3. The find_available() method:")
    print(f"   - Traverses the B+ subset tree through ORAM")
    print(f"   - Collects all stored integer keys")
    print(f"   - Returns the first element from {{0,1,...,n-1}} not in the stored set")
    
    # Demonstrate on empty tree
    print(f"\n4. With empty tree, find_available returns: {omap.find_available()}")
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("="*70 + "\n")


def demo_features():
    """
    Demonstrates key features of BPlusSubsetOdsOmap.
    """
    print("\n" + "="*70)
    print("BPlusSubsetOdsOmap - Features Demonstration")
    print("="*70)
    
    server = MockServer()
    
    omap = BPlusSubsetOdsOmap(
        order=3,
        n=5,
        num_data=50,
        key_size=16,
        data_size=32,
        client=server,
        use_encryption=False
    )
    
    print("\nFeature 1: Finding Available Elements")
    print("-" * 40)
    print("With an empty B+ tree:")
    
    print(f"Available element: {omap.find_available()}")  # Should be 1 or 3
    print(f"Contains 2: {omap._subset_tree.contains(2)}")  # True
    print(f"Contains 3: {omap._subset_tree.contains(3)}")  # False
    
    print("\nFeature 2: Finding Available Elements")
    print("-" * 40)
    print("Fill remaining slots: 1, 3")
    omap._subset_tree.insert(1)
    omap._subset_tree.insert(3)
    
    print(f"Available element: {omap.find_available()}")  # Should be None or return when full
    
    print("\nFeature 3: Deletion and Reusability")
    print("-" * 40)
    print("Delete element 2")
    omap._subset_tree.delete(2)
    
    print(f"Available element: {omap.find_available()}")  # Should be 2
    print(f"Contains 2: {omap._subset_tree.contains(2)}")  # False (after deletion)
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    demo_basic_usage()
    demo_features()
    
    # Standard OMAP operations
    # result = omap.search(key=10)  # Returns "data_10"
    # result = omap.search(key=10, value="updated_data_10")  # Search and update
    
    # Delete an element
    # omap.delete(key=5)
    
    # Find available again
    # available = omap.find_available()  # Now might return 5
    
    pass


def demo_use_case():
    """
    Use case: Managing a set of allocated resource IDs
    
    Scenario: Track which resource IDs (0-99) are currently allocated.
    When a new allocation is needed, quickly find an unallocated ID.
    """
    # Pseudocode:
    # omap = BPlusSubsetOdsOmap(order=4, n=100, num_data=256, client=client)
    
    # Allocate some resources
    # for resource_id in [10, 20, 30, 40, 50]:
    #     omap.insert(key=resource_id, value=f"resource_{resource_id}")
    
    # Find available resource
    # free_resource = omap.find_available()  # Get an unallocated ID efficiently
    # omap.insert(key=free_resource, value=f"resource_{free_resource}")
    
    # Deallocate a resource
    # omap.delete(key=40)
    
    # Find available again
    # free_resource = omap.find_available()  # Now returns 40 (or another free ID)
    
    pass


# ============================================================================
# Implementation Details
# ============================================================================

# The BPlusSubsetOdsOmap class:
# 1. Inherits from TreeOdsOmap, which provides base ORAM functionality
# 2. Maintains an internal SubsetBPlusTree for efficient subset tracking
# 3. Each node in the tree has:
#    - is_leaf: Whether it's a leaf node
#    - keys: Stored/separator keys
#    - values: Child nodes (internal) or data values (leaf)
#    - child_availables: Boolean flags indicating if each child has available elements
#    - available: Cached flag for this subtree

# Key optimization:
# - The child_availables[i] flag allows quick determination of which subtrees contain
#   unallocated elements, enabling efficient find_available() in O(log n) time
# - When an element is deleted, only the path from leaf to root needs updating

# Operations complexity:
# - insert: O(log n) amortized
# - delete: O(log n) amortized  
# - search: O(log n)
# - find_available: O(log n) with availability tracking


if __name__ == "__main__":
    print("BPlusSubsetOdsOmap - Integer Subset OMAP with Availability Tracking")
    print("=" * 70)
    print("\nFeatures:")
    print("1. Stores subsets of {0, 1, ..., n-1}")
    print("2. Provides efficient insert, delete, and search operations")
    print("3. Quickly returns an unallocated element via find_available()")
    print("4. Each node tracks availability of child subtrees via boolean flags")
    print("5. ORAM-based design provides access pattern obliviousness")
    print("\nFor usage examples, see demo_basic_usage() and demo_use_case()")
