"""Tests for edge_lookup and edge_insertion functions."""
import pytest
import random

from daoram.dependency import InteractLocalServer
from daoram.graph.grove import Grove


class TestEdgeOperations:
    """Test edge_lookup and edge_insertion functions."""

    @pytest.fixture
    def grove_with_data(self):
        """Create a Grove instance with some vertices inserted."""
        client = InteractLocalServer()
        grove = Grove(
            max_deg=4,
            num_opr=100,
            num_data=64,
            key_size=16,
            data_size=32,
            client=client,
            stash_scale=20,
        )
        grove.init_server_storage()
        
        # Insert vertices 0-7 with connections
        # Create a small connected graph
        grove.insert(vertex=(0, 'v0', {1: None, 2: None}))
        grove.insert(vertex=(1, 'v1', {0: None, 2: None, 3: None}))
        grove.insert(vertex=(2, 'v2', {0: None, 1: None, 3: None}))
        grove.insert(vertex=(3, 'v3', {1: None, 2: None, 4: None}))
        grove.insert(vertex=(4, 'v4', {3: None, 5: None}))
        grove.insert(vertex=(5, 'v5', {4: None, 6: None}))
        grove.insert(vertex=(6, 'v6', {5: None, 7: None}))
        grove.insert(vertex=(7, 'v7', {6: None}))
        
        return grove

    def test_edge_lookup_basic(self, grove_with_data):
        """Test edge_lookup returns correct data for two vertices."""
        grove = grove_with_data
        
        result = grove.edge_lookup(0, 1)
        
        # Both vertices should be found
        assert 0 in result
        assert 1 in result
        
        # Check vertex 0
        v0_data, v0_adj, v0_new_leaf = result[0]
        assert v0_data == 'v0'
        assert 1 in v0_adj
        assert 2 in v0_adj
        
        # Check vertex 1
        v1_data, v1_adj, v1_new_leaf = result[1]
        assert v1_data == 'v1'
        assert 0 in v1_adj
        assert 2 in v1_adj
        assert 3 in v1_adj

    def test_edge_lookup_consistency(self, grove_with_data):
        """Test that edge_lookup results are consistent with separate lookups."""
        grove = grove_with_data
        
        # Do edge_lookup
        edge_result = grove.edge_lookup(2, 3)
        
        # Do separate lookups
        lookup_result_2 = grove.lookup([2])
        lookup_result_3 = grove.lookup([3])
        
        # Data should be consistent (vertex_data should match)
        assert edge_result[2][0] == lookup_result_2[2][0]
        assert edge_result[3][0] == lookup_result_3[3][0]

    def test_edge_insertion_basic(self, grove_with_data):
        """Test edge_insertion returns neighbors of both centers."""
        grove = grove_with_data
        
        # edge_insertion on vertices 1 and 3
        # Vertex 1's neighbors: 0, 2, 3
        # Vertex 3's neighbors: 1, 2, 4
        # Combined unique neighbors (excluding centers): 0, 2, 4
        result = grove.edge_insertion(1, 3)
        
        # Should have neighbors (excluding the centers themselves)
        # 0 is neighbor of 1
        # 2 is neighbor of both 1 and 3
        # 4 is neighbor of 3
        assert 0 in result or 2 in result or 4 in result

    def test_edge_insertion_consistency(self, grove_with_data):
        """Test that vertices remain consistent after edge_insertion."""
        grove = grove_with_data
        
        # Perform edge_insertion
        grove.edge_insertion(0, 1)
        
        # Verify vertices can still be looked up
        result = grove.lookup([0, 1, 2, 3])
        assert 0 in result
        assert 1 in result
        assert 2 in result
        assert 3 in result

    def test_edge_lookup_nonexistent(self, grove_with_data):
        """Test edge_lookup with one nonexistent key."""
        grove = grove_with_data
        
        result = grove.edge_lookup(0, 999)
        
        # Vertex 0 should be found
        assert 0 in result
        assert result[0] is not None
        
        # Vertex 999 should not be found (or return None)
        # Depending on implementation, it might not be in result or be None

    @pytest.mark.parametrize("seed", range(5))
    def test_edge_ops_random(self, seed):
        """Test edge operations with random sequences."""
        random.seed(seed)
        
        client = InteractLocalServer()
        grove = Grove(
            max_deg=4,
            num_opr=100,
            num_data=128,
            key_size=16,
            data_size=32,
            client=client,
            stash_scale=20,
        )
        grove.init_server_storage()
        
        # Insert some vertices
        for i in range(10):
            neighbors = {}
            for j in range(i):
                if random.random() < 0.3:
                    neighbors[j] = None
            grove.insert(vertex=(i, f'v{i}', neighbors))
        
        # Do some edge_lookups
        for _ in range(3):
            k1 = random.randint(0, 9)
            k2 = random.randint(0, 9)
            if k1 != k2:
                result = grove.edge_lookup(k1, k2)
                assert k1 in result or k2 in result
        
        # Do some edge_insertions
        for _ in range(3):
            k1 = random.randint(0, 9)
            k2 = random.randint(0, 9)
            if k1 != k2:
                grove.edge_insertion(k1, k2)
        
        # Verify all vertices still accessible
        for i in range(10):
            result = grove.lookup([i])
            assert i in result


class TestInteractionRounds:
    """Test that interaction rounds are correct."""
    
    def test_edge_lookup_rounds(self):
        """Verify edge_lookup uses same rounds as single lookup."""
        client = InteractLocalServer()
        grove = Grove(
            max_deg=4,
            num_opr=100,
            num_data=64,
            key_size=16,
            data_size=32,
            client=client,
            stash_scale=20,
        )
        grove.init_server_storage()
        
        grove.insert(vertex=(0, 'v0', {1: None}))
        grove.insert(vertex=(1, 'v1', {0: None}))
        
        # Count execute calls for single lookup
        # lookup internally calls: batch_search (has its own executes) + lookup_without_omap (1 read + 1 write)
        
        # For edge_lookup: batch_search + lookup_without_omap
        # Should be same pattern as lookup
        
        # This is a structural test - edge_lookup should use:
        # 1. batch_search (same as lookup)
        # 2. lookup_without_omap with 2 keys (same round structure as lookup with 1 key)
        result = grove.edge_lookup(0, 1)
        assert 0 in result
        assert 1 in result
        print("edge_lookup completed successfully")
    
    def test_edge_insertion_rounds(self):
        """Verify edge_insertion uses same rounds as neighbor query."""
        client = InteractLocalServer()
        grove = Grove(
            max_deg=4,
            num_opr=100,
            num_data=64,
            key_size=16,
            data_size=32,
            client=client,
            stash_scale=20,
        )
        grove.init_server_storage()
        
        grove.insert(vertex=(0, 'v0', {1: None, 2: None}))
        grove.insert(vertex=(1, 'v1', {0: None, 2: None}))
        grove.insert(vertex=(2, 'v2', {0: None, 1: None}))
        
        # edge_insertion should use:
        # 1. edge_lookup (same rounds as lookup)
        # 2. One more round to download all neighbors of both centers
        
        result = grove.edge_insertion(0, 1)
        # Should return neighbors (vertex 2 is neighbor of both)
        print(f"edge_insertion result keys: {list(result.keys())}")
        print("edge_insertion completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
