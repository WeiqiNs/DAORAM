"""
Complex tests for Grove insert and lookup operations.
"""
import pytest
import secrets
import random
from daoram.graph.grove import Grove
from daoram.dependency import InteractLocalServer


def create_grove(num_data=256, max_deg=5, stash_scale=20):
    """Helper to create Grove instance with consistent parameters."""
    client = InteractLocalServer()
    grove = Grove(
        max_deg=max_deg,
        num_opr=100,
        num_data=num_data,
        key_size=16,
        data_size=64,
        client=client,
        encryptor=None,  # No encryption for efficiency
        stash_scale=stash_scale
    )
    # Initialize storage
    grove._pos_omap.init_server_storage()
    grove._graph_oram.init_server_storage()
    grove._graph_meta.init_server_storage()
    grove._pos_meta.init_server_storage()
    return grove


class TestGroveComplex:
    """Complex integration tests for Grove."""

    def test_insert_then_lookup(self):
        """Insert vertices one by one, then lookup to verify."""
        grove = create_grove(num_data=256, max_deg=5)
        
        # Insert 10 isolated vertices first
        for i in range(10):
            grove.insert((i, f"data_{i}", {}))
        
        # Lookup each vertex to verify
        for i in range(10):
            result = grove.lookup([i])
            assert i in result, f"Vertex {i} not found after insert"
            assert result[i][0] == f"data_{i}", f"Wrong data for vertex {i}"
    
    def test_insert_with_connections(self):
        """Insert vertices with connections and verify adjacency."""
        grove = create_grove(num_data=256, max_deg=5)
        
        # Insert vertex 0 (isolated)
        grove.insert((0, "node_0", {}))
        
        # Insert vertex 1 connected to 0
        grove.insert((1, "node_1", {0: None}))
        
        # Insert vertex 2 connected to both 0 and 1
        grove.insert((2, "node_2", {0: None, 1: None}))
        
        # Lookup vertex 0 and check neighbors
        result = grove.lookup([0])
        assert 0 in result
        adj_0 = result[0][1]
        assert 1 in adj_0, "Vertex 0 should have neighbor 1"
        assert 2 in adj_0, "Vertex 0 should have neighbor 2"
        
        # Lookup vertex 1 and check neighbors
        result = grove.lookup([1])
        assert 1 in result
        adj_1 = result[1][1]
        assert 0 in adj_1, "Vertex 1 should have neighbor 0"
        assert 2 in adj_1, "Vertex 1 should have neighbor 2"
        
        # Lookup vertex 2 and check neighbors
        result = grove.lookup([2])
        assert 2 in result
        adj_2 = result[2][1]
        assert 0 in adj_2, "Vertex 2 should have neighbor 0"
        assert 1 in adj_2, "Vertex 2 should have neighbor 1"
    
    def test_star_topology(self):
        """Test star topology: one central node connected to many."""
        grove = create_grove(num_data=256, max_deg=10)
        
        # Insert central node
        grove.insert((0, "center", {}))
        
        # Insert 8 nodes all connected to center
        for i in range(1, 9):
            grove.insert((i, f"leaf_{i}", {0: None}))
        
        # Verify center has all neighbors
        result = grove.lookup([0])
        assert 0 in result
        adj = result[0][1]
        for i in range(1, 9):
            assert i in adj, f"Center should have neighbor {i}"
        
        # Verify each leaf has center as neighbor
        for i in range(1, 9):
            result = grove.lookup([i])
            assert i in result
            assert 0 in result[i][1], f"Leaf {i} should have center as neighbor"
    
    def test_chain_topology(self):
        """Test chain topology: 0-1-2-3-4-5..."""
        grove = create_grove(num_data=256, max_deg=5)
        
        # Insert first node
        grove.insert((0, "node_0", {}))
        
        # Insert remaining nodes in chain
        for i in range(1, 10):
            grove.insert((i, f"node_{i}", {i-1: None}))
        
        # Verify chain structure
        # Node 0 should only have neighbor 1
        result = grove.lookup([0])
        assert 1 in result[0][1]
        
        # Middle nodes should have two neighbors
        for i in range(1, 9):
            result = grove.lookup([i])
            assert i-1 in result[i][1], f"Node {i} should have neighbor {i-1}"
            assert i+1 in result[i][1], f"Node {i} should have neighbor {i+1}"
        
        # Last node should only have neighbor 8
        result = grove.lookup([9])
        assert 8 in result[9][1]
    
    def test_clique(self):
        """Test fully connected clique."""
        grove = create_grove(num_data=256, max_deg=10)
        
        # Insert 5 nodes forming a clique
        grove.insert((0, "node_0", {}))
        grove.insert((1, "node_1", {0: None}))
        grove.insert((2, "node_2", {0: None, 1: None}))
        grove.insert((3, "node_3", {0: None, 1: None, 2: None}))
        grove.insert((4, "node_4", {0: None, 1: None, 2: None, 3: None}))
        
        # Verify each node has all other nodes as neighbors
        for i in range(5):
            result = grove.lookup([i])
            assert i in result
            adj = result[i][1]
            for j in range(5):
                if i != j:
                    assert j in adj, f"Node {i} should have neighbor {j}"
    
    def test_batch_insert_then_batch_lookup(self):
        """Insert many vertices then do batch lookups."""
        grove = create_grove(num_data=1024, max_deg=5)
        
        # Insert 50 isolated vertices
        for i in range(50):
            grove.insert((i, f"data_{i}", {}))
        
        # Batch lookup 10 at a time
        for batch_start in range(0, 50, 10):
            keys = list(range(batch_start, batch_start + 10))
            result = grove.lookup(keys)
            for k in keys:
                assert k in result, f"Vertex {k} not found"
                assert result[k][0] == f"data_{k}"
    
    def test_repeated_insert_lookup_cycles(self):
        """Alternate between insert and lookup operations."""
        grove = create_grove(num_data=256, max_deg=5)
        
        for cycle in range(5):
            # Insert 5 vertices
            for i in range(cycle * 5, (cycle + 1) * 5):
                neighbors = {j: None for j in range(max(0, i - 2), i)}  # Connect to previous 2
                grove.insert((i, f"data_{i}", neighbors))
            
            # Lookup all vertices inserted so far
            all_keys = list(range((cycle + 1) * 5))
            result = grove.lookup(all_keys)
            for k in all_keys:
                assert k in result, f"Vertex {k} not found in cycle {cycle}"
    
    def test_high_degree_stress(self):
        """Test with max_deg neighbors."""
        max_deg = 8
        grove = create_grove(num_data=1024, max_deg=max_deg)
        
        # Insert initial nodes
        for i in range(max_deg):
            grove.insert((i, f"node_{i}", {}))
        
        # Insert a hub connected to all initial nodes
        hub_neighbors = {i: None for i in range(max_deg)}
        grove.insert((100, "hub", hub_neighbors))
        
        # Verify hub has all neighbors
        result = grove.lookup([100])
        assert 100 in result
        adj = result[100][1]
        for i in range(max_deg):
            assert i in adj, f"Hub should have neighbor {i}"
        
        # Verify each initial node has hub as neighbor
        for i in range(max_deg):
            result = grove.lookup([i])
            assert 100 in result[i][1], f"Node {i} should have hub as neighbor"
    
    def test_random_graph_structure(self):
        """Build a random graph and verify consistency."""
        grove = create_grove(num_data=1024, max_deg=6)
        
        n_vertices = 30
        expected_adj = {i: set() for i in range(n_vertices)}
        
        # Insert vertices with random connections
        for i in range(n_vertices):
            # Connect to random subset of existing vertices
            if i > 0:
                n_neighbors = min(i, secrets.randbelow(4) + 1)  # 1 to 4 neighbors
                neighbors = set()
                while len(neighbors) < n_neighbors:
                    neighbor = secrets.randbelow(i)
                    neighbors.add(neighbor)
                neighbor_dict = {n: None for n in neighbors}
            else:
                neighbor_dict = {}
                neighbors = set()
            
            grove.insert((i, f"vertex_{i}", neighbor_dict))
            
            # Update expected adjacency
            for n in neighbors:
                expected_adj[i].add(n)
                expected_adj[n].add(i)
        
        # Verify all adjacencies
        for i in range(n_vertices):
            result = grove.lookup([i])
            assert i in result, f"Vertex {i} not found"
            actual_neighbors = set(result[i][1].keys())
            assert actual_neighbors == expected_adj[i], \
                f"Vertex {i}: expected neighbors {expected_adj[i]}, got {actual_neighbors}"


class TestGroveStress:
    """Stress tests for Grove."""
    
    def test_many_lookups_after_inserts(self):
        """Insert many vertices, then do many lookups."""
        grove = create_grove(num_data=1024, max_deg=5)
        
        # Insert 100 vertices
        for i in range(100):
            neighbors = {j: None for j in range(max(0, i - 2), i)}
            grove.insert((i, f"v{i}", neighbors))
        
        # Do 200 random lookups
        success = 0
        for _ in range(200):
            key = secrets.randbelow(100)
            result = grove.lookup([key])
            if key in result and result[key][0] == f"v{key}":
                success += 1
        
        assert success == 200, f"Only {success}/200 lookups succeeded"
    
    def test_interleaved_operations(self):
        """Interleave inserts and lookups randomly."""
        grove = create_grove(num_data=1024, max_deg=5)
        
        inserted = set()
        
        for _ in range(100):
            op = secrets.randbelow(3)  # 0: insert, 1-2: lookup
            
            if op == 0 or len(inserted) == 0:
                # Insert
                new_key = len(inserted)
                neighbors = {}
                if inserted:
                    # Connect to 1-2 random existing vertices
                    n_neighbors = min(len(inserted), secrets.randbelow(2) + 1)
                    neighbor_keys = list(inserted)
                    random.shuffle(neighbor_keys)
                    neighbors = {k: None for k in neighbor_keys[:n_neighbors]}
                
                grove.insert((new_key, f"data_{new_key}", neighbors))
                inserted.add(new_key)
            else:
                # Lookup
                key = random.choice(list(inserted))
                result = grove.lookup([key])
                assert key in result, f"Key {key} not found"
                assert result[key][0] == f"data_{key}", f"Wrong data for key {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
