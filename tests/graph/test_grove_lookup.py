"""Large-scale lookup tests for Grove."""
import random
import secrets
import pytest
from typing import Dict, List, Tuple

from daoram.graph.grove import Grove
from daoram.dependency import Data, InteractLocalServer


class TestGroveLookupLargeScale:
    """
    Test Grove lookup operations with a large-scale graph (2^10 vertices).
    """
    
    NUM_VERTICES = 2 ** 10  # 1024 vertices
    MAX_DEGREE = 10
    
    @pytest.fixture
    def initialized_grove(self):
        """
        Initialize Grove with a pre-populated graph.
        
        Strategy:
        1. Create Grove instance
        2. Generate random graph with NUM_VERTICES vertices, each with up to MAX_DEGREE neighbors
        3. Directly populate Graph ORAM with vertex data
        4. Initialize PosMap ORAM with corresponding paths
        """
        num_data = self.NUM_VERTICES
        max_deg = self.MAX_DEGREE
        num_opr = 100
        key_size = 16
        data_size = 64
        
        client = InteractLocalServer()
        
        grove = Grove(
            max_deg=max_deg,
            num_opr=num_opr,
            num_data=num_data,
            key_size=key_size,
            data_size=data_size,
            client=client,
            encryptor=None,  # No encryption for efficiency
            stash_scale=20  # Larger stash for meta ORAMs to handle duplications
        )
        
        # Generate random graph structure
        # Each vertex has random neighbors (up to MAX_DEGREE)
        graph_data: Dict[int, Tuple[str, Dict[int, int]]] = {}
        
        for vertex_key in range(num_data):
            # Random vertex data
            vertex_data = f"vertex_{vertex_key}_data"
            
            # Random neighbors (up to MAX_DEGREE, excluding self)
            possible_neighbors = [i for i in range(num_data) if i != vertex_key]
            num_neighbors = random.randint(0, min(max_deg, len(possible_neighbors)))
            neighbors = random.sample(possible_neighbors, num_neighbors)
            
            # Initially, adjacency dict will have placeholder graph_leaf values
            # We'll update them after assigning actual graph_leaf to each vertex
            graph_data[vertex_key] = (vertex_data, neighbors)
        
        # Assign random graph_leaf for each vertex
        leaf_range = grove._leaf_range
        vertex_graph_leaf: Dict[int, int] = {}
        for vertex_key in range(num_data):
            vertex_graph_leaf[vertex_key] = secrets.randbelow(leaf_range)
        
        # Assign random pos_leaf for each vertex (for PosMap ORAM)
        vertex_pos_leaf: Dict[int, int] = {}
        for vertex_key in range(num_data):
            vertex_pos_leaf[vertex_key] = secrets.randbelow(leaf_range)
        
        # Build final vertex data with proper adjacency dict
        # Format: Data(key=vertex_key, leaf=graph_leaf, value=(vertex_data, adjacency_dict, pos_leaf))
        graph_oram_data: List[Data] = []
        for vertex_key in range(num_data):
            vertex_data, neighbors = graph_data[vertex_key]
            
            # Build adjacency dict: {neighbor_key: neighbor_graph_leaf}
            adjacency_dict = {n: vertex_graph_leaf[n] for n in neighbors}
            
            # Get this vertex's pos_leaf
            pos_leaf = vertex_pos_leaf[vertex_key]
            
            graph_oram_data.append(Data(
                key=vertex_key,
                leaf=vertex_graph_leaf[vertex_key],
                value=(vertex_data, adjacency_dict, pos_leaf)
            ))
        
        # Build PosMap data: {vertex_key: graph_leaf}
        posmap_data = [(vertex_key, vertex_graph_leaf[vertex_key]) 
                       for vertex_key in range(num_data)]
        
        # Initialize PosMap ORAM with the path data
        grove._pos_omap.init_server_storage(data=posmap_data)
        
        # Initialize Graph ORAM
        # We need to build a path_map for init_server_storage
        path_map = {vertex_key: vertex_graph_leaf[vertex_key] for vertex_key in range(num_data)}
        data_map = {vertex_key: (graph_data[vertex_key][0], 
                                  {n: vertex_graph_leaf[n] for n in graph_data[vertex_key][1]},
                                  vertex_pos_leaf[vertex_key])
                    for vertex_key in range(num_data)}
        grove._graph_oram.init_server_storage(data_map=data_map, path_map=path_map)
        
        # Initialize meta ORAMs (empty, no duplications needed)
        grove._graph_meta.init_server_storage()
        grove._pos_meta.init_server_storage()
        # _pos_omap._meta is initialized automatically via init_server_storage
        
        # Store test data for verification
        grove._test_graph_data = graph_data
        grove._test_vertex_graph_leaf = vertex_graph_leaf
        grove._test_vertex_pos_leaf = vertex_pos_leaf
        
        print(f"\nInitialized Grove with {num_data} vertices, max_degree={max_deg}")
        print(f"Graph ORAM leaf_range: {leaf_range}")
        
        return grove
    
    def test_single_lookup(self, initialized_grove):
        """Test looking up a single vertex."""
        grove = initialized_grove
        
        # Pick a random vertex to lookup
        vertex_key = random.randint(0, self.NUM_VERTICES - 1)
        
        print(f"\nLooking up vertex {vertex_key}...")
        result = grove.lookup([vertex_key])
        
        assert vertex_key in result, f"Vertex {vertex_key} not found in lookup result"
        
        vertex_data, adjacency_dict = result[vertex_key]
        expected_data, expected_neighbors = grove._test_graph_data[vertex_key]
        
        assert vertex_data == expected_data, f"Vertex data mismatch for {vertex_key}"
        assert set(adjacency_dict.keys()) == set(expected_neighbors), \
            f"Adjacency mismatch for {vertex_key}: got {set(adjacency_dict.keys())}, expected {set(expected_neighbors)}"
        
        print(f"Vertex {vertex_key} lookup successful!")
        print(f"  Data: {vertex_data[:30]}...")
        print(f"  Neighbors: {list(adjacency_dict.keys())[:5]}...")
    
    def test_batch_lookup(self, initialized_grove):
        """Test looking up multiple vertices at once."""
        grove = initialized_grove
        
        # Pick random vertices to lookup
        num_lookups = 5
        vertex_keys = random.sample(range(self.NUM_VERTICES), num_lookups)
        
        print(f"\nBatch looking up vertices: {vertex_keys}...")
        result = grove.lookup(vertex_keys)
        
        for vertex_key in vertex_keys:
            assert vertex_key in result, f"Vertex {vertex_key} not found in lookup result"
            
            vertex_data, adjacency_dict = result[vertex_key]
            expected_data, expected_neighbors = grove._test_graph_data[vertex_key]
            
            assert vertex_data == expected_data, f"Vertex data mismatch for {vertex_key}"
            assert set(adjacency_dict.keys()) == set(expected_neighbors), \
                f"Adjacency mismatch for {vertex_key}"
        
        print(f"Batch lookup of {num_lookups} vertices successful!")
    
    def test_repeated_lookups(self, initialized_grove):
        """Test multiple consecutive lookups to verify consistency."""
        grove = initialized_grove
        
        num_rounds = 10
        lookups_per_round = 3
        
        print(f"\nPerforming {num_rounds} rounds of lookups...")
        
        for round_idx in range(num_rounds):
            vertex_keys = random.sample(range(self.NUM_VERTICES), lookups_per_round)
            
            result = grove.lookup(vertex_keys)
            
            for vertex_key in vertex_keys:
                assert vertex_key in result, f"Round {round_idx}: Vertex {vertex_key} not found"
                
                vertex_data, adjacency_dict = result[vertex_key]
                expected_data, expected_neighbors = grove._test_graph_data[vertex_key]
                
                assert vertex_data == expected_data, \
                    f"Round {round_idx}: Vertex data mismatch for {vertex_key}"
                assert set(adjacency_dict.keys()) == set(expected_neighbors), \
                    f"Round {round_idx}: Adjacency mismatch for {vertex_key}"
            
            if (round_idx + 1) % 5 == 0:
                print(f"  Round {round_idx + 1}/{num_rounds} completed")
        
        print(f"All {num_rounds} rounds of lookups successful!")
    
    def test_lookup_same_vertex_twice(self, initialized_grove):
        """Test looking up the same vertex multiple times."""
        grove = initialized_grove
        
        vertex_key = random.randint(0, self.NUM_VERTICES - 1)
        
        print(f"\nLooking up vertex {vertex_key} multiple times...")
        
        for i in range(5):
            result = grove.lookup([vertex_key])
            
            assert vertex_key in result, f"Iteration {i}: Vertex {vertex_key} not found"
            
            vertex_data, adjacency_dict = result[vertex_key]
            expected_data, expected_neighbors = grove._test_graph_data[vertex_key]
            
            assert vertex_data == expected_data
            assert set(adjacency_dict.keys()) == set(expected_neighbors)
        
        print(f"Vertex {vertex_key} looked up successfully 5 times!")
    
    def test_lookup_many_sequential(self, initialized_grove):
        """Test looking up many sequential vertices."""
        grove = initialized_grove
        
        # Lookup first 50 vertices sequentially
        num_lookups = 50
        
        # Store expected graph_leaf for debugging
        grove._expected_graph_leaf = grove._test_vertex_graph_leaf.copy()
        
        print(f"\nLooking up first {num_lookups} vertices sequentially...")
        
        success_count = 0
        fail_count = 0
        for vertex_key in range(num_lookups):
            result = grove.lookup([vertex_key])
            
            if vertex_key in result:
                vertex_data, adjacency_dict = result[vertex_key]
                expected_data, expected_neighbors = grove._test_graph_data[vertex_key]
                
                if (vertex_data == expected_data and 
                    set(adjacency_dict.keys()) == set(expected_neighbors)):
                    success_count += 1
                else:
                    print(f"  Data mismatch for vertex {vertex_key}")
                    fail_count += 1
            else:
                fail_count += 1
                print(f"  Vertex {vertex_key} not found")
        
        print(f"Successfully looked up {success_count}/{num_lookups} vertices ({100*success_count/num_lookups:.1f}%)")
        print(f"Failed: {fail_count}")
        assert success_count == num_lookups, f"Only {success_count}/{num_lookups} vertices looked up successfully"

    def test_high_skewed_zipf(self, initialized_grove):
        """Test lookup with Zipf distribution (high skew towards low-index vertices)."""
        grove = initialized_grove
        
        # Generate Zipf-distributed accesses
        # Zipf distribution: probability of accessing item i is proportional to 1/i^s
        # We use s=1.5 for high skew
        num_lookups = 100
        num_vertices = self.NUM_VERTICES
        
        # Generate Zipf weights
        s = 1.5  # Skew parameter (higher = more skewed)
        weights = [1.0 / (i + 1) ** s for i in range(num_vertices)]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        # Sample according to Zipf distribution
        import random
        random.seed(42)  # For reproducibility
        access_sequence = random.choices(range(num_vertices), weights=probs, k=num_lookups)
        
        # Count access frequencies
        from collections import Counter
        access_counts = Counter(access_sequence)
        print(f"\nZipf distribution test (s={s}):")
        print(f"  Total lookups: {num_lookups}")
        print(f"  Unique vertices accessed: {len(access_counts)}")
        print(f"  Top 5 most accessed: {access_counts.most_common(5)}")
        
        # Perform lookups
        success_count = 0
        fail_count = 0
        
        for i, vertex_key in enumerate(access_sequence):
            result = grove.lookup([vertex_key])
            
            if vertex_key in result:
                vertex_data, adjacency_dict = result[vertex_key]
                expected_data, expected_neighbors = grove._test_graph_data[vertex_key]
                
                if (vertex_data == expected_data and 
                    set(adjacency_dict.keys()) == set(expected_neighbors)):
                    success_count += 1
                else:
                    print(f"  Lookup {i}: Data mismatch for vertex {vertex_key}")
                    fail_count += 1
            else:
                fail_count += 1
                print(f"  Lookup {i}: Vertex {vertex_key} not found")
        
        print(f"Successfully looked up {success_count}/{num_lookups} ({100*success_count/num_lookups:.1f}%)")
        assert success_count == num_lookups, f"Only {success_count}/{num_lookups} lookups succeeded"

    def test_hotspot_access(self, initialized_grove):
        """Test lookup with hotspot pattern (90% accesses to 10% of vertices)."""
        grove = initialized_grove
        
        num_lookups = 100
        num_vertices = self.NUM_VERTICES
        hotspot_size = num_vertices // 10  # 10% of vertices are "hot"
        
        import random
        random.seed(123)
        
        # Generate access sequence: 90% to hotspot, 10% to rest
        access_sequence = []
        for _ in range(num_lookups):
            if random.random() < 0.9:
                # Access hotspot (vertices 0 to hotspot_size-1)
                access_sequence.append(random.randint(0, hotspot_size - 1))
            else:
                # Access cold vertices
                access_sequence.append(random.randint(hotspot_size, num_vertices - 1))
        
        # Count access frequencies
        from collections import Counter
        access_counts = Counter(access_sequence)
        hotspot_accesses = sum(1 for v in access_sequence if v < hotspot_size)
        print(f"\nHotspot access test:")
        print(f"  Total lookups: {num_lookups}")
        print(f"  Hotspot size: {hotspot_size} vertices (indices 0-{hotspot_size-1})")
        print(f"  Hotspot accesses: {hotspot_accesses} ({100*hotspot_accesses/num_lookups:.1f}%)")
        print(f"  Unique vertices accessed: {len(access_counts)}")
        print(f"  Top 5 most accessed: {access_counts.most_common(5)}")
        
        # Perform lookups
        success_count = 0
        fail_count = 0
        
        for i, vertex_key in enumerate(access_sequence):
            result = grove.lookup([vertex_key])
            
            if vertex_key in result:
                vertex_data, adjacency_dict = result[vertex_key]
                expected_data, expected_neighbors = grove._test_graph_data[vertex_key]
                
                if (vertex_data == expected_data and 
                    set(adjacency_dict.keys()) == set(expected_neighbors)):
                    success_count += 1
                else:
                    print(f"  Lookup {i}: Data mismatch for vertex {vertex_key}")
                    fail_count += 1
            else:
                fail_count += 1
                print(f"  Lookup {i}: Vertex {vertex_key} not found")
        
        print(f"Successfully looked up {success_count}/{num_lookups} ({100*success_count/num_lookups:.1f}%)")
        assert success_count == num_lookups, f"Only {success_count}/{num_lookups} lookups succeeded"

    def test_single_vertex_repeated(self, initialized_grove):
        """Test repeatedly accessing the same single vertex many times."""
        grove = initialized_grove
        
        # Pick a vertex and access it many times
        target_vertex = 42
        num_lookups = 50
        
        print(f"\nSingle vertex repeated access test:")
        print(f"  Target vertex: {target_vertex}")
        print(f"  Number of accesses: {num_lookups}")
        
        success_count = 0
        for i in range(num_lookups):
            result = grove.lookup([target_vertex])
            
            if target_vertex in result:
                vertex_data, adjacency_dict = result[target_vertex]
                expected_data, expected_neighbors = grove._test_graph_data[target_vertex]
                
                if (vertex_data == expected_data and 
                    set(adjacency_dict.keys()) == set(expected_neighbors)):
                    success_count += 1
                else:
                    print(f"  Access {i}: Data mismatch")
            else:
                print(f"  Access {i}: Vertex not found")
        
        print(f"Successfully looked up {success_count}/{num_lookups} ({100*success_count/num_lookups:.1f}%)")
        assert success_count == num_lookups, f"Only {success_count}/{num_lookups} lookups succeeded"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
