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
        2. Generate a random graph whose physical adjacency records appear at both endpoints
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
        
        # Grove's delayed notifications require a physical adjacency entry at
        # both endpoints. Logical direction, when needed, is represented in
        # the entry rather than by omitting the reverse physical record.
        rng = random.Random(0x47524F56)
        neighbor_sets = {vertex_key: set() for vertex_key in range(num_data)}
        for vertex_key in range(num_data):
            target_degree = rng.randint(0, max_deg)
            candidates = [
                candidate
                for candidate in range(num_data)
                if candidate != vertex_key
                and candidate not in neighbor_sets[vertex_key]
                and len(neighbor_sets[candidate]) < max_deg
            ]
            rng.shuffle(candidates)
            for neighbor in candidates[:max(0, target_degree - len(neighbor_sets[vertex_key]))]:
                neighbor_sets[vertex_key].add(neighbor)
                neighbor_sets[neighbor].add(vertex_key)

        graph_data: Dict[int, Tuple[str, List[int]]] = {
            vertex_key: (f"vertex_{vertex_key}_data", sorted(neighbor_sets[vertex_key]))
            for vertex_key in range(num_data)
        }
        
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
        
        # Build the graph maps used by Grove's public initialization boundary.
        path_map = {vertex_key: vertex_graph_leaf[vertex_key] for vertex_key in range(num_data)}
        data_map = {vertex_key: (graph_data[vertex_key][0], 
                                  {n: vertex_graph_leaf[n] for n in graph_data[vertex_key][1]},
                                  vertex_pos_leaf[vertex_key])
                    for vertex_key in range(num_data)}
        grove.init_server_storage(
            posmap_data=posmap_data,
            graph_data_map=data_map,
            graph_path_map=path_map,
        )
        
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
        
        vertex_data, adjacency_dict = result[vertex_key][:2]
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
            
            vertex_data, adjacency_dict = result[vertex_key][:2]
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
                
                vertex_data, adjacency_dict = result[vertex_key][:2]
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
            
            vertex_data, adjacency_dict = result[vertex_key][:2]
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
                vertex_data, adjacency_dict = result[vertex_key][:2]
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
                vertex_data, adjacency_dict = result[vertex_key][:2]
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
                vertex_data, adjacency_dict = result[vertex_key][:2]
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
                vertex_data, adjacency_dict = result[target_vertex][:2]
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


@pytest.mark.slow
class TestGroveMassiveLookup:
    """
    Stress test: massive lookup operations to verify dedup fix.
    Uses 2^10 vertices, performs 500+ lookups with various distributions.
    This is the key test for the dup application dedup bug fix in grove.py.
    """
    
    NUM_VERTICES = 2 ** 10  # 1024 vertices
    MAX_DEGREE = 10
    
    @pytest.fixture
    def initialized_grove(self):
        """Initialize Grove with a pre-populated circulant-like graph."""
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
            encryptor=None,
            stash_scale=20
        )
        
        # Build a circulant-like graph (each vertex connected to its d nearest neighbors)
        d = max_deg
        graph_data: Dict[int, Tuple[str, List[int]]] = {}
        for v in range(num_data):
            vertex_data = f"vertex_{v}_data"
            neighbors = []
            for offset in range(1, d // 2 + 1):
                neighbors.append((v + offset) % num_data)
                neighbors.append((v - offset) % num_data)
            # Keep only max_deg neighbors
            neighbors = neighbors[:max_deg]
            graph_data[v] = (vertex_data, neighbors)
        
        leaf_range = grove._leaf_range
        vertex_graph_leaf: Dict[int, int] = {}
        vertex_pos_leaf: Dict[int, int] = {}
        for v in range(num_data):
            vertex_graph_leaf[v] = secrets.randbelow(leaf_range)
            vertex_pos_leaf[v] = secrets.randbelow(leaf_range)
        
        # Build data structures for initialization
        data_map = {}
        path_map = {}
        for v in range(num_data):
            vertex_data, neighbors = graph_data[v]
            adjacency_dict = {n: vertex_graph_leaf[n] for n in neighbors}
            pos_leaf = vertex_pos_leaf[v]
            data_map[v] = (vertex_data, adjacency_dict, pos_leaf)
            path_map[v] = vertex_graph_leaf[v]
        
        posmap_data = [(v, vertex_graph_leaf[v]) for v in range(num_data)]
        
        grove.init_server_storage(
            posmap_data=posmap_data,
            graph_data_map=data_map,
            graph_path_map=path_map,
        )
        
        grove._test_graph_data = graph_data
        grove._test_vertex_graph_leaf = vertex_graph_leaf
        grove._test_vertex_pos_leaf = vertex_pos_leaf
        
        print(f"\n[Stress Test] Initialized Grove: {num_data} vertices, max_degree={max_deg}")
        return grove
    
    def _verify_lookup(self, grove, vertex_key, result, lookup_idx):
        """Verify a single lookup result. Returns True if correct."""
        if vertex_key not in result:
            print(f"  [FAIL] Lookup #{lookup_idx}: vertex {vertex_key} not found in result")
            return False
        
        res = result[vertex_key]
        # lookup returns (vertex_data, adjacency_dict) or (vertex_data, adjacency_dict, pos_leaf)
        vertex_data = res[0]
        adjacency_dict = res[1]
        expected_data, expected_neighbors = grove._test_graph_data[vertex_key]
        
        if vertex_data != expected_data:
            print(f"  [FAIL] Lookup #{lookup_idx}: vertex {vertex_key} data mismatch: "
                  f"got '{vertex_data[:30]}', expected '{expected_data[:30]}'")
            return False
        
        got_neighbors = set(adjacency_dict.keys())
        expected_neighbor_set = set(expected_neighbors)
        if got_neighbors != expected_neighbor_set:
            missing = expected_neighbor_set - got_neighbors
            extra = got_neighbors - expected_neighbor_set
            print(f"  [FAIL] Lookup #{lookup_idx}: vertex {vertex_key} adjacency mismatch: "
                  f"missing={missing}, extra={extra}")
            return False
        
        return True
    
    def test_massive_uniform_lookups(self, initialized_grove):
        """500 uniform random lookups - the primary dedup stress test."""
        grove = initialized_grove
        num_lookups = 500
        
        random.seed(2024)
        access_sequence = [random.randint(0, self.NUM_VERTICES - 1) for _ in range(num_lookups)]
        
        from collections import Counter
        counts = Counter(access_sequence)
        print(f"\n[Uniform] {num_lookups} lookups, {len(counts)} unique vertices")
        print(f"  Most accessed: {counts.most_common(3)}")
        
        success = 0
        first_fail = None
        for i, vertex_key in enumerate(access_sequence):
            result = grove.lookup([vertex_key])
            if self._verify_lookup(grove, vertex_key, result, i):
                success += 1
            elif first_fail is None:
                first_fail = i
            
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{num_lookups}, success so far: {success}/{i + 1}")
        
        print(f"\n[Uniform] Result: {success}/{num_lookups} ({100*success/num_lookups:.1f}%)")
        if first_fail is not None:
            print(f"  First failure at lookup #{first_fail}")
        assert success == num_lookups, \
            f"Only {success}/{num_lookups} uniform lookups succeeded (first fail at #{first_fail})"
    
    def test_massive_repeated_single_vertex(self, initialized_grove):
        """Repeatedly access the SAME vertex 200 times - worst case for dup accumulation."""
        grove = initialized_grove
        num_lookups = 200
        target = 42
        
        print(f"\n[Repeated] {num_lookups} lookups on vertex {target}")
        
        success = 0
        first_fail = None
        for i in range(num_lookups):
            result = grove.lookup([target])
            if self._verify_lookup(grove, target, result, i):
                success += 1
            elif first_fail is None:
                first_fail = i
            
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{num_lookups}, success so far: {success}/{i + 1}")
        
        print(f"\n[Repeated] Result: {success}/{num_lookups} ({100*success/num_lookups:.1f}%)")
        assert success == num_lookups, \
            f"Only {success}/{num_lookups} repeated lookups succeeded (first fail at #{first_fail})"
    
    def test_massive_zipf_lookups(self, initialized_grove):
        """500 Zipf-distributed lookups (heavy skew, alpha=0.99)."""
        grove = initialized_grove
        num_lookups = 500
        
        random.seed(9999)
        alpha = 0.99
        weights = [1.0 / (i + 1) ** alpha for i in range(self.NUM_VERTICES)]
        total = sum(weights)
        probs = [w / total for w in weights]
        access_sequence = random.choices(range(self.NUM_VERTICES), weights=probs, k=num_lookups)
        
        from collections import Counter
        counts = Counter(access_sequence)
        print(f"\n[Zipf α={alpha}] {num_lookups} lookups, {len(counts)} unique vertices")
        print(f"  Most accessed: {counts.most_common(5)}")
        
        success = 0
        first_fail = None
        for i, vertex_key in enumerate(access_sequence):
            result = grove.lookup([vertex_key])
            if self._verify_lookup(grove, vertex_key, result, i):
                success += 1
            elif first_fail is None:
                first_fail = i
            
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{num_lookups}, success so far: {success}/{i + 1}")
        
        print(f"\n[Zipf] Result: {success}/{num_lookups} ({100*success/num_lookups:.1f}%)")
        assert success == num_lookups, \
            f"Only {success}/{num_lookups} Zipf lookups succeeded (first fail at #{first_fail})"
    
    def test_massive_sequential_lookups(self, initialized_grove):
        """Sequentially access vertices 0,1,2,...,N-1 then repeat - 500 total."""
        grove = initialized_grove
        num_lookups = 500
        
        access_sequence = [i % self.NUM_VERTICES for i in range(num_lookups)]
        
        print(f"\n[Sequential] {num_lookups} lookups, cycling through {self.NUM_VERTICES} vertices")
        
        success = 0
        first_fail = None
        for i, vertex_key in enumerate(access_sequence):
            result = grove.lookup([vertex_key])
            if self._verify_lookup(grove, vertex_key, result, i):
                success += 1
            elif first_fail is None:
                first_fail = i
            
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{num_lookups}, success so far: {success}/{i + 1}")
        
        print(f"\n[Sequential] Result: {success}/{num_lookups} ({100*success/num_lookups:.1f}%)")
        assert success == num_lookups, \
            f"Only {success}/{num_lookups} sequential lookups succeeded (first fail at #{first_fail})"
    
    def test_massive_mixed_batch_lookups(self):
        """300 batch lookups (each batch 2-5 vertices) - tests multi-key dedup."""
        # Use larger stash_scale for batch operations that generate more dups
        num_data = self.NUM_VERTICES
        max_deg = self.MAX_DEGREE
        client = InteractLocalServer()
        
        grove = Grove(
            max_deg=max_deg,
            num_opr=100,
            num_data=num_data,
            key_size=16,
            data_size=64,
            client=client,
            encryptor=None,
            stash_scale=200  # Larger stash for batch ops (each batch generates many dups)
        )
        
        # Build circulant graph
        d = max_deg
        graph_data = {}
        for v in range(num_data):
            neighbors = []
            for offset in range(1, d // 2 + 1):
                neighbors.append((v + offset) % num_data)
                neighbors.append((v - offset) % num_data)
            neighbors = neighbors[:max_deg]
            graph_data[v] = (f"vertex_{v}_data", neighbors)
        
        leaf_range = grove._leaf_range
        vertex_graph_leaf = {v: secrets.randbelow(leaf_range) for v in range(num_data)}
        vertex_pos_leaf = {v: secrets.randbelow(leaf_range) for v in range(num_data)}
        
        data_map = {}
        path_map = {}
        for v in range(num_data):
            vd, nb = graph_data[v]
            data_map[v] = (vd, {n: vertex_graph_leaf[n] for n in nb}, vertex_pos_leaf[v])
            path_map[v] = vertex_graph_leaf[v]
        
        grove.init_server_storage(
            posmap_data=[(v, vertex_graph_leaf[v]) for v in range(num_data)],
            graph_data_map=data_map,
            graph_path_map=path_map,
        )
        
        grove._test_graph_data = graph_data
        grove._test_vertex_graph_leaf = vertex_graph_leaf
        grove._test_vertex_pos_leaf = vertex_pos_leaf
        
        print(f"\n[Stress Test] Initialized Grove: {num_data} vertices, stash_scale=50")
        
        num_rounds = 300
        
        random.seed(7777)
        
        print(f"\n[Mixed Batch] {num_rounds} rounds of batch lookups")
        
        total_lookups = 0
        success = 0
        first_fail = None
        
        for round_idx in range(num_rounds):
            batch_size = random.randint(2, 5)
            vertex_keys = random.sample(range(self.NUM_VERTICES), batch_size)
            
            result = grove.lookup(vertex_keys)
            
            for vk in vertex_keys:
                total_lookups += 1
                if self._verify_lookup(grove, vk, result, total_lookups):
                    success += 1
                elif first_fail is None:
                    first_fail = total_lookups
            
            if (round_idx + 1) % 60 == 0:
                print(f"  Round {round_idx + 1}/{num_rounds}, "
                      f"total lookups: {total_lookups}, success: {success}")
        
        print(f"\n[Mixed Batch] Result: {success}/{total_lookups} ({100*success/total_lookups:.1f}%)")
        assert success == total_lookups, \
            f"Only {success}/{total_lookups} batch lookups succeeded (first fail at #{first_fail})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
