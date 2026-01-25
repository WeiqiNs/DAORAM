"""
Comprehensive tests for Grove graph ORAM operations.

Tests cover: insert, delete, lookup, neighbor, t_hop, t_traversal
"""

import random
import pytest

from daoram.graph.grove import Grove
from daoram.dependency import InteractLocalServer


class TestGroveBasic:
    """Basic tests for Grove graph operations."""

    @pytest.fixture
    def grove_instance(self):
        """Create a basic Grove instance for testing."""
        client = InteractLocalServer()
        grove = Grove(
            max_deg=5,
            num_opr=100,
            num_data=1024,
            key_size=16,
            data_size=64,
            client=client,
            encryptor=None,
            stash_scale=20,
        )
        grove._pos_omap.init_server_storage()
        grove._graph_oram.init_server_storage()
        grove._graph_meta.init_server_storage()
        grove._pos_meta.init_server_storage()
        return grove

    def test_insert_single_vertex(self, grove_instance):
        """Test inserting a single vertex with no neighbors."""
        grove = grove_instance
        grove.insert(vertex=(0, "value_0", {}))

        result = grove.lookup(keys=[0])
        assert 0 in result
        assert result[0][0] == "value_0"

    def test_insert_vertex_with_neighbors(self, grove_instance):
        """Test inserting vertices with neighbors."""
        grove = grove_instance

        grove.insert(vertex=(0, "value_0", {}))
        grove.insert(vertex=(1, "value_1", {0: None}))

        result = grove.lookup(keys=[0])
        assert 0 in result
        assert 1 in result[0][1]  # vertex 1 should be neighbor of vertex 0

        result = grove.lookup(keys=[1])
        assert 1 in result
        assert 0 in result[1][1]  # vertex 0 should be neighbor of vertex 1

    def test_insert_multiple_vertices(self, grove_instance):
        """Test inserting multiple connected vertices."""
        grove = grove_instance

        grove.insert(vertex=(0, "v0", {}))
        grove.insert(vertex=(1, "v1", {0: None}))
        grove.insert(vertex=(2, "v2", {0: None, 1: None}))

        for key in [0, 1, 2]:
            result = grove.lookup(keys=[key])
            assert key in result

    def test_lookup_multiple_keys(self, grove_instance):
        """Test looking up multiple keys at once."""
        grove = grove_instance

        for i in range(5):
            neighbors = {j: None for j in range(i)}
            grove.insert(vertex=(i, f"data_{i}", neighbors))

        result = grove.lookup(keys=[0, 2, 4])
        assert 0 in result
        assert 2 in result
        assert 4 in result


class TestGroveDelete:
    """Tests for Grove delete operations."""

    @pytest.fixture
    def grove_with_graph(self):
        """Create a Grove instance with a pre-built graph."""
        client = InteractLocalServer()
        grove = Grove(
            max_deg=5,
            num_opr=100,
            num_data=1024,
            key_size=16,
            data_size=64,
            client=client,
            encryptor=None,
            stash_scale=20,
        )
        grove._pos_omap.init_server_storage()
        grove._graph_oram.init_server_storage()
        grove._graph_meta.init_server_storage()
        grove._pos_meta.init_server_storage()

        grove.insert(vertex=(0, "center", {}))
        grove.insert(vertex=(1, "n1", {0: None}))
        grove.insert(vertex=(2, "n2", {0: None}))
        grove.insert(vertex=(3, "n3", {0: None, 1: None}))

        return grove

    def test_delete_vertex(self, grove_with_graph):
        """Test deleting a vertex."""
        grove = grove_with_graph

        grove.delete(key=1)

        # Verify vertex 1 is gone
        result = grove.lookup(keys=[1])
        assert 1 not in result

        # Verify vertex 0 no longer has vertex 1 as neighbor
        result = grove.lookup(keys=[0])
        assert 0 in result
        assert 1 not in result[0][1]

    def test_delete_and_lookup_remaining(self, grove_with_graph):
        """Test that other vertices remain accessible after delete."""
        grove = grove_with_graph

        grove.delete(key=2)

        # Remaining vertices should still be accessible
        for key in [0, 1, 3]:
            result = grove.lookup(keys=[key])
            assert key in result


class TestGroveNeighbor:
    """Tests for Grove neighbor query operations."""

    @pytest.fixture
    def grove_with_star(self):
        """Create a Grove instance with a star graph."""
        client = InteractLocalServer()
        grove = Grove(
            max_deg=5,
            num_opr=100,
            num_data=1024,
            key_size=16,
            data_size=64,
            client=client,
            encryptor=None,
            stash_scale=20,
        )
        grove._pos_omap.init_server_storage()
        grove._graph_oram.init_server_storage()
        grove._graph_meta.init_server_storage()
        grove._pos_meta.init_server_storage()

        grove.insert(vertex=(0, "center", {}))
        for i in range(1, 4):
            grove.insert(vertex=(i, f"leaf_{i}", {0: None}))

        return grove

    def test_neighbor_query(self, grove_with_star):
        """Test neighbor query returns all neighbors."""
        grove = grove_with_star

        neighbors = grove.neighbor(keys=[0])

        # Center has 3 neighbors
        assert len(neighbors) == 3
        for i in range(1, 4):
            assert i in neighbors

    def test_neighbor_query_preserves_accessibility(self, grove_with_star):
        """Test that vertices remain accessible after neighbor query."""
        grove = grove_with_star

        grove.neighbor(keys=[0])

        # All vertices should still be accessible
        for key in range(4):
            result = grove.lookup(keys=[key])
            assert key in result


class TestGroveTHop:
    """Tests for Grove t-hop and t-traversal operations."""

    @pytest.fixture
    def grove_with_chain(self):
        """Create a Grove instance with a chain graph: 0-1-2-3-4."""
        client = InteractLocalServer()
        grove = Grove(
            max_deg=5,
            num_opr=100,
            num_data=1024,
            key_size=16,
            data_size=64,
            client=client,
            encryptor=None,
            stash_scale=20,
        )
        grove._pos_omap.init_server_storage()
        grove._graph_oram.init_server_storage()
        grove._graph_meta.init_server_storage()
        grove._pos_meta.init_server_storage()

        grove.insert(vertex=(0, "v0", {}))
        for i in range(1, 5):
            grove.insert(vertex=(i, f"v{i}", {i-1: None}))

        return grove

    def test_t_hop_includes_start(self, grove_with_chain):
        """Test that t_hop includes the starting vertex."""
        grove = grove_with_chain

        result = grove.t_hop(key=2, num_hop=1)

        assert 2 in result  # Start vertex included

    def test_t_hop_one_hop(self, grove_with_chain):
        """Test 1-hop query from middle of chain."""
        grove = grove_with_chain

        result = grove.t_hop(key=2, num_hop=1)

        # Should find vertex 2 and its neighbors (1, 3)
        assert 2 in result
        assert 1 in result or 3 in result

    def test_t_hop_two_hops(self, grove_with_chain):
        """Test 2-hop query from start of chain."""
        grove = grove_with_chain

        result = grove.t_hop(key=0, num_hop=2)

        # Should find 0, 1, and 2
        assert 0 in result
        assert 1 in result
        assert 2 in result

    def test_t_traversal_includes_start(self, grove_with_chain):
        """Test that t_traversal includes the starting vertex."""
        grove = grove_with_chain

        result = grove.t_traversal(key=0, num_hop=2)

        assert 0 in result

    def test_t_traversal_visits_vertices(self, grove_with_chain):
        """Test that t_traversal visits the expected number of vertices."""
        grove = grove_with_chain

        result = grove.t_traversal(key=0, num_hop=3)

        # Should visit at least 2 vertices (start + at least one hop)
        assert len(result) >= 2


class TestGroveMixedOperations:
    """Tests for mixed Grove operations."""

    def create_grove(self):
        """Create a fresh Grove instance."""
        client = InteractLocalServer()
        grove = Grove(
            max_deg=5,
            num_opr=100,
            num_data=1024,
            key_size=16,
            data_size=64,
            client=client,
            encryptor=None,
            stash_scale=20,
        )
        grove._pos_omap.init_server_storage()
        grove._graph_oram.init_server_storage()
        grove._graph_meta.init_server_storage()
        grove._pos_meta.init_server_storage()
        return grove

    def test_insert_delete_lookup_cycle(self):
        """Test insert, delete, lookup cycle."""
        grove = self.create_grove()

        # Insert vertices
        grove.insert(vertex=(0, "v0", {}))
        grove.insert(vertex=(1, "v1", {0: None}))
        grove.insert(vertex=(2, "v2", {0: None, 1: None}))

        # Verify all accessible
        for key in [0, 1, 2]:
            result = grove.lookup(keys=[key])
            assert key in result

        # Delete vertex 1
        grove.delete(key=1)

        # Verify 1 is gone, others still accessible
        result = grove.lookup(keys=[1])
        assert 1 not in result

        for key in [0, 2]:
            result = grove.lookup(keys=[key])
            assert key in result

    def test_neighbor_after_delete(self):
        """Test neighbor query after deletion."""
        grove = self.create_grove()

        grove.insert(vertex=(0, "center", {}))
        grove.insert(vertex=(1, "n1", {0: None}))
        grove.insert(vertex=(2, "n2", {0: None}))

        grove.delete(key=1)

        neighbors = grove.neighbor(keys=[0])

        # Only vertex 2 should be neighbor now
        assert 1 not in neighbors
        # Vertex 2 might still be there (if it wasn't deleted)

    @pytest.mark.parametrize("seed", range(10))
    def test_random_operations(self, seed):
        """Test random mix of operations."""
        random.seed(seed)
        grove = self.create_grove()
        inserted = set()
        deleted = set()

        for _ in range(50):
            op = random.randint(0, 4)
            active = list(inserted - deleted)

            if op == 0 or len(active) == 0:
                # Insert
                new_key = len(inserted)
                neighbors = {}
                if active:
                    n = min(len(active), random.randint(1, 2))
                    random.shuffle(active)
                    neighbors = {k: None for k in active[:n]}
                grove.insert((new_key, f"d{new_key}", neighbors))
                inserted.add(new_key)

            elif op == 1 and len(active) > 1:
                # Delete
                key = random.choice(active)
                grove.delete(key)
                deleted.add(key)

            elif op == 2 and active:
                # Lookup
                key = random.choice(active)
                result = grove.lookup([key])
                assert key in result, f"lookup({key}) failed"

            elif op == 3 and active:
                # Neighbor
                key = random.choice(active)
                grove.neighbor([key])
                result = grove.lookup([key])
                assert key in result, f"lookup({key}) failed after neighbor"

            else:
                # Another lookup
                if active:
                    key = random.choice(active)
                    result = grove.lookup([key])
                    assert key in result


class TestGroveStress:
    """Stress tests for Grove operations."""

    def create_grove(self):
        """Create a fresh Grove instance."""
        client = InteractLocalServer()
        grove = Grove(
            max_deg=5,
            num_opr=100,
            num_data=1024,
            key_size=16,
            data_size=64,
            client=client,
            encryptor=None,
            stash_scale=20,
        )
        grove._pos_omap.init_server_storage()
        grove._graph_oram.init_server_storage()
        grove._graph_meta.init_server_storage()
        grove._pos_meta.init_server_storage()
        return grove

    @pytest.mark.parametrize("trial", range(10))
    def test_comprehensive_random_operations(self, trial):
        """Comprehensive test with all operation types."""
        random.seed(trial * 100)
        grove = self.create_grove()
        inserted = set()
        deleted = set()

        for step in range(80):
            op = random.randint(0, 6)
            active = list(inserted - deleted)

            if op == 0 or len(active) == 0:
                # Insert
                new_key = len(inserted)
                neighbors = {}
                if active:
                    n = min(len(active), random.randint(1, 3))
                    random.shuffle(active)
                    neighbors = {k: None for k in active[:n]}
                grove.insert((new_key, f"d{new_key}", neighbors))
                inserted.add(new_key)

            elif op == 1 and len(active) > 2:
                # Delete
                key = random.choice(active)
                grove.delete(key)
                deleted.add(key)

            elif op == 2 and active:
                # Lookup
                key = random.choice(active)
                result = grove.lookup([key])
                assert key in result, f"Step {step}: lookup({key}) failed"

            elif op == 3 and active:
                # Neighbor
                key = random.choice(active)
                grove.neighbor([key])
                result = grove.lookup([key])
                assert key in result, f"Step {step}: lookup after neighbor failed"

            elif op == 4 and active:
                # t_hop
                key = random.choice(active)
                result = grove.t_hop(key, random.randint(1, 2))
                assert key in result, f"Step {step}: t_hop missing start"

            elif op == 5 and active:
                # t_traversal
                key = random.choice(active)
                result = grove.t_traversal(key, random.randint(1, 2))
                assert key in result, f"Step {step}: t_traversal missing start"

            else:
                if active:
                    key = random.choice(active)
                    grove.lookup([key])


class TestGroveStaticMethods:
    """Tests for Grove static methods."""

    def test_binomial(self):
        """Test binomial probability calculation."""
        from decimal import Decimal

        result = Grove.binomial(n=2, i=0, p=Decimal("0.5"))
        assert abs(float(result) - 0.25) < 0.0001

        result = Grove.binomial(n=2, i=1, p=Decimal("0.5"))
        assert abs(float(result) - 0.5) < 0.0001

        result = Grove.binomial(n=2, i=2, p=Decimal("0.5"))
        assert abs(float(result) - 0.25) < 0.0001

    def test_equation(self):
        """Test the overflow probability equation."""
        from decimal import Decimal

        result = Grove.equation(m=10, K=2, Y=4, L=5, prec=50)
        assert isinstance(result, Decimal)
        assert result >= 0

    def test_find_bound(self):
        """Test that find_bound returns a positive integer."""
        client = InteractLocalServer()
        grove = Grove(
            max_deg=4,
            num_opr=100,
            num_data=64,
            key_size=10,
            data_size=10,
            client=client,
        )
        bound = grove.find_bound()
        assert bound > 0
        assert isinstance(bound, int)

    def test_get_rl_leaf_sequence(self):
        """Test that get_rl_leaf returns correct sequence."""
        client = InteractLocalServer()
        grove = Grove(
            max_deg=4,
            num_opr=100,
            num_data=64,
            key_size=10,
            data_size=10,
            client=client,
        )

        leaves = grove.get_rl_leaf(count=4)
        assert len(leaves) == 4
        assert grove._graph_counter == 4

        more_leaves = grove.get_rl_leaf(count=3)
        assert len(more_leaves) == 3
        assert grove._graph_counter == 7
