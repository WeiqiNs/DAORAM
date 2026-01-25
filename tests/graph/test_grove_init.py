import pytest
from daoram.graph.grove import Grove
from daoram.dependency import InteractLocalServer

class TestGroveStepByStep:
    """
    Test Grove initialization and connection step by step as suggested.
    """

    @pytest.fixture
    def grove_setup(self):
        num_data = 2**12  # 4096
        max_deg = 10
        num_opr = 100
        key_size = 16
        data_size = 32
        client = InteractLocalServer()

        grove = Grove(
            max_deg=max_deg,
            num_opr=num_opr,
            num_data=num_data,
            key_size=key_size,
            data_size=data_size,
            client=client
        )
        
        # Initialize storage
        grove._pos_omap.init_server_storage()
        grove._graph_oram.init_server_storage()
        grove._graph_meta.init_server_storage()
        grove._pos_meta.init_server_storage()
        
        return grove

    def test_step_1_insert_vertex_1(self, grove_setup):
        grove = grove_setup
        print("\nStep 1: Inserting isolated vertex 1...")
        grove.insert((1, "v1_data", []))
        
        res = grove.lookup([1])
        assert 1 in res
        assert res[1][0] == "v1_data"
        assert res[1][1] == {}
        print("Step 1 passed.")

    def test_step_2_insert_vertex_2(self, grove_setup):
        grove = grove_setup
        grove.insert((1, "v1_data", [])) # Ensure 1 is there
        
        print("\nStep 2: Inserting isolated vertex 2...")
        grove.insert((2, "v2_data", []))
        
        res = grove.lookup([2])
        assert 2 in res
        assert res[2][0] == "v2_data"
        assert res[2][1] == {}
        print("Step 2 passed.")

    def test_step_3_connect_1_and_2(self, grove_setup):
        grove = grove_setup
        # Pre-insert both as isolated
        grove.insert((1, "v1_data", []))
        grove.insert((2, "v2_data", []))
        
        print("\nStep 3: Connecting 1 and 2 by re-inserting 1 with neighbor 2...")
        # This should trigger neighbor update for 2 because 2 is already in pos_map
        grove.insert((1, "v1_data", [2]))
        
        print("Verifying connectivity...")
        results = grove.lookup([1, 2])
        
        assert 1 in results
        assert 2 in results
        
        # Check if 1 knows about 2
        assert 2 in results[1][1]
        # Check if 2 was updated to know about 1 (Grove.insert handles this)
        assert 1 in results[2][1]
        print("Step 3 passed: Connection established successfully.")

if __name__ == "__main__":
    pytest.main([__file__])
