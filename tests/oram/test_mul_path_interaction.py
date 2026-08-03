import pytest
import secrets
from daoram.dependency import InteractLocalServer, Data
from daoram.oram.mul_path_oram import MulPathOram

class TestMulPathOramInteraction:
    """
    Test the interaction between MulPathOram and InteractServer, 
    specifically focusing on batch queuing and process_read_result.
    """

    @pytest.fixture
    def setup_orams(self):
        client = InteractLocalServer()
        num_data = 16
        data_size = 10
        
        # Create two separate MulPathOram instances
        oram1 = MulPathOram(
            name="oram1",
            num_data=num_data,
            data_size=data_size,
            client=client,
            bucket_size=4
        )
        oram2 = MulPathOram(
            name="oram2",
            num_data=num_data,
            data_size=data_size,
            client=client,
            bucket_size=4
        )
        
        # Give the two stores distinguishable payloads while retaining the
        # same integer key domain.
        oram1.init_server_storage(data_map={i: ("oram1", i) for i in range(num_data)})
        oram2.init_server_storage(data_map={i: ("oram2", i) for i in range(num_data)})
        
        return client, oram1, oram2

    def test_batch_read_and_process(self, setup_orams):
        client, oram1, oram2 = setup_orams
        
        # 1. Prepare some keys and their leaves
        # MulPathOram.init_server_storage initializes keys 0 to num_data-1
        key1 = 5
        leaf1 = oram1._look_up_pos_map(key1)
        
        key2 = 10
        leaf2 = oram2._look_up_pos_map(key2)
        
        # 2. Queue reads for both ORAMs
        oram1.queue_read(leaves=[leaf1])
        oram2.queue_read(leaves=[leaf2])
        
        # Verify that client has both labels in its read queue
        assert "oram1" in client._read_paths
        assert "oram2" in client._read_paths
        assert leaf1 in client._read_paths["oram1"]
        assert leaf2 in client._read_paths["oram2"]
        
        # 3. Execute the batch read
        result = client.execute()
        
        # Verify result contains data for both ORAMs
        assert result.success is True
        assert "oram1" in result.results
        assert "oram2" in result.results
        
        # 4. Process results individually
        # Initially, stashes should be empty (or only contain init data if any)
        # In this implementation, init_server_storage fills the tree, stash is empty.
        initial_stash_size1 = len(oram1.stash)
        initial_stash_size2 = len(oram2.stash)
        
        oram1.process_read_result(result)
        oram2.process_read_result(result)
        
        # 5. Verify that data was moved to respective stashes
        # Each path read should bring at least one real block (the one we asked for)
        # plus any other real blocks on that path.
        assert len(oram1.stash) > initial_stash_size1
        assert len(oram2.stash) > initial_stash_size2
        
        # Check if our specific keys are now in the stash
        assert any(d.key == key1 for d in oram1.stash)
        assert any(d.key == key2 for d in oram2.stash)
        
        # Both ORAMs intentionally use the same key domain. Isolation is
        # therefore checked by payload provenance, not key absence.
        assert all(d.value[0] == "oram1" for d in oram1.stash)
        assert all(d.value[0] == "oram2" for d in oram2.stash)

    def test_batch_write_after_process(self, setup_orams):
        client, oram1, oram2 = setup_orams
        
        # Setup: Read keys into stash first
        key1, key2 = 1, 2
        leaf1 = oram1._look_up_pos_map(key1)
        leaf2 = oram2._look_up_pos_map(key2)
        
        oram1.queue_read(leaves=[leaf1])
        oram2.queue_read(leaves=[leaf2])
        result = client.execute()
        oram1.process_read_result(result)
        oram2.process_read_result(result)
        
        # 1. Modify data in stash
        new_val1 = b"new_data1"
        new_val2 = b"new_data2"
        
        for d in oram1.stash:
            if d.key == key1:
                d.value = new_val1
        
        for d in oram2.stash:
            if d.key == key2:
                d.value = new_val2
                
        # 2. Queue writes (Eviction)
        # queue_write uses the leaves stored in _tmp_leaves from queue_read
        oram1.queue_write()
        oram2.queue_write()
        
        # Verify client has write paths queued
        assert "oram1" in client._write_paths
        assert "oram2" in client._write_paths
        
        # 3. Execute batch write
        client.execute()
        
        # 4. Verify data was actually written by reading it back
        # We need to clear stash to force a fresh read from server
        oram1.stash = []
        oram2.stash = []
        
        oram1.queue_read(leaves=[leaf1])
        oram2.queue_read(leaves=[leaf2])
        result = client.execute()
        oram1.process_read_result(result)
        oram2.process_read_result(result)
        
        # Find the keys in stash and check values
        val1_read = next(d.value for d in oram1.stash if d.key == key1)
        val2_read = next(d.value for d in oram2.stash if d.key == key2)
        
        assert val1_read == new_val1
        assert val2_read == new_val2
