from daoram.dependency.binary_tree import BinaryTree
from daoram.dependency.interact_server import InteractLocalServer


class TestInteractLocalServer:
    def test_init_storage(self):
        server = InteractLocalServer()
        tree = BinaryTree(num_data=4, bucket_size=2)

        server.init_storage({
            'tree1': tree,
            'list1': [10, 20, 30, 40, 50]
        })
        assert 'tree1' in server._storage
        assert 'list1' in server._storage

    def test_list_read_write(self):
        server = InteractLocalServer()
        server.init_storage({'mylist': [10, 20, 30, 40, 50]})

        # Read
        server.add_read_list(label='mylist', indices=[0, 2, 4])
        result = server.execute()
        assert result.success
        assert result.results['mylist'] == {0: 10, 2: 30, 4: 50}

        # Write then read
        server.add_write_list(label='mylist', data={1: 100, 3: 300})
        server.add_read_list(label='mylist', indices=[1, 3])
        result = server.execute()
        assert result.success
        assert result.results['mylist'] == {1: 100, 3: 300}

    def test_tree_read_write_path(self):
        server = InteractLocalServer()
        tree = BinaryTree(num_data=4, bucket_size=2)
        server.init_storage({'tree': tree})

        # Read path
        server.add_read_path(label='tree', leaves=[0])
        result = server.execute()
        assert result.success
        assert isinstance(result.results['tree'], dict)

        # Write path back
        path_data = result.results['tree']
        server.add_write_path(label='tree', data=path_data)
        result = server.execute()
        assert result.success

    def test_writes_before_reads(self):
        server = InteractLocalServer()
        server.init_storage({'mylist': [0, 0, 0]})

        # Add write and read - writes should execute first
        server.add_write_list(label='mylist', data={0: 999})
        server.add_read_list(label='mylist', indices=[0])
        result = server.execute()

        assert result.success
        assert result.results['mylist'] == {0: 999}

    def test_error_handling(self):
        server = InteractLocalServer()
        server.add_read_list(label='nonexistent', indices=[0])
        result = server.execute()

        assert not result.success
        assert result.error is not None

    def test_bandwidth_tracking(self):
        server = InteractLocalServer()
        server.init_storage({'mylist': [10, 20, 30, 40, 50]})

        # Initial bandwidth should be zero.
        assert server.get_bandwidth() == (0, 0)

        # Execute a read.
        server.add_read_list(label='mylist', indices=[0, 1, 2])
        server.execute()

        # Bandwidth should be non-zero after read.
        bytes_read, bytes_written = server.get_bandwidth()
        assert bytes_read > 0

        # Reset and verify.
        server.reset_bandwidth()
        assert server.get_bandwidth() == (0, 0)
