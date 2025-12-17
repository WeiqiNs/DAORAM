from daoram.dependency.binary_tree import BinaryTree
from daoram.dependency.interact_server import InteractLocalServer
from daoram.dependency.types import (
    Query,
    TreeReadPathPayload,
    TreeWritePathPayload,
    TreeReadBucketPayload,
    ListReadPayload,
    ListWritePayload,
    TREE_READ_PATH,
    TREE_WRITE_PATH,
    TREE_READ_BUCKET,
    LIST_READ,
    LIST_WRITE,
)


class TestInteractLocalServer:
    def test_init_storage(self):
        server = InteractLocalServer()
        tree = BinaryTree(num_data=4, bucket_size=2)

        # Init with multiple storages
        server.init_storage({
            'tree1': tree,
            'list1': [10, 20, 30, 40, 50]
        })
        assert 'tree1' in server._storage
        assert 'list1' in server._storage

        # Init additional storage
        server.init_storage({'list2': [1, 2, 3]})
        assert 'list2' in server._storage

    def test_tree_queries(self):
        server = InteractLocalServer()
        tree = BinaryTree(num_data=4, bucket_size=2)
        server.init_storage({'tree': tree})

        # Read path
        read_query = Query(
            payload=TreeReadPathPayload(leaves=[0, 1]),
            query_type=TREE_READ_PATH,
            storage_label='tree'
        )
        path_data = server.execute_query(read_query)
        assert isinstance(path_data, dict)
        assert len(path_data) > 0

        # Write path
        write_query = Query(
            payload=TreeWritePathPayload(data=path_data),
            query_type=TREE_WRITE_PATH,
            storage_label='tree'
        )
        assert server.execute_query(write_query) is None

        # Read bucket
        bucket_query = Query(
            payload=TreeReadBucketPayload(keys=[(0, 0), (0, 1)]),
            query_type=TREE_READ_BUCKET,
            storage_label='tree'
        )
        bucket_data = server.execute_query(bucket_query)
        assert isinstance(bucket_data, dict)

    def test_list_queries(self):
        server = InteractLocalServer()
        server.init_storage({'mylist': [10, 20, 30, 40, 50]})

        # Read
        read_query = Query(
            payload=ListReadPayload(indices=[0, 2, 4]),
            query_type=LIST_READ,
            storage_label='mylist'
        )
        assert server.execute_query(read_query) == {0: 10, 2: 30, 4: 50}

        # Write
        write_query = Query(
            payload=ListWritePayload(data={1: 100, 3: 300}),
            query_type=LIST_WRITE,
            storage_label='mylist'
        )
        server.execute_query(write_query)

        # Verify write
        verify_query = Query(
            payload=ListReadPayload(indices=[1, 3]),
            query_type=LIST_READ,
            storage_label='mylist'
        )
        assert server.execute_query(verify_query) == {1: 100, 3: 300}

    def test_batch_queries_write_before_read(self):
        server = InteractLocalServer()
        server.init_storage({'mylist': [10, 20, 30, 40, 50]})

        # Add queries in any order - writes should execute first
        read_q1 = Query(payload=ListReadPayload(indices=[0]), query_type=LIST_READ, storage_label='mylist')
        read_q2 = Query(payload=ListReadPayload(indices=[2]), query_type=LIST_READ, storage_label='mylist')
        write_q = Query(payload=ListWritePayload(data={0: 999, 2: 888}), query_type=LIST_WRITE, storage_label='mylist')

        server.add_query(read_q1)
        server.add_query(write_q)
        server.add_query(read_q2)

        assert len(server.read_queries) == 2
        assert len(server.write_queries) == 1

        # Execute - writes happen first, so reads see updated values
        results = server.execute_queries()

        assert results == [{0: 999}, {2: 888}]
        assert len(server.read_queries) == 0
        assert len(server.write_queries) == 0

    def test_multiple_writes_before_reads(self):
        server = InteractLocalServer()
        server.init_storage({'mylist': [0, 0, 0, 0, 0]})

        # Multiple writes and reads
        server.add_query(Query(payload=ListWritePayload(data={0: 1}), query_type=LIST_WRITE, storage_label='mylist'))
        server.add_query(Query(payload=ListWritePayload(data={1: 2}), query_type=LIST_WRITE, storage_label='mylist'))
        server.add_query(Query(payload=ListReadPayload(indices=[0, 1, 2]), query_type=LIST_READ, storage_label='mylist'))

        results = server.execute_queries()

        # Read should see both writes applied
        assert results == [{0: 1, 1: 2, 2: 0}]

    def test_clear_queries(self):
        server = InteractLocalServer()
        server.init_storage({'mylist': [10, 20, 30]})

        server.add_query(Query(payload=ListReadPayload(indices=[0]), query_type=LIST_READ, storage_label='mylist'))
        server.add_query(Query(payload=ListWritePayload(data={0: 100}), query_type=LIST_WRITE, storage_label='mylist'))

        server.clear_queries()

        assert len(server.read_queries) == 0
        assert len(server.write_queries) == 0

    def test_storage_not_found(self):
        server = InteractLocalServer()
        query = Query(payload=ListReadPayload(indices=[0]), query_type=LIST_READ, storage_label='nonexistent')

        try:
            server.execute_query(query)
            assert False, "Should have raised KeyError"
        except KeyError:
            pass
