import pickle

import pytest

from oblivlib.dependency import (
    BinaryTree,
    Data,
    DuplicateLabelError,
    InteractLocalServer,
    ListPopBack,
    ListPushFront,
    ListWrite,
    MissingResultError,
    UnknownLabelError,
)
from oblivlib.dependency.codec import DefaultCodec
from oblivlib.dependency.types import Bucket

_CODEC = DefaultCodec(block_size=128)


class TestInteractLocalServer:
    def test_init_storage(self):
        server = InteractLocalServer()
        tree = BinaryTree(num_data=4, bucket_size=2, codec=_CODEC)
        server.init_storage({"tree1": tree, "list1": [10, 20, 30, 40, 50]})

        assert server._require_tree("tree1") is tree
        server.add_read_list(label="list1", indices=[0, 4])
        assert server.execute().require("list1") == {0: 10, 4: 50}

    def test_init_storage_rejects_duplicate_label(self):
        server = InteractLocalServer()
        server.init_storage({"taken": [1, 2]})

        with pytest.raises(DuplicateLabelError, match="taken"):
            server.init_storage({"fresh": [3], "taken": [4]})
        with pytest.raises(UnknownLabelError):
            server._require_list("fresh")

        server.add_read_list(label="taken", indices=None)
        assert server.execute().require("taken") == [1, 2]

    def test_list_read_write(self):
        server = InteractLocalServer()
        server.init_storage({"mylist": [10, 20, 30, 40, 50]})

        server.add_read_list(label="mylist", indices=[0, 2, 4])
        assert server.execute().require("mylist") == {0: 10, 2: 30, 4: 50}

        server.add_write_list(label="mylist", ops=[ListWrite(1, 100), ListWrite(3, 300)])
        server.add_read_list(label="mylist", indices=[1, 3])
        assert server.execute().require("mylist") == {1: 100, 3: 300}

    def test_list_write_rejects_negative_index(self):
        with pytest.raises(ValueError):
            ListWrite(-1, "x")

    def test_list_ops_apply_in_staging_order(self):
        server = InteractLocalServer()
        server.init_storage({"mylist": [1, 2, 3]})

        server.add_write_list(label="mylist", ops=[ListPushFront("a"), ListPushFront("b")])
        server.add_write_list(label="mylist", ops=[ListPopBack(), ListWrite(0, "c")])
        server.add_read_list(label="mylist", indices=None)
        assert server.execute().require("mylist") == ["c", "a", 1, 2]

    def test_tree_read_write_path_round_trip(self):
        server = InteractLocalServer()
        tree = BinaryTree(num_data=4, bucket_size=2, codec=_CODEC)
        server.init_storage({"tree": tree})

        server.add_read_path(label="tree", leaves=[0])
        path_data = server.execute().require("tree")
        assert isinstance(path_data, dict)

        leaf_index = max(path_data)
        path_data[leaf_index] = [Data(key=7, leaf=0, value="written")]
        server.add_write_path(label="tree", data=path_data)
        server.execute()

        server.add_read_path(label="tree", leaves=[0])
        reread = server.execute().require("tree")
        assert reread[leaf_index] == [Data(key=7, leaf=0, value="written")]

    def test_results_and_writes_do_not_alias_server_storage(self):
        server = InteractLocalServer()
        server.init_storage({"tree": BinaryTree(num_data=4, bucket_size=2, codec=_CODEC), "mylist": [1, 2]})

        bucket: Bucket = [Data(key=7, leaf=0, value="stored")]
        server.add_write_path(label="tree", data={3: bucket})
        server.execute()
        written = bucket[0]
        assert isinstance(written, Data)
        written.value = "client edit after write"
        bucket.append(Data(key=8, leaf=0, value="extra"))

        server.add_read_path(label="tree", leaves=[0])
        server.add_read_list(label="mylist", indices=None)
        result = server.execute()
        result.require("tree")[3][0].value = "client edit after read"
        result.require("mylist").append(3)

        server.add_read_path(label="tree", leaves=[0])
        server.add_read_list(label="mylist", indices=None)
        reread = server.execute()
        assert reread.require("tree")[3] == [Data(key=7, leaf=0, value="stored")]
        assert reread.require("mylist") == [1, 2]

    def test_require_accessor(self):
        server = InteractLocalServer()
        server.init_storage({"mylist": [10, 20, 30]})

        server.add_read_list(label="mylist", indices=[0])
        result = server.execute()
        assert result.require("mylist") == {0: 10}
        with pytest.raises(MissingResultError):
            result.require("other")

        server.add_read_list(label="nonexistent", indices=[0])
        failed = server.execute()
        for outcome in (failed, pickle.loads(pickle.dumps(failed))):
            with pytest.raises(UnknownLabelError, match="nonexistent"):
                outcome.require("nonexistent")

        server.add_read_list(label="mylist", indices=[2])
        assert server.execute().require("mylist") == {2: 30}

    def test_bandwidth_tracking(self, remote_client):
        server = InteractLocalServer()
        for endpoint in (server, remote_client):
            endpoint.init_storage({"mylist": [10, 20, 30, 40, 50]})
            assert endpoint.get_bandwidth() == (0, 0)

            endpoint.add_write_list(label="mylist", ops=[ListWrite(0, 11)])
            endpoint.add_read_list(label="mylist", indices=[0, 1, 2])
            endpoint.execute()

        bytes_read, bytes_written = server.get_bandwidth()
        assert bytes_read > 0 and bytes_written > 0
        assert remote_client.get_bandwidth() == (bytes_read, bytes_written)

        server.reset_bandwidth()
        assert server.get_bandwidth() == (0, 0)


class TestInteractRemoteServer:
    def test_remote_init_error_is_reraised(self, remote_client):
        remote_client.init_storage({"mylist": [1, 2]})

        with pytest.raises(DuplicateLabelError, match="mylist"):
            remote_client.init_storage({"mylist": [3]})

        remote_client.add_read_list(label="mylist", indices=None)
        assert remote_client.execute().require("mylist") == [1, 2]
