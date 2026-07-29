import pytest

from oblivlib.dependency import BinaryTree, Data, InteractLocalServer


class TestInteractLocalServer:
    def test_init_storage(self):
        server = InteractLocalServer()
        tree = BinaryTree(num_data=4, bucket_size=2)
        server.init_storage({"tree1": tree, "list1": [10, 20, 30, 40, 50]})

        assert "tree1" in server._storage
        assert "list1" in server._storage
        server.add_read_list(label="list1", indices=[0, 4])
        assert server.execute().results["list1"] == {0: 10, 4: 50}

    def test_list_read_write(self):
        server = InteractLocalServer()
        server.init_storage({"mylist": [10, 20, 30, 40, 50]})

        server.add_read_list(label="mylist", indices=[0, 2, 4])
        result = server.execute()
        assert result.success
        assert result.results["mylist"] == {0: 10, 2: 30, 4: 50}

        server.add_write_list(label="mylist", data={1: 100, 3: 300})
        server.add_read_list(label="mylist", indices=[1, 3])
        result = server.execute()
        assert result.success
        assert result.results["mylist"] == {1: 100, 3: 300}

    def test_tree_read_write_path_round_trip(self):
        server = InteractLocalServer()
        tree = BinaryTree(num_data=4, bucket_size=2)
        server.init_storage({"tree": tree})

        server.add_read_path(label="tree", leaves=[0])
        path_data = server.execute().results["tree"]
        assert isinstance(path_data, dict)

        leaf_index = max(path_data)
        path_data[leaf_index] = [Data(key=7, leaf=0, value="written")]
        server.add_write_path(label="tree", data=path_data)
        server.execute()

        server.add_read_path(label="tree", leaves=[0])
        reread = server.execute().results["tree"]
        assert reread[leaf_index] == [Data(key=7, leaf=0, value="written")]

    def test_writes_before_reads(self):
        server = InteractLocalServer()
        server.init_storage({"mylist": [0, 0, 0]})

        server.add_write_list(label="mylist", data={0: 999})
        server.add_read_list(label="mylist", indices=[0])
        result = server.execute()
        assert result.success
        assert result.results["mylist"] == {0: 999}

    def test_error_handling(self):
        server = InteractLocalServer()
        server.add_read_list(label="nonexistent", indices=[0])
        result = server.execute()
        assert not result.success
        assert result.error is not None

    def test_require_accessor(self):
        server = InteractLocalServer()
        server.init_storage({"mylist": [10, 20, 30]})

        server.add_read_list(label="mylist", indices=[0])
        result = server.execute()
        assert result.require("mylist") == {0: 10}
        with pytest.raises(KeyError):
            result.require("other")

        server.add_read_list(label="nonexistent", indices=[0])
        failed = server.execute()
        with pytest.raises(RuntimeError):
            failed.require("nonexistent")

    def test_bandwidth_tracking(self):
        server = InteractLocalServer()
        server.init_storage({"mylist": [10, 20, 30, 40, 50]})
        assert server.get_bandwidth() == (0, 0)

        server.add_read_list(label="mylist", indices=[0, 1, 2])
        server.execute()
        bytes_read, _ = server.get_bandwidth()
        assert bytes_read > 0

        server.reset_bandwidth()
        assert server.get_bandwidth() == (0, 0)
