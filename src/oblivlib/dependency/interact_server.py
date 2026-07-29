"""Client-server interaction layer for local and remote ORAM storage."""

import pickle
from abc import ABC, abstractmethod
from typing import Any, cast, override

from oblivlib.dependency.binary_tree import BinaryTree
from oblivlib.dependency.sockets import BaseSocket
from oblivlib.dependency.types import BlockData, BlockKey, BucketData, BucketKey, ExecuteResult, PathData

SERVER_DEFAULT_RESPONSE = "Done!"
PORT = 10000
ServerStorage = dict[str, BinaryTree | list]


class InteractServer(ABC):
    def __init__(self):
        # Reads accumulate keys/indices; writes accumulate data (a later write overwrites an earlier one).
        self._read_paths: dict[str, list[int]] = {}
        self._read_buckets: dict[str, list[BucketKey]] = {}
        self._read_blocks: dict[str, list[BlockKey]] = {}
        self._read_lists: dict[str, list[int] | None] = {}  # None means "read the whole list".

        self._write_paths: dict[str, PathData] = {}
        self._write_buckets: dict[str, BucketData] = {}
        self._write_blocks: dict[str, BlockData] = {}
        self._write_lists: dict[str, dict[int, Any]] = {}

        self._bytes_read: int = 0
        self._bytes_written: int = 0

    def clear_queries(self) -> None:
        self._read_paths.clear()
        self._read_buckets.clear()
        self._read_blocks.clear()
        self._read_lists.clear()
        self._write_paths.clear()
        self._write_buckets.clear()
        self._write_blocks.clear()
        self._write_lists.clear()

    def get_bandwidth(self) -> tuple[int, int]:
        """Return (bytes_read, bytes_written)."""
        return self._bytes_read, self._bytes_written

    def reset_bandwidth(self) -> None:
        self._bytes_read = 0
        self._bytes_written = 0

    def add_read_path(self, label: str, leaves: list[int]) -> None:
        self._read_paths.setdefault(label, []).extend(leaves)

    def add_read_bucket(self, label: str, keys: list[BucketKey]) -> None:
        self._read_buckets.setdefault(label, []).extend(keys)

    def add_read_block(self, label: str, keys: list[BlockKey]) -> None:
        self._read_blocks.setdefault(label, []).extend(keys)

    def add_read_list(self, label: str, indices: list[int] | None) -> None:
        """``indices=None`` reads the whole list; otherwise index calls accumulate."""
        if indices is None:
            self._read_lists[label] = None
            return
        existing = self._read_lists.get(label)
        if existing is None:
            existing = []
            self._read_lists[label] = existing
        existing.extend(indices)

    def add_write_path(self, label: str, data: PathData) -> None:
        self._write_paths.setdefault(label, {}).update(data)

    def add_write_bucket(self, label: str, data: BucketData) -> None:
        self._write_buckets.setdefault(label, {}).update(data)

    def add_write_block(self, label: str, data: BlockData) -> None:
        self._write_blocks.setdefault(label, {}).update(data)

    def add_write_list(self, label: str, data: dict[int, Any]) -> None:
        self._write_lists.setdefault(label, {}).update(data)

    @abstractmethod
    def init_connection(self, client: BaseSocket) -> None:
        raise NotImplementedError

    @abstractmethod
    def close_connection(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def init_storage(self, storage: ServerStorage) -> None:
        raise NotImplementedError

    @abstractmethod
    def execute(self) -> ExecuteResult:
        """Run all pending queries — writes first, then reads — and return an ExecuteResult."""
        raise NotImplementedError


class InteractLocalServer(InteractServer):
    """Local server: storage lives in the same process."""

    def __init__(self):
        super().__init__()
        self._storage: ServerStorage = {}

    @override
    def init_connection(self, client: BaseSocket | None = None) -> None:
        pass

    @override
    def close_connection(self) -> None:
        pass

    @override
    def init_storage(self, storage: ServerStorage) -> None:
        self._storage.update(storage)

    def _get_tree(self, label: str) -> BinaryTree:
        if label not in self._storage:
            raise KeyError(f"Label {label} is not hosted in the server storage.")
        # Tree labels always map to a BinaryTree (callers keep tree and list labels disjoint).
        return cast(BinaryTree, self._storage[label])

    def _get_list(self, label: str) -> list:
        if label not in self._storage:
            raise KeyError(f"Label {label} is not hosted in the server storage.")
        # List labels always map to a list (callers keep tree and list labels disjoint).
        return cast(list, self._storage[label])

    @override
    def execute(self) -> ExecuteResult:
        try:
            results = {}

            request = (
                self._read_paths,
                self._read_buckets,
                self._read_blocks,
                self._read_lists,
                self._write_paths,
                self._write_buckets,
                self._write_blocks,
                self._write_lists,
            )
            self._bytes_written += len(pickle.dumps(request))

            for label, data in self._write_paths.items():
                self._get_tree(label).write_path(data)

            for label, data in self._write_buckets.items():
                self._get_tree(label).write_bucket(data)

            for label, data in self._write_blocks.items():
                self._get_tree(label).write_block(data)

            for label, data in self._write_lists.items():
                lst = self._get_list(label)
                for idx, val in data.items():
                    if idx > -1:
                        lst[idx] = val
                    elif idx == -1:
                        lst.insert(0, val)
                    elif idx == -2:
                        lst.pop()
                    else:
                        raise ValueError(f"Invalid list index: {idx}")

            for label, leaves in self._read_paths.items():
                results[label] = self._get_tree(label).read_path(list(set(leaves)))

            for label, keys in self._read_buckets.items():
                results[label] = self._get_tree(label).read_bucket(list(set(keys)))

            for label, keys in self._read_blocks.items():
                results[label] = self._get_tree(label).read_block(list(set(keys)))

            for label, indices in self._read_lists.items():
                lst = self._get_list(label)
                results[label] = lst if indices is None else {i: lst[i] for i in set(indices)}

            result = ExecuteResult(success=True, results=results)
            self._bytes_read += len(pickle.dumps(result))
            return result

        except Exception as e:
            return ExecuteResult(success=False, error=str(e))

        finally:
            self.clear_queries()


class InteractRemoteServer(InteractServer):
    """Remote client that sends queries to a remote server."""

    def __init__(self):
        super().__init__()
        self._client: BaseSocket | None = None

    def _check_client(self) -> BaseSocket:
        if self._client is None:
            raise ValueError("Client has not been initialized; call init_connection() first.")
        return self._client

    @override
    def init_connection(self, client: BaseSocket) -> None:
        self._client = client

    @override
    def close_connection(self) -> None:
        self._check_client().close()

    @override
    def init_storage(self, storage: ServerStorage) -> None:
        client = self._check_client()
        client.send(("init", storage))
        response = client.recv()
        if response != SERVER_DEFAULT_RESPONSE:
            raise ValueError("Server failed to initialize storage.")

    @override
    def execute(self) -> ExecuteResult:
        try:
            client = self._check_client()

            request = {
                "read_paths": self._read_paths,
                "read_buckets": self._read_buckets,
                "read_blocks": self._read_blocks,
                "read_lists": self._read_lists,
                "write_paths": self._write_paths,
                "write_buckets": self._write_buckets,
                "write_blocks": self._write_blocks,
                "write_lists": self._write_lists,
            }

            self._bytes_written += len(pickle.dumps(request))

            client.send(("execute", request))
            response = client.recv()

            self._bytes_read += len(pickle.dumps(response))
            return response

        finally:
            self.clear_queries()


class RemoteServer(InteractLocalServer):
    """Server that listens for and answers remote client requests."""

    def __init__(self):
        super().__init__()
        self._server: BaseSocket | None = None

    def _process_request(self, request: tuple[str, Any]) -> Any:
        cmd, data = request

        if cmd == "init":
            self.init_storage(data)
            return SERVER_DEFAULT_RESPONSE

        elif cmd == "execute":
            self._read_paths = data.get("read_paths", {})
            self._read_buckets = data.get("read_buckets", {})
            self._read_blocks = data.get("read_blocks", {})
            self._read_lists = data.get("read_lists", {})
            self._write_paths = data.get("write_paths", {})
            self._write_buckets = data.get("write_buckets", {})
            self._write_blocks = data.get("write_blocks", {})
            self._write_lists = data.get("write_lists", {})

            return self.execute()

        else:
            raise ValueError(f"Unknown command: {cmd}")

    def run(self, server: BaseSocket) -> None:
        """Listen for client queries on a bound socket (e.g. ZMQSocket with is_server=True)."""
        self._server = server

        while True:
            request = self._server.recv()
            if request:
                self._server.send(self._process_request(request))
            else:
                break

        self._server.close()
