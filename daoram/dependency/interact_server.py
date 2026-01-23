"""Client-server interaction layer for local and remote ORAM storage."""

import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

from daoram.dependency.binary_tree import BinaryTree
from daoram.dependency.sockets import BaseSocket
from daoram.dependency.types import PathData, BucketKey, BucketData, BlockKey, BlockData, ExecuteResult

# Set a default response for the server.
SERVER_DEFAULT_RESPONSE = "Done!"
# Set a default port for the server and client to connect to.
PORT = 10000
# Define the type of storage the server should hold.
ServerStorage = Dict[str, Union[BinaryTree, List]]


class InteractServer(ABC):
    """Abstract class defining the interface for interacting with storage server."""

    def __init__(self):
        """Initialize the query dictionaries for batching."""
        # Read queries: accumulate keys/indices to read.
        self._read_paths: Dict[str, List[int]] = {}
        self._read_buckets: Dict[str, List[BucketKey]] = {}
        self._read_blocks: Dict[str, List[BlockKey]] = {}
        self._read_queues: Dict[str, List[int] | None] = {}

        # Write queries: accumulate data to write (later overwrites earlier).
        self._write_paths: Dict[str, PathData] = {}
        self._write_buckets: Dict[str, BucketData] = {}
        self._write_blocks: Dict[str, BlockData] = {}
        # Queue writes: list of (operation, value) tuples.
        # Operations: "push_front", "push_back", "pop_front", "pop_back", ("set", index)
        self._write_queues: Dict[str, List[Tuple[str | Tuple[str, int], Any]]] = {}

        # Bandwidth counters (serialized bytes).
        self._bytes_read: int = 0
        self._bytes_written: int = 0

    def clear_queries(self) -> None:
        """Clear all pending queries."""
        self._read_paths.clear()
        self._read_buckets.clear()
        self._read_blocks.clear()
        self._read_queues.clear()
        self._write_paths.clear()
        self._write_buckets.clear()
        self._write_blocks.clear()
        self._write_queues.clear()

    def get_bandwidth(self) -> Tuple[int, int]:
        """Return (bytes_read, bytes_written)."""
        return self._bytes_read, self._bytes_written

    def reset_bandwidth(self) -> None:
        """Reset bandwidth counters to zero."""
        self._bytes_read = 0
        self._bytes_written = 0

    def add_read_path(self, label: str, leaves: List[int]) -> None:
        """Queue path read."""
        if label not in self._read_paths:
            self._read_paths[label] = []
        self._read_paths[label].extend(leaves)

    def add_read_bucket(self, label: str, keys: List[BucketKey]) -> None:
        """Queue bucket read."""
        if label not in self._read_buckets:
            self._read_buckets[label] = []
        self._read_buckets[label].extend(keys)

    def add_read_block(self, label: str, keys: List[BlockKey]) -> None:
        """Queue block read."""
        if label not in self._read_blocks:
            self._read_blocks[label] = []
        self._read_blocks[label].extend(keys)

    def add_queue_read(self, label: str, indices: List[int] = None) -> None:
        """Queue read. If indices is None, read entire queue."""
        if indices is None:
            self._read_queues[label] = None
        else:
            if label not in self._read_queues or self._read_queues[label] is None:
                self._read_queues[label] = []
            self._read_queues[label].extend(indices)

    def add_write_path(self, label: str, data: PathData) -> None:
        """Queue path write."""
        if label not in self._write_paths:
            self._write_paths[label] = {}
        self._write_paths[label].update(data)

    def add_write_bucket(self, label: str, data: BucketData) -> None:
        """Queue bucket write."""
        if label not in self._write_buckets:
            self._write_buckets[label] = {}
        self._write_buckets[label].update(data)

    def add_write_block(self, label: str, data: BlockData) -> None:
        """Queue block write."""
        if label not in self._write_blocks:
            self._write_blocks[label] = {}
        self._write_blocks[label].update(data)

    def add_queue_push_front(self, label: str, value: Any) -> None:
        """Push value to front of queue."""
        if label not in self._write_queues:
            self._write_queues[label] = []
        self._write_queues[label].append(("push_front", value))

    def add_queue_push_back(self, label: str, value: Any) -> None:
        """Push value to back of queue."""
        if label not in self._write_queues:
            self._write_queues[label] = []
        self._write_queues[label].append(("push_back", value))

    def add_queue_pop_front(self, label: str) -> None:
        """Pop value from front of queue."""
        if label not in self._write_queues:
            self._write_queues[label] = []
        self._write_queues[label].append(("pop_front", None))

    def add_queue_pop_back(self, label: str) -> None:
        """Pop value from back of queue."""
        if label not in self._write_queues:
            self._write_queues[label] = []
        self._write_queues[label].append(("pop_back", None))

    def add_queue_set(self, label: str, index: int, value: Any) -> None:
        """Set value at index in queue."""
        if label not in self._write_queues:
            self._write_queues[label] = []
        self._write_queues[label].append((("set", index), value))

    @abstractmethod
    def init_connection(self, client: BaseSocket) -> None:
        """Initialize the connection to the server."""
        raise NotImplementedError

    @abstractmethod
    def close_connection(self) -> None:
        """Close the connection to the server."""
        raise NotImplementedError

    @abstractmethod
    def init_storage(self, storage: ServerStorage) -> None:
        """Initialize storages from a dictionary."""
        raise NotImplementedError

    @abstractmethod
    def execute(self) -> ExecuteResult:
        """Execute all pending queries (writes first, then reads)."""
        raise NotImplementedError


class InteractLocalServer(InteractServer):
    """Local server implementation - storage is in the same process."""

    def __init__(self):
        """Create an instance for the local server with storages."""
        super().__init__()
        self._storage: ServerStorage = {}

    def init_connection(self, client: BaseSocket = None) -> None:
        """Since the server is local, no connection needed."""
        pass

    def close_connection(self) -> None:
        """Since the server is local, no connection to close."""
        pass

    def init_storage(self, storage: ServerStorage) -> None:
        """Register storages from a dictionary."""
        self._storage.update(storage)

    def _get_tree(self, label: str) -> BinaryTree:
        """Get tree storage by label."""
        if label not in self._storage:
            raise KeyError(f"Label {label} is not hosted in the server storage.")
        return self._storage[label]

    def _get_queue(self, label: str) -> List:
        """Get queue storage by label."""
        if label not in self._storage:
            raise KeyError(f"Label {label} is not hosted in the server storage.")
        return self._storage[label]

    def execute(self) -> ExecuteResult:
        """Execute all pending queries (writes first, then reads)."""
        try:
            results = {}

            # Track bytes written (queries + data sent to server).
            request = (
                self._read_paths, self._read_buckets, self._read_blocks, self._read_queues,
                self._write_paths, self._write_buckets, self._write_blocks, self._write_queues,
            )
            self._bytes_written += len(pickle.dumps(request))

            # Execute all writes.
            for label, data in self._write_paths.items():
                self._get_tree(label).write_path(data)

            for label, data in self._write_buckets.items():
                self._get_tree(label).write_bucket(data)

            for label, data in self._write_blocks.items():
                self._get_tree(label).write_block(data)

            for label, operations in self._write_queues.items():
                queue = self._get_queue(label)
                for op, value in operations:
                    if op == "push_front":
                        queue.insert(0, value)
                    elif op == "push_back":
                        queue.append(value)
                    elif op == "pop_front":
                        # Store popped value in results.
                        results[label] = queue.pop(0)
                    elif op == "pop_back":
                        # Store popped value in results.
                        results[label] = queue.pop()
                    elif isinstance(op, tuple) and op[0] == "set":
                        queue[op[1]] = value
                    else:
                        raise ValueError(f"Invalid queue operation: {op}")

            # Execute all reads (deduplicate keys).
            for label, leaves in self._read_paths.items():
                results[label] = self._get_tree(label).read_path(list(set(leaves)))

            for label, keys in self._read_buckets.items():
                results[label] = self._get_tree(label).read_bucket(list(set(keys)))

            for label, keys in self._read_blocks.items():
                results[label] = self._get_tree(label).read_block(list(set(keys)))

            for label, indices in self._read_queues.items():
                queue = self._get_queue(label)
                if indices is None:
                    results[label] = list(queue)
                else:
                    results[label] = {i: queue[i] for i in set(indices)}

            # Track bytes read (data received from server).
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
        """Create an instance to talk to the remote server."""
        super().__init__()
        self._client: BaseSocket | None = None

    def _check_client(self) -> None:
        """Check if the client is connected to the server."""
        if self._client is None:
            raise ValueError("Client has not been initialized; call init_connection() first.")

    def init_connection(self, client: BaseSocket) -> None:
        """Initialize the connection to the server."""
        self._client = client

    def close_connection(self) -> None:
        """Close the connection to the server."""
        self._client.close()

    def init_storage(self, storage: ServerStorage) -> None:
        """Initialize storages on the remote server."""
        self._check_client()
        # Send storage init request.
        self._client.send(("init", storage))
        response = self._client.recv()
        if response != SERVER_DEFAULT_RESPONSE:
            raise ValueError("Server failed to initialize storage.")

    def execute(self) -> ExecuteResult:
        """Execute all pending queries on the remote server."""
        self._check_client()

        # Package all queries into a single request.
        request = {
            "read_paths": self._read_paths,
            "read_buckets": self._read_buckets,
            "read_blocks": self._read_blocks,
            "read_queues": self._read_queues,
            "write_paths": self._write_paths,
            "write_buckets": self._write_buckets,
            "write_blocks": self._write_blocks,
            "write_queues": self._write_queues,
        }

        # Track bytes written (request sent to server).
        self._bytes_written += len(pickle.dumps(request))

        # Send request and receive response.
        self._client.send(("execute", request))
        response = self._client.recv()

        # Track bytes read (response from server).
        self._bytes_read += len(pickle.dumps(response))
        self.clear_queries()
        return response


class RemoteServer(InteractLocalServer):
    """Server that listens for remote client requests."""

    def __init__(self):
        """Creates a remote server instance that will answer client's queries."""
        super().__init__()
        self._server: BaseSocket | None = None

    def _process_request(self, request: Tuple[str, Any]) -> Any:
        """Process client request."""
        cmd, data = request

        if cmd == "init":
            # Initialize storage.
            self.init_storage(data)
            return SERVER_DEFAULT_RESPONSE

        elif cmd == "execute":
            # Load queries from request into our query dicts.
            self._read_paths = data.get("read_paths", {})
            self._read_buckets = data.get("read_buckets", {})
            self._read_blocks = data.get("read_blocks", {})
            self._read_queues = data.get("read_queues", {})
            self._write_paths = data.get("write_paths", {})
            self._write_buckets = data.get("write_buckets", {})
            self._write_blocks = data.get("write_blocks", {})
            self._write_queues = data.get("write_queues", {})

            # Execute and return results.
            return self.execute()

        else:
            raise ValueError(f"Unknown command: {cmd}")

    def run(self, server: BaseSocket) -> None:
        """Run the server to listen for client queries."""
        self._server = server

        while True:
            request = self._server.recv()
            if request:
                self._server.send(self._process_request(request))
            else:
                break

        self._server.close()
