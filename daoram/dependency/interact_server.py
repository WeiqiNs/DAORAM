from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from daoram.dependency.binary_tree import BinaryTree
from daoram.dependency.sockets import Socket
from daoram.dependency.types import (
    Query,
    TREE_READ_PATH,
    TREE_WRITE_PATH,
    TREE_READ_BUCKET,
    TREE_WRITE_BUCKET,
    TREE_READ_BLOCK,
    TREE_WRITE_BLOCK,
    TREE_INIT,
    LIST_READ,
    LIST_WRITE,
    LIST_INIT,
    TreeReadPathPayload,
    TreeWritePathPayload,
    Buckets,
    PathData,
)

# Set a default response for the server.
SERVER_DEFAULT_RESPONSE = "Done!"
# Set a default port for the server and client to connect to.
PORT = 10000
# Define the type of storage the server should hold.
ServerStorage = Dict[str, Union[BinaryTree, List]]


class InteractServer(ABC):
    """This abstract class defines the interface for interacting with the server."""

    def __init__(self):
        """Initialize the query lists for batching."""
        self._read_queries: List[Query] = []
        self._write_queries: List[Query] = []

    @property
    def read_queries(self) -> List[Query]:
        """Get the list of pending read queries."""
        return self._read_queries

    @property
    def write_queries(self) -> List[Query]:
        """Get the list of pending write queries."""
        return self._write_queries

    def add_query(self, query: Query) -> None:
        """
        Add a query to the appropriate pending list for batch execution.

        :param query: The query to add.
        """
        # Note that we don't handle duplications, and allow for duplicated read.
        if query.query_type in (TREE_READ_PATH, TREE_READ_BUCKET, TREE_READ_BLOCK, LIST_READ):
            self._read_queries.append(query)
        else:
            # Duplicated writes will be overwritten with the later ones.
            self._write_queries.append(query)

    def clear_queries(self) -> None:
        """Clear all pending queries."""
        self._read_queries.clear()
        self._write_queries.clear()

    @abstractmethod
    def init_connection(self) -> None:
        """Initialize the connection to the server."""
        raise NotImplementedError

    @abstractmethod
    def close_connection(self) -> None:
        """Close the connection to the server."""
        raise NotImplementedError

    @abstractmethod
    def init_storage(self, storage: ServerStorage) -> None:
        """
        Initialize storages from a dictionary.

        :param storage: Dict mapping labels to storage objects (BinaryTree or List).
        """
        raise NotImplementedError

    @abstractmethod
    def execute_query(self, query: Query) -> Any:
        """
        Execute a single query against the appropriate storage.

        :param query: The query to execute.
        :return: The result of the query (if any).
        """
        raise NotImplementedError

    @abstractmethod
    def execute_queries(self) -> List[Any]:
        """
        Execute all pending queries (writes first, then reads) and clear the lists.

        :return: List of results from read queries.
        """
        raise NotImplementedError

    def read_query(self, label: str, leaf: Union[int, List[int]]) -> Buckets:
        """
        Convenience method: Read path(s) from a BinaryTree storage.

        :param label: The storage label.
        :param leaf: A single leaf index or a list of leaf indices.
        :return: The buckets read from the path(s).
        """
        # Normalize leaf to list
        leaves = [leaf] if isinstance(leaf, int) else leaf

        query = Query(
            payload=TreeReadPathPayload(leaves=leaves),
            query_type=TREE_READ_PATH,
            storage_label=label
        )
        return self.execute_query(query)

    def write_query(self, label: str, data: PathData) -> None:
        """
        Convenience method: Write path(s) to a BinaryTree storage.

        :param label: The storage label.
        :param data: PathData dict mapping storage index to bucket.
        """
        query = Query(
            payload=TreeWritePathPayload(data=data),
            query_type=TREE_WRITE_PATH,
            storage_label=label
        )
        self.execute_query(query)


class InteractRemoteServer(InteractServer):
    def __init__(self, ip: str = "localhost", port: int = PORT):
        """Create an instance to talk to the remote server.

        :param ip: The IP address of the remote server.
        :param port: The port the remote server is listening on.
        """
        super().__init__()
        self.__ip = ip
        self.__port = port
        self.__client = None

    def update_ip(self, ip: str) -> None:
        """Updates the ip address of the server."""
        self.__ip = ip

    def update_port(self, port: int) -> None:
        """Updates the port of the server."""
        self.__port = port

    def __check_client(self) -> None:
        """Check if the client is connected to the server."""
        if self.__client is None:
            raise ValueError("Client has not been initialized; call init_connection() first.")

    def __check_response(self) -> None:
        """Check whether the server successfully executes the request."""
        response = self.__client.recv()
        if response != SERVER_DEFAULT_RESPONSE:
            raise ValueError("Server did not give an expected response, the operation may have failed.")

    def init_connection(self) -> None:
        """Initialize the connection to the server."""
        self.__client = Socket(ip=self.__ip, port=self.__port, is_server=False)

    def close_connection(self) -> None:
        """Close the connection to the server."""
        self.__client.close()

    def init_storage(self, storage: ServerStorage) -> None:
        """Initialize storages on the remote server."""
        self.__check_client()

        for label, store in storage.items():
            if isinstance(store, BinaryTree):
                query = Query(payload=store, query_type=TREE_INIT, storage_label=label)
            else:
                query = Query(payload=store, query_type=LIST_INIT, storage_label=label)

            self.__client.send(query)
            self.__check_response()

    def execute_query(self, query: Query) -> Any:
        """Execute a single query on the remote server."""
        self.__check_client()

        # Send the query directly to the server
        self.__client.send(query)

        # For write operations, just check response; for reads, return the data
        if query.query_type in (TREE_WRITE_PATH, TREE_WRITE_BUCKET, TREE_WRITE_BLOCK, LIST_WRITE):
            self.__check_response()
            return None
        else:
            return self.__client.recv()

    def execute_queries(self) -> List[Any]:
        """Execute all pending queries (writes first, then reads) and clear the lists."""
        self.__check_client()

        # Combine writes and reads (writes first, then reads)
        all_queries = self._write_queries + self._read_queries

        if not all_queries:
            return []

        # Send all queries as a batch
        self.__client.send(all_queries)

        # Receive the response
        response = self.__client.recv()

        # Clear the query lists
        self.clear_queries()

        # Return results (server returns list of read results)
        return response if isinstance(response, list) else []


class InteractLocalServer(InteractServer):
    def __init__(self):
        """Create an instance for the local server with storages."""
        super().__init__()
        self._storage: ServerStorage = {}

    def init_connection(self) -> None:
        """Since the server is local, no connection needed."""
        pass

    def close_connection(self) -> None:
        """Since the server is local, no connection to close."""
        pass

    def init_storage(self, storage: ServerStorage) -> None:
        """Register storages from a dictionary."""
        self._storage.update(storage)

    def _get_storage(self, label: str) -> Union[BinaryTree, List]:
        """Get storage by label, raising KeyError if not found."""
        if label not in self._storage:
            raise KeyError(f"Label {label} is not hosted in the server storage.")
        return self._storage[label]

    def execute_query(self, query: Query) -> Any:
        """Execute a single query against the appropriate storage."""
        storage = self._get_storage(query.storage_label)

        # Dispatch based on storage type
        if isinstance(storage, BinaryTree):
            return self._handle_tree_query(storage, query)
        elif isinstance(storage, list):
            return self._handle_list_query(storage, query)
        else:
            raise TypeError(f"Unknown storage type: {type(storage)}")

    def execute_queries(self) -> List[Any]:
        """Execute all pending queries (writes first, then reads) and clear the lists."""
        results = []

        # Execute write queries first
        for query in self._write_queries:
            self.execute_query(query)

        # Execute read queries
        for query in self._read_queries:
            result = self.execute_query(query)
            results.append(result)

        # Clear the query lists
        self.clear_queries()

        return results

    @staticmethod
    def _handle_tree_query(tree: BinaryTree, query: Query) -> Any:
        """Handle queries for BinaryTree storage."""
        payload = query.payload
        query_type = query.query_type

        if query_type == TREE_READ_PATH:
            return tree.read_path(payload.leaves)
        elif query_type == TREE_WRITE_PATH:
            tree.write_path(payload.data)
        elif query_type == TREE_READ_BUCKET:
            return tree.read_bucket(payload.keys)
        elif query_type == TREE_WRITE_BUCKET:
            tree.write_bucket(payload.data)
        elif query_type == TREE_READ_BLOCK:
            return tree.read_block(payload.keys)
        elif query_type == TREE_WRITE_BLOCK:
            tree.write_block(payload.data)
        else:
            raise ValueError(f"Unknown tree query type: {query_type}")

    @staticmethod
    def _handle_list_query(lst: list, query: Query) -> Any:
        """Handle queries for List storage."""
        payload = query.payload
        query_type = query.query_type

        if query_type == LIST_READ:
            return {i: lst[i] for i in payload.indices}

        elif query_type == LIST_WRITE:
            for idx, val in payload.data.items():
                lst[idx] = val
            return None

        else:
            raise ValueError(f"Unknown list query type: {query_type}")


class RemoteServer(InteractLocalServer):
    def __init__(self, ip: str = "localhost", port: int = PORT):
        """Creates a remote server instance that will answer client's queries.

        :param ip: The ip address of the remote server.
        :param port: The port the remote server is listening on.
        """
        super().__init__()
        self.__server = None
        self.__port = port
        self.__ip = ip

    def update_ip(self, ip: str) -> None:
        """Updates the ip address of the server."""
        self.__ip = ip

    def update_port(self, port: int) -> None:
        """Updates the port of the server."""
        self.__port = port

    def __init_connection(self) -> None:
        """Create the server instance and have it running."""
        self.__server = Socket(ip=self.__ip, port=self.__port, is_server=True)

    def __close_connection(self) -> None:
        """Shuts off the server."""
        self.__server.close()

    def __process_query(self, query_data: Union[Query, List[Query]]) -> Any:
        """Process client's queries based on their types."""
        queries = query_data if isinstance(query_data, list) else [query_data]

        # Handle init queries separately, add others to batch
        for query in queries:
            if query.query_type in (TREE_INIT, LIST_INIT):
                self.init_storage({query.storage_label: query.payload})
            else:
                self.add_query(query)

        # Execute all batched queries and get results
        results = self.execute_queries()

        return results if results else SERVER_DEFAULT_RESPONSE

    def run(self) -> None:
        """Run the server to listen to client queries."""
        self.__init_connection()

        while True:
            query = self.__server.recv()
            if query:
                self.__server.send(self.__process_query(query_data=query))
            else:
                break

        self.__close_connection()
