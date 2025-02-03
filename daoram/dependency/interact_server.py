from abc import ABC, abstractmethod
from typing import Dict, List, Union

from daoram.dependency.binary_tree import BinaryTree, Buckets
from daoram.dependency.helper import Block
from daoram.dependency.sockets import Socket

# Set a default response for the server.
SERVER_DEFAULT_RESPONSE = "Done!"
# Set a default port for the server and client to connect to.
PORT = 10000
# Define the type of storage the server should hold.
ServerStorage = Dict[str, BinaryTree]


class InteractServer(ABC):
    @abstractmethod
    def init_connection(self) -> None:
        """Initialize the connection to the server."""
        raise NotImplementedError

    @abstractmethod
    def close_connection(self) -> None:
        """Close the connection to the server."""
        raise NotImplementedError

    @abstractmethod
    def init_query(self, storage: ServerStorage) -> None:
        """
        Issues an init query; sending some storage over to the server.

        Note that storage is either one BinaryTree or a list of BinaryTrees, in the case of position map oram storages.
        :param storage: the storage client wants server to hold.
        :return: no return, but should check whether the query is successfully issues.
        """
        raise NotImplementedError

    @abstractmethod
    def read_query(self, label: str, leaf: Union[int, List[int]]) -> Buckets:
        """
        Issues a read query; telling the server to read one/multiple path from some storage.

        :param label: the label indicating what kind of storage is being loaded.
        :param leaf: the leaf of one path or leaves of multiples paths to retrieve data.
        :return: the desired path data.
        """
        raise NotImplementedError

    @abstractmethod
    def write_query(self, label: str, leaf: Union[int, List[int]], data: Buckets) -> None:
        """
        Issues a write query; telling the server to write one/multiple paths to some storage.

        :param label: the label indicating what kind of storage is being loaded.
        :param leaf: the leaf of one path or leaves of multiples paths to retrieve data.
        :param data: the data to write to the path(s).
        :return: no return, but should check whether the query is successfully issues.
        """
        raise NotImplementedError

    @abstractmethod
    def read_block_query(self, label: str, leaf: int, bucket_id: int, block_id: int) -> Block:
        """
        Issues a read query; telling the server to read one block from some storage.

        :param label: the label indicating what kind of storage is being loaded.
        :param leaf: the leaf of one path or leaves of multiples paths to retrieve data.
        :param bucket_id: the index of which bucket on the path is of interest.
        :param block_id: the index of which block in the bucket is of interest.
        :return: the desired block data.
        """
        raise NotImplementedError

    @abstractmethod
    def write_block_query(self, label: str, leaf: int, bucket_id: int, block_id: int, data: Block) -> None:
        """
        Issues a write query; telling the server to write one block to some storage.

        :param label: the label indicating what kind of storage is being loaded.
        :param leaf: the leaf of one path or leaves of multiples paths to retrieve data.
        :param bucket_id: the index of which bucket on the path is of interest.
        :param block_id: the index of which block in the bucket is of interest.
        :param data: the data to write to the block.
        :return: no return, but should check whether the query is successfully issues.
        """
        raise NotImplementedError


class InteractRemoteServer(InteractServer):
    def __init__(self, ip: str = "localhost", port: int = PORT):
        """Create an instance to talk to the remote server.

        :param ip: the IP address of the remote server.
        :param port: the port the remote server is listening on.
        """
        # Save the connection information.
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
        # Get response from server.
        response = self.__client.recv()
        # Check if it is the default response.
        if response != SERVER_DEFAULT_RESPONSE:
            raise ValueError("Server did not give an expected response, the operation may have failed.")

    def init_connection(self) -> None:
        """Initialize the connection to the server."""
        # To make sure the socket won't close, we return it to the place where this is called.
        self.__client = Socket(ip=self.__ip, port=self.__port, is_server=False)

    def close_connection(self) -> None:
        """Close the connection to the server."""
        self.__client.close()

    def init_query(self, storage: ServerStorage) -> None:
        """Issues an init query; sending some storage over to the server."""
        # Check for connection.
        self.__check_client()

        # Create the init query.
        query = {"type": "i", "storage": storage}

        # Send the query to server.
        self.__client.send(query)

        # Check for response.
        self.__check_response()

    def read_query(self, label: str, leaf: Union[int, List[int]]) -> Buckets:
        """Issues a read query; telling the server to read one/multiple paths from some storage."""
        # Check for connection.
        self.__check_client()

        # Create the init query.
        query = {"type": "r", "label": label, "leaf": leaf}

        # Send the query to server.
        self.__client.send(query)

        # Get the server response.
        response = self.__client.recv()

        return response

    def write_query(self, label: str, leaf: Union[int, List[int]], data: Buckets) -> None:
        """Issues a write query; telling the server to write one/multiple paths to some storage."""
        # Check for connection.
        self.__check_client()

        # Create the init query.
        query = {"type": "w", "label": label, "leaf": leaf, "data": data}

        # Send the query to server.
        self.__client.send(query)

        # Check for response.
        self.__check_response()

    def read_block_query(self, label: str, leaf: int, bucket_id: int, block_id: int) -> Block:
        """Issues a read block query; telling the server to read one block from some storage."""
        # Check for connection.
        self.__check_client()

        # Create the init query.
        query = {"type": "rb", "label": label, "leaf": leaf, "bucket_id": bucket_id, "block_id": block_id}

        # Send the query to server.
        self.__client.send(query)

        # Get the server response.
        return self.__client.recv()

    def write_block_query(self, label: str, leaf: int, bucket_id: int, block_id: int, data: Block) -> None:
        """Issues a write block query; telling the server to write one block to some storage."""
        # Check for connection.
        self.__check_client()

        # Create the init query.
        query = {"type": "wb", "label": label, "leaf": leaf, "bucket_id": bucket_id, "block_id": block_id, "data": data}

        # Send the query to server.
        self.__client.send(query)

        # Check for response.
        self.__check_response()


class InteractLocalServer(InteractServer):
    def __init__(self):
        """Create an instance for the local server with storages."""
        self.__storage: Dict[str, BinaryTree] = {}

    def init_connection(self) -> None:
        """Since the server is local, just pass."""
        pass

    def close_connection(self) -> None:
        """Since the server is local, just pass."""
        pass

    def init_query(self, storage: ServerStorage) -> None:
        """Issues an init query; sending some storage over to the server."""
        # Save the storage.
        self.__storage.update(storage)

    def read_query(self, label: str, leaf: Union[int, List[int]]) -> Buckets:
        """Issues a read query; telling the server to read one/multiple paths from some storage."""
        # Check if the requested storage exists.
        if label not in self.__storage:
            raise KeyError(f"Label {label} is not hosted in the server storage.")

        # Depends on the label, read the storage.
        return self.__storage[label].read_path(leaf=leaf)

    def write_query(self, label: str, leaf: Union[int, List[int]], data: Buckets) -> None:
        """Issues a write query; telling the server to write one/multiple paths from some storage."""
        # Check if the requested storage exists.
        if label not in self.__storage:
            raise KeyError(f"Label {label} is not hosted in the server storage.")

        # Depends on the label, write the storage.
        self.__storage[label].write_path(leaf=leaf, data=data)

    def read_block_query(self, label: str, leaf: int, bucket_id: int, block_id: int) -> Block:
        """Issues a read block query; telling the server to read one block from some storage."""
        # Check if the requested storage exists.
        if label not in self.__storage:
            raise KeyError(f"Label {label} is not hosted in the server storage.")

        # Depends on the label, read the storage.
        return self.__storage[label].read_block(leaf=leaf, block_id=block_id, bucket_id=bucket_id)

    def write_block_query(self, label: str, leaf: int, bucket_id: int, block_id: int, data: Block) -> None:
        """Issues a write block query; telling the server to write one block to some storage."""
        # Check if the requested storage exists.
        if label not in self.__storage:
            raise KeyError(f"Label {label} is not hosted in the server storage.")

        # Depends on the label, write the storage.
        self.__storage[label].write_block(leaf=leaf, block_id=block_id, bucket_id=bucket_id, data=data)


class RemoteServer(InteractLocalServer):
    def __init__(self, ip: str = "localhost", port: int = PORT):
        """Creates a remote server instance that will answer client's queries.

        :param ip: the ip address of the remote server.
        :param port: the port the remote server is listening on.
        """
        # Initialize the super class to create the storages.
        super().__init__()
        # Create the server instance and have it running.
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

    def __process_query(self, query: dict) -> Union[str, Block, Buckets]:
        """Process client's queries based on their types."""
        if query["type"] == "i":
            self.init_query(storage=query["storage"])
            return SERVER_DEFAULT_RESPONSE

        elif query["type"] == "r":
            return self.read_query(label=query["label"], leaf=query["leaf"])

        elif query["type"] == "w":
            self.write_query(label=query["label"], leaf=query["leaf"], data=query["data"])
            return SERVER_DEFAULT_RESPONSE

        elif query["type"] == "rb":
            return self.read_block_query(
                label=query["label"],
                leaf=query["leaf"],
                block_id=query["block_id"],
                bucket_id=query["bucket_id"],
            )

        elif query["type"] == "wb":
            self.write_block_query(
                label=query["label"],
                leaf=query["leaf"],
                bucket_id=query["bucket_id"],
                block_id=query["block_id"],
                data=query["data"]
            )
            return SERVER_DEFAULT_RESPONSE

        else:
            raise ValueError("Invalid query type was given.")

    def run(self) -> None:
        """Run the server to listen to client queries."""
        # Initialize the socket and hearing to port.
        self.__init_connection()

        # Keep receiving queries.
        while True:
            # Receive query from the client.
            query = self.__server.recv()
            # If query has content, we process it.
            if query:
                self.__server.send(self.__process_query(query=query))
            else:
                break

        # Close the connection.
        self.__close_connection()
