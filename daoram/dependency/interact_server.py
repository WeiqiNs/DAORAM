from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from daoram.dependency.binary_tree import BinaryTree, Buckets
from daoram.dependency.helper import Block, Query
from daoram.dependency.sockets import Socket

# Set a default response for the server.
SERVER_DEFAULT_RESPONSE = "Done!"
# Set a default port for the server and client to connect to.
PORT = 10000
# Define the type of storage the server should hold.
ServerStorage = Dict[str, Union[BinaryTree, List]]


class InteractServer(ABC):
    # All class inherits this list.
    read_queries: List[Query] = []
    write_queries: List[Query] = []

    """This abstract class defines the interface for interacting with the server."""
    @abstractmethod
    def init_connection(self) -> None:
        """Initialize the connection to the server."""
        raise NotImplementedError

    @abstractmethod
    def close_connection(self) -> None:
        """Close the connection to the server."""
        raise NotImplementedError

    @abstractmethod
    def init(self, storage: ServerStorage) -> None:
        """
        Issues an init query; sending some storage over to the server.

        Note that storage is a map, each storage type has a unique label.
        :param storage: The storage client wants server to hold.
        :return: No return, but should check whether the query is successfully issued.
        """
        raise NotImplementedError

    @abstractmethod
    def list_insert(self, label: str, index:int = 0, value:Any = None) -> None :
        raise NotImplementedError

    @abstractmethod
    def list_pop(self, label: str, index:int = -1) -> Any :
       raise NotImplementedError
    
    @abstractmethod
    def list_get(self, label: str, index:int) -> Any :
        raise NotImplementedError
    
    @abstractmethod
    def list_update(self, label: str, index:int, value:Any) -> None :
        raise NotImplementedError

    # @abstractmethod
    # def add_query(self, query: Query) -> None:
    #     raise NotImplementedError

    # @abstractmethod
    # def merge_and_sort_query(self) -> None:
    #     raise NotImplementedError

    # @abstractmethod
    # def execute_query(self) -> None:
    #     raise NotImplementedError
    
class InteractRemoteServer(InteractServer):
    def __init__(self, ip: str = "localhost", port: int = PORT):
        """Create an instance to talk to the remote server.

        :param ip: The IP address of the remote server.
        :param port: The port the remote server is listening on.
        """
        # Save the connection information.
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

    def init(self, storage: ServerStorage) -> None:
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

    def read_mul_query(self, label: List[str], leaf: Union[List[int], List[List[int]]]) -> List[Buckets]:
        """Issues a read query; telling the server to read one/multiple paths from some storage."""
        # Check for connection.
        self.__check_client()

        # Create the init query.
        query = [{"type": "r", "label": label[index], "leaf": each_leaf} for index, each_leaf in enumerate(leaf)]

        # Send the query to server.
        self.__client.send(query)

        # Get the server response.
        response = self.__client.recv()

        return response

    def write_query(self, label: str, leaf: Union[int, List[int]], data: Buckets) -> None:
            """Issues a "write" query; telling the server to write one/multiple paths to some storage."""
            # Check for connection.
            self.__check_client()

            # Create the init query.
            query = {"type": "w", "label": label, "leaf": leaf, "data": data}

            # Send the query to server.
            self.__client.send(query)

            # Check for response.
            self.__check_response()

    def write_mul_query(self, label: List[str], leaf: Union[List[int], List[List[int]]], data: List[Buckets]) -> None:
        # Check for connection.
        self.__check_client()

        # Create the init query.
        query = [
            {"type": "w", "label": label[index], "leaf": each_leaf, "data": data[index]}
            for index, each_leaf in enumerate(leaf)
        ]

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
        """Issues a "write" block query; telling the server to write one block to some storage."""
        # Check for connection.
        self.__check_client()

        # Create the init query.
        query = {"type": "wb", "label": label, "leaf": leaf, "bucket_id": bucket_id, "block_id": block_id, "data": data}

        # Send the query to server.
        self.__client.send(query)

        # Check for response.
        self.__check_response()

    def list_insert(self, label: str, index:int = 0, value: Any = None) -> None :
        """Issues a list insert query; telling the server to insert a value to some list."""
        # Check if the requested storage exists.
        self.__check_client()

        # Create the insert query.
        query = {"type": "li", "label": label, "index": index, "value": value}

        # Send the query to server.
        self.__client.send(query)
        self.__check_response()

    def list_pop(self, label: str, index:int = -1) -> Any :
        """Issues a list pop query; telling the server to pop a value from some list."""
        # Check if the requested storage exists.
        self.__check_client()

        # Create the pop query.
        query = {"type": "lp", "label": label, "index": index}

        # Send the query to server.
        self.__client.send(query)
        return self.__client.recv()
    
    def list_get(self, label: str, index:int) -> Any :
        """Issues a list get query; telling the server to get a value from some list."""
        # Check if the requested storage exists.
        self.__check_client()

        # Create the get query.
        query = {"type": "lg", "label": label, "index": index}
        self.__client.send(query)
        return self.__client.recv()

    def list_update(self, label: str, index:int, value: Any) -> None :
        """Issues a list upgrade query; telling the server to upgrade a value from some list."""
        # Check if the requested storage exists.
        self.__check_client()

        # Create the upgrade query.
        query = {"type": "lu", "label": label, "index": index, "value": value}

        # Send the query to server.
        self.__client.send(query)
        self.__check_response()



class InteractLocalServer(InteractServer):
    def __init__(self):
        """Create an instance for the local server with storages."""
        self.__storage: ServerStorage = {}

    def init_connection(self) -> None:
        """Since the server is local, pass."""
        pass

    def close_connection(self) -> None:
        """Since the server is local, pass."""
        pass

    def init(self, storage: ServerStorage) -> None:
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
        """Issues a "write" query; telling the server to write one/multiple paths from some storage."""
        # Skip it when writing a dummy 
        if data is None:
            return
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
        """Issues a "write" block query; telling the server to write one block to some storage."""
        # Check if the requested storage exists.
        if label not in self.__storage:
            raise KeyError(f"Label {label} is not hosted in the server storage.")

        # Depends on the label, write the storage.
        self.__storage[label].write_block(leaf=leaf, block_id=block_id, bucket_id=bucket_id, data=data)

    def list_insert(self, label: str, index:int = 0, value: Any = None) -> None :
        """Issues a list insert query; telling the server to insert a value to some list."""
        # Check if the requested storage exists.
        if label not in self.__storage:
            raise KeyError(f"Label {label} is not hosted in the server storage.")

        # Depends on the label, write the storage.
        self.__storage[label].insert(index, value)

    def list_pop(self, label: str, index:int = -1) -> Any :
        """Issues a list pop query; telling the server to pop a value from some list."""
        # Check if the requested storage exists.
        if label not in self.__storage:
            raise KeyError(f"Label {label} is not hosted in the server storage.")

        # Depends on the label, write the storage.
        return self.__storage[label].pop(index)

    def list_get(self, label:str, index:int) -> Any :
        if label not in self.__storage:
            raise KeyError(f"Label {label} is not hosted in the server storage.")

        # Depends on the label, write the storage.
        return self.__storage[label][index]
    
    def list_update(self, label:str, index:int, value:Any) -> None :
        if label not in self.__storage:
            raise KeyError(f"Label {label} is not hosted in the server storage.")

        # Depends on the label, write the storage.
        self.__storage[label][index] = value
class RemoteServer(InteractLocalServer):
    def __init__(self, ip: str = "localhost", port: int = PORT):
        """Creates a remote server instance that will answer client's queries.

        :param ip: The ip address of the remote server.
        :param port: The port the remote server is listening on.
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

    def __process_query(self, query: Union[dict, List[dict]]) -> Union[str, Block, Buckets, List[Buckets]]:
        """Process client's queries based on their types."""
        print(f"Received query: {query}")
        if query["type"] == "i":
            self.init(storage=query["storage"])
        elif query["type"] == "r":
            return self.read_query(label=query["label"], leaf=query["leaf"])
        elif query["type"] == "rm":
            return [self.read_query(label=each_query["label"], leaf=each_query["leaf"]) for each_query in query]
        elif query["type"] == "w":
            self.write_query(label=query["label"], leaf=query["leaf"], data=query["data"])
        elif query["type"] == "rw":
            for each_query in query:
                self.write_query(label=each_query["label"], leaf=each_query["leaf"], data=each_query["data"])
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
        elif query["type"] == "li":
                self.list_insert(label=query["label"], index=query["index"], value=query["value"])
        elif query["type"] == "lp":
                return self.list_pop(label=query["label"], index=query["index"])
        elif query["type"] == "lg":
                return self.list_get(label=query["label"], index=query["index"])
        elif query["type"] == "lu":
                self.list_update(label=query["label"], index=query["index"], value=query["value"])
        else:
            raise ValueError("Invalid query type was given.")
        return SERVER_DEFAULT_RESPONSE
    def run(self) -> None:
        """Run the server to listen to client queries."""
        # Initialize the socket and hearing to port.
        self.__init_connection()

        # Keep receiving queries.
        while True:
            # Receive a query from the client.
            query = self.__server.recv()
            # If the query has content, we process it.
            if query:
                self.__server.send(self.__process_query(query=query))
            else:
                break

        # Close the connection.
        self.__close_connection()
