from abc import ABC, abstractmethod
from typing import List, Optional, Union

from daoram.dependency.binary_tree import BinaryTree, Bucket, Buckets, Data
from daoram.dependency.sockets import Socket

# Set a default response for the server.
SERVER_DEFAULT_RESPONSE = "Done!"


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
    def init_query(self, label: str, storage: Union[BinaryTree, List[BinaryTree]]) -> None:
        """
        Issues an init query; sending some storage over to the server.

        Note that storage is either one BinaryTree or a list of BinaryTrees, in the case of position map oram storages.
        :param label: the label indicating what kind of storage is being sent.
        :param storage: the storage client wants server to hold.
        :return: no return, but should check whether the query is successfully issues.
        """
        raise NotImplementedError

    @abstractmethod
    def load_query(self, label: str, file: str) -> None:
        """
        Issues a load query; telling the server to load some storage from the disk.

        :param label: the label indicating what kind of storage is being loaded.
        :param file: the name of the file to load.
        :return: no return, but should check whether the query is successfully issues.
        """
        raise NotImplementedError

    @abstractmethod
    def read_query(self, label: str, leaf: Union[int, List[int]], index: Optional[int] = None) -> Buckets:
        """
        Issues a read query; telling the server to read one/multiple path from some storage.

        :param label: the label indicating what kind of storage is being loaded.
        :param leaf: the leaf of one path or leaves of multiples paths to retrieve data.
        :param index: the index of which position map oram storage to read; can be None if not reading this type.
        :return: the desired path data.
        """
        raise NotImplementedError

    @abstractmethod
    def write_query(self, label: str, leaf: Union[int, List[int]], data: Buckets, index: Optional[int] = None) -> None:
        """
        Issues a write query; telling the server to write one/multiple paths to some storage.

        :param label: the label indicating what kind of storage is being loaded.
        :param leaf: the leaf of one path or leaves of multiples paths to retrieve data.
        :param data: the data to write to the path(s).
        :param index: the index of which position map oram storage to read; can be None if not reading this type.
        :return: no return, but should check whether the query is successfully issues.
        """
        raise NotImplementedError

    @abstractmethod
    def read_block_query(
            self, label: str, leaf: int, bucket_id: int, block_id: int, index: Optional[int] = None) -> Buckets:
        """
        Issues a read query; telling the server to read one block from some storage.

        :param label: the label indicating what kind of storage is being loaded.
        :param leaf: the leaf of one path or leaves of multiples paths to retrieve data.
        :param bucket_id: the index of which bucket on the path is of interest.
        :param block_id: the index of which block in the bucket is of interest.
        :param index: the index of which position map oram storage to read; can be None if not reading this type.
        :return: the desired block data.
        """
        raise NotImplementedError

    @abstractmethod
    def read_meta_block_query(
            self, label: str, leaf: int, bucket_id: int, block_id: int, index: Optional[int] = None) -> Buckets:
        """
        Issues a read query; telling the server to read metadata of one block from some storage.

        :param label: the label indicating what kind of storage is being loaded.
        :param leaf: the leaf of one path or leaves of multiples paths to retrieve data.
        :param bucket_id: the index of which bucket on the path is of interest.
        :param block_id: the index of which block in the bucket is of interest.
        :param index: the index of which position map oram storage to read; can be None if not reading this type.
        :return: the desired block metadata.
        """
        raise NotImplementedError

    @abstractmethod
    def write_block_query(
            self, label: str, leaf: int, bucket_id: int, block_id: int, data: Buckets, index: Optional[int] = None
    ) -> None:
        """
        Issues a write query; telling the server to write one block to some storage.

        :param label: the label indicating what kind of storage is being loaded.
        :param leaf: the leaf of one path or leaves of multiples paths to retrieve data.
        :param bucket_id: the index of which bucket on the path is of interest.
        :param block_id: the index of which block in the bucket is of interest.
        :param data: the data to write to the block.
        :param index: the index of which position map oram storage to read; can be None if not reading this type.
        :return: no return, but should check whether the query is successfully issues.
        """
        raise NotImplementedError

    @abstractmethod
    def read_stash_query(self, label: str, stash: int, index: Optional[int] = None) -> Data:
        """
        Issues a read query; telling the server to read one data from the stash.

        :param label: the label indicating what kind of storage is being loaded.
        :param stash: the index of which data in the stash in of interest.
        :param index: the index of which position map oram stash to read; can be None if not reading this type.
        :return: the desired data in stash.
        """
        raise NotImplementedError

    @abstractmethod
    def write_stash_query(self, label: str, stash: int, data: Data, index: Optional[int] = None) -> None:
        """
        Issues a write query; telling the server to write one data to the stash.

        :param label: the label indicating what kind of storage is being loaded.
        :param stash: the index of which data in the stash in of interest.
        :param data: the data to write to the stash.
        :param index: the index of which position map oram stash to read; can be None if not reading this type.
        :return: the desired data in stash.
        """
        raise NotImplementedError


class InteractRemoteServer(InteractServer):
    def __init__(self, ip: str = "localhost", port: int = 7779):
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

    def init_query(self, label: str, storage: Union[BinaryTree, List[BinaryTree]]) -> None:
        """Issues an init query; sending some storage over to the server."""
        # Check for connection.
        self.__check_client()

        # Create the init query.
        query = {"type": "init", "label": label, "storage": storage}

        # Send the query to server.
        self.__client.send(query)

        # Check for response.
        self.__check_response()

    def load_query(self, label: str, file: str) -> None:
        """Issues a load query; telling the server to load some storage from the disk."""
        # Check for connection.
        self.__check_client()

        # Create the load query.
        query = {"type": "load", "label": label, "file": file}

        # Send the query to server.
        self.__client.send(query)

        # Check for response.
        self.__check_response()

    def read_query(self, label: str, leaf: Union[int, List[int]], index: Optional[int] = None) -> Buckets:
        """Issues a read query; telling the server to read one/multiple paths from some storage."""
        # Check for connection.
        self.__check_client()

        # Create the init query.
        query = {"type": "read", "label": label, "leaf": leaf, "index": index}

        # Send the query to server.
        self.__client.send(query)

        # Get the server response.
        response = self.__client.recv()

        return response

    def write_query(self, label: str, leaf: Union[int, List[int]], data: Buckets, index: Optional[int] = None) -> None:
        """Issues a write query; telling the server to write one/multiple paths to some storage."""
        # Check for connection.
        self.__check_client()

        # Create the init query.
        query = {"type": "write", "label": label, "leaf": leaf, "data": data, "index": index}

        # Send the query to server.
        self.__client.send(query)

        # Check for response.
        self.__check_response()

    def read_block_query(
            self, label: str, leaf: int, bucket_id: int, block_id: int, index: Optional[int] = None) -> Buckets:
        """Issues a read block query; telling the server to read one block from some storage."""
        # Check for connection.
        self.__check_client()

        # Create the init query.
        query = {
            "type": "read_block",
            "label": label,
            "leaf": leaf,
            "bucket_id": bucket_id,
            "block_id": block_id,
            "index": index
        }

        # Send the query to server.
        self.__client.send(query)

        # Get the server response.
        return self.__client.recv()

    def read_meta_block_query(
            self, label: str, leaf: int, bucket_id: int, block_id: int, index: Optional[int] = None) -> Bucket:
        """Issues a read block query; telling the server to read metadata of one block from some storage."""
        # Check for connection.
        self.__check_client()

        # Create the init query.
        query = {
            "type": "read_meta_block",
            "label": label,
            "leaf": leaf,
            "bucket_id": bucket_id,
            "block_id": block_id,
            "index": index
        }

        # Send the query to server.
        self.__client.send(query)

        # Get the server response.
        return self.__client.recv()

    def write_block_query(
            self, label: str, leaf: int, bucket_id: int, block_id: int, data: Buckets, index: Optional[int] = None
    ) -> None:
        """Issues a write block query; telling the server to write one block to some storage."""
        # Check for connection.
        self.__check_client()

        # Create the init query.
        query = {
            "type": "write_block",
            "label": label,
            "leaf": leaf,
            "bucket_id": bucket_id,
            "block_id": block_id,
            "data": data,
            "index": index
        }

        # Send the query to server.
        self.__client.send(query)

        # Check for response.
        self.__check_response()

    def read_stash_query(self, label: str, stash: int, index: Optional[int] = None) -> Data:
        """Issues a read stash query; telling the server to read one data from some stash."""
        # Check for connection.
        self.__check_client()

        # Create the init query.
        query = {"type": "read_stash", "label": label, "stash": stash, "index": index}

        # Send the query to server.
        self.__client.send(query)

        # Get the server response.
        return self.__client.recv()

    def write_stash_query(self, label: str, stash: int, data: Data, index: Optional[int] = None) -> None:
        """Issues a write stash query; telling the server to write one data to some stash."""
        # Check for connection.
        self.__check_client()

        # Create the init query.
        query = {"type": "read_stash", "label": label, "stash": stash, "data": data, "index": index}

        # Send the query to server.
        self.__client.send(query)

        # Check for response.
        self.__check_response()


class InteractLocalServer(InteractServer):
    def __init__(self):
        """Create an instance for the local server with storages."""
        self.__ods_storage: Optional[BinaryTree] = None
        self.__oram_storage: Optional[BinaryTree] = None
        self.__pos_map_storage: Optional[List[BinaryTree]] = None
        self.__oram_stash: Optional[Bucket] = None
        self.__pos_map_stash: Optional[List[Bucket]] = None

    def init_connection(self) -> None:
        """Since the server is local, just pass."""
        pass

    def close_connection(self) -> None:
        """Since the server is local, just pass."""
        pass

    def init_query(self, label: str, storage: Union[BinaryTree, List[BinaryTree]]) -> None:
        """Issues an init query; sending some storage over to the server."""
        # Depends on the label, save the storage.
        if label == "ods":
            self.__ods_storage = storage
        elif label == "oram":
            self.__oram_storage = storage
        elif label == "pos_map":
            self.__pos_map_storage = storage
        elif label == "oram_stash":
            self.__oram_stash = storage
        elif label == "pos_map_stash":
            self.__pos_map_stash = storage
        else:
            raise ValueError("Invalid label was given.")

    def load_query(self, label: str, file: str) -> None:
        """Issues a load query; telling the server to load some storage from the disk."""
        # Depends on the label, save the storage.
        with open(file, "rb") as storage:
            # Depends on the label, save the storage.
            if label == "ods":
                self.__ods_storage = storage
            elif label == "oram":
                self.__oram_storage = storage
            elif label == "pos_map":
                self.__pos_map_storage = storage
            elif label == "oram_stash":
                self.__oram_stash = storage
            elif label == "pos_map_stash":
                self.__pos_map_stash = storage
            else:
                raise ValueError("Invalid label was given.")

    def read_query(self, label: str, leaf: Union[int, List[int]], index: Optional[int] = None) -> Buckets:
        """Issues a read query; telling the server to read one/multiple paths from some storage."""
        # Depends on the label, save the storage.
        if label == "ods":
            return self.__ods_storage.read_path(leaf=leaf)

        elif label == "oram":
            return self.__oram_storage.read_path(leaf=leaf)

        elif label == "pos_map":
            # In this case, we need to check that index is not None.
            if index is None:
                raise ValueError("Index must be provided when reading position map oram storage.")
            return self.__pos_map_storage[index].read_path(leaf=leaf)

        else:
            raise ValueError("Invalid label was given.")

    def write_query(self, label: str, leaf: Union[int, List[int]], data: Buckets, index: Optional[int] = None) -> None:
        """Issues a write query; telling the server to write one/multiple paths from some storage."""
        # Depends on the label, save the storage.
        if label == "ods":
            self.__ods_storage.write_path(leaf=leaf, data=data)

        elif label == "oram":
            self.__oram_storage.write_path(leaf=leaf, data=data)

        elif label == "pos_map":
            # In this case, we need to check that index is not None.
            if index is None:
                raise ValueError("Index must be provided when reading position map oram storage.")
            self.__pos_map_storage[index].write_path(leaf=leaf, data=data)

        else:
            raise ValueError("Invalid label was given.")

    def read_block_query(
            self, label: str, leaf: int, bucket_id: int, block_id: int, index: Optional[int] = None
    ) -> Bucket:
        """Issues a read block query; telling the server to read one block from some storage."""
        # Depends on the label, read the block; ODS will not perform linear scans on blocks.
        if label == "oram":
            return self.__oram_storage.read_block(leaf=leaf, bucket_id=bucket_id, block_id=block_id)

        elif label == "pos_map":
            return self.__pos_map_storage[index].read_block(leaf=leaf, bucket_id=bucket_id, block_id=block_id)

        else:
            raise ValueError("Invalid label was given, it must be either \"oram\" or \"pos_map\".")

    def read_meta_block_query(
            self, label: str, leaf: int, bucket_id: int, block_id: int, index: Optional[int] = None
    ) -> Bucket:
        """Issues a read block query; telling the server to read metadata of one block from some storage."""
        # Depends on the label, read the block; ODS will not perform linear scans on blocks.
        if label == "oram":
            return self.__oram_storage.read_block(leaf=leaf, bucket_id=bucket_id, block_id=block_id)

        elif label == "pos_map":
            return self.__pos_map_storage[index].read_block(leaf=leaf, bucket_id=bucket_id, block_id=block_id)

        else:
            raise ValueError("Invalid label was given, it must be either \"oram\" or \"pos_map\".")

    def write_block_query(
            self, label: str, leaf: int, bucket_id: int, block_id: int, data: Bucket, index: Optional[int] = None
    ) -> None:
        """Issues a write block query; telling the server to write one block to some storage."""
        # Depends on the label, read the block; ODS will not perform linear scans on blocks.
        if label == "oram":
            self.__oram_storage.write_block(leaf=leaf, bucket_id=bucket_id, block_id=block_id, data=data)

        elif label == "pos_map":
            self.__pos_map_storage[index].write_block(leaf=leaf, bucket_id=bucket_id, block_id=block_id, data=data)

        else:
            raise ValueError("Invalid label was given, it must be either \"oram\" or \"pos_map\".")

    def read_stash_query(self, label: str, stash: int, index: Optional[int] = None) -> Data:
        """Issues a read stash query; telling the server to read one data from some stash."""
        # Depends on the label, read the stash data; ODS will not perform linear scans on stash.
        if label == "oram":
            return self.__oram_stash[stash]

        elif label == "pos_map":
            return self.__pos_map_stash[index][stash]

        else:
            raise ValueError("Invalid label was given, it must be either \"oram\" or \"pos_map\".")

    def write_stash_query(self, label: str, stash: int, data: Data, index: Optional[int] = None) -> None:
        """Issues a write stash query; telling the server to write one data to some stash."""
        # Depends on the label, write the stash data; ODS will not perform linear scans on stash.
        if label == "oram":
            self.__oram_stash[stash] = data

        elif label == "pos_map":
            self.__pos_map_stash[index][stash] = data

        else:
            raise ValueError("Invalid label was given, it must be either \"oram\" or \"pos_map\".")


class RemoteServer(InteractLocalServer):
    def __init__(self, ip: str = "localhost", port: int = 7779):
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

    def __process_query(self, query: dict) -> Union[str, Buckets]:
        """Process client's queries based on their types."""
        if query["type"] == "init":
            self.init_query(label=query["label"], storage=query["storage"])
            return SERVER_DEFAULT_RESPONSE

        elif query["type"] == "load":
            self.load_query(label=query["label"], file=query["file"])
            return SERVER_DEFAULT_RESPONSE

        elif query["type"] == "read":
            return self.read_query(label=query["label"], leaf=query["leaf"], index=query["index"])

        elif query["type"] == "write":
            self.write_query(label=query["label"], leaf=query["leaf"], data=query["data"], index=query["index"])
            return SERVER_DEFAULT_RESPONSE

        elif query["type"] == "read_block":
            return self.read_block_query(
                label=query["label"],
                leaf=query["leaf"],
                bucket_id=query["bucket_id"],
                block_id=query["block_id"],
                index=query["index"]
            )

        elif query["type"] == "read_meta_block":
            return self.read_meta_block_query(
                label=query["label"],
                leaf=query["leaf"],
                bucket_id=query["bucket_id"],
                block_id=query["block_id"],
                index=query["index"]
            )

        elif query["type"] == "write_block":
            self.write_block_query(
                label=query["label"],
                leaf=query["leaf"],
                bucket_id=query["bucket_id"],
                block_id=query["block_id"],
                data=query["data"],
                index=query["index"]
            )
            return SERVER_DEFAULT_RESPONSE

        elif query["type"] == "read_stash":
            return self.read_stash_query(
                label=query["label"], stash=query["stash"], index=query["index"]
            )

        elif query["type"] == "write_stash":
            self.write_stash_query(
                label=query["label"], stash=query["stash"], data=query["data"], index=query["index"]
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
