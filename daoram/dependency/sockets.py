"""This module defines the socket that allows client server interation over the WAN.

The code is from the following post with minor modifications:
https://stackoverflow.com/questions/64549275/server-client-app-and-jsondecodeerror-unterminated-string-python
"""

import pickle
import socket
import struct
from typing import Any, Optional


class Socket(object):
    def __init__(self, ip: str, port: int, is_server: bool):
        """Initialize the socket object for client or server.

        :param ip: the IP address of the server/client.
        :param port: the port the server/client is listening/talking.
        :param is_server: a boolean value indicating whether this socket is a server.
        """
        if is_server:
            self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__socket.bind((ip, port))
            self.__socket.listen()
            self.__connected_socket, _ = self.__socket.accept()
        else:
            self.__connected_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__connected_socket.connect((ip, port))

    def __recv_all(self, length: int) -> Optional[bytes]:
        """Receive all bytes from the connected socket."""
        data = bytearray()
        # When read data is less than desired length, keep reading.
        while len(data) < length:
            packet = self.__connected_socket.recv(length - len(data))
            # When no more data is there to receive, return None.
            if not packet:
                return None
            data.extend(packet)
        # Otherwise return the read data.
        return data

    def send(self, msg: Any) -> None:
        """Send a message to the connected socket."""
        # Dump the message to some bytes.
        msg_pack = pickle.dumps(msg)
        # Prefix each message with a 4-byte length (network byte order).
        msg = struct.pack('>I', len(msg_pack)) + msg_pack
        # Send the message to connected socket.
        self.__connected_socket.sendall(msg)

    def recv(self) -> Any:
        """Receive a message from the connected socket."""
        # Read message length and unpack it into an integer
        raw_msg_len = self.__recv_all(4)
        if not raw_msg_len:
            return None
        msg_len = struct.unpack('>I', raw_msg_len)[0]
        # Read the message data
        return pickle.loads(self.__recv_all(msg_len))

    def close(self):
        """Close the socket."""
        self.__connected_socket.close()
