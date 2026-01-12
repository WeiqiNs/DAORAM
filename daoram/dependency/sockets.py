"""Socket implementations for client-server interaction over WAN."""

import pickle
from abc import ABC, abstractmethod
from typing import Any

import zmq


class BaseSocket(ABC):
    """Abstract base class for socket implementations."""

    @abstractmethod
    def send(self, msg: Any) -> None:
        """Send a message to the connected socket.

        :param msg: The message to send (any picklable object).
        """
        pass

    @abstractmethod
    def recv(self) -> Any:
        """Receive a message from the connected socket.

        :return: The received message.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the socket connection."""
        pass


class ZMQSocket(BaseSocket):
    """ZeroMQ socket implementation with automatic message framing."""

    def __init__(self, ip: str, port: int, is_server: bool):
        """Initialize the ZeroMQ socket for client or server.

        :param ip: The IP address of the server/client.
        :param port: The port the server/client is listening/talking.
        :param is_server: Whether this socket is a server.
        """
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP if is_server else zmq.REQ)

        if is_server:
            self._socket.bind(f"tcp://{ip}:{port}")
        else:
            self._socket.connect(f"tcp://{ip}:{port}")

    def send(self, msg: Any) -> None:
        """Send a message to the connected socket."""
        self._socket.send(pickle.dumps(msg))

    def recv(self) -> Any:
        """Receive a message from the connected socket."""
        return pickle.loads(self._socket.recv())

    def close(self) -> None:
        """Close the socket and context."""
        self._socket.close()
        self._context.term()
