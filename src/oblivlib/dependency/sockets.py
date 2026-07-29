"""Socket implementations for client-server interaction over WAN."""

import pickle
from abc import ABC, abstractmethod
from typing import Any, override

import zmq


class BaseSocket(ABC):
    @abstractmethod
    def send(self, msg: Any) -> None:
        pass

    @abstractmethod
    def recv(self) -> Any:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class ZMQSocket(BaseSocket):
    def __init__(self, ip: str, port: int, is_server: bool):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP if is_server else zmq.REQ)

        if is_server:
            self._socket.bind(f"tcp://{ip}:{port}")
        else:
            self._socket.connect(f"tcp://{ip}:{port}")

    @override
    def send(self, msg: Any) -> None:
        self._socket.send(pickle.dumps(msg))

    @override
    def recv(self) -> Any:
        return pickle.loads(self._socket.recv())

    @override
    def close(self) -> None:
        self._socket.close()
        self._context.term()
