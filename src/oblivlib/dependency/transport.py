from abc import ABC, abstractmethod
from typing import override

import zmq

from oblivlib.dependency.errors import ProtocolError, TransportError
from oblivlib.dependency.protocol import DEFAULT_TIMEOUT_MS, Shutdown, decode_request, encode, error_reply
from oblivlib.dependency.storage_server import StorageServer


class Transport(ABC):
    """Moves one encoded request to a server and returns its encoded reply."""

    @abstractmethod
    def request(self, data: bytes) -> bytes: ...

    @abstractmethod
    def close(self) -> None: ...


def handle_wire(server: StorageServer, data: bytes) -> tuple[bytes, bool]:
    """Answer one encoded request; the flag is whether it was a ``Shutdown``. Never raises on bad input."""
    if len(data) > server.max_message_bytes:
        error = ProtocolError(f"message of {len(data)} bytes exceeds the {server.max_message_bytes}-byte limit.")
        return encode(error_reply(error)), False
    try:
        message = decode_request(data)
    except ProtocolError as exc:
        return encode(error_reply(exc)), False
    return encode(server.handle(message)), isinstance(message, Shutdown)


class LoopbackTransport(Transport):
    def __init__(self, server: StorageServer):
        self._server = server

    @override
    def request(self, data: bytes) -> bytes:
        return handle_wire(self._server, data)[0]

    @override
    def close(self) -> None:
        pass


class ZmqTransport(Transport):
    """A ZeroMQ ``REQ`` socket. A request that gets no reply within ``timeout_ms`` raises
    ``TransportError`` and recreates the socket, so the transport stays usable."""

    def __init__(self, endpoint: str, *, timeout_ms: int = DEFAULT_TIMEOUT_MS):
        self._endpoint = endpoint
        self._timeout_ms = timeout_ms
        self._context = zmq.Context()
        self._socket = self._connect()

    def _connect(self) -> zmq.Socket[bytes]:
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self._endpoint)
        return socket

    @override
    def request(self, data: bytes) -> bytes:
        try:
            self._socket.send(data)
            return self._socket.recv()
        except zmq.ZMQError as exc:
            self._socket.close()
            self._socket = self._connect()
            if isinstance(exc, zmq.Again):
                raise TransportError(
                    f"ZmqTransport: no reply from {self._endpoint} within {self._timeout_ms} ms."
                ) from exc
            raise TransportError(f"ZmqTransport: request to {self._endpoint} failed: {exc}") from exc

    @override
    def close(self) -> None:
        self._socket.close()
        self._context.term()


class ZmqListener:
    """The server side: a bound ZeroMQ ``REP`` socket. ``endpoint`` is the bound address, so binding to
    ``tcp://127.0.0.1:*`` picks a free port."""

    def __init__(self, endpoint: str):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(endpoint)

    @property
    def endpoint(self) -> str:
        return self._socket.getsockopt_string(zmq.LAST_ENDPOINT)

    def recv(self) -> bytes:
        return self._socket.recv()

    def send(self, data: bytes) -> None:
        self._socket.send(data)

    def close(self) -> None:
        self._socket.close()
        self._context.term()


def serve(listener: ZmqListener, server: StorageServer) -> None:
    """Answer requests on ``listener`` until a ``Shutdown`` has been answered."""
    while True:
        reply, shutdown = handle_wire(server, listener.recv())
        listener.send(reply)
        if shutdown:
            return


class SimulatedNetwork(Transport):
    """Wraps a transport and accumulates modeled network time, ``rtt_s`` plus the request and reply bytes
    over ``bytes_per_s`` per request, without sleeping."""

    def __init__(self, inner: Transport, *, rtt_s: float, bytes_per_s: float):
        self._inner = inner
        self._rtt_s = rtt_s
        self._bytes_per_s = bytes_per_s
        self._modeled_seconds = 0.0
        self._request_count = 0

    @property
    def modeled_seconds(self) -> float:
        return self._modeled_seconds

    @property
    def request_count(self) -> int:
        return self._request_count

    @override
    def request(self, data: bytes) -> bytes:
        reply = self._inner.request(data)
        self._request_count += 1
        self._modeled_seconds += self._rtt_s + (len(data) + len(reply)) / self._bytes_per_s
        return reply

    @override
    def close(self) -> None:
        self._inner.close()
