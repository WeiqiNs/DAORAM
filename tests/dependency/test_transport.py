import os
import pickle
import threading
from typing import override

import pytest

from oblivlib.dependency import (
    Client,
    DuplicateLabelError,
    LoopbackTransport,
    ProtocolError,
    RowSizeError,
    SimulatedNetwork,
    StorageServer,
    TransportBackend,
    TransportError,
    ZmqListener,
    ZmqTransport,
    serve,
)
from oblivlib.dependency.protocol import ErrorReply, Hello, HelloReply, Ok, Shutdown, decode_reply, encode, expect_reply
from oblivlib.dependency.tree_builder import MemoryImage

ROW = 4


def _row(tag: int) -> bytes:
    return bytes([tag]) * ROW


def _workload(client: Client) -> None:
    client.host_tree("t", MemoryImage(level=3, row_bytes=ROW, rows=[_row(i + 1) for i in range(7)]))
    client.reset_metrics()
    client.add_write_path("t", {0: _row(9), 2: _row(9), 6: _row(9)})
    client.execute()
    client.add_read_path("t", [0, 3])
    assert client.execute().require("t")[6] == _row(9)


def _serve_in_thread(listener: ZmqListener, server: StorageServer) -> threading.Thread:
    thread = threading.Thread(target=serve, args=(listener, server), daemon=True)
    thread.start()
    return thread


def _shutdown(endpoint: str, thread: threading.Thread) -> None:
    control = ZmqTransport(endpoint, timeout_ms=5000)
    expect_reply(decode_reply(control.request(encode(Shutdown()))), Ok)
    control.close()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_error_replies_raise_mapped_errors_over_zmq():
    listener = ZmqListener("tcp://127.0.0.1:*")
    thread = _serve_in_thread(listener, StorageServer())
    client = Client.connect(listener.endpoint, timeout_ms=5000)

    client.create_tree("t", level=2, row_bytes=ROW)
    with pytest.raises(DuplicateLabelError, match="'t'"):
        client.create_tree("t", level=2, row_bytes=ROW)

    client.add_write_path("t", {0: b"short", 2: _row(1)})
    client.execute()
    client.add_read_path("t", [1])
    with pytest.raises(RowSizeError):
        client.execute()

    client.add_write_path("t", {0: _row(5), 2: _row(6)})
    client.add_read_path("t", [1])
    assert client.execute().require("t") == {0: _row(5), 2: _row(6)}

    client.close()
    _shutdown(listener.endpoint, thread)
    listener.close()


def test_timeout_raises_transport_error_then_recovers():
    listener = ZmqListener("tcp://127.0.0.1:*")
    transport = ZmqTransport(listener.endpoint, timeout_ms=100)
    with pytest.raises(TransportError, match="within 100 ms"):
        transport.request(encode(Hello(version=1)))

    thread = _serve_in_thread(listener, StorageServer())
    assert isinstance(decode_reply(transport.request(encode(Hello(version=1)))), HelloReply)
    transport.close()
    _shutdown(listener.endpoint, thread)
    listener.close()


class _MakesDirectoryOnUnpickle:
    def __init__(self, path: str):
        self._path = path

    @override
    def __reduce__(self):
        return (os.mkdir, (self._path,))


def test_server_never_unpickles_and_bounds_message_size(tmp_path):
    marker = tmp_path / "unpickled"
    transport = LoopbackTransport(StorageServer(max_message_bytes=64))

    reply = decode_reply(transport.request(pickle.dumps(_MakesDirectoryOnUnpickle(str(marker)))))
    assert isinstance(reply, ErrorReply) and reply.kind == "ProtocolError"
    assert not marker.exists()

    with pytest.raises(ProtocolError, match="exceeds the 64-byte limit"):
        expect_reply(decode_reply(transport.request(encode(Hello(version=1)) + bytes(64))), HelloReply)


def test_simulated_network_models_rtt_and_bandwidth():
    network = SimulatedNetwork(LoopbackTransport(StorageServer()), rtt_s=0.01, bytes_per_s=1000)
    backend = TransportBackend(network)
    _workload(Client(backend))
    assert network.request_count > 0
    assert network.modeled_seconds == pytest.approx(network.request_count * 0.01 + backend.wire_bytes / 1000)


def test_metrics_match_between_local_and_loopback():
    local = Client.local()
    remote = Client(TransportBackend(LoopbackTransport(StorageServer())))
    for client in (local, remote):
        _workload(client)

    assert local.metrics.rounds == remote.metrics.rounds == 1
    assert local.metrics.payload_bytes == remote.metrics.payload_bytes == 8 * ROW
    assert local.metrics.wire_bytes == 0
    assert remote.metrics.wire_bytes > remote.metrics.payload_bytes
