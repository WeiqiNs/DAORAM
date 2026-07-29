import pickle
from typing import override

from oblivlib.dependency import InteractRemoteServer, PathOramConfig, RemoteServer
from oblivlib.dependency.sockets import BaseSocket
from oblivlib.oram import PathOram


class _LoopbackSocket(BaseSocket):
    """In-process stand-in for ZMQSocket: pickles each message like the wire and runs it through a
    real RemoteServer, exercising the full client/server protocol without sockets or threads."""

    def __init__(self, server: RemoteServer):
        self._server = server
        self._response = None

    @override
    def send(self, msg):
        request = pickle.loads(pickle.dumps(msg))
        self._response = pickle.loads(pickle.dumps(self._server._process_request(request)))

    @override
    def recv(self):
        return self._response

    @override
    def close(self):
        pass


def test_remote_client_server_round_trip():
    client = InteractRemoteServer()
    client.init_connection(_LoopbackSocket(RemoteServer()))

    oram = PathOram(PathOramConfig(num_data=256, data_size=10, client=client))
    oram.init_server_storage()

    for i in range(256):
        oram.operate_on_key(key=i, value=i)

    for i in range(256):
        assert oram.operate_on_key(key=i) == i


def test_multiple_orams_share_one_client(client):
    a = PathOram(PathOramConfig(num_data=64, data_size=10, client=client, name="a"))
    b = PathOram(PathOramConfig(num_data=64, data_size=10, client=client, name="b"))
    a.init_server_storage()
    b.init_server_storage()

    for i in range(64):
        a.operate_on_key(key=i, value=i)
        b.operate_on_key(key=i, value=i * 10)

    for i in range(64):
        assert a.operate_on_key(key=i) == i
        assert b.operate_on_key(key=i) == i * 10
