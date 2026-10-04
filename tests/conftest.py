import os
import pickle
from typing import override

import pytest

from oblivlib.dependency import AesGcm, InteractLocalServer, InteractRemoteServer, RemoteServer
from oblivlib.dependency.sockets import BaseSocket

DEFAULT_NUM_DATA = 2**12


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


@pytest.fixture
def num_data() -> int:
    return int(os.environ.get("NUM_DATA", DEFAULT_NUM_DATA))


@pytest.fixture
def test_file(tmp_path):
    return tmp_path / "test.bin"


@pytest.fixture
def client():
    return InteractLocalServer()


@pytest.fixture
def remote_client():
    remote = InteractRemoteServer()
    remote.init_connection(_LoopbackSocket(RemoteServer()))
    return remote


@pytest.fixture
def encryptor():
    return AesGcm()
