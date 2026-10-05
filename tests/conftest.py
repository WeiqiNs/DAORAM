import os
import random
import secrets
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Literal, override

import pytest

from oblivlib.dependency import (
    AesGcm,
    Backend,
    Client,
    LocalBackend,
    LoopbackTransport,
    StorageServer,
    TransportBackend,
)
from oblivlib.dependency.protocol import Batch, ReadPath, Reply, Request, WritePath, WriteRange
from oblivlib.oram import freecursive_oram

DEFAULT_NUM_DATA = 2**12

AccessShape = tuple[dict[str, frozenset[int]], dict[str, tuple[int, ...]]]


class RecordingBackend(Backend):
    """Delegates to ``inner`` and records every ``Batch`` and ``WriteRange`` it forwards."""

    def __init__(self, inner: Backend):
        self._inner = inner
        self.messages: list[Batch | WriteRange] = []

    @override
    def call(self, message: Request) -> Reply:
        if isinstance(message, Batch | WriteRange):
            self.messages.append(message)
        return self._inner.call(message)

    @property
    @override
    def wire_bytes(self) -> int:
        return self._inner.wire_bytes

    @property
    @override
    def storage_dir(self) -> Path | None:
        return self._inner.storage_dir

    @override
    def close(self) -> None:
        self._inner.close()

    @property
    def batches(self) -> list[Batch]:
        return [message for message in self.messages if isinstance(message, Batch)]


def _access_shape(batch: Batch) -> AccessShape:
    writes = {op.label: frozenset(op.leaves) for op in batch.writes if isinstance(op, WritePath)}
    reads = {op.label: tuple(op.leaves) for op in batch.reads if isinstance(op, ReadPath)}
    return writes, reads


def _coalesce(shapes: list[AccessShape]) -> list[AccessShape]:
    """What deferral should turn a sequence of undeferred batches into: each run of batches without
    reads folds into the next batch's writes (leaves unioned per label), while an empty batch with no
    writes pending is still sent."""
    coalesced: list[AccessShape] = []
    pending: dict[str, frozenset[int]] = {}
    for writes, reads in shapes:
        if not writes and not reads and not pending:
            coalesced.append(({}, {}))
            continue
        for label, leaves in writes.items():
            pending[label] = pending.get(label, frozenset()) | leaves
        if reads:
            coalesced.append((pending, reads))
            pending = {}
    if pending:
        coalesced.append((pending, {}))
    return coalesced


def _seed(monkeypatch: pytest.MonkeyPatch, seed: int) -> None:
    rng = random.Random(seed)
    monkeypatch.setattr(secrets, "randbelow", rng.randrange)
    monkeypatch.setattr(os, "urandom", rng.randbytes)
    monkeypatch.setattr(freecursive_oram, "_CSPRNG", random.Random(seed))


@pytest.fixture
def num_data() -> int:
    return int(os.environ.get("NUM_DATA", DEFAULT_NUM_DATA))


@pytest.fixture
def test_file(tmp_path):
    return tmp_path / "test.bin"


@pytest.fixture
def client():
    return Client.local()


@pytest.fixture
def remote_client():
    return Client(TransportBackend(LoopbackTransport(StorageServer())))


@pytest.fixture
def recorded_client():
    """Factory for a ``Client`` over a ``RecordingBackend``; returns ``(client, recorder)``."""

    def make(*, defer_writes: bool = True, server: StorageServer | None = None) -> tuple[Client, RecordingBackend]:
        recorder = RecordingBackend(LocalBackend(server or StorageServer()))
        return Client(recorder, defer_writes=defer_writes), recorder

    return make


@pytest.fixture
def handover_client(tmp_path):
    """Factory ``handover_client(mode)``: a ``Client`` whose ``host_tree`` hands a build file over by
    ``mode``: ``"adopt"`` (a local server moves it in), ``"attach"`` (shipped into the server's directory,
    then attached), or ``"stream"`` (sent as rows)."""

    def make(mode: Literal["adopt", "attach", "stream"]) -> Client:
        server_dir = tmp_path / f"{mode}_server"
        match mode:
            case "adopt":
                return Client.local(storage_dir=server_dir)
            case "stream":
                return Client(TransportBackend(LoopbackTransport(StorageServer())))
            case "attach":

                def ship(path: Path) -> str:
                    shutil.copy(path, server_dir / path.name)
                    return path.name

                return Client(TransportBackend(LoopbackTransport(StorageServer(server_dir))), ship_file=ship)

    return make


@pytest.fixture
def assert_deferral_preserves_access(monkeypatch, recorded_client):
    """Run ``workload(client)`` twice under the same seed, with write deferral off and on, and assert the
    deferred batches are exactly the undeferred ones coalesced. Returns the rounds each run took before
    its final flush, as ``(off, on)``."""

    def run(workload: Callable[[Client], None]) -> tuple[int, int]:
        shapes: dict[bool, list[AccessShape]] = {}
        rounds: dict[bool, int] = {}
        for defer_writes in (False, True):
            _seed(monkeypatch, 7)
            client, recorder = recorded_client(defer_writes=defer_writes)
            workload(client)
            rounds[defer_writes] = client.metrics.rounds
            client.flush()
            shapes[defer_writes] = [_access_shape(batch) for batch in recorder.batches]
        assert _coalesce(shapes[False]) == shapes[True]
        return rounds[False], rounds[True]

    return run


@pytest.fixture
def encryptor():
    return AesGcm()
