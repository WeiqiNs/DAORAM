import secrets
import shutil
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self, override

from oblivlib.dependency.errors import MissingResultError, UnknownLabelError
from oblivlib.dependency.heap_index import leaf_index, path_indices
from oblivlib.dependency.protocol import (
    DEFAULT_MAX_MESSAGE_BYTES,
    DEFAULT_TIMEOUT_MS,
    PROTOCOL_VERSION,
    WRITE_RANGE_CHUNK_BYTES,
    AttachTree,
    Batch,
    BatchReply,
    Close,
    CreateList,
    CreateTree,
    DropTree,
    Hello,
    HelloReply,
    ListOps,
    Ok,
    ReadList,
    ReadOp,
    ReadPath,
    Reply,
    Request,
    Resize,
    WriteOp,
    WritePath,
    WriteRange,
    decode_reply,
    encode,
    expect_reply,
)
from oblivlib.dependency.storage_server import StorageServer
from oblivlib.dependency.transport import Transport, ZmqTransport
from oblivlib.dependency.tree_builder import FileImage, MemoryImage, TreeImage
from oblivlib.dependency.types import ListOp, ListPopBack, PathRows


@dataclass(frozen=True)
class Metrics:
    """``rounds`` counts the batches sent; ``payload_bytes`` the row and list-value bytes written and
    read; ``wire_bytes`` the encoded message bytes a transport moved (zero in process)."""

    rounds: int
    payload_bytes: int
    wire_bytes: int


@dataclass(frozen=True)
class ExecuteResult:
    """The reads of one ``Client.execute()``, keyed by label: a path read maps each node's heap index to
    its row (``b""`` for an absent node); a list read is the whole list, or ``{index: value}``."""

    results: dict[str, Any]

    def require(self, label: str) -> Any:
        if label not in self.results:
            raise MissingResultError(f"ExecuteResult has no result for label {label!r}.")
        return self.results[label]


class Backend(ABC):
    @abstractmethod
    def call(self, message: Request) -> Reply: ...

    @property
    @abstractmethod
    def wire_bytes(self) -> int: ...

    @property
    @abstractmethod
    def storage_dir(self) -> Path | None:
        """Where the server keeps hosted files, when the client can place a file there directly."""

    @abstractmethod
    def close(self) -> None: ...


class LocalBackend(Backend):
    def __init__(self, server: StorageServer):
        self._server = server

    @override
    def call(self, message: Request) -> Reply:
        return self._server.handle(message)

    @property
    @override
    def wire_bytes(self) -> int:
        return 0

    @property
    @override
    def storage_dir(self) -> Path | None:
        return self._server.storage_dir

    @override
    def close(self) -> None:
        pass


class TransportBackend(Backend):
    def __init__(self, transport: Transport):
        self._transport = transport
        self._wire_bytes = 0

    @override
    def call(self, message: Request) -> Reply:
        data = encode(message)
        reply = self._transport.request(data)
        self._wire_bytes += len(data) + len(reply)
        return decode_reply(reply)

    @property
    @override
    def wire_bytes(self) -> int:
        return self._wire_bytes

    @property
    @override
    def storage_dir(self) -> None:
        return None

    @override
    def close(self) -> None:
        self._transport.close()


def _list_op_bytes(op: ListOp) -> int:
    return 0 if isinstance(op, ListPopBack) else len(op.value)


def _path_leaves(indices: Iterable[int], level: int) -> list[int]:
    first_leaf = leaf_index(0, level)
    return sorted(index - first_leaf for index in indices if index >= first_leaf)


def _file_rows(image: FileImage) -> Iterator[bytes]:
    with image.path.open("rb") as file:
        while row := file.read(image.row_bytes):
            yield row


class Client:
    """The client-side handle to a storage server; every scheme does all its I/O through one.

    ``add_*`` calls only stage queries, keyed by storage label. ``execute()`` sends them as one batch
    whose writes (path writes merged per label, a later write winning; list ops in staging order) apply
    before its reads (deduplicated per label), and returns their ``ExecuteResult``. A failed batch raises
    the server's error and leaves nothing staged. An ``execute()`` with nothing staged is still a round.

    With ``defer_writes`` (the default), an ``execute()`` that stages only writes sends nothing: its writes
    ride on the next batch, ahead of that batch's reads. ``flush()`` sends them on their own, every
    lifecycle call flushes first, and ``close()`` flushes. An error from a deferred write surfaces on the
    call that sends it.

    ``host_tree`` hands a built tree to the server: a file image is adopted in place when the server's
    storage directory is local, placed out of band by ``ship_file`` (which returns its name inside the
    server's storage directory) and attached, or else streamed; the local build file is gone afterwards.

    A client is not thread-safe: use one per thread.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        defer_writes: bool = True,
        ship_file: Callable[[Path], str] | None = None,
    ) -> None:
        self._backend = backend
        self._defer_writes = defer_writes
        self._ship_file = ship_file

        self._session: str | None = None
        self._max_message_bytes = DEFAULT_MAX_MESSAGE_BYTES
        self._levels: dict[str, int] = {}
        self._lists: set[str] = set()

        self._read_paths: dict[str, set[int]] = {}
        self._read_lists: dict[str, set[int] | None] = {}
        self._write_paths: dict[str, PathRows] = {}
        self._write_lists: dict[str, list[ListOp]] = {}

        self._rounds = 0
        self._payload_bytes = 0
        self._wire_baseline = 0
        self._closed = False

    @classmethod
    def local(cls, storage_dir: str | Path | None = None, *, defer_writes: bool = True) -> Self:
        return cls(LocalBackend(StorageServer(storage_dir)), defer_writes=defer_writes)

    @classmethod
    def connect(cls, endpoint: str, *, timeout_ms: int = DEFAULT_TIMEOUT_MS, defer_writes: bool = True) -> Self:
        return cls(TransportBackend(ZmqTransport(endpoint, timeout_ms=timeout_ms)), defer_writes=defer_writes)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()

    @property
    def metrics(self) -> Metrics:
        return Metrics(
            rounds=self._rounds,
            payload_bytes=self._payload_bytes,
            wire_bytes=self._backend.wire_bytes - self._wire_baseline,
        )

    def reset_metrics(self) -> None:
        self._rounds = 0
        self._payload_bytes = 0
        self._wire_baseline = self._backend.wire_bytes

    def level_of(self, label: str) -> int:
        if label not in self._levels:
            raise UnknownLabelError(f"Client: tree label {label!r} is not hosted by this client.")
        return self._levels[label]

    def _require_list(self, label: str) -> None:
        if label not in self._lists:
            raise UnknownLabelError(f"Client: list label {label!r} is not hosted by this client.")

    def add_read_path(self, label: str, leaves: Iterable[int]) -> None:
        self.level_of(label)
        self._read_paths.setdefault(label, set()).update(leaves)

    def add_read_list(self, label: str, indices: Iterable[int] | None) -> None:
        """``indices=None`` reads the whole list; otherwise index calls accumulate."""
        self._require_list(label)
        staged = self._read_lists.get(label, set())
        if indices is None or staged is None:
            self._read_lists[label] = None
            return
        staged.update(indices)
        self._read_lists[label] = staged

    def add_write_path(self, label: str, data: PathRows) -> None:
        """Stage rows for whole root-to-leaf paths: ``data`` must hold exactly the nodes of the paths to
        the leaves among its indices."""
        level = self.level_of(label)
        if set(data) != set(path_indices(_path_leaves(data, level), level)):
            raise ValueError(f"Client: a write to {label!r} must cover exactly the paths to its leaves.")
        self._write_paths.setdefault(label, {}).update(data)

    def add_write_list(self, label: str, ops: Sequence[ListOp]) -> None:
        self._require_list(label)
        self._write_lists.setdefault(label, []).extend(ops)

    def execute(self) -> ExecuteResult:
        reads = self._take_reads()
        if self._defer_writes and not reads and self._has_pending_writes():
            return ExecuteResult({})
        return self._send_batch(self._take_writes(), reads)

    def flush(self) -> None:
        if self._has_pending_writes():
            self._send_batch(self._take_writes(), [])

    def create_tree(self, label: str, *, level: int, row_bytes: int | None) -> None:
        self.flush()
        self._expect_ok(CreateTree(session=self._require_session(), label=label, level=level, row_bytes=row_bytes))
        self._levels[label] = level

    def host_tree(self, label: str, image: TreeImage) -> None:
        match image:
            case MemoryImage(level=level, row_bytes=row_bytes, rows=rows):
                self.create_tree(label, level=level, row_bytes=row_bytes)
                self._stream_rows(label, rows)
            case FileImage():
                self._host_file(label, image)

    def create_list(self, label: str) -> None:
        self.flush()
        self._expect_ok(CreateList(session=self._require_session(), label=label))
        self._lists.add(label)

    def drop_tree(self, label: str) -> None:
        self.flush()
        self.level_of(label)
        self._expect_ok(DropTree(session=self._require_session(), label=label))
        del self._levels[label]

    def resize(self, label: str, *, level: int) -> None:
        self.flush()
        self.level_of(label)
        self._expect_ok(Resize(session=self._require_session(), label=label, level=level))
        self._levels[label] = level

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.flush()
        finally:
            try:
                if self._session is not None:
                    self._expect_ok(Close(session=self._session))
            finally:
                self._backend.close()

    def _require_session(self) -> str:
        if self._session is None:
            reply = expect_reply(self._backend.call(Hello(version=PROTOCOL_VERSION)), HelloReply)
            self._session = reply.session
            self._max_message_bytes = reply.max_message_bytes
        return self._session

    def _expect_ok(self, message: Request) -> None:
        expect_reply(self._backend.call(message), Ok)

    def _host_file(self, label: str, image: FileImage) -> None:
        self.flush()
        session = self._require_session()
        filename = self._hand_over(image.path)
        if filename is None:
            self.create_tree(label, level=image.level, row_bytes=image.row_bytes)
            self._stream_rows(label, _file_rows(image))
            image.path.unlink()
            return
        attach = AttachTree(
            session=session, label=label, filename=filename, level=image.level, row_bytes=image.row_bytes
        )
        self._expect_ok(attach)
        self._levels[label] = image.level

    def _hand_over(self, path: Path) -> str | None:
        if self._ship_file is not None:
            filename = self._ship_file(path)
            path.unlink()
            return filename
        storage_dir = self._backend.storage_dir
        if storage_dir is None:
            return None
        filename = f"adopt-{secrets.token_hex(8)}.tree"
        shutil.move(path, storage_dir / filename)
        return filename

    def _stream_rows(self, label: str, rows: Iterable[bytes]) -> None:
        chunk_bytes = min(WRITE_RANGE_CHUNK_BYTES, self._max_message_bytes // 2)
        start = 0
        chunk: list[bytes] = []
        size = 0
        for row in rows:
            if chunk and size + len(row) > chunk_bytes:
                self._write_range(label, start, chunk)
                start += len(chunk)
                chunk, size = [], 0
            chunk.append(row)
            size += len(row)
        if chunk:
            self._write_range(label, start, chunk)

    def _write_range(self, label: str, start: int, rows: list[bytes]) -> None:
        self._expect_ok(WriteRange(session=self._require_session(), label=label, start=start, rows=rows))
        self._payload_bytes += sum(len(row) for row in rows)

    def _has_pending_writes(self) -> bool:
        return bool(self._write_paths or self._write_lists)

    def _take_writes(self) -> list[WriteOp]:
        writes: list[WriteOp] = []
        for label, data in self._write_paths.items():
            level = self._levels[label]
            leaves = _path_leaves(data, level)
            writes.append(WritePath(label=label, leaves=leaves, rows=[data[i] for i in path_indices(leaves, level)]))
        writes.extend(ListOps(label=label, ops=ops) for label, ops in self._write_lists.items())
        self._write_paths = {}
        self._write_lists = {}
        return writes

    def _take_reads(self) -> list[ReadOp]:
        reads: list[ReadOp] = [
            ReadPath(label=label, leaves=sorted(leaves)) for label, leaves in self._read_paths.items()
        ]
        reads.extend(
            ReadList(label=label, indices=None if indices is None else sorted(indices))
            for label, indices in self._read_lists.items()
        )
        self._read_paths = {}
        self._read_lists = {}
        return reads

    def _send_batch(self, writes: list[WriteOp], reads: list[ReadOp]) -> ExecuteResult:
        batch = Batch(session=self._require_session(), writes=writes, reads=reads)
        self._rounds += 1
        reply = expect_reply(self._backend.call(batch), BatchReply)

        payload = 0
        for op in writes:
            if isinstance(op, WritePath):
                payload += sum(len(row) for row in op.rows)
            else:
                payload += sum(_list_op_bytes(list_op) for list_op in op.ops)

        results: dict[str, Any] = {}
        for op, rows in zip(reads, reply.results, strict=True):
            payload += sum(len(row) for row in rows)
            if isinstance(op, ReadPath):
                results[op.label] = dict(zip(path_indices(op.leaves, self._levels[op.label]), rows, strict=True))
            elif op.indices is None:
                results[op.label] = rows
            else:
                results[op.label] = dict(zip(op.indices, rows, strict=True))
        self._payload_bytes += payload
        return ExecuteResult(results)
