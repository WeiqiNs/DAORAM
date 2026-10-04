"""Client-server interaction layer for local and remote ORAM storage."""

import pickle
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, override

from oblivlib.dependency.binary_tree import BinaryTree
from oblivlib.dependency.errors import DuplicateLabelError, UnknownLabelError
from oblivlib.dependency.sockets import BaseSocket
from oblivlib.dependency.types import (
    ExecuteResult,
    ListOp,
    ListPopBack,
    ListPushFront,
    ListWrite,
    PathData,
    Request,
)

PORT = 10000
ServerStorage = dict[str, BinaryTree | list]


class InteractServer(ABC):
    """Client-side handle to server storage; every scheme does all its I/O through one.

    Batching contract: ``add_*`` calls only stage queries, keyed by storage label. ``execute()`` ships
    them as one ``Request`` and clears the staging buffers whether or not it succeeds. Within one
    execute the server applies path writes, then list ops in staging order, then reads (deduplicated).
    Results and stored writes are copies, never aliases of server storage. Bandwidth counts the pickled
    ``Request`` and ``ExecuteResult``.
    """

    def __init__(self):
        self._read_paths: dict[str, list[int]] = {}
        self._read_lists: dict[str, list[int] | None] = {}
        self._write_paths: dict[str, PathData] = {}
        self._write_lists: dict[str, list[ListOp]] = {}

        self._bytes_read: int = 0
        self._bytes_written: int = 0

    def get_bandwidth(self) -> tuple[int, int]:
        """Return (bytes_read, bytes_written)."""
        return self._bytes_read, self._bytes_written

    def reset_bandwidth(self) -> None:
        self._bytes_read = 0
        self._bytes_written = 0

    def add_read_path(self, label: str, leaves: list[int]) -> None:
        self._read_paths.setdefault(label, []).extend(leaves)

    def add_read_list(self, label: str, indices: list[int] | None) -> None:
        """``indices=None`` reads the whole list; otherwise index calls accumulate."""
        if indices is None:
            self._read_lists[label] = None
            return
        existing = self._read_lists.get(label)
        if existing is None:
            existing = []
            self._read_lists[label] = existing
        existing.extend(indices)

    def add_write_path(self, label: str, data: PathData) -> None:
        self._write_paths.setdefault(label, {}).update(data)

    def add_write_list(self, label: str, ops: Sequence[ListOp]) -> None:
        self._write_lists.setdefault(label, []).extend(ops)

    def _take_request(self) -> Request:
        request = Request(
            read_paths=self._read_paths,
            read_lists=self._read_lists,
            write_paths=self._write_paths,
            write_lists=self._write_lists,
        )
        self._read_paths = {}
        self._read_lists = {}
        self._write_paths = {}
        self._write_lists = {}
        return request

    @abstractmethod
    def init_connection(self, client: BaseSocket) -> None:
        raise NotImplementedError

    @abstractmethod
    def close_connection(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def init_storage(self, storage: ServerStorage) -> None:
        raise NotImplementedError

    @abstractmethod
    def execute(self) -> ExecuteResult:
        """Run all pending queries — writes first, then reads — and return an ExecuteResult."""
        raise NotImplementedError


class InteractLocalServer(InteractServer):
    """In-process server storage, used by tests and local runs.

    It pickles each request and result exactly as the wire would, so a scheme sees the same copies and
    bandwidth numbers as against ``InteractRemoteServer``. ``init_storage`` takes ownership of the
    given trees and lists without copying and raises ``DuplicateLabelError`` for a hosted label.
    """

    def __init__(self):
        super().__init__()
        self._trees: dict[str, BinaryTree] = {}
        self._lists: dict[str, list] = {}

    @override
    def init_connection(self, client: BaseSocket | None = None) -> None:
        pass

    @override
    def close_connection(self) -> None:
        pass

    @override
    def init_storage(self, storage: ServerStorage) -> None:
        trees = {label: store for label, store in storage.items() if isinstance(store, BinaryTree)}
        lists = {label: store for label, store in storage.items() if isinstance(store, list)}
        if len(trees) + len(lists) != len(storage):
            raise TypeError(f"{type(self).__name__}: every hosted store must be a BinaryTree or a list.")
        for label in storage:
            if label in self._trees or label in self._lists:
                raise DuplicateLabelError(f"{type(self).__name__}: label {label!r} is already hosted.")

        self._trees.update(trees)
        self._lists.update(lists)

    def _require_tree(self, label: str) -> BinaryTree:
        if label not in self._trees:
            raise UnknownLabelError(f"{type(self).__name__}: tree label {label!r} is not hosted.")
        return self._trees[label]

    def _require_list(self, label: str) -> list:
        if label not in self._lists:
            raise UnknownLabelError(f"{type(self).__name__}: list label {label!r} is not hosted.")
        return self._lists[label]

    def _run(self, request: Request) -> ExecuteResult:
        try:
            for label, data in request.write_paths.items():
                self._require_tree(label).write_path(data)

            for label, ops in request.write_lists.items():
                lst = self._require_list(label)
                for op in ops:
                    match op:
                        case ListWrite(index=index, value=value):
                            lst[index] = value
                        case ListPushFront(value=value):
                            lst.insert(0, value)
                        case ListPopBack():
                            lst.pop()

            results: dict[str, Any] = {}
            for label, leaves in request.read_paths.items():
                results[label] = self._require_tree(label).read_path(list(set(leaves)))

            for label, indices in request.read_lists.items():
                lst = self._require_list(label)
                results[label] = lst if indices is None else {i: lst[i] for i in set(indices)}

            return ExecuteResult(results=results)

        except Exception as e:
            return ExecuteResult(error=e)

    @override
    def execute(self) -> ExecuteResult:
        blob = pickle.dumps(self._take_request())
        self._bytes_written += len(blob)

        out = pickle.dumps(self._run(pickle.loads(blob)))
        self._bytes_read += len(out)
        return pickle.loads(out)


class InteractRemoteServer(InteractServer):
    """Client side of a remote deployment: ships each ``Request`` over a ``BaseSocket`` to a
    ``RemoteServer`` and returns its ``ExecuteResult``. ``init_storage`` pickles the trees (file-backed
    trees cannot be shipped) and re-raises the server's error."""

    def __init__(self):
        super().__init__()
        self._client: BaseSocket | None = None

    def _check_client(self) -> BaseSocket:
        if self._client is None:
            raise ValueError("Client has not been initialized; call init_connection() first.")
        return self._client

    @override
    def init_connection(self, client: BaseSocket) -> None:
        self._client = client

    @override
    def close_connection(self) -> None:
        self._check_client().close()

    @override
    def init_storage(self, storage: ServerStorage) -> None:
        client = self._check_client()
        client.send(("init", storage))
        reply: ExecuteResult = client.recv()
        if reply.error is not None:
            raise reply.error

    @override
    def execute(self) -> ExecuteResult:
        request = self._take_request()
        client = self._check_client()
        self._bytes_written += len(pickle.dumps(request))

        client.send(("execute", request))
        response: ExecuteResult = client.recv()

        self._bytes_read += len(pickle.dumps(response))
        return response


class RemoteServer(InteractLocalServer):
    """Server side of a remote deployment: answers ``("init", storage)`` and ``("execute", Request)``
    messages against in-process storage. Every failure is replied as ``ExecuteResult(error=...)``, so
    the serving loop never dies on a bad request. It does not count bandwidth."""

    def __init__(self):
        super().__init__()
        self._server: BaseSocket | None = None

    def _process_request(self, request: tuple[str, Any]) -> ExecuteResult:
        try:
            cmd, data = request
            if cmd == "init":
                self.init_storage(data)
                return ExecuteResult()
            if cmd == "execute":
                return self._run(data)
            raise ValueError(f"{type(self).__name__}: unknown command {cmd!r}.")

        except Exception as e:
            return ExecuteResult(error=e)

    def run(self, server: BaseSocket) -> None:
        """Listen for client queries on a bound socket (e.g. ZMQSocket with is_server=True)."""
        self._server = server

        while True:
            request = self._server.recv()
            if request:
                self._server.send(self._process_request(request))
            else:
                break

        self._server.close()
