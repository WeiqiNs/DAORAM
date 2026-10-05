"""The client/server wire protocol: every message the client and ``StorageServer`` exchange.

A message is a frozen dataclass of primitives (ints, str, bytes, None, lists, and nested ops), framed as
the msgpack array ``[ClassName, *fields in declaration order]``. Decoding is the trust boundary: it
checks the tag, the arity, and every field's type against the dataclass annotations, and raises
``ProtocolError`` for anything else, so a peer never gets more than these primitives across.
"""

import types
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, get_args, get_origin, get_type_hints

import msgpack

from oblivlib.dependency.errors import (
    DuplicateLabelError,
    OblivlibError,
    ProtocolError,
    RowSizeError,
    ScaleDownError,
    ServerError,
    StorageError,
    UnknownLabelError,
)
from oblivlib.dependency.types import ListPopBack, ListPushFront, ListWrite

PROTOCOL_VERSION = 1
DEFAULT_MAX_MESSAGE_BYTES = 256 * 2**20
WRITE_RANGE_CHUNK_BYTES = 32 * 2**20
DEFAULT_TIMEOUT_MS = 30_000


@dataclass(frozen=True, slots=True)
class ReadPath:
    label: str
    leaves: list[int]


@dataclass(frozen=True, slots=True)
class WritePath:
    label: str
    leaves: list[int]
    rows: list[bytes]


@dataclass(frozen=True, slots=True)
class ReadList:
    label: str
    indices: list[int] | None


@dataclass(frozen=True, slots=True)
class ListOps:
    label: str
    ops: list[ListWrite | ListPushFront | ListPopBack]


@dataclass(frozen=True, slots=True)
class Hello:
    version: int


@dataclass(frozen=True, slots=True)
class CreateTree:
    session: str
    label: str
    level: int
    row_bytes: int | None


@dataclass(frozen=True, slots=True)
class CreateList:
    session: str
    label: str


@dataclass(frozen=True, slots=True)
class AttachTree:
    session: str
    label: str
    filename: str
    level: int
    row_bytes: int


@dataclass(frozen=True, slots=True)
class DropTree:
    session: str
    label: str


@dataclass(frozen=True, slots=True)
class Resize:
    session: str
    label: str
    level: int


@dataclass(frozen=True, slots=True)
class WriteRange:
    session: str
    label: str
    start: int
    rows: list[bytes]


@dataclass(frozen=True, slots=True)
class Batch:
    session: str
    writes: list[WritePath | ListOps]
    reads: list[ReadPath | ReadList]


@dataclass(frozen=True, slots=True)
class Close:
    session: str


@dataclass(frozen=True, slots=True)
class Shutdown:
    pass


@dataclass(frozen=True, slots=True)
class HelloReply:
    session: str
    max_message_bytes: int


@dataclass(frozen=True, slots=True)
class Ok:
    pass


@dataclass(frozen=True, slots=True)
class BatchReply:
    results: list[list[bytes]]


@dataclass(frozen=True, slots=True)
class ErrorReply:
    kind: str
    message: str


WriteOp = WritePath | ListOps
ReadOp = ReadPath | ReadList
Request = Hello | CreateTree | CreateList | AttachTree | DropTree | Resize | WriteRange | Batch | Close | Shutdown
Reply = HelloReply | Ok | BatchReply | ErrorReply

_Decoder = Callable[[Any], Any]


def _tag_table[T](classes: tuple[type[T], ...]) -> dict[str, type[T]]:
    table: dict[str, type[T]] = {}
    for cls in classes:
        if cls.__name__ in table:
            raise ValueError(f"protocol: duplicate tag {cls.__name__!r}.")
        table[cls.__name__] = cls
    return table


ERROR_KINDS: dict[str, type[OblivlibError]] = _tag_table(
    (UnknownLabelError, DuplicateLabelError, RowSizeError, ProtocolError, ScaleDownError, StorageError, ServerError)
)


def _decode_instance(kind: type) -> _Decoder:
    def decode(raw: Any) -> Any:
        if type(raw) is not kind:
            raise ProtocolError(f"expected {kind.__name__}, got {type(raw).__name__}.")
        return raw

    return decode


def _decode_list(item: _Decoder) -> _Decoder:
    def decode(raw: Any) -> list[Any]:
        if type(raw) is not list:
            raise ProtocolError(f"expected a list, got {type(raw).__name__}.")
        return [item(value) for value in raw]

    return decode


def _decode_optional(inner: _Decoder) -> _Decoder:
    return lambda raw: None if raw is None else inner(raw)


def _decode_tagged(table: dict[str, type]) -> _Decoder:
    def decode(raw: Any) -> Any:
        if type(raw) is not list or not raw or type(raw[0]) is not str:
            raise ProtocolError("expected a tagged message array.")
        cls = table.get(raw[0])
        if cls is None:
            raise ProtocolError(f"unexpected message tag {raw[0]!r}.")
        decoders = _FIELD_DECODERS[cls]
        if len(raw) - 1 != len(decoders):
            raise ProtocolError(f"{raw[0]} takes {len(decoders)} fields, got {len(raw) - 1}.")
        return cls(*(decoder(value) for decoder, value in zip(decoders, raw[1:], strict=True)))

    return decode


def _decoder_for(hint: Any) -> _Decoder:
    if hint in (int, str, bytes):
        return _decode_instance(hint)
    if get_origin(hint) is list:
        return _decode_list(_decoder_for(get_args(hint)[0]))
    if get_origin(hint) is types.UnionType:
        members = get_args(hint)
        if type(None) in members:
            (inner,) = (member for member in members if member is not type(None))
            return _decode_optional(_decoder_for(inner))
        return _decode_tagged(_tag_table(members))
    raise TypeError(f"protocol: no decoder for field type {hint!r}.")


_FIELD_DECODERS: dict[type, list[_Decoder]] = {}
for _cls in (
    ListWrite,
    ListPushFront,
    ListPopBack,
    ReadPath,
    WritePath,
    ReadList,
    ListOps,
    *get_args(Request),
    *get_args(Reply),
):
    _FIELD_DECODERS[_cls] = [_decoder_for(hint) for hint in get_type_hints(_cls).values()]

_FIELD_NAMES: dict[type, tuple[str, ...]] = {cls: tuple(f.name for f in fields(cls)) for cls in _FIELD_DECODERS}
_REQUEST_DECODER = _decode_tagged(_tag_table(get_args(Request)))
_REPLY_DECODER = _decode_tagged(_tag_table(get_args(Reply)))


def _to_wire(value: Any) -> Any:
    if type(value) is list:
        return [_to_wire(item) for item in value]
    names = _FIELD_NAMES.get(type(value))
    if names is None:
        return value
    return [type(value).__name__, *(_to_wire(getattr(value, name)) for name in names)]


def encode(message: Request | Reply) -> bytes:
    return msgpack.packb(_to_wire(message))


def _decode(data: bytes, decoder: _Decoder) -> Any:
    try:
        raw = msgpack.unpackb(data, raw=False)
    except Exception as exc:
        raise ProtocolError(f"undecodable message: {exc}") from exc
    try:
        return decoder(raw)
    except ValueError as exc:
        raise ProtocolError(f"invalid message field: {exc}") from exc


def decode_request(data: bytes) -> Request:
    return _decode(data, _REQUEST_DECODER)


def decode_reply(data: bytes) -> Reply:
    return _decode(data, _REPLY_DECODER)


def error_reply(exc: Exception) -> ErrorReply:
    for cls in type(exc).__mro__:
        if ERROR_KINDS.get(cls.__name__) is cls:
            return ErrorReply(kind=cls.__name__, message=str(exc))
    if isinstance(exc, OSError):
        return ErrorReply(kind=StorageError.__name__, message=str(exc))
    return ErrorReply(kind=ServerError.__name__, message=f"{type(exc).__name__}: {exc}")


def expect_reply[R](reply: Reply, kind: type[R]) -> R:
    if isinstance(reply, ErrorReply):
        error = ERROR_KINDS.get(reply.kind)
        if error is None:
            raise ProtocolError(f"unknown error kind {reply.kind!r}: {reply.message}")
        raise error(reply.message)
    if not isinstance(reply, kind):
        raise ProtocolError(f"expected {kind.__name__}, got {type(reply).__name__}.")
    return reply
