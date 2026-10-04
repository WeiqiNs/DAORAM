"""Type definitions and data classes used throughout the ORAM library."""

import pickle
from dataclasses import Field, dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar, Self

from oblivlib.dependency.errors import MissingResultError

UNSET = object()
"""Sentinel for "no value given": ``operate_on_key(key)`` reads, while ``operate_on_key(key, None)`` writes
``None``. Compare with ``is``."""


class FieldTuplePickle:
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    if TYPE_CHECKING:

        def __init__(self, *args: Any) -> None: ...

    @classmethod
    def load(cls, data: bytes) -> Self:
        return cls(*pickle.loads(data))

    def dump(self) -> bytes:
        return pickle.dumps(tuple(getattr(self, f.name) for f in fields(self)))


@dataclass
class Data(FieldTuplePickle):
    key: Any = None
    leaf: int | None = None
    value: Any = None

    def dump_pad(self, length: int) -> bytes:
        payload = self.dump()
        if len(payload) > length:
            raise ValueError(f"Data: the {len(payload)}-byte pickle exceeds the {length}-byte slot.")
        return payload + b"\x00" * (length - len(payload))

    def is_real(self) -> bool:
        return self.key is not None

    def require_leaf(self) -> int:
        if self.leaf is None:
            raise ValueError("Data block has no leaf assigned.")
        return self.leaf


@dataclass
class KVPair:
    key: Any
    value: Any


Block = Data | bytes
Bucket = list[Block]

PathData = dict[int, Bucket]

PosMap = dict[int, int]
DataMap = dict[int, Any]


@dataclass(frozen=True)
class ListWrite:
    index: int
    value: Any

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"ListWrite index must be >= 0, got {self.index}.")


@dataclass(frozen=True)
class ListPushFront:
    value: Any


@dataclass(frozen=True)
class ListPopBack:
    pass


ListOp = ListWrite | ListPushFront | ListPopBack


@dataclass(frozen=True)
class Request:
    read_paths: dict[str, list[int]]
    read_lists: dict[str, list[int] | None]
    write_paths: dict[str, PathData]
    write_lists: dict[str, list[ListOp]]


@dataclass
class ExecuteResult:
    """The outcome of one ``InteractServer.execute()``.

    ``results`` maps each read label to its result (a ``PathData`` for a path read; the whole list, or
    ``{index: value}``, for a list read). ``error`` is the exception a failed execute caught, else
    ``None``; a failed execute may already have applied some of its writes. Read through ``require``,
    which re-raises ``error`` and raises ``MissingResultError`` for a label with no result.
    """

    results: dict[str, Any] = field(default_factory=dict)
    error: Exception | None = None

    def require(self, label: str) -> Any:
        if self.error is not None:
            raise self.error
        if label not in self.results:
            raise MissingResultError(f"ExecuteResult has no result for label {label!r}.")
        return self.results[label]
