"""Type definitions and data classes used throughout the ORAM library."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import Field, dataclass, fields
from typing import TYPE_CHECKING, Any, ClassVar, Self

UNSET = object()
"""Sentinel for "no value given": ``operate_on_key(key)`` reads, while ``operate_on_key(key, value)``
writes. Compare with ``is``."""


class FieldTuple:
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    if TYPE_CHECKING:

        def __init__(self, *args: Any) -> None: ...

    @classmethod
    def from_fields(cls, values: Sequence[Any]) -> Self:
        return cls(*values)

    def to_fields(self) -> list[Any]:
        return [getattr(self, f.name) for f in fields(self)]


@dataclass
class Data(FieldTuple):
    key: Any = None
    leaf: int | None = None
    value: Any = None

    def require_leaf(self) -> int:
        if self.leaf is None:
            raise ValueError("Data block has no leaf assigned.")
        return self.leaf


@dataclass
class KVPair:
    key: Any
    value: Any


Bucket = list[Data]

PathData = dict[int, Bucket]
PathRows = dict[int, bytes]

PosMap = dict[int, int]
DataMap = dict[int, Any]
InitData = Mapping[int, Any] | Iterable[tuple[int, Any]]


@dataclass(frozen=True)
class ListWrite:
    index: int
    value: bytes

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"ListWrite index must be >= 0, got {self.index}.")


@dataclass(frozen=True)
class ListPushFront:
    value: bytes


@dataclass(frozen=True)
class ListPopBack:
    pass


ListOp = ListWrite | ListPushFront | ListPopBack
