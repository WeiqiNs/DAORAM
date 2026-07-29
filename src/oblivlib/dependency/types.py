"""Type definitions and data classes used throughout the ORAM library."""

from dataclasses import dataclass, field
from typing import Any, NamedTuple

from oblivlib.dependency.helper import Data

# Distinguishes "value not provided" from an explicit None.
UNSET = object()


class BucketKey(NamedTuple):
    leaf: int
    bucket_id: int


class BlockKey(NamedTuple):
    leaf: int
    bucket_id: int
    block_id: int


Block = Data | bytes
Bucket = list[Block]
Buckets = list[Bucket]

PathData = dict[int, Bucket]
BucketData = dict[BucketKey, Bucket]
BlockData = dict[BlockKey, Block]

# Position map: {key -> leaf}. Data map: {key -> arbitrary value}.
PosMap = dict[int, int]
DataMap = dict[int, Any]


@dataclass
class KVPair:
    key: Any
    value: Any


@dataclass
class ExecuteResult:
    success: bool
    results: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def require(self, label: str) -> Any:
        # Raise the original execute() error rather than the masking KeyError a bare results[label] hits.
        if not self.success:
            raise RuntimeError(f"execute() failed: {self.error}")
        if label not in self.results:
            raise KeyError(f"No result for label {label!r}.")
        return self.results[label]
