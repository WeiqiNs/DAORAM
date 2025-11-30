from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Union, List, TypeVar, Generic, Any

from daoram.dependency import Data


BucketKey = Tuple[int, int]
BlockKey = Tuple[int, int, int]

Block = Union[Data, bytes]
Bucket = List[Block]
Buckets = List[Bucket]

PathData = Dict[int, Bucket]
BucketData = Dict[BucketKey, Bucket]
BlockData = Dict[BlockKey, Block]

# Define the payload type.
PL = TypeVar("PL")


@dataclass(frozen=True)
class Query(Generic[PL]):
    payload: PL
    query_type: str
    storage_label: str

    def envelope(self) -> Dict[str, Any]:
        """Envelope the payload of the query to the storage."""
        return {"payload": self.payload, "query_type": self.query_type, "storage_label": self.storage_label}
