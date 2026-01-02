from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Tuple, Union, Optional

from daoram.dependency.helper import Data


class BucketKey(NamedTuple):
    """Key for accessing a bucket: (leaf, bucket_id)."""
    leaf: int
    bucket_id: int


class BlockKey(NamedTuple):
    """Key for accessing a block: (leaf, bucket_id, block_id)."""
    leaf: int
    bucket_id: int
    block_id: int


Block = Union[Data, bytes]
Bucket = List[Block]
Buckets = List[Bucket]

PathData = Dict[int, Bucket]
BucketData = Dict[BucketKey, Bucket]
BlockData = Dict[BlockKey, Block]


@dataclass
class KVPair:
    """A key-value pair with named access."""
    key: Any
    value: Any

    def to_tuple(self) -> Tuple[Any, Any]:
        """Convert to a tuple (key, value)."""
        return self.key, self.value


@dataclass
class ExecuteResult:
    """Result of executing batched queries."""
    success: bool
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
