from dataclasses import dataclass
from typing import Tuple, Dict, Union, List, TypeVar, Generic, Any

from daoram.dependency.helper import Data

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

# Tree operations
TREE_READ_PATH = "tree_read_path"
TREE_WRITE_PATH = "tree_write_path"
TREE_READ_BUCKET = "tree_read_bucket"
TREE_WRITE_BUCKET = "tree_write_bucket"
TREE_READ_BLOCK = "tree_read_block"
TREE_WRITE_BLOCK = "tree_write_block"
TREE_INIT = "tree_init"

# List operations
LIST_READ = "list_read"
LIST_WRITE = "list_write"
LIST_INIT = "list_init"


@dataclass
class TreeReadPathPayload:
    """Payload for reading paths from a BinaryTree."""
    leaves: List[int]


@dataclass
class TreeWritePathPayload:
    """Payload for writing paths to a BinaryTree."""
    data: PathData  # Dict mapping tree index to bucket


@dataclass
class TreeReadBucketPayload:
    """Payload for reading buckets from a BinaryTree."""
    keys: List[BucketKey]  # List of (leaf, bucket_id)


@dataclass
class TreeWriteBucketPayload:
    """Payload for writing buckets to a BinaryTree."""
    data: BucketData  # Dict[BucketKey, Bucket]


@dataclass
class TreeReadBlockPayload:
    """Payload for reading blocks from a BinaryTree."""
    keys: List[BlockKey]  # List of (leaf, bucket_id, block_id)


@dataclass
class TreeWriteBlockPayload:
    """Payload for writing blocks to a BinaryTree."""
    data: BlockData  # Dict[BlockKey, Block]


@dataclass
class ListReadPayload:
    """Payload for reading from a List storage."""
    indices: List[int]


@dataclass
class ListWritePayload:
    """Payload for writing to a List storage."""
    data: Dict[int, Any]  # index -> value


@dataclass
class KVPair:
    """A key-value pair with named access."""
    key: Any
    value: Any

    def to_tuple(self) -> Tuple[Any, Any]:
        """Convert to a tuple (key, value)."""
        return self.key, self.value


@dataclass(frozen=True)
class Query(Generic[PL]):
    """A query to be executed against a storage."""
    payload: PL
    query_type: str
    storage_label: str
