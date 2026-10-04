from oblivlib.dependency.codec import BlockCodec
from oblivlib.dependency.crypto import Encryptor
from oblivlib.dependency.heap_index import compute_level, leaf_index, path_to_root, union_of_paths
from oblivlib.dependency.storage import Storage
from oblivlib.dependency.types import Data, PathData


class BinaryTree:
    def __init__(
        self,
        num_data: int,
        bucket_size: int,
        codec: BlockCodec,
        encryptor: Encryptor | None = None,
        filename: str | None = None,
    ) -> None:
        self._bucket_size = bucket_size

        self._level = compute_level(num_data)
        self._size = (1 << self._level) - 1

        self._storage = Storage(
            size=self._size, bucket_size=bucket_size, codec=codec, encryptor=encryptor, filename=filename
        )

    @property
    def size(self) -> int:
        return self._size

    @property
    def level(self) -> int:
        return self._level

    @property
    def storage(self) -> Storage:
        return self._storage

    def fill_data_to_storage_leaf(self, data: Data) -> bool:
        for path_index in path_to_root(leaf_index(data.require_leaf(), self._level)):
            bucket = self._storage[path_index]
            if len(bucket) < self._bucket_size:
                bucket.append(data)
                self._storage[path_index] = bucket
                return True

        return False

    def read_path(self, leaves: list[int]) -> PathData:
        indices = union_of_paths([leaf_index(leaf, self._level) for leaf in leaves])
        return {idx: self._storage[idx] for idx in indices}

    def write_path(self, data: PathData) -> None:
        for idx, bucket in data.items():
            self._storage[idx] = bucket
