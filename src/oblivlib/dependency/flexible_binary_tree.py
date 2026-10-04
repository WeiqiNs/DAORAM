from oblivlib.dependency.codec import BlockCodec
from oblivlib.dependency.crypto import Encryptor
from oblivlib.dependency.errors import ScaleDownError
from oblivlib.dependency.heap_index import compute_level, leaf_index, path_to_root, union_of_paths
from oblivlib.dependency.storage import Storage
from oblivlib.dependency.types import Data, PathData


class FlexibleBinaryTree:
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
        size = (1 << self._level) - 1
        self._present = bytearray(b"\x01") * size
        self._storage = Storage(size=size, bucket_size=bucket_size, codec=codec, encryptor=encryptor, filename=filename)

    @property
    def level(self) -> int:
        return self._level

    @property
    def storage(self) -> Storage:
        return self._storage

    def _path_nodes(self, leaves: list[int]) -> list[int]:
        return sorted(union_of_paths([leaf_index(leaf, self._level) for leaf in leaves]))

    def fill_data_to_storage_leaf(self, data: Data) -> bool:
        for path_index in path_to_root(leaf_index(data.require_leaf(), self._level)):
            bucket = self._storage[path_index]
            if len(bucket) < self._bucket_size:
                bucket.append(data)
                self._storage[path_index] = bucket
                self._present[path_index] = 1
                return True

        return False

    def read_path(self, leaves: list[int]) -> PathData:
        return {index: self._storage[index] for index in self._path_nodes(leaves) if self._present[index]}

    def write_path(self, leaves: list[int], data: PathData) -> None:
        nodes = self._path_nodes(leaves)
        off_path = sorted(set(data) - set(nodes))
        if off_path:
            raise ValueError(f"{type(self).__name__}: indices {off_path} are not on the paths of leaves {leaves}.")

        for index in nodes:
            if index in data:
                self._storage[index] = data[index]
                self._present[index] = 1
            else:
                self._storage[index] = []
                self._present[index] = 0

    def scale_up(self) -> None:
        self._level += 1
        size = (1 << self._level) - 1
        self._present.extend(bytes(size - len(self._present)))
        self._storage.resize(size)

    def scale_down(self) -> None:
        if self._level == 1:
            raise ScaleDownError(f"{type(self).__name__}: cannot scale down below level 1.")
        size = (1 << (self._level - 1)) - 1
        if any(self._present[size:]):
            raise ScaleDownError(f"{type(self).__name__}: the bottom layer of level {self._level} still holds nodes.")

        del self._present[size:]
        self._storage.resize(size)
        self._level -= 1
