import pickle

import pytest

from oblivlib.dependency import AesGcm, Data
from oblivlib.dependency.codec import DefaultCodec
from oblivlib.dependency.storage import Storage
from oblivlib.dependency.types import Bucket

_DATA_SIZE = 160
_BUCKET_SIZE = 3
_CODEC = DefaultCodec(block_size=_DATA_SIZE)


@pytest.fixture(params=["memory-plain", "memory-enc", "disk-plain", "disk-enc"])
def storage_mode(request, test_file):
    """Build a 2x3 Storage in each (memory|disk) x (plain|encrypted) mode; return (storage, encryptor)."""
    encryptor = AesGcm() if request.param.endswith("enc") else None
    filename = str(test_file) if request.param.startswith("disk") else None
    storage = Storage(size=2, bucket_size=_BUCKET_SIZE, codec=_CODEC, encryptor=encryptor, filename=filename)
    return storage, encryptor


def _seal_if_encrypted(storage: Storage, encryptor: AesGcm | None) -> None:
    if encryptor is not None:
        storage.seal(encryptor=encryptor)


def _read_blocks(storage: Storage, index: int, encryptor: AesGcm | None) -> list[Data]:
    row = storage[index]
    if encryptor is None:
        return [data for data in row if isinstance(data, Data)]
    blob = row[0]
    assert isinstance(blob, bytes)
    return _CODEC.open_bucket(encryptor, blob)


class TestStorage:
    def test_write_read_round_trip(self, storage_mode):
        storage, encryptor = storage_mode
        storage[0] = [Data(key=1, leaf=1, value=1)]
        _seal_if_encrypted(storage, encryptor)

        assert _read_blocks(storage, 0, encryptor) == [Data(key=1, leaf=1, value=1)]
        assert _read_blocks(storage, 1, encryptor) == []

    def test_full_bucket_multi_row_round_trip(self, storage_mode):
        storage, encryptor = storage_mode
        full = [Data(key=1, leaf=1, value="a"), Data(key=2, leaf=2, value="b"), Data(key=3, leaf=3, value="c")]
        storage[0] = list(full)
        storage[1] = [Data(key=4, leaf=4, value="d")]
        _seal_if_encrypted(storage, encryptor)

        assert _read_blocks(storage, 0, encryptor) == full
        assert _read_blocks(storage, 1, encryptor) == [Data(key=4, leaf=4, value="d")]

    def test_reopen_starts_empty(self, test_file):
        storage = Storage(size=2, bucket_size=_BUCKET_SIZE, codec=_CODEC, filename=str(test_file))
        storage[0] = [Data(key=1, leaf=1, value=1)]
        storage.close()

        reopened = Storage(size=2, bucket_size=_BUCKET_SIZE, codec=_CODEC, filename=str(test_file))
        assert reopened[0] == []
        assert reopened[1] == []
        reopened.close()

    @pytest.mark.parametrize("sealed", [False, True], ids=["disk-plain", "disk-enc"])
    def test_oversized_block_raises_and_leaves_neighbor_intact(self, test_file, sealed):
        encryptor = AesGcm() if sealed else None
        storage = Storage(size=2, bucket_size=_BUCKET_SIZE, codec=_CODEC, encryptor=encryptor, filename=str(test_file))
        storage[0] = [Data(key=0, leaf=0, value="left")]
        storage[1] = [Data(key=1, leaf=1, value="neighbor")]
        _seal_if_encrypted(storage, encryptor)
        before = (storage[0], storage[1])

        oversized: list[Bucket] = (
            [[b"\x01" * (encryptor.ciphertext_length(_BUCKET_SIZE * _DATA_SIZE) + 1)]]
            if encryptor is not None
            else [
                [Data(key=0, leaf=0, value=b"x" * _DATA_SIZE)],
                [Data(key=i, leaf=0, value=i) for i in range(_BUCKET_SIZE + 1)],
            ]
        )
        for bucket in oversized:
            with pytest.raises(ValueError):
                storage[0] = bucket

        assert (storage[0], storage[1]) == before
        assert _read_blocks(storage, 1, encryptor) == [Data(key=1, leaf=1, value="neighbor")]

    def test_resize_extends_and_truncates(self, storage_mode):
        storage, encryptor = storage_mode
        storage[0] = [Data(key=0, leaf=0, value="kept")]
        storage[1] = [Data(key=1, leaf=1, value="dropped")]
        _seal_if_encrypted(storage, encryptor)

        storage.resize(4)
        assert storage[2] == [] and storage[3] == []
        assert _read_blocks(storage, 1, encryptor) == [Data(key=1, leaf=1, value="dropped")]

        storage.resize(1)
        storage.resize(2)
        assert storage[1] == []
        assert _read_blocks(storage, 0, encryptor) == [Data(key=0, leaf=0, value="kept")]

    def test_storage_pickles_without_encryptor(self):
        encryptor = AesGcm()
        storage = Storage(size=2, bucket_size=_BUCKET_SIZE, codec=_CODEC, encryptor=encryptor)
        storage[0] = [Data(key=1, leaf=1, value="secret")]
        storage.seal(encryptor=encryptor)

        blob = pickle.dumps(storage)
        assert encryptor.key not in blob
        assert _read_blocks(pickle.loads(blob), 0, encryptor) == [Data(key=1, leaf=1, value="secret")]
