import pytest

from oblivlib.dependency import AesGcm, Data, Storage

_DATA_SIZE = 160
_BUCKET_SIZE = 3


class TestData:
    def test_pad_length_exact_fit(self):
        data = Data(key=255, leaf=255, value=b"\x00\x01\x02\x03")
        padded = data.dump_pad(length=len(data.dump()))
        assert Data.load_unpad(padded) == data


@pytest.fixture(params=["memory-plain", "memory-enc", "disk-plain", "disk-enc"])
def storage_mode(request, test_file):
    """Build a 2x3 Storage in each (memory|disk) x (plain|encrypted) mode; return (storage, encryptor)."""
    encryption = request.param.endswith("enc")
    on_disk = request.param.startswith("disk")
    encryptor = AesGcm() if encryption else None
    disk_size = None
    if on_disk:
        disk_size = encryptor.ciphertext_length(_BUCKET_SIZE * _DATA_SIZE) if encryptor is not None else _DATA_SIZE
    storage = Storage(
        size=2,
        bucket_size=_BUCKET_SIZE,
        encryption=encryption,
        data_size=_DATA_SIZE if encryption else None,
        disk_size=disk_size,
        filename=test_file if on_disk else None,
    )
    return storage, encryptor


def _read_as_data(bucket):
    """Normalize a read bucket to Data objects (encrypted on-disk reads come back as raw bytes)."""
    return [elem if isinstance(elem, Data) else Data.load_unpad(elem) for elem in bucket]


class TestStorage:
    def test_write_read_round_trip(self, storage_mode):
        storage, encryptor = storage_mode
        storage[0] = [Data(key=1, leaf=1, value=1)]

        if encryptor is not None:
            storage.encrypt(encryptor=encryptor)
            storage.decrypt(encryptor=encryptor)

        bucket0 = _read_as_data(storage[0])
        bucket1 = _read_as_data(storage[1])

        assert bucket0[0] == Data(key=1, leaf=1, value=1)
        assert all(elem == Data() for elem in bucket0[1:])
        assert all(elem == Data() for elem in bucket1)

    def test_full_bucket_multi_row_round_trip(self, storage_mode):
        storage, encryptor = storage_mode
        storage[0] = [Data(key=1, leaf=1, value="a"), Data(key=2, leaf=2, value="b"), Data(key=3, leaf=3, value="c")]
        storage[1] = [Data(key=4, leaf=4, value="d")]

        if encryptor is not None:
            storage.encrypt(encryptor=encryptor)
            storage.decrypt(encryptor=encryptor)

        bucket0 = _read_as_data(storage[0])
        bucket1 = _read_as_data(storage[1])
        assert bucket0[:3] == [
            Data(key=1, leaf=1, value="a"),
            Data(key=2, leaf=2, value="b"),
            Data(key=3, leaf=3, value="c"),
        ]
        assert bucket1[0] == Data(key=4, leaf=4, value="d")
        assert all(elem == Data() for elem in bucket1[1:])

    def test_disk_persists_across_reopen(self, test_file):
        storage = Storage(size=2, bucket_size=3, disk_size=_DATA_SIZE, filename=test_file, encryption=False)
        storage[0] = [Data(key=1, leaf=1, value=1)]
        storage.close()

        reopened = Storage(size=2, bucket_size=3, disk_size=_DATA_SIZE, filename=test_file, encryption=False)
        assert reopened[0] == [Data(key=1, leaf=1, value=1)]
        assert reopened[1] == []
        reopened.close()


class TestStorageValidation:
    def test_encryption_requires_data_size(self):
        with pytest.raises(ValueError):
            Storage(size=2, bucket_size=3, encryption=True)

    def test_file_requires_disk_size(self, test_file):
        with pytest.raises(ValueError):
            Storage(size=2, bucket_size=3, encryption=False, filename=test_file)
