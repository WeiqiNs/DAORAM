import pickle
from dataclasses import astuple

from daoram.dependency import AesGcm, Data, Storage, Helper


class TestData:
    def test_init_data(self):
        # Test the dataclass.
        data = Data(key="Key", leaf=0)
        # Assert the fields.
        assert data.key == "Key"
        assert data.leaf == 0
        assert data.value is None
        # Check the pickle dump and load works.
        assert Data.load_unpad(data.dump_pad(length=1000)) == data

    def test_dummy_data(self):
        # Test dummy data.
        dummy = Data()
        # Assert the fields.
        assert dummy.key is None
        assert dummy.leaf is None
        assert dummy.value is None

    def test_update_data(self):
        # Test the dataclass.
        data = Data(key="Key", leaf=0, value="Value")
        # Update data.
        data.key = "New Key"
        data.leaf = 100

        # Assert the fields.
        assert data.key == "New Key"
        assert data.leaf == 100

    def test_pad_length(self):
        # Some random data.
        data = Data(key="Key", leaf=0, value="Testing some value.")

        # Get the actual pickled length.
        length = len(pickle.dumps(astuple(data)))

        # The dump_pad method length should have a header.
        assert length == len(data.dump_pad(length=length + Helper.LENGTH_HEADER_SIZE)) - Helper.LENGTH_HEADER_SIZE


class TestStorage:
    def test_internal_storage(self):
        # In this case the byte size could be different.
        storage = Storage(size=2, bucket_size=3, encryption=False)
        # Write via bracket notation
        storage[0] = [Data(key=1, leaf=1, value=1)]
        storage[1] = [Data(key=1, leaf=1, value=1), Data(key=2, leaf=2, value=2)]
        # Read via bracket notation
        assert storage[0] == [Data(key=1, leaf=1, value=1)]
        assert storage[1] == [Data(key=1, leaf=1, value=1), Data(key=2, leaf=2, value=2)]

    def test_internal_storage_enc(self):
        # In this case the byte size could be different.
        storage = Storage(size=2, bucket_size=3, data_size=160, encryption=True)
        storage[0] = [Data(key=1, leaf=1, value=1)]
        # Get an AES instance.
        encryptor = AesGcm()
        # Write via bracket notation
        storage[0] = [Data(key=1, leaf=1, value=1)]
        # Encrypt the storage and then decrypt.
        storage.encrypt(encryptor=encryptor)
        # Decrypt the storage.
        storage.decrypt(encryptor=encryptor)
        # Assert the storage.
        assert storage[0][0] == Data(key=1, leaf=1, value=1)
        assert storage[0][1] == Data()

    def test_disk_storage(self, test_file):
        # Create a 2x3 matrix, each element 4 bytes, stored on disk.
        storage = Storage(size=2, bucket_size=3, disk_size=160, filename=test_file, encryption=False)
        # Write via bracket notation.
        storage[0] = [Data(key=1, leaf=1, value=1)]
        # Read via bracket notation, note that padding will be present.
        assert storage[0] == [Data(key=1, leaf=1, value=1)]
        assert storage[1] == []
        # Close the file when done.
        storage.close()

        # Perform another read.
        storage = Storage(size=2, bucket_size=3, disk_size=160, filename=test_file, encryption=False)
        # Read via bracket notation, note that padding will be present.
        assert storage[0] == [Data(key=1, leaf=1, value=1)]
        # Close the file when done (removal handled by fixture).
        storage.close()

    def test_disk_storage_enc(self, test_file):
        # Get an AES instance.
        encryptor = AesGcm()

        # Compute the disk size.
        disk_size = encryptor.ciphertext_length(plaintext_length=160)

        # Create a 2x3 matrix, each element 4 bytes, stored on disk.
        storage = Storage(
            size=2, bucket_size=3, data_size=160, disk_size=disk_size, encryption=True, filename=test_file
        )

        # Write via bracket notation.
        storage[0] = [Data(key=1, leaf=1, value=1)]
        # Encrypt the storage and then decrypt.
        storage.encrypt(encryptor=encryptor)
        # Decrypt the storage.
        storage.decrypt(encryptor=encryptor)

        # Testing the file locations.
        assert Data.load_unpad(storage[0][0]) == Data(key=1, leaf=1, value=1)
        assert Data.load_unpad(storage[0][1]) == Data()
        assert Data.load_unpad(storage[1][0]) == Data()
        # Close the file when done (removal handled by fixture).
        storage.close()
