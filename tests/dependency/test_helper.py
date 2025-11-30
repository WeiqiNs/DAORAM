import os

from daoram.dependency import Aes, Data, Helper, Storage

TEST_FILE = "test.bin"


class TestData:
    def test_init_data(self):
        # Test the dataclass.
        data = Data(key="Key", leaf=0)
        # Assert the fields.
        assert data.key == "Key"
        assert data.leaf == 0
        assert data.value is None
        # Check the pickle dump and load works.
        assert Data.from_pickle(data.dump()) == data

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


class TestStorage:
    def test_internal_storage(self):
        # In this case the byte size could be different.
        storage = Storage(size=2, bucket_size=3)
        # Write via bracket notation
        storage[0] = [Data(key=1, leaf=1, value=1)]
        storage[1] = [Data(key=1, leaf=1, value=1), Data(key=2, leaf=2, value=2)]
        # Read via bracket notation
        assert storage[0] == [Data(key=1, leaf=1, value=1)]
        assert storage[1] == [Data(key=1, leaf=1, value=1), Data(key=2, leaf=2, value=2)]

    def test_internal_storage_enc(self):
        # In this case the byte size could be different.
        storage = Storage(size=2, bucket_size=3, data_size=160)
        # Get an AES instance.
        aes = Aes()
        # Write via bracket notation
        storage[0] = [Data(key=1, leaf=1, value=1)]
        # Encrypt the storage and then decrypt.
        storage.encrypt(aes=aes)
        # Decrypt the storage.
        storage.decrypt(aes=aes)
        # Assert the storage.
        assert storage[0][0] == Data(key=1, leaf=1, value=1)
        assert storage[0][1] == Data()

    def test_disk_storage(self):
        # Create a 2x3 matrix, each element 4 bytes, stored on disk.
        storage = Storage(size=2, bucket_size=3, data_size=100, filename=TEST_FILE)
        # Write via bracket notation.
        storage[0] = [Data(key=1, leaf=1, value=1)]
        # Read via bracket notation, note that padding will be present.
        assert storage[0] == [Data(key=1, leaf=1, value=1)]
        assert storage[1] == []
        # Close the file when done.
        storage.close()

        # Perform another read.
        storage = Storage(size=2, bucket_size=3, data_size=100, filename=TEST_FILE)
        # Read via bracket notation, note that padding will be present.
        assert storage[0] == [Data(key=1, leaf=1, value=1)]
        # Close and remove the file when done.
        storage.close()
        os.remove(TEST_FILE)

    def test_disk_storage_enc(self):
        # Create a 2x3 matrix, each element 4 bytes, stored on disk.
        storage = Storage(size=2, bucket_size=3, data_size=160, filename=TEST_FILE, enc_key_size=16)
        # Get an AES instance.
        aes = Aes()
        # Write via bracket notation.
        storage[0] = [Data(key=1, leaf=1, value=1)]
        # Encrypt the storage and then decrypt.
        storage.encrypt(aes=aes)
        # Decrypt the storage.
        storage.decrypt(aes=aes)

        # Testing the file locations.
        assert Data.from_pickle(storage[0][0]) == Data(key=1, leaf=1, value=1)
        assert Data.from_pickle(storage[0][1]) == Data()
        assert Data.from_pickle(storage[1][0]) == Data()
        # Close and remove the file when done.
        storage.close()
        os.remove(TEST_FILE)


class TestHelper:
    def test_binary_str_conversion(self):
        # Declare a string representing a binary number.
        binary_str = "100100100"
        # Convert to bytes and then convert it back.
        assert Helper.bytes_to_binary_str(
            binary_bytes=Helper.binary_str_to_bytes(binary_str=binary_str)
        ) == binary_str

    def test_pad_pickle(self):
        empty_str = b""
        assert Helper.unpad_pickle(Helper.pad_pickle(data=empty_str, length=100)) == empty_str

        data = b"data"
        assert Helper.unpad_pickle(Helper.pad_pickle(data=data, length=100)) == data

