from daoram.dependency import Helper

TEST_FILE = "test.bin"


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
