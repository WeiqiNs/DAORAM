from daoram.dependency import Blake2Prf, Helper

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

    def test_hash_data_to_leaf(self):
        prf = Blake2Prf()
        map_size = 100

        # Test various input types and verify deterministic output.
        test_cases = [42, "hello", b"hello", (1, 2, 3), [1, 2, 3]]
        for data in test_cases:
            result = Helper.hash_data_to_leaf(prf=prf, map_size=map_size, data=data)
            assert 0 <= result < map_size
            assert result == Helper.hash_data_to_leaf(prf=prf, map_size=map_size, data=data)

        # Test custom object with __str__.
        class CustomKey:
            def __init__(self, value):
                self.value = value

            def __str__(self):
                return f"CustomKey({self.value})"

        result = Helper.hash_data_to_leaf(prf=prf, map_size=map_size, data=CustomKey(42))
        assert 0 <= result < map_size
        assert result == Helper.hash_data_to_leaf(prf=prf, map_size=map_size, data=CustomKey(42))
