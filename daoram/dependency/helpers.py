from typing import Any, List, Tuple, Union

from daoram.dependency.crypto import Prf


class Helper:
    @staticmethod
    def binary_str_to_bytes(binary_str: str) -> bytes:
        """Given some binary integer, convert it to a byte string."""
        return int(binary_str, 2).to_bytes((len(binary_str) + 7) // 8, byteorder="big")

    @staticmethod
    def bytes_to_binary_str(binary_bytes: bytes) -> str:
        """Given some byte string, convert it to a binary string."""
        return bin(int.from_bytes(binary_bytes, byteorder="big"))[2:]

    @staticmethod
    def hash_data_to_leaf(prf: Prf, map_size: int, data: Union[str, int, bytes]) -> int:
        """Compute H(data) % map_size."""
        # Convert data to bytes depend on their types.
        if type(data) is int:
            byte_data = data.to_bytes(16, byteorder="big")
        elif type(data) is str:
            byte_data = data.encode("utf-8")
        elif type(data) is bytes:
            byte_data = data
        else:
            raise TypeError(f"Data must be either a string or an integer.")

        # Use the prf as hash and compute mod map size.
        return prf.digest_mod_n(message=byte_data, mod=map_size)

    @staticmethod
    def hash_data_to_map(prf: Prf, map_size: int, data: List[Tuple[Union[str, int, bytes], Any]]) -> dict:
        """
        Given a list of data, map them to the correct integer bucket.

        :param prf: the PRF instance defined in crypto.
        :param map_size: the total number of buckets; note that some buckets might be empty.
        :param data: a list of key-value pairs.
        :return: a dictionary where each integer corresponds to a bucket of key-value pairs.
        """
        # Create data map with empty buckets.
        data_map = {i: [] for i in range(map_size)}

        # Map each data to the correct buckets.
        for data in data:
            data_key = Helper.hash_data_to_leaf(prf=prf, data=data[0], map_size=map_size)
            data_map[data_key].append(data)

        # Remove the empty buckets.
        return data_map
