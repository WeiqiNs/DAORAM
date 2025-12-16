from __future__ import annotations

import math
import pickle
import struct
from dataclasses import astuple, dataclass
from typing import List, Tuple, Union, Any, Optional

from daoram.dependency.crypto import PseudoRandomFunction, Encryptor


@dataclass
class Data:
    """
    Create the data structure to hold a data record that should be put into a complete binary tree.

    It has three fields: key, leaf, and value, where key and value could be anything, but the leaf needs to be an int.
    By default, (when used as a dummy), when initialize the fields to None.
    """
    key: Optional[Any] = None
    leaf: Optional[int] = None
    value: Optional[Any] = None

    @classmethod
    def load_unpad(cls, data: bytes) -> Data:
        """Unpad some bytes and load into the Data structure."""
        # Load from pickle and remove padding if necessary.
        return cls(*pickle.loads(Helper.unpad_pickle(data=data)))

    def dump_pad(self, length: int) -> bytes:
        """Dump the data structure to desired number of bytes."""
        return Helper.pad_pickle(data=pickle.dumps(astuple(self)), length=length)


class Helper:
    """A wrapper for the helper functions. They are wrapped in a class for neater importing statements."""

    # Fixed-size header for storing payload length; keeps unpadding lossless.
    LENGTH_HEADER_SIZE = 4

    @staticmethod
    def pad_pickle(data: bytes, length: int) -> bytes:
        """
        Pad pickled data to the desired length with a length header and trailing zeros.

        :param data: Data to pad.
        :param length: Desired length of the padded data, the desired length include the header!
        :return: Padded data.
        """
        # Note that for the desired length, we only count the body of the data.
        if len(data) + Helper.LENGTH_HEADER_SIZE > length:
            raise ValueError("The desired length is too short, must be at least 4 + data length.")

        # Header to denote the original length for safe unpad.
        return struct.pack("!I", len(data)) + data + b"\x00" * (length - len(data) - Helper.LENGTH_HEADER_SIZE)

    @staticmethod
    def unpad_pickle(data: bytes) -> bytes:
        """Restore original bytes from padded data using the stored length header."""
        if len(data) < Helper.LENGTH_HEADER_SIZE:
            raise ValueError("Padded data too short to contain length header.")

        # Read the original length.
        orig_len, = struct.unpack("!I", data[:Helper.LENGTH_HEADER_SIZE])

        # Check if the object has been trimmed.
        if orig_len > len(data) - Helper.LENGTH_HEADER_SIZE:
            raise ValueError("Invalid length header in padded data.")

        # Return the unpadded data.
        return data[Helper.LENGTH_HEADER_SIZE: Helper.LENGTH_HEADER_SIZE + orig_len]

    @staticmethod
    def compute_data_size(data: Data) -> int:
        """Given a data record, find the size of the data record."""
        return len(pickle.dumps(astuple(data))) + Helper.LENGTH_HEADER_SIZE

    @staticmethod
    def compute_disk_size(data: Data, encryptor: Optional[Encryptor] = None) -> int:
        """Given a data record and an encryptor, find the size required to store the encrypted data on disk."""
        if encryptor is None:
            return Helper.compute_data_size(data)
        else:
            return encryptor.ciphertext_length(plaintext_length=Helper.compute_data_size(data))

    @staticmethod
    def binary_str_to_bytes(binary_str: str) -> bytes:
        """Given some binary integer, convert it to a byte string."""
        return int(binary_str, 2).to_bytes((len(binary_str) + 7) // 8, byteorder="big")

    @staticmethod
    def bytes_to_binary_str(binary_bytes: bytes) -> str:
        """Given some byte string, convert it to a binary string."""
        return bin(int.from_bytes(binary_bytes, byteorder="big"))[2:]

    @staticmethod
    def hash_data_to_leaf(prf: PseudoRandomFunction, map_size: int, data: Union[str, int, bytes]) -> int:
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

        # Use the prf as a hash and compute mod map size.
        return prf.digest_mod_n(message=byte_data, mod=map_size)

    @staticmethod
    def hash_data_to_map(
            prf: PseudoRandomFunction,
            map_size: int,
            data: List[Tuple[Union[str, int, bytes], Any]]
    ) -> dict:
        """
        Given a list of data, map them to the correct integer bucket.

        :param prf: The PRF instance is defined in crypto.
        :param map_size: The total number of buckets; note that some buckets might be empty.
        :param data: A list of key-value pairs.
        :return: A dictionary where each integer corresponds to a bucket of key-value pairs.
        """
        # Create a data map with empty buckets.
        data_map = {i: [] for i in range(map_size)}

        # Map each data to the correct buckets.
        for data in data:
            data_key = Helper.hash_data_to_leaf(prf=prf, data=data[0], map_size=map_size)
            data_map[data_key].append(data)

        # Remove the empty buckets.
        return data_map

    @staticmethod
    def lambert_w(x: float, tol: float = 1e-10, max_iter: int = 100):
        """
        Calculate the Lambert W function for a given value of x using an iterative Newton's method approach.

        The Lambert W function satisfies the equation w * e^w = x for a given x.
        :param x: The input value for which to calculate the Lambert W function.
        :param tol: The error tolerance for the convergence criterion.
        :param max_iter: The maximum number of iterations allowed for the Newton's method.
        :return: The value of the Lambert W function at the given input x.
        """
        # Handle special cases.
        if x == 0:
            return 0.0
        if x < -math.exp(-1):
            raise ValueError("lambert_w(x) is not defined for x < -1/e.")

        # Initial approximation.
        w = 0 if x <= 1 else math.log(x) - math.log(math.log(x))

        # Newton's method.
        for _ in range(max_iter):
            ew = math.exp(w)
            wew = w * ew
            w_next = w - (wew - x) / (ew * (w + 1) - (w + 2) * (wew - x) / (2 * w + 2))

            # Stop if meet the tolerance bound.
            if abs(w_next - w) < tol:
                return w_next

            # Otherwise update w.
            w = w_next

        raise RuntimeError("Lambert W function did not converge")
