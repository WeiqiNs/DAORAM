from __future__ import annotations

import os
import pickle
from dataclasses import astuple, dataclass
from typing import Any, BinaryIO, List, Optional, Tuple, Union

from daoram.dependency.crypto import Aes, Prf


@dataclass
class Data:
    """
    Create the data structure to hold a data record that should be put into a complete binary tree.

    It has three fields: key, leaf, and value, where key and value could be anything but leaf needs to be an integer.
    By default, (when used as a dummy), when initialize the fields to None.
    """
    key: Optional[Any] = None
    leaf: Optional[int] = None
    value: Optional[Any] = None

    @classmethod
    def from_pickle(cls, data: bytes) -> Data:
        """Given some pickled data, convert it to a Data typed object"""
        # Load from pickle and remove padding if necessary.
        return cls(*pickle.loads(Helper.unpad_pickle(data)))

    def dump(self) -> bytes:
        """Dump the data structure to bytes."""
        return pickle.dumps(astuple(self))  # type: ignore


# Define some common structures that would be used.
Block = Union[Data, bytes]
Bucket = List[Block]
Buckets = List[Bucket]


class Storage:
    """
    A class that manages an n x m matrix of fixed-size byte strings.
    In our use case, n is the size of the binary tree, m is the size of each bucket.
    Since we always read the entire bucket, the read/write access is with respect to one row of the matrix.
    If filename is provided, data is stored on disk, otherwise data is stored in memory (list of lists).
    """

    def __init__(self,
                 size: int,
                 bucket_size: int,
                 filename: str = None,
                 data_size: int = None,
                 enc_key_size: int = None) -> None:
        """
        :param size: number of rows.
        :param data_size: size (in bytes) of each element.
        :param bucket_size: number of columns.
        :param filename: optional path for on-disk storage.
        :param enc_key_size: the key size for the encryption, if set to 0 it means encryption will not be used.
        """
        # Store the information useful for accessing data.
        self.__size: int = size
        self.__data_size: int = data_size
        self.__bucket_size: int = bucket_size

        # If encryption key size is not provided.
        if enc_key_size is None:
            self.__total_size: int = data_size
            self.__encryption: bool = False
        else:
            if data_size is None:
                raise ValueError("Data size is required when encrypting.")
            # Compute the total size, we need plus 2 for padding and IV.
            self.__total_size: int = (data_size // enc_key_size + 2) * enc_key_size
            # Set the encryption to True.
            self.__encryption: bool = True

        # Store the file name.
        self.__filename: str = filename

        # Whether the filename is provided determines how we store things.
        if self.__filename is None:
            # Simply create a list of empty lists.
            self.__internal_data: Buckets = [[] for _ in range(size)]
        else:
            # When file name is not None, data size must be provided.
            if data_size is None:
                raise ValueError("Data size is required when storing to a file.")

            # Allocate or confirm existence of the file.
            if not os.path.exists(self.__filename):
                with open(self.__filename, 'wb') as file:
                    # Compute the total number of bytes required.
                    total_bytes = size * bucket_size * self.__total_size
                    # Declare the space.
                    file.seek(total_bytes - 1)
                    file.write(b"\x00")

            # Open the file for read/write in binary mode
            self.__file: BinaryIO = open(self.__filename, 'r+b')

    @property
    def __dummy_byte(self) -> bytes:
        """Get a dummy all zero string of the desired length."""
        return b"\x00" * self.__total_size

    @property
    def __dummy_data(self) -> bytes:
        """Get a dummy all zero string of the desired length."""
        return Helper.pad_pickle(data=Data().dump(), length=self.__data_size)

    def __getitem__(self, index: int) -> Bucket:
        """storage[i] => returns the i-th bucket."""
        return self.read_row(index=index)

    def __setitem__(self, index: int, data: Bucket) -> None:
        """storage[i] = override the i-th bucket."""
        self.write_row(index=index, data=data)

    def __row_offset(self, index: int) -> int:
        """Compute the starting byte offset for row i: offset = i * (m * x)."""
        return index * self.__bucket_size * self.__total_size

    def read_row(self, index: int) -> Bucket:
        """Reads and returns the entire row i as a list."""
        # Determine where to read.
        if self.__filename is None:
            # In-memory read.
            return self.__internal_data[index]

        # On-disk read.
        self.__file.seek(self.__row_offset(index))
        row_bytes = self.__file.read(self.__bucket_size * self.__total_size)

        # Split this big chunk into m elements each of length x.
        read_data = [
            row_bytes[i * self.__total_size: (i + 1) * self.__total_size]
            for i in range(self.__bucket_size)
        ]

        # Filter out the dummy chunks and transform data:
        # - If encryption is True, we just return the raw chunk.
        # - If encryption is False, we unpad/unpickle the data into a Data object.
        return [
            data if self.__encryption else Data.from_pickle(data=data)
            for data in read_data if data != self.__dummy_byte
        ]

    def write_row(self, index: int, data: Bucket) -> None:
        """Writes the entire row i from a list.

        The tricky thing here is that, when writing to internal list, you don't need padding, but you need it for
        writing to the disk.
        """
        # Determines where to write.
        if self.__filename is None:
            # In-memory write and terminate the function.
            self.__internal_data[index] = data
            return

        # On-disk write. First convert and pad the data to desired length.
        write_data = [
            Helper.pad_pickle(data=elem.dump() if isinstance(elem, Data) else elem, length=self.__total_size)
            for elem in data
        ]

        # Check if more dummy data is needed.
        dummy_needed = self.__bucket_size - len(data)

        # Add dummy bytes if necessary.
        data_to_write = b"".join(write_data) if dummy_needed == 0 \
            else b"".join(write_data) + dummy_needed * self.__dummy_byte

        # Seek on-disk write position.
        self.__file.seek(self.__row_offset(index))
        # Join the row data into a single byte string
        self.__file.write(data_to_write)

    def encrypt(self, aes: Aes) -> None:
        """
        Given an aes instance, encrypt the entire storage.

        Note that each of the bucket should be made full (dummy data maybe added).
        """
        # If we do not need to load from the file.
        if self.__filename is None:
            # Iterate through the buckets.
            for i, bucket in enumerate(self.__internal_data):
                # Encrypt the existing data.
                encrypted_data = [
                    aes.enc(plaintext=Helper.pad_pickle(data=data.dump(), length=self.__data_size))
                    for data in bucket
                ]

                # If the bucket is not full, append more data to it.
                dummy_needed = self.__bucket_size - len(encrypted_data)
                # Make the bucket full.
                if dummy_needed > 0:
                    encrypted_data.extend([aes.enc(plaintext=self.__dummy_data) for _ in range(dummy_needed)])

                # Replace the bucket.
                self.__internal_data[i] = encrypted_data

            # Terminate the function.
            return

        # Otherwise we load from the disk and encrypt.
        for i in range(self.__size * self.__bucket_size):
            # Compute the location to read.
            location = i * self.__total_size
            # Seek to the correct position.
            self.__file.seek(location)
            # Get the data from that location.
            data = self.__file.read(self.__total_size)
            # Decide what data to encrypt.
            data = data[:self.__data_size] if data != self.__dummy_byte else self.__dummy_data

            # Move back and write the encrypted data back.
            self.__file.seek(location)
            self.__file.write(aes.enc(plaintext=data))

    def decrypt(self, aes: Aes) -> None:
        """Given an aes instance, decrypt the entire storage."""
        # If we do not need to load from the file
        if self.__filename is None:
            for bucket in self.__internal_data:
                # Encrypt the existing data.
                for index, data in enumerate(bucket):
                    bucket[index] = Data.from_pickle(data=aes.dec(ciphertext=data))

            # Terminate the function.
            return

        # Otherwise we load from the disk and encrypt.
        for i in range(self.__size * self.__bucket_size):
            # Compute the location to read.
            location = i * self.__total_size
            # Seek to the correct position.
            self.__file.seek(location)
            # Get the data from that location and decrypt.
            data = aes.dec(ciphertext=self.__file.read(self.__total_size))

            # Move back and write the data.
            self.__file.seek(location)
            self.__file.write(Helper.pad_pickle(data=data, length=self.__total_size))

    def close(self):
        """Close the file if using file-based storage."""
        if self.__file is not None:
            self.__file.close()
            self.__file = None


class Helper:
    @staticmethod
    def pad_pickle(data: bytes, length: int) -> bytes:
        """
        Pad pickled data to desired length with trailing zeros.

        Note that if the data to pad has trailing zeros already, the padding would fail.
        :param data: data to pad.
        :param length: desired length of the padded data.
        :return: padded data.
        """
        if len(data) > length:
            # If the data length is too long, return error.
            raise ValueError("Data length is longer than the desired padded length.")
        else:
            return data + b"\x00" * (length - len(data))

    @staticmethod
    def unpad_pickle(data: bytes) -> bytes:
        """Remove trailing zeros from padded data."""
        return data.rstrip(b"\x00")

    """A wrapper for the helper functions. They are wrapped in a class for neater importing statements."""

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
