import os
from typing import BinaryIO, Optional, List

from daoram.dependency.crypto import Encryptor
from daoram.dependency.helper import Data
from daoram.dependency.types import Bucket


class Storage:
    """
    A class that manages an n x m matrix of fixed-size byte strings.
    In our use case, n is the size of the binary tree, m is the size of each bucket.
    Since we always read the entire bucket, the read/write access is with respect to one row of the matrix.
    If the filename is provided, data is stored on disk, otherwise data is stored in memory (list of lists).
    """

    def __init__(
            self,
            size: int,
            bucket_size: int,
            encryption: bool,
            filename: str = None,
            data_size: int = None,
            disk_size: int = None,
    ) -> None:
        """
        :param size: Number of rows (tree nodes) in the matrix.
        :param bucket_size: Number of blocks stored in each bucket (columns).
        :param encryption: Whether encryption is enabled or not.
        :param filename: Optional path for on-disk backing; if provided, storage is pre-allocated.
        :param data_size: Number of bytes in the data structure (padding header will be added automatically).
        :param disk_size: Byte length of each block (after padding and encryption), required for file-backed storage.
        """
        # Store the information useful for accessing data.
        self._size: int = size
        self._filename: str = filename
        self._encryption: bool = encryption
        self._bucket_size: int = bucket_size
        self._file: Optional[BinaryIO] = None

        # Whether encryption is needed or not determines how we generate dummy data.
        if self._encryption:
            # When the encryption is used, data size must be provided (to hide length).
            if data_size is None:
                raise ValueError("Data size is required to be provided for encryption.")

            # Add the padding header so callers do not need to account for it.
            self._data_size: int = data_size

        # Whether the filename is provided determines how we store things.
        if self._filename is None:
            # Create a list of empty lists.
            self._internal_data: List[list] = [[] for _ in range(size)]
        else:
            # When the file name is not None, disk size must be provided.
            if disk_size is None:
                raise ValueError("Disk size is required to be provided when storing to a file.")

            # Store the disk size.
            # Add the padding header so callers do not need to account for it.
            self._disk_size: int = disk_size

            # Compute the total number of bytes required for the backing file.
            total_bytes = size * bucket_size * self._disk_size

            # Allocate or resize the file to the exact size we need.
            if not os.path.exists(self._filename) or os.path.getsize(self._filename) != total_bytes:
                with open(self._filename, 'wb') as file:
                    if total_bytes > 0:
                        file.seek(total_bytes - 1)
                        file.write(b"\x00")

            # Open the file for read/write in binary mode
            self._file: BinaryIO = open(self._filename, 'r+b')

    @property
    def _data_size_dummy(self) -> bytes:
        """Get a dummy all zero string of the desired length."""
        return Data().dump_pad(self._data_size)

    @property
    def _disk_size_dummy(self) -> bytes:
        """Get a dummy all zero string of the desired length."""
        return Data().dump_pad(self._disk_size)

    def __getitem__(self, index: int) -> Bucket:
        """storage[i] => returns the i-th bucket."""
        return self.read_row(index=index)

    def __setitem__(self, index: int, data: Bucket) -> None:
        """storage[i] = override the i-th bucket."""
        self.write_row(index=index, data=data)

    def _row_offset(self, index: int) -> int:
        """Compute the starting byte offset for row i: offset = i * (m * x)."""
        return index * self._bucket_size * self._disk_size

    def read_row(self, index: int) -> Bucket:
        """Reads and returns the entire row i as a list."""
        # Determine where to read.
        if self._filename is None:
            # In-memory read.
            return self._internal_data[index]

        # On-disk read.
        self._file.seek(self._row_offset(index))
        row_bytes = self._file.read(self._bucket_size * self._disk_size)

        # Split this big chunk into m elements each of lengths x.
        read_data = [
            row_bytes[i * self._disk_size: (i + 1) * self._disk_size] for i in range(self._bucket_size)
        ]

        # Filter out the dummy chunks and transform data:
        # - If encryption is True, we just return the raw chunk.
        # - If encryption is False, we unpickle the data into a Data object.
        return [
            data if self._encryption else Data.load_unpad(data=data)
            for data in read_data if not data == b"\x00" * self._disk_size
        ]

    def write_row(self, index: int, data: Bucket) -> None:
        """Writes the entire row i from a list.

        The tricky thing here is that, when writing to an internal list, you don't need padding, but you need it for
        writing to the disk.
        """
        # Determines where to write.
        if self._filename is None:
            # In-memory write and terminate the function.
            self._internal_data[index] = data
            return

        # On-disk write. First, convert and pad the data to the desired length.
        write_data = [elem.dump_pad(self._disk_size) if isinstance(elem, Data) else elem for elem in data]

        # Add dummy bytes if necessary.
        data_to_write = b"".join(write_data) + b"\x00" * self._disk_size * (self._bucket_size - len(data))

        # Seek on-disk write position.
        self._file.seek(self._row_offset(index))
        # Join the row data into a single byte string
        self._file.write(data_to_write)

    def encrypt(self, encryptor: Encryptor) -> None:
        """
        Given an encryptor instance, encrypt the entire storage.

        Note that each of the buckets should be made full (dummy data maybe added).
        """
        # If we do not need to load from the file.
        if self._filename is None:
            # Iterate through the buckets.
            for i, bucket in enumerate(self._internal_data):
                # Encrypt the existing data.
                encrypted_data = [encryptor.enc(plaintext=data.dump_pad(length=self._data_size)) for data in bucket]

                # If the bucket is not full, append more data to it.
                dummy_needed = self._bucket_size - len(encrypted_data)
                # Make the bucket full.
                if dummy_needed > 0:
                    encrypted_data.extend(
                        [encryptor.enc(plaintext=self._data_size_dummy) for _ in range(dummy_needed)])

                # Replace the bucket.
                self._internal_data[i] = encrypted_data

            # Terminate the function.
            return

        # Otherwise, we load from the disk and encrypt.
        for i in range(self._size * self._bucket_size):
            # Compute the location to read.
            location = i * self._disk_size

            # Seek to the correct position.
            self._file.seek(location)
            # Get the data from that location.
            data = self._file.read(self._disk_size)
            # Decide what data to encrypt.
            data = data[:self._data_size] if data.strip(b"\x00") else self._data_size_dummy

            # Move back and write the encrypted data back.
            self._file.seek(location)
            self._file.write(encryptor.enc(plaintext=data))

    def decrypt(self, encryptor: Encryptor) -> None:
        """
        Given an encryptor instance, decrypt the entire storage.

        The existence of this function is more for testing purposes or one-time use.
        """
        # If we do not need to load from the file
        if self._filename is None:
            for bucket in self._internal_data:
                # Decrypt the existing data.
                for index, data in enumerate(bucket):
                    bucket[index] = Data.load_unpad(data=encryptor.dec(ciphertext=data))

            # Terminate the function.
            return

        # Otherwise, we load from the disk and decrypt.
        for i in range(self._size * self._bucket_size):
            # Compute the location to read.
            location = i * self._disk_size
            # Seek to the correct position.
            self._file.seek(location)
            # Get the data from that location and decrypt.
            data = encryptor.dec(ciphertext=self._file.read(self._disk_size))
            # Move back and write the data.
            self._file.seek(location)
            # Load the data as object and re-dump it to proper length.
            self._file.write(Data.load_unpad(data=data).dump_pad(length=self._disk_size))

    def close(self):
        """Close the file if using file-based storage."""
        if self._file is not None:
            self._file.close()
            self._file = None
