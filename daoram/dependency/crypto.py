from typing import Optional

from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.Padding import pad, unpad


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
        # If the data is not empty and its trailing byte is zero.
        if data and data[-1] == 0:
            raise ValueError("Padding is broken because of trailing null byte.")

        return data + b"\x00" * (length - len(data))


def unpad_pickle(data: bytes) -> bytes:
    """Remove trailing zeros from padded data."""
    return data.rstrip(b"\x00")


class Aes:
    def __init__(self, aes_mode=AES.MODE_CBC, key: Optional[bytes] = None, key_byte_length: int = 16):
        """Class for performing AES encryption and decryption.

        :param aes_mode: the mode of AES encryption/decryption, default is CBC.
        :param key: the AES key to use; it will be randomly sampled unless provided here.
        :param key_byte_length: the length of AES key to use, default is 16.
        """
        # Check if the key length is supported.
        if key_byte_length not in [16, 24, 32]:
            raise ValueError("The AES key length must be 16, 24, or 32 bytes.")

        # Save generate a random key or use the provided key.
        self.__key = get_random_bytes(key_byte_length) if key is None else key

        # Save other AES configurations.
        self.__key_byte_length = key_byte_length
        self.__aes_mode = aes_mode

    @property
    def key(self) -> bytes:
        """Get the current AES key."""
        return self.__key

    def enc(self, plaintext: bytes) -> bytes:
        """Perform AES encryption on the provided plaintext."""
        # Sample a new IV to use.
        iv = get_random_bytes(self.__key_byte_length)

        # Create a new AES instance.
        cipher = AES.new(self.__key, self.__aes_mode, iv)

        # Pad the plaintext and encrypt.
        ciphertext = cipher.encrypt(pad(data_to_pad=plaintext, block_size=self.__key_byte_length, style="pkcs7"))

        # Prepend the iv and return the ciphertext.
        return iv + ciphertext

    def dec(self, ciphertext: bytes) -> bytes:
        """Perform AES decryption on the provided ciphertext."""
        # Separate the IV and the ciphertext.
        iv = ciphertext[:self.__key_byte_length]
        ciphertext = ciphertext[self.__key_byte_length:]

        # Create a new AES instance.
        cipher = AES.new(self.__key, self.__aes_mode, iv)

        # Decrypt and un-pad the plaintext.
        return unpad(padded_data=cipher.decrypt(ciphertext), block_size=self.__key_byte_length, style="pkcs7")


class Prf:
    def __init__(self, key: Optional[bytes] = None):
        """Class for a pseudo-random function.

        :param key: the 16 byte key to use; it will be randomly sampled unless provided here.
        """
        # Check if the key length is supported.
        if key is not None and len(key) != 16:
            raise ValueError("The PRF key length must be 16 bytes.")

        # Generate a random key or use the provided key.
        self.__key = get_random_bytes(16) if key is None else key

    @property
    def key(self) -> bytes:
        """Get the current PRF key."""
        return self.__key

    def digest(self, message: bytes) -> bytes:
        """Run PRF on the provided message."""
        # Create a new AES instance; fix the IV to make the algorithm deterministic.
        cipher = AES.new(self.__key, AES.MODE_CBC, iv=b"\x00" * 16)

        # Pad the plaintext and encrypt.
        return cipher.encrypt(pad(data_to_pad=message, block_size=16, style="pkcs7"))

    def digest_mod_n(self, message: bytes, mod: int) -> int:
        """Run PRF on the provided message and compute mod n."""
        return int.from_bytes(self.digest(message=message), "big") % mod
