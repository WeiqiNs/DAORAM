import hashlib
import os
from abc import ABC, abstractmethod
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class Encryptor(ABC):
    """Abstract base class for encryption implementations."""

    @property
    @abstractmethod
    def key(self) -> bytes:
        """Get the encryption key."""
        pass

    @abstractmethod
    def ciphertext_length(self, plaintext_length: int) -> int:
        """Calculate the ciphertext length for a given plaintext length.

        :param plaintext_length: Length of the plaintext in bytes.
        :return: Length of the ciphertext in bytes.
        """
        pass

    @abstractmethod
    def enc(self, plaintext: bytes) -> bytes:
        """Encrypt the plaintext."""
        pass

    @abstractmethod
    def dec(self, ciphertext: bytes) -> bytes:
        """Decrypt the ciphertext."""
        pass


class AesGcm(Encryptor):
    """AES-GCM authenticated encryption using the cryptography package."""

    # GCM standard sizes
    NONCE_SIZE = 12  # 96 bits (recommended)
    TAG_SIZE = 16  # 128 bits (default, most secure)

    def __init__(self, key: Optional[bytes] = None, key_byte_length: int = 16):
        """Class for performing AES-GCM authenticated encryption and decryption.

        :param key: The AES key to use; it will be randomly generated if not provided.
        :param key_byte_length: The length of AES key to use (16, 24, or 32 bytes).
        """
        if key_byte_length not in [16, 24, 32]:
            raise ValueError("The AES key length must be 16, 24, or 32 bytes.")

        self.__key = os.urandom(key_byte_length) if key is None else key
        self.__key_byte_length = key_byte_length
        self.__aes_gcm = AESGCM(self.__key)

    @property
    def key(self) -> bytes:
        """Get the current AES key."""
        return self.__key

    def ciphertext_length(self, plaintext_length: int) -> int:
        """Calculate the ciphertext length for a given plaintext length.

        Ciphertext = nonce (12 bytes) + encrypted_data + tag (16 bytes)
        Note: GCM does not require padding, so encrypted_data length equals plaintext length.

        :param plaintext_length: Length of the plaintext in bytes.
        :return: Length of the ciphertext in bytes.
        """
        return self.NONCE_SIZE + plaintext_length + self.TAG_SIZE

    def enc(self, plaintext: bytes) -> bytes:
        """Perform AES-GCM encryption on the provided plaintext.

        :param plaintext: The plaintext to encrypt.
        :return: nonce + ciphertext + tag
        """
        nonce = os.urandom(self.NONCE_SIZE)
        # encrypt() returns ciphertext + tag concatenated
        ciphertext_with_tag = self.__aes_gcm.encrypt(nonce, plaintext, None)
        return nonce + ciphertext_with_tag

    def dec(self, ciphertext: bytes) -> bytes:
        """Perform AES-GCM decryption on the provided ciphertext.

        :param ciphertext: nonce + ciphertext + tag
        :return: The decrypted plaintext.
        :raises InvalidTag: If authentication fails.
        """
        nonce = ciphertext[:self.NONCE_SIZE]
        ciphertext_with_tag = ciphertext[self.NONCE_SIZE:]
        return self.__aes_gcm.decrypt(nonce, ciphertext_with_tag, None)


class PseudoRandomFunction(ABC):
    """Abstract base class for pseudo-random function implementations."""

    @property
    @abstractmethod
    def key(self) -> bytes:
        """Get the PRF key."""
        pass

    @abstractmethod
    def digest(self, message: bytes) -> bytes:
        """Compute PRF on the message."""
        pass

    @abstractmethod
    def digest_mod_n(self, message: bytes, mod: int) -> int:
        """Compute PRF on the message and return result mod n."""
        pass


class Blake2Prf(PseudoRandomFunction):
    """Pseudo-random function using BLAKE2b keyed hashing."""
    # Set BLAKE2b key size.
    KEY_SIZE = 32
    # Set output digest size.
    DIGEST_SIZE = 64

    def __init__(self, key: Optional[bytes] = None):
        """Class for a pseudo-random function using BLAKE2b.

        :param key: The 32-byte key to use; it will be randomly generated if not provided.
        """
        if key is not None and len(key) != self.KEY_SIZE:
            raise ValueError(f"The PRF key length must be {self.KEY_SIZE} bytes.")

        self.__key = os.urandom(self.KEY_SIZE) if key is None else key

    @property
    def key(self) -> bytes:
        """Get the current PRF key."""
        return self.__key

    def digest(self, message: bytes) -> bytes:
        """Compute keyed BLAKE2b hash of the message.

        :param message: The message to hash.
        :return: 64-byte digest.
        """
        return hashlib.blake2b(message, key=self.__key, digest_size=self.DIGEST_SIZE).digest()

    def digest_mod_n(self, message: bytes, mod: int) -> int:
        """Compute keyed BLAKE2b hash and return result mod n.

        :param message: The message to hash.
        :param mod: The modulus.
        :return: Hash value mod n.
        """
        return int.from_bytes(self.digest(message), "big") % mod


# Backward compatibility alias
Aes = AesGcm
