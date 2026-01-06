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

        self._key = os.urandom(key_byte_length) if key is None else key
        self._key_byte_length = key_byte_length
        self._aes_gcm = AESGCM(self._key)

    @property
    def key(self) -> bytes:
        """Get the current AES key."""
        return self._key

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
        ciphertext_with_tag = self._aes_gcm.encrypt(nonce, plaintext, None)
        return nonce + ciphertext_with_tag

    def dec(self, ciphertext: bytes) -> bytes:
        """Perform AES-GCM decryption on the provided ciphertext.

        :param ciphertext: nonce + ciphertext + tag
        :return: The decrypted plaintext.
        :raises InvalidTag: If authentication fails.
        """
        nonce = ciphertext[:self.NONCE_SIZE]
        ciphertext_with_tag = ciphertext[self.NONCE_SIZE:]
        return self._aes_gcm.decrypt(nonce, ciphertext_with_tag, None)


class PseudoRandomFunction(ABC):
    """Abstract base class for pseudo-random function implementations."""

    @property
    @abstractmethod
    def key(self) -> bytes:
        """Get the PRF key."""
        pass

    @abstractmethod
    def digest(self, message: bytes) -> bytes:
        """Compute PRF on the message.

        :param message: The message to hash.
        :return: The PRF output digest.
        """
        pass

    @abstractmethod
    def digest_mod_n(self, message: bytes, mod: int) -> int:
        """Compute PRF on the message and return result mod n.

        :param message: The message to hash.
        :param mod: The modulus.
        :return: PRF output mod n.
        """
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

        self._key = os.urandom(self.KEY_SIZE) if key is None else key

    @property
    def key(self) -> bytes:
        """Get the current PRF key."""
        return self._key

    def digest(self, message: bytes) -> bytes:
        """Compute keyed BLAKE2b hash of the message.

        :param message: The message to hash.
        :return: 64-byte digest.
        """
        return hashlib.blake2b(message, key=self._key, digest_size=self.DIGEST_SIZE).digest()

    def digest_mod_n(self, message: bytes, mod: int) -> int:
        """Compute keyed BLAKE2b hash and return result mod n.

        :param message: The message to hash.
        :param mod: The modulus.
        :return: Hash value mod n.
        """
        return int.from_bytes(self.digest(message), "big") % mod


class PseudoRandomPermutation(ABC):
    """Abstract base class for pseudo-random permutation implementations."""

    @property
    @abstractmethod
    def key(self) -> bytes:
        """Get the PRP key."""
        pass

    @property
    @abstractmethod
    def domain_size(self) -> int:
        """Get the domain size n; permutation maps [0, n) -> [0, n)."""
        pass

    @abstractmethod
    def permute(self, x: int) -> int:
        """Apply the permutation to an integer in [0, domain_size).

        :param x: The input integer.
        :return: The permuted integer.
        """
        pass

    @abstractmethod
    def inverse(self, y: int) -> int:
        """Apply the inverse permutation to an integer in [0, domain_size).

        :param y: The permuted integer.
        :return: The original integer.
        """
        pass


class FeistelPrp(PseudoRandomPermutation):
    """Pseudo-random permutation using a Feistel network with cycle-walking."""

    # Minimum key size in bytes.
    KEY_SIZE = 16
    # Number of Feistel rounds (4 rounds for security).
    NUM_ROUNDS = 4

    def __init__(self, domain_size: int, key: Optional[bytes] = None):
        """Initialize a pseudo-random permutation using a balanced Feistel network.

        Uses a 4-round Feistel cipher with SHA-256 as the round function.
        Cycle-walking ensures outputs stay within [0, domain_size) for
        non-power-of-2 domain sizes.

        :param domain_size: The size of the permutation domain [0, domain_size).
        :param key: The key to use; it will be randomly generated if not provided.
        """
        if domain_size <= 1:
            raise ValueError("Domain size must be >= 2.")
        if key is not None and len(key) < self.KEY_SIZE:
            raise ValueError(f"Key should be at least {self.KEY_SIZE} bytes.")

        self._key = os.urandom(self.KEY_SIZE) if key is None else key
        self._domain_size = domain_size

        # Compute the bit length needed to represent domain, rounded up to even for a balanced Feistel.
        raw_bits = (domain_size - 1).bit_length()
        self._bit_length = raw_bits + (raw_bits % 2)

    @property
    def key(self) -> bytes:
        """Get the current PRP key."""
        return self._key

    @property
    def domain_size(self) -> int:
        """Get the domain size."""
        return self._domain_size

    def _round_function(self, round_num: int, value: int, output_bits: int) -> int:
        """Compute the Feistel round function using SHA-256.

        :param round_num: The round number (used for domain separation).
        :param value: The input value to the round function.
        :param output_bits: Number of output bits needed.
        :return: The round function output, truncated to output_bits.
        """
        # Concatenate key, round number, and value for domain separation.
        value_bytes = value.to_bytes((value.bit_length() + 7) // 8 or 1, "big")
        data = self._key + round_num.to_bytes(1, "big") + value_bytes
        h = hashlib.sha256(data).digest()

        # Truncate hash to required number of bits.
        num_bytes = (output_bits + 7) // 8
        return int.from_bytes(h[:num_bytes], "big") % (1 << output_bits)

    def _feistel_forward(self, x: int) -> int:
        """Apply the forward Feistel network transformation.

        :param x: The input integer.
        :return: The transformed integer.
        """
        half = self._bit_length // 2
        mask = (1 << half) - 1

        # Split input into left (high bits) and right (low bits) halves.
        left = x >> half
        right = x & mask

        # Apply Feistel rounds: swap and XOR with round function output.
        for i in range(self.NUM_ROUNDS):
            f_out = self._round_function(i, right, half)
            left, right = right, left ^ f_out

        # Recombine halves into output.
        return ((left << half) | right) & ((1 << self._bit_length) - 1)

    def _feistel_inverse(self, y: int) -> int:
        """Apply the inverse Feistel network transformation.

        :param y: The transformed integer.
        :return: The original integer.
        """
        half = self._bit_length // 2
        mask = (1 << half) - 1

        # Split input into left (high bits) and right (low bits) halves.
        left = y >> half
        right = y & mask

        # Apply inverse Feistel rounds in reverse order.
        for i in reversed(range(self.NUM_ROUNDS)):
            f_out = self._round_function(i, left, half)
            right, left = left, right ^ f_out

        # Recombine halves into output.
        return ((left << half) | right) & ((1 << self._bit_length) - 1)

    def permute(self, x: int) -> int:
        """Apply the permutation to an integer.

        :param x: Integer in [0, domain_size).
        :return: Permuted integer in [0, domain_size).
        :raises ValueError: If x is not in [0, domain_size).
        """
        if not (0 <= x < self._domain_size):
            raise ValueError(f"Input must be in [0, {self._domain_size}).")

        # Cycle-walking: repeatedly apply Feistel until output is in domain.
        # This handles non-power-of-2 domain sizes while preserving bijectivity.
        y = self._feistel_forward(x)
        while y >= self._domain_size:
            y = self._feistel_forward(y)
        return y

    def inverse(self, y: int) -> int:
        """Apply the inverse permutation to an integer.

        :param y: Integer in [0, domain_size).
        :return: Original integer in [0, domain_size).
        :raises ValueError: If y is not in [0, domain_size).
        """
        if not (0 <= y < self._domain_size):
            raise ValueError(f"Input must be in [0, {self._domain_size}).")

        # Cycle-walking: repeatedly apply inverse Feistel until output is in domain.
        x = self._feistel_inverse(y)
        while x >= self._domain_size:
            x = self._feistel_inverse(x)
        return x
