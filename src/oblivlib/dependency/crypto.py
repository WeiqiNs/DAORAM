"""Cryptographic primitives: encryption, PRF, and PRP implementations."""

import hashlib
import os
from abc import ABC, abstractmethod
from typing import override

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class Encryptor(ABC):
    @property
    @abstractmethod
    def key(self) -> bytes:
        pass

    @abstractmethod
    def ciphertext_length(self, plaintext_length: int) -> int:
        pass

    @abstractmethod
    def enc(self, plaintext: bytes) -> bytes:
        pass

    @abstractmethod
    def dec(self, ciphertext: bytes) -> bytes:
        pass


class AesGcm(Encryptor):
    NONCE_SIZE = 12  # 96 bits, the GCM-recommended nonce size.
    TAG_SIZE = 16  # 128-bit auth tag.

    def __init__(self, key: bytes | None = None, key_byte_length: int = 16):
        if key_byte_length not in [16, 24, 32]:
            raise ValueError("The AES key length must be 16, 24, or 32 bytes.")

        self._key = os.urandom(key_byte_length) if key is None else key
        self._key_byte_length = key_byte_length
        self._aes_gcm = AESGCM(self._key)

    @property
    @override
    def key(self) -> bytes:
        return self._key

    @override
    def ciphertext_length(self, plaintext_length: int) -> int:
        # Layout is nonce + ciphertext + tag; GCM adds no padding, so ciphertext == plaintext length.
        return self.NONCE_SIZE + plaintext_length + self.TAG_SIZE

    @override
    def enc(self, plaintext: bytes) -> bytes:
        """Encrypt under a fresh random nonce; returns nonce + ciphertext + tag."""
        nonce = os.urandom(self.NONCE_SIZE)
        ciphertext_with_tag = self._aes_gcm.encrypt(nonce, plaintext, None)
        return nonce + ciphertext_with_tag

    @override
    def dec(self, ciphertext: bytes) -> bytes:
        """Decrypt a nonce + ciphertext + tag blob; raises InvalidTag if authentication fails."""
        nonce = ciphertext[: self.NONCE_SIZE]
        ciphertext_with_tag = ciphertext[self.NONCE_SIZE :]
        return self._aes_gcm.decrypt(nonce, ciphertext_with_tag, None)


class PseudoRandomFunction(ABC):
    @property
    @abstractmethod
    def key(self) -> bytes:
        pass

    @abstractmethod
    def digest(self, message: bytes) -> bytes:
        pass

    @abstractmethod
    def digest_mod_n(self, message: bytes, mod: int) -> int:
        pass


class Blake2Prf(PseudoRandomFunction):
    KEY_SIZE = 32
    DIGEST_SIZE = 64

    def __init__(self, key: bytes | None = None):
        if key is not None and len(key) != self.KEY_SIZE:
            raise ValueError(f"The PRF key length must be {self.KEY_SIZE} bytes.")

        self._key = os.urandom(self.KEY_SIZE) if key is None else key

    @property
    @override
    def key(self) -> bytes:
        return self._key

    @override
    def digest(self, message: bytes) -> bytes:
        return hashlib.blake2b(message, key=self._key, digest_size=self.DIGEST_SIZE).digest()

    @override
    def digest_mod_n(self, message: bytes, mod: int) -> int:
        return int.from_bytes(self.digest(message), "big") % mod


class PseudoRandomPermutation(ABC):
    @property
    @abstractmethod
    def key(self) -> bytes:
        pass

    @property
    @abstractmethod
    def domain_size(self) -> int:
        pass

    @abstractmethod
    def permute(self, x: int) -> int:
        pass

    @abstractmethod
    def inverse(self, y: int) -> int:
        pass


class FeistelPrp(PseudoRandomPermutation):
    """PRP over [0, domain_size) via a balanced Feistel network with cycle-walking."""

    KEY_SIZE = 16
    NUM_ROUNDS = 4  # 4 rounds suffice for a secure pseudo-random permutation.

    def __init__(self, domain_size: int, key: bytes | None = None):
        if domain_size <= 1:
            raise ValueError("Domain size must be >= 2.")
        if key is not None and len(key) < self.KEY_SIZE:
            raise ValueError(f"Key should be at least {self.KEY_SIZE} bytes.")

        self._key = os.urandom(self.KEY_SIZE) if key is None else key
        self._domain_size = domain_size

        # Compute the bit length needed to represent domain, rounded up to even for a balanced Feistel.
        raw_bits = (domain_size - 1).bit_length()
        self._bit_length = raw_bits + (raw_bits % 2)

        self._half = self._bit_length // 2
        self._half_mask = (1 << self._half) - 1
        self._full_mask = (1 << self._bit_length) - 1
        self._base_hash = hashlib.sha256(self._key)
        self._round_bytes = [r.to_bytes(1, "big") for r in range(self.NUM_ROUNDS)]

    @property
    @override
    def key(self) -> bytes:
        return self._key

    @property
    @override
    def domain_size(self) -> int:
        return self._domain_size

    def _round_function(self, round_num: int, value: int, output_bits: int) -> int:
        # round_num gives per-round domain separation.
        value_bytes = value.to_bytes((value.bit_length() + 7) // 8 or 1, "big")
        # Clone the key-seeded hash and absorb only round + value; identical to hashing key||round||value.
        h = self._base_hash.copy()
        h.update(self._round_bytes[round_num])
        h.update(value_bytes)

        num_bytes = (output_bits + 7) // 8
        return int.from_bytes(h.digest()[:num_bytes], "big") % (1 << output_bits)

    def _feistel_forward(self, x: int) -> int:
        left = x >> self._half
        right = x & self._half_mask

        for i in range(self.NUM_ROUNDS):
            f_out = self._round_function(i, right, self._half)
            left, right = right, left ^ f_out

        return ((left << self._half) | right) & self._full_mask

    def _feistel_inverse(self, y: int) -> int:
        left = y >> self._half
        right = y & self._half_mask

        for i in reversed(range(self.NUM_ROUNDS)):
            f_out = self._round_function(i, left, self._half)
            right, left = left, right ^ f_out

        return ((left << self._half) | right) & self._full_mask

    @override
    def permute(self, x: int) -> int:
        if not (0 <= x < self._domain_size):
            raise ValueError(f"Input must be in [0, {self._domain_size}).")

        # Cycle-walking: re-apply Feistel until the output lands in [0, domain_size) (handles
        # non-power-of-2 domains while staying a bijection).
        y = self._feistel_forward(x)
        while y >= self._domain_size:
            y = self._feistel_forward(y)
        return y

    @override
    def inverse(self, y: int) -> int:
        if not (0 <= y < self._domain_size):
            raise ValueError(f"Input must be in [0, {self._domain_size}).")

        x = self._feistel_inverse(y)
        while x >= self._domain_size:
            x = self._feistel_inverse(x)
        return x
