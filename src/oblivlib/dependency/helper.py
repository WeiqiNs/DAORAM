"""Helper utilities: data structures, padding, hashing, and math functions."""

from __future__ import annotations

import math
import pickle
from dataclasses import astuple, dataclass
from typing import Any, Self

from oblivlib.dependency.crypto import Encryptor, PseudoRandomFunction


@dataclass
class Data:
    """A block stored in the tree; an all-None instance (key is None) is a dummy block."""

    key: Any = None
    leaf: int | None = None
    value: Any = None

    @classmethod
    def load_unpad(cls, data: bytes) -> Self:
        return cls(*pickle.loads(data))

    def dump(self) -> bytes:
        return pickle.dumps(astuple(self))

    def dump_pad(self, length: int) -> bytes:
        return Helper.pad_pickle(data=self.dump(), length=length)

    def is_real(self) -> bool:
        return self.key is not None

    def is_dummy(self) -> bool:
        return self.key is None

    def require_leaf(self) -> int:
        if self.leaf is None:
            raise ValueError("Data block has no leaf assigned.")
        return self.leaf


class Helper:
    """Namespace for stateless helper functions, grouped in a class for neater imports."""

    @staticmethod
    def encrypt_bucket(encryptor: Encryptor, blocks: list[bytes], dummy: bytes, bucket_size: int) -> bytes:
        """Pad ``blocks`` to ``bucket_size`` with ``dummy`` and seal the whole bucket as one ciphertext
        (one nonce+tag per bucket, not per block); ``decrypt_bucket`` splits it back."""
        if len(blocks) < bucket_size:
            blocks = blocks + [dummy] * (bucket_size - len(blocks))
        return encryptor.enc(plaintext=b"".join(blocks))

    @staticmethod
    def decrypt_bucket(encryptor: Encryptor, blob: bytes, block_size: int) -> list[bytes]:
        """Decrypt a bucket ciphertext into its fixed-width blocks (real and dummy; caller drops dummies)."""
        plaintext = encryptor.dec(ciphertext=blob)
        return [plaintext[i : i + block_size] for i in range(0, len(plaintext), block_size)]

    @staticmethod
    def pad_pickle(data: bytes, length: int) -> bytes:
        # No length header: pickle.loads stops at the STOP opcode and ignores the trailing zeros.
        if len(data) > length:
            raise ValueError(f"Desired length {length} is shorter than the {len(data)}-byte data.")
        return data + b"\x00" * (length - len(data))

    @staticmethod
    def binary_str_to_bytes(binary_str: str) -> bytes:
        return int(binary_str, 2).to_bytes((len(binary_str) + 7) // 8, byteorder="big")

    @staticmethod
    def bytes_to_binary_str(binary_bytes: bytes) -> str:
        return bin(int.from_bytes(binary_bytes, byteorder="big"))[2:]

    @staticmethod
    def hash_data_to_leaf(prf: PseudoRandomFunction, map_size: int, data: str | int | bytes) -> int:
        if type(data) is int:
            byte_data = data.to_bytes(16, byteorder="big")
        elif type(data) is str:
            byte_data = data.encode("utf-8")
        elif type(data) is bytes:
            byte_data = data
        else:
            raise TypeError("Data must be a string, an integer, or bytes.")

        return prf.digest_mod_n(message=byte_data, mod=map_size)

    @staticmethod
    def hash_data_to_map(prf: PseudoRandomFunction, map_size: int, data: list[tuple[str | int | bytes, Any]]) -> dict:
        """PRF-hash each key into one of ``map_size`` buckets; returns {bucket: [pairs]} (some empty)."""
        data_map = {i: [] for i in range(map_size)}

        for pair in data:
            data_key = Helper.hash_data_to_leaf(prf=prf, data=pair[0], map_size=map_size)
            data_map[data_key].append(pair)

        return data_map

    @staticmethod
    def lambert_w(x: float, tol: float = 1e-10, max_iter: int = 100) -> float:
        """Lambert W (the w satisfying w * e^w = x), by Newton's method."""
        if x == 0:
            return 0.0
        if x < -math.exp(-1):
            raise ValueError("lambert_w(x) is not defined for x < -1/e.")

        w = 0 if x <= 1 else math.log(x) - math.log(math.log(x))

        # Newton's method on f(w) = w * e^w - x (Halley-corrected update).
        for _ in range(max_iter):
            ew = math.exp(w)
            wew = w * ew
            w_next = w - (wew - x) / (ew * (w + 1) - (w + 2) * (wew - x) / (2 * w + 2))

            if abs(w_next - w) < tol:
                return w_next

            w = w_next

        raise RuntimeError("Lambert W function did not converge")
