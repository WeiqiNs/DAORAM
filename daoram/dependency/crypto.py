from typing import Optional

from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util.Padding import pad, unpad

# Global flag to enable Mock Encryption for benchmarking
MOCK_ENCRYPTION = False
# Global flag to simulate CPU cost during Mock Encryption (by encrypting dummy data)
SIMULATE_CPU_COST = False

class Aes:
    def __init__(self, aes_mode=AES.MODE_CBC, key: Optional[bytes] = None, key_byte_length: int = 16):
        """Class for performing AES encryption and decryption.

        :param aes_mode: The mode of AES encryption/decryption, default is CBC.
        :param key: The AES key to use; it will be randomly sampled unless provided here.
        :param key_byte_length: The length of AES key to use, default is 16.
        """
        # Check if the key length is supported.
        if key_byte_length not in [16, 24, 32]:
            raise ValueError("The AES key length must be 16, 24, or 32 bytes.")

        # Save generate a random key or use the provided key.
        self.__key = get_random_bytes(key_byte_length) if key is None else key

        # Save other AES configurations.
        self.__key_byte_length = key_byte_length
        self.__aes_mode = aes_mode
        
        # Pre-allocate a dummy cipher for simulation to avoid initialization overhead?
        # No, initialization is part of the cost.

    @property
    def key(self) -> bytes:
        """Get the current AES key."""
        return self.__key

    @property
    def key_byte_length(self) -> int:
        """Get the current AES key byte length."""
        return self.__key_byte_length

    def enc(self, plaintext: bytes) -> bytes:
        """Perform AES encryption on the provided plaintext."""
        # MOCK MODE: Skip actual AES for speed, but keep padding/size logic
        if MOCK_ENCRYPTION:
            iv = b'\x00' * self.__key_byte_length
            padded = pad(data_to_pad=plaintext, block_size=self.__key_byte_length, style="pkcs7")
            
            if SIMULATE_CPU_COST:
                # Simulate the CPU cost by encrypting the padded data with a real cipher
                # We use a static key/iv to avoid randomness cost if desired, or random if we want full emulation.
                # To be fair, we should include cipher create cost.
                dummy_cipher = AES.new(self.__key, self.__aes_mode, iv)
                dummy_cipher.encrypt(padded) # Discard result

            return iv + padded

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

        # MOCK MODE: Just unpad
        if MOCK_ENCRYPTION:
            if SIMULATE_CPU_COST:
                # Simulate CPU cost
                dummy_cipher = AES.new(self.__key, self.__aes_mode, iv)
                dummy_cipher.decrypt(ciphertext) # Discard result
                
            return unpad(padded_data=ciphertext, block_size=self.__key_byte_length, style="pkcs7")

        # Create a new AES instance.
        cipher = AES.new(self.__key, self.__aes_mode, iv)

        # Decrypt and un-pad the plaintext.
        return unpad(padded_data=cipher.decrypt(ciphertext), block_size=self.__key_byte_length, style="pkcs7")


class Prf:
    def __init__(self, key: Optional[bytes] = None):
        """Class for a pseudo-random function.

        :param key: The 16-byte key to use; it will be randomly sampled unless provided here.
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

import hashlib
import math
import os

def prp_prf(key: bytes, x: int, output_bits: int) -> int:
    data = x.to_bytes((x.bit_length() + 7) // 8 or 1, 'big')
    h = hashlib.sha256(key + data).digest()
    # 取足够字节
    num_bytes = (output_bits + 7) // 8
    truncated = h[:num_bytes]
    val = int.from_bytes(truncated, 'big')
    return val % (1 << output_bits)

class PRP:
    def __init__(self, key: bytes, n: int):
        if n <= 1:
            raise ValueError("n must be >= 2")
        self.__key = get_random_bytes(16) if key is None else key
        if key is not None and len(key) < 16:
            raise ValueError("Key should be at least 16 bytes")
        self.n = n
        self.b = (n - 1).bit_length()

    def _feistel_encrypt(self, x: int) -> int:
        half = self.b // 2
        mask_right = (1 << half) - 1
        left = x >> half
        right = x & mask_right

        for i in range(4):
            round_key = self.key + i.to_bytes(1, 'big')
            f_out = prp_prf(round_key, right, self.b - half)
            left, right = right, left ^ f_out

        return ((right << half) | left) & ((1 << self.b) - 1)

    def _feistel_decrypt(self, y: int) -> int:
        half = self.b // 2
        mask_right = (1 << half) - 1
        left = y >> half
        right = y & mask_right

        for i in reversed(range(4)):
            round_key = self.key + i.to_bytes(1, 'big')
            f_out = prp_prf(round_key, left, self.b - half)
            right, left = left, right ^ f_out

        return ((right << half) | left) & ((1 << self.b) - 1)

    def encrypt(self, x: int) -> int:
        if not (0 <= x < self.n):
            raise ValueError(f"Input must be in [0, {self.n})")
        y = x
        while y >= self.n:
            y = self._feistel_encrypt(y)
        return y

    def decrypt(self, y: int) -> int:
        if not (0 <= y < self.n):
            raise ValueError(f"Input must be in [0, {self.n})")
        x = y
        while x >= self.n:
            x = self._feistel_decrypt(x)
        return x

if __name__ == "__main__":
    n = pow(2, 20)  # 任意 n，比如 100 万
    key = os.urandom(32)
    prp = PRP(key, n)

    test_vals = [0, 1, 123456, n - 1]
    for x in test_vals:
        y = prp.encrypt(x)
        x2 = prp.decrypt(y)
        print(f"x={x:8d} → y={y:8d} → x2={x2:8d} {'✅' if x == x2 else '❌'}")

    seen = set()
    for x in range(n):
        y = prp.encrypt(x)
        assert 0 <= y < n
        if y in seen:
            print("Collision!")
            break
        seen.add(y)
    else:
        print(f"No collision in values.")