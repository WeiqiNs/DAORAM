def binary_str_to_bytes(binary_str: str) -> bytes:
    return int(binary_str, 2).to_bytes((len(binary_str) + 7) // 8, byteorder="big")


def bytes_to_binary_str(binary_bytes: bytes) -> str:
    return bin(int.from_bytes(binary_bytes, byteorder="big"))[2:]
