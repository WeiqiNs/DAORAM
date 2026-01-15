"""Calculate ORAM block sizes for different parameters."""
import os
import pickle
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from daoram.dependency.helper import Data


def calc_block_size(key_size: int, value_size: int, num_data: int = 16383) -> dict:
    """
    Calculate the actual block size used in ORAM.
    
    Returns dict with:
    - somap_block_size: Size used in SOMAP (key=bytes, leaf=int, value=bytes)
    - oram_block_size: Size used in base ORAM (key=int, leaf=int, value=bytes)
    """
    # SOMAP style (key is bytes, value is bytes)
    somap_data = Data(key=b'k' * key_size, leaf=0, value=b'v' * value_size)
    somap_block_size = len(somap_data.dump())
    
    # Base ORAM style (key is int, value is bytes) - from tree_base_oram.py
    oram_data = Data(key=num_data - 1, leaf=num_data - 1, value=os.urandom(value_size))
    oram_block_size = len(oram_data.dump())
    
    return {
        "somap_block_size": somap_block_size,
        "oram_block_size": oram_block_size,
    }


def main():
    print("=== ORAM Block Size Calculator ===\n")
    
    # Default parameters from benchmark
    key_size = 16
    num_data = 16383  # 2^14 - 1
    
    print(f"Parameters: key_size={key_size}, num_data={num_data}\n")
    
    print("Block size breakdown:")
    print("-" * 60)
    print(f"{'value_size':<12} {'block_size':<12} {'overhead':<12} notes")
    print("-" * 60)
    
    for value_size in [16, 32, 64, 128, 256, 512, 1024]:
        result = calc_block_size(key_size, value_size, num_data)
        block_size = result["somap_block_size"]
        overhead = block_size - value_size
        print(f"{value_size:<12} {block_size:<12} {overhead:<12} (key+leaf+pickle header)")
    
    print("\n" + "=" * 60)
    print("\nNote: Block size = pickle.dumps((key, leaf, value))")
    print("  - key: bytes of length key_size")
    print("  - leaf: int (leaf label)")
    print("  - value: bytes of length value_size")
    print("  - overhead: ~30-40 bytes for pickle serialization + key")


if __name__ == "__main__":
    main()
