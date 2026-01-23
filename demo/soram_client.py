"""SORAM Client Demo.

Usage:
    python soram_client.py [--num-data N] [--cache-size N]

Examples:
    python soram_client.py
    python soram_client.py --num-data 512 --cache-size 100
"""

import argparse
import time

from daoram.dependency import InteractLocalServer
from daoram.soram import Soram


def run_demo(num_data: int, cache_size: int, data_size: int):
    """Run SORAM demo: setup, write, read, and verify."""
    # Create local client (SORAM uses local server).
    client = InteractLocalServer()

    # Create and setup SORAM.
    soram = Soram(
        num_data=num_data,
        cache_size=cache_size,
        data_size=data_size,
        client=client,
        name="demo_soram"
    )

    # Initialize with data.
    data_map = {i: f"value_{i}".encode() for i in range(num_data)}
    soram.setup(data_map=data_map)
    print(f"Initialized SORAM with {num_data} entries, cache_size={cache_size}.")

    # Write phase.
    start = time.time()
    for i in range(num_data):
        soram.access(key=i, op="write", value=f"updated_{i}".encode())
    write_time = time.time() - start

    # Read phase.
    start = time.time()
    errors = 0
    for i in range(num_data):
        value = soram.access(key=i, op="read")
        if value != f"updated_{i}".encode():
            errors += 1
    read_time = time.time() - start

    # Summary.
    print(f"Write: {write_time:.2f}s ({num_data / write_time:.0f} ops/s)")
    print(f"Read:  {read_time:.2f}s ({num_data / read_time:.0f} ops/s)")
    print(f"Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(description="SORAM Client Demo")
    parser.add_argument("--num-data", type=int, default=1024, help="Number of entries (default: 1024)")
    parser.add_argument("--cache-size", type=int, default=200, help="Cache size (default: 200)")
    parser.add_argument("--data-size", type=int, default=64, help="Data size in bytes (default: 64)")
    args = parser.parse_args()

    run_demo(args.num_data, args.cache_size, args.data_size)


if __name__ == "__main__":
    main()
