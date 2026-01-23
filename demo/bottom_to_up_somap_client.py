"""Bottom-Up SOMAP Client Demo.

Usage:
    python bottom_to_up_somap_client.py [--num-data N] [--cache-size N]

Examples:
    python bottom_to_up_somap_client.py
    python bottom_to_up_somap_client.py --num-data 512 --cache-size 100
"""

import argparse
import time

from daoram.dependency import InteractLocalServer
from daoram.soram import BottomUpSomap


def run_demo(num_data: int, cache_size: int, data_size: int):
    """Run Bottom-Up SOMAP demo: setup, write, read, and verify."""
    # Create local client (SOMAP uses local server).
    client = InteractLocalServer()

    # Create and setup SOMAP.
    somap = BottomUpSomap(
        num_data=num_data,
        cache_size=cache_size,
        data_size=data_size,
        client=client,
        name="demo_somap"
    )

    # Initialize with data.
    data_map = {i: i for i in range(num_data)}
    somap.setup(data_map=data_map)
    print(f"Initialized BottomUpSomap with {num_data} entries, cache_size={cache_size}.")

    # Write phase.
    start = time.time()
    for i in range(num_data):
        somap.access(key=i, op="write", value=i + 1)
    write_time = time.time() - start

    # Read phase.
    start = time.time()
    errors = 0
    for i in range(num_data):
        value = somap.access(key=i, op="read")
        if value != i + 1:
            errors += 1
    read_time = time.time() - start

    # Summary.
    print(f"Write: {write_time:.2f}s ({num_data / write_time:.0f} ops/s)")
    print(f"Read:  {read_time:.2f}s ({num_data / read_time:.0f} ops/s)")
    print(f"Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(description="Bottom-Up SOMAP Client Demo")
    parser.add_argument("--num-data", type=int, default=1024, help="Number of entries (default: 1024)")
    parser.add_argument("--cache-size", type=int, default=200, help="Cache size (default: 200)")
    parser.add_argument("--data-size", type=int, default=64, help="Data size in bytes (default: 64)")
    args = parser.parse_args()

    run_demo(args.num_data, args.cache_size, args.data_size)


if __name__ == "__main__":
    main()
