"""ORAM Client Demo.

Usage:
    python oram_client.py <type> [--num-data N] [--ip IP] [--port PORT]

Examples:
    python oram_client.py path
    python oram_client.py da --num-data 512
"""

import argparse
import time

from daoram.dependency import InteractRemoteServer, ZMQSocket
from daoram.oram import DAOram, FreecursiveOram, PathOram, RecursivePathOram

# Available ORAM types.
ORAM_TYPES = {
    "path": PathOram,
    "recursive": RecursivePathOram,
    "freecursive": FreecursiveOram,
    "da": DAOram,
}


def run_demo(oram_type: str, num_data: int, ip: str, port: int):
    """Run ORAM demo: write all values, then read and verify."""
    # Connect to server.
    client = InteractRemoteServer()
    client.init_connection(client=ZMQSocket(ip=ip, port=port, is_server=False))

    # Create and initialize ORAM.
    oram = ORAM_TYPES[oram_type](num_data=num_data, data_size=10, client=client)
    oram.init_server_storage()
    print(f"Initialized {oram_type} ORAM with {num_data} entries.")

    # Write phase.
    start = time.time()
    for i in range(num_data):
        oram.operate_on_key(key=i, value=i)
    write_time = time.time() - start

    # Read phase.
    start = time.time()
    errors = sum(1 for i in range(num_data) if oram.operate_on_key(key=i) != i)
    read_time = time.time() - start

    # Summary.
    print(f"Write: {write_time:.2f}s ({num_data/write_time:.0f} ops/s)")
    print(f"Read:  {read_time:.2f}s ({num_data/read_time:.0f} ops/s)")
    print(f"Errors: {errors}")

    client.close_connection()


def main():
    parser = argparse.ArgumentParser(description="ORAM Client Demo")
    parser.add_argument("type", choices=ORAM_TYPES.keys(), help="ORAM type")
    parser.add_argument("--num-data", type=int, default=1024, help="Number of entries (default: 1024)")
    parser.add_argument("--ip", default="localhost", help="Server IP (default: localhost)")
    parser.add_argument("--port", type=int, default=5555, help="Server port (default: 5555)")
    args = parser.parse_args()

    run_demo(args.type, args.num_data, args.ip, args.port)


if __name__ == "__main__":
    main()
