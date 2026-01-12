"""OMAP Client Demo.

Usage:
    python omap_client.py <type> [--num-data N] [--ip IP] [--port PORT]

Examples:
    python omap_client.py avl
    python omap_client.py daoram-bplus --num-data 256
"""

import argparse
import time

from daoram.dependency import InteractRemoteServer, ZMQSocket
from daoram.omap import AVLOmap, BPlusOmap, OramOstOmap
from daoram.oram import DAOram

# Available OMAP types.
OMAP_TYPES = ["avl", "bplus", "daoram-avl", "daoram-bplus"]


def create_omap(omap_type: str, num_data: int, client: InteractRemoteServer):
    """Create OMAP instance based on type."""
    if omap_type == "avl":
        return AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)
    elif omap_type == "bplus":
        return BPlusOmap(order=10, num_data=num_data, key_size=10, data_size=10, client=client)
    elif omap_type == "daoram-avl":
        ods = AVLOmap(num_data=num_data, key_size=10, data_size=10, client=client)
        oram = DAOram(num_data=num_data, data_size=10, client=client)
        return OramOstOmap(num_data=num_data, ost=ods, oram=oram)
    elif omap_type == "daoram-bplus":
        ods = BPlusOmap(order=10, num_data=num_data, key_size=10, data_size=10, client=client)
        oram = DAOram(num_data=num_data, data_size=10, client=client)
        return OramOstOmap(num_data=num_data, ost=ods, oram=oram)


def run_demo(omap_type: str, num_data: int, ip: str, port: int):
    """Run OMAP demo: insert all values, then search and verify."""
    # Connect to server.
    client = InteractRemoteServer()
    client.init_connection(client=ZMQSocket(ip=ip, port=port, is_server=False))

    # Create and initialize OMAP.
    omap = create_omap(omap_type, num_data, client)
    omap.init_server_storage()
    print(f"Initialized {omap_type} OMAP with {num_data} entries.")

    # Insert phase.
    start = time.time()
    for i in range(num_data):
        omap.insert(key=i, value=i)
    insert_time = time.time() - start

    # Search phase.
    start = time.time()
    errors = sum(1 for i in range(num_data) if omap.search(key=i) != i)
    search_time = time.time() - start

    # Summary.
    print(f"Insert: {insert_time:.2f}s ({num_data/insert_time:.0f} ops/s)")
    print(f"Search: {search_time:.2f}s ({num_data/search_time:.0f} ops/s)")
    print(f"Errors: {errors}")

    client.close_connection()


def main():
    parser = argparse.ArgumentParser(description="OMAP Client Demo")
    parser.add_argument("type", choices=OMAP_TYPES, help="OMAP type")
    parser.add_argument("--num-data", type=int, default=512, help="Number of entries (default: 512)")
    parser.add_argument("--ip", default="localhost", help="Server IP (default: localhost)")
    parser.add_argument("--port", type=int, default=5555, help="Server port (default: 5555)")
    args = parser.parse_args()

    run_demo(args.type, args.num_data, args.ip, args.port)


if __name__ == "__main__":
    main()
