"""
SORAM client example.
"""
import os
import random
import sys

from daoram.dependency.interact_server import InteractLocalServer

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("project_root: ", project_root)
sys.path.insert(0, project_root)

import time
from daoram.so.soram import Soram


def main():
    # Initialize server
    # client = InteractRemoteServer()
    client = InteractLocalServer()

    # SORAM parameters
    num_data = pow(2, 10)
    cache_size = 200
    data_size = 1024

    # Create SORAM instance
    soram = Soram(
        num_data=num_data,
        cache_size=cache_size,
        data_size=data_size,
        client=client,
        name="demo_soram"
    )
    # client.init_connection()

    data_map = {}
    for i in range(num_data):
        data_map[i] = f"Data value {i}".encode()

    soram.setup(data_map=data_map)

    print("Performing operations...")
    for i in range(num_data):
        new_value = f"Updated value {i}".encode()
        value = soram.access(key=i, op='write', value=new_value)
        value = soram.access(key=i, op='read')
        print(f"Updated key {i} value:{value}")

    for i in range(num_data):
        value = soram.access(key=i, op='read')
        print(f"Read key {i}: {value}...")

    for i in range(num_data):
        # Get random value between 0-20
        random_key = random.randint(0, 120)
        # Write random value
        soram.access(key=random_key, op="write", value=f"Random value {random_key}".encode())
        # Read and verify
        read_value = soram.access(key=random_key, op="read")
        print(f"Loop progress: {i}, Random key: {random_key}, Read value: {read_value}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
