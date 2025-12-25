"""
BottomUpSomap Client Example
Algorithm 3: Bottom-to-Up SOMAP (Snapshot-Oblivious Map with Dynamic Security)

This client demonstrates the usage of the Bottom-to-Up SOMAP protocol, which provides
oblivious access to a key-value store with dynamic security level adjustment.
"""
import os
import random
import sys

from daoram.dependency.interact_server import InteractLocalServer
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("project_root: ", project_root)
sys.path.insert(0, project_root)

import time
from daoram.dependency import InteractRemoteServer
from daoram.so import BottomUpSomap


def main():
    # Initialize server connection
    # client = InteractRemoteServer()
    client = InteractLocalServer()
    # BottomUpSomap parameters
    num_data = pow(2, 14)
    cache_size = 200  # Cache size (window parameter c)
    data_size = 1024  # Data block size in bytes
    
    # Create BottomUpSomap instance
    somap = BottomUpSomap(
        num_data=num_data,
        cache_size=cache_size,
        data_size=data_size,
        client=client,
        name="demo_bottomup_somap"
    )
    # somap.client.init_connection()

    data_map = {}
    for i in range(num_data):
        data_map[i] = i
    somap.setup(data_map=data_map)

    for i in range(num_data):
        start_time = time.time()
        value = somap.access(key=i, op="write", value=i+1)
        value = somap.access(key=i, op="read")
        end_time = time.time() - start_time
        print(f"Updated key {i} value:{value} in {end_time:.2f} seconds")
        
    somap.adjust_cache_size(40)

    for i in range(num_data):
        value = somap.access(key=i, op="read")
        print(f"Read key {i}: {value}...")
    for i in range(num_data):
        # Get random value between 0-20
        random_key = random.randint(0, 120)
        # Write random value
        somap.access(key=random_key, op="write", value=f"Random value {random_key}".encode())
        # Read and verify
        read_value = somap.access(key=random_key, op="read")
        print(f"Loop progress: {i}, Random key: {random_key}, Read value: {read_value}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time() - start_time
    print(f"Total time: {end_time:.2f} seconds")
