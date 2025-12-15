"""
SORAM client example.
"""
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("project_root: ", project_root)
sys.path.insert(0, project_root)

import time
from daoram.dependency import InteractRemoteServer
from daoram.so.soram import Soram
import sys
import os


def main():
    # Initialize server
    client = InteractRemoteServer()
    
    # SORAM parameters
    num_data = 200
    cache_size = 20
    data_size = 1024
    
    # Create SORAM instance
    soram = Soram(
        num_data=num_data,
        cache_size=cache_size,
        data_size=data_size,
        client=client,
        name="demo_soram"
    )
    soram._client.init_connection()
    # Setup with some data
    print("Setting up SORAM...")
    data_map = {}
    for i in range(num_data):
        data_map[i] = f"Data value {i}".encode()
    start_time = time.time()
    soram.setup(data_map=data_map)
    setup_time = time.time() - start_time
    print(f"Setup completed in {setup_time:.2f} seconds")
    
    # Perform some operations
    print("\nPerforming operations...")
    for i in range(10):
        new_value = f"Updated value {i}".encode()
        start_time = time.time()
        value = soram.access(key=i, op='write', value=new_value)
        value = soram.access(key=i, op='read')
        access_time = time.time() - start_time
        print(f"Updated key {i} value:{value} (took {access_time:.4f}s)")
    
    for i in range(100):
        print("start read:")
        start_time = time.time()
        value = soram.access(key=i, op='read')
        access_time = time.time() - start_time
        print(f"Read key {i}: {value}... (took {access_time:.4f}s)")



if __name__ == "__main__":
    main()
