import argparse
import time
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daoram.dependency import InteractRemoteServer, crypto
from daoram.so.top_down_somap_fixed_cache import TopDownSomapFixedCache
from daoram.dependency.helper import Data

"""
Benchmark Top-Down SOMAP Insert Efficiency
------------------------------------------
This script specifically targets the INSERT performance of Top-Down SOMAP.
It connects to a running server (loaded with pre-built storage) and performs
a series of insert operations, measuring latency and bandwidth.

We focus on Insert because previous experiments used 'mix' mode (read/write).
"""

def main():
    parser = argparse.ArgumentParser(description="Benchmark Top-Down Insert Only")
    parser.add_argument("--server-ip", type=str, default="localhost", help="Server IP")
    parser.add_argument("--port", type=int, default=8888, help="Server Port")
    parser.add_argument("--num-data", type=int, default=16383, help="N")
    parser.add_argument("--value-size", type=int, default=64, help="Value size (bytes)")
    parser.add_argument("--cache-size", type=int, default=1024, help="Cache Size")
    parser.add_argument("--order", type=int, default=8, help="B+ Tree Order")
    parser.add_argument("--ops", type=int, default=100, help="Number of insert operations")
    parser.add_argument("--load-storage", type=str, default=None, help="If provided, indicates server is loaded from this file (for restore)")
    parser.add_argument("--mock-crypto", action="store_true", help="Use mock encryption")
    
    args = parser.parse_args()

    if args.mock_crypto:
        print("[Using Mock Encryption] Fast init enabled.")
        crypto.MOCK_ENCRYPTION = True

    # 1. Connect to Server
    print(f"[*] Connecting to {args.server_ip}:{args.port}...")
    client = InteractRemoteServer(ip=args.server_ip, port=args.port)
    client.init_connection()
    
    # 2. Initialize Protocol Wrapper (Client Side)
    print(f"[*] Initializing Top-Down Client (N={args.num_data}, Cache={args.cache_size}, Order={args.order})...")
    td_omap = TopDownSomapFixedCache(
        num_data=args.num_data,
        cache_size=args.cache_size,
        data_size=args.value_size,
        client=client,
        order=args.order,
        use_encryption=True, # Always use encryption (Mock handled by crypto.MOCK_ENCRYPTION)
        aes_key=b'0'*16 if args.mock_crypto else os.urandom(16)
    )

    # 3. Restore State (if server pre-loaded) or Setup (if fresh)
    # Since we usually run this after prebuild, we assume Restore.
    if args.load_storage:
         print(f"[*] Restoring client state (assuming server loaded {args.load_storage})...")
         # We need to tell the server to load? 
         # In our architecture, the server.py argument --storage loads it on startup.
         # So here we just send "load_storage" command just in case, OR just restore if already connected.
         # Actually, benchmark_remote.py does client.load_storage(filename). Let's do that.
         try:
             client.load_storage(args.load_storage)
             td_omap.restore_client_state(force_reset_caches=False) 
         except Exception as e:
             print(f"[-] Load/Restore Warning: {e}")
             # If server already loaded it via command line, this might fail or be no-op. 
             # We try restore_client_state anyway.
             td_omap.restore_client_state(force_reset_caches=False)

    # 4. Perform Inserts
    print(f"\n[*] Starting Benchmark: {args.ops} Inserts")
    
    start_time = time.time()
    
    # Reset bandwidth counter
    client.reset_bandwidth()
    
    for i in range(args.ops):
        # Insert a new key (or update existing)
        # We use keys outside initial range to force inserts, or random keys inside
        # Top-Down supports appending if key is new.
        # Let's update existing keys to ensure stable tree size, OR insert new.
        # SOMAP 'insert' is usually 'access(key, op=write, value)'.
        
        # Using a simple key pattern
        key = i % args.num_data 
        if i % 10 == 0:
            print(f"    Op {i}/{args.ops}...", end="\r")
            
        # Top-Down access interface: access(op, key, value)
        # op='write' is insert/update.
        new_value = os.urandom(args.value_size)
        td_omap.access('insert', key, new_value) # Use 'insert' explicitly for Top-Down optimization
        
    total_time = time.time() - start_time
    
    bw_sent, bw_recv = client.get_bandwidth()
    total_bw_mb = (bw_sent + bw_recv) / 1024 / 1024
    
    print(f"\n{'='*60}")
    print(f"RESULTS: Top-Down Insert Only (Order={args.order})")
    print(f"{'='*60}")
    print(f"Total Time      : {total_time:.4f} s")
    print(f"Latency         : {total_time / args.ops * 1000:.2f} ms/op")
    print(f"Total Bandwidth : {total_bw_mb:.2f} MB")
    print(f"BW per Op       : {total_bw_mb / args.ops * 1024:.2f} KB/op")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
