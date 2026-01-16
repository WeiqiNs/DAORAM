import argparse
import time
import os
import sys
import random

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daoram.dependency import InteractRemoteServer, crypto
from daoram.so.top_down_somap_fixed_cache import TopDownSomapFixedCache
from daoram.dependency.helper import Data

def main():
    parser = argparse.ArgumentParser(description="Benchmark Search vs Insert (Top-Down)")
    parser.add_argument("--server-ip", type=str, default="localhost", help="Server IP")
    parser.add_argument("--port", type=int, default=8888, help="Server Port")
    parser.add_argument("--num-data", type=int, default=16383, help="N")
    parser.add_argument("--value-size", type=int, default=64, help="Value size (bytes)")
    parser.add_argument("--cache-size", type=int, default=1024, help="Cache Size")
    parser.add_argument("--order", type=int, default=8, help="B+ Tree Order")
    parser.add_argument("--ops", type=int, default=50, help="Number of operations for each phase")
    parser.add_argument("--load-storage", type=str, default=None, help="Server storage file")
    parser.add_argument("--mock-crypto", action="store_true", help="Use mock encryption")
    
    args = parser.parse_args()

    if args.mock_crypto:
        print("[Using Mock Encryption]")
        crypto.MOCK_ENCRYPTION = True

    # 1. Connect
    print(f"[*] Connecting to {args.server_ip}:{args.port}...")
    client = InteractRemoteServer(ip=args.server_ip, port=args.port)
    client.init_connection()
    
    # 2. Initialize
    print(f"[*] Initializing Top-Down Client (N={args.num_data}, Order={args.order})...")
    td_omap = TopDownSomapFixedCache(
        num_data=args.num_data,
        cache_size=args.cache_size,
        data_size=args.value_size,
        client=client,
        order=args.order,
        use_encryption=True, 
        aes_key=b'0'*16 if args.mock_crypto else os.urandom(16)
    )

    # 3. Restore
    if args.load_storage:
         print(f"[*] Restoring client state...")
         try:
             client.load_storage(args.load_storage)
             td_omap.restore_client_state(force_reset_caches=False) 
         except Exception as e:
             print(f"[-] Load/Restore Warning: {e}")
             td_omap.restore_client_state(force_reset_caches=False)

    # ==========================
    # PHASE 1: SEARCH
    # ==========================
    print(f"\n[*] Starting Phase 1: {args.ops} Search Operations")
    client.reset_bandwidth()
    start_time = time.time()
    
    for i in range(args.ops):
        # Search for random keys in range
        key = i % args.num_data
        td_omap.access('search', key)
        if (i+1) % 10 == 0: print(f"    Search {i+1}/{args.ops}...", end="\r")
        
    search_time = time.time() - start_time
    s_bw_sent, s_bw_recv = client.get_bandwidth()
    s_bw_total_mb = (s_bw_sent + s_bw_recv) / 1024 / 1024
    s_rounds = client.get_rounds()
    
    # ==========================
    # PHASE 2: INSERT
    # ==========================
    print(f"\n[*] Starting Phase 2: {args.ops} Insert Operations")
    client.reset_bandwidth()
    start_time = time.time()
    
    for i in range(args.ops):
        key = i % args.num_data
        new_value = os.urandom(args.value_size)
        td_omap.access('insert', key, new_value)
        if (i+1) % 10 == 0: print(f"    Insert {i+1}/{args.ops}...", end="\r")
        
    insert_time = time.time() - start_time
    i_bw_sent, i_bw_recv = client.get_bandwidth()
    i_bw_total_mb = (i_bw_sent + i_bw_recv) / 1024 / 1024
    i_rounds = client.get_rounds()

    # ==========================
    # REPORT
    # ==========================
    print(f"\n{'='*70}")
    print(f"Comparison: Search vs Insert (Top-Down, Order={args.order})")
    print(f"{'='*70}")
    print(f"{'Metric':<15} | {'Search':<20} | {'Insert':<20} | {'Ratio (S/I)':<10}")
    print(f"{'-'*70}")
    
    s_latency = search_time / args.ops * 1000
    i_latency = insert_time / args.ops * 1000
    ratio_lat = s_latency / i_latency if i_latency > 0 else 0
    print(f"{'Latency':<15} | {s_latency:.2f} ms/op        | {i_latency:.2f} ms/op        | {ratio_lat:.2f}x")
    
    s_bw_op = s_bw_total_mb / args.ops * 1024 # KB
    i_bw_op = i_bw_total_mb / args.ops * 1024 # KB
    ratio_bw = s_bw_op / i_bw_op if i_bw_op > 0 else 0
    print(f"{'Bandwidth':<15} | {s_bw_op:.2f} KB/op        | {i_bw_op:.2f} KB/op        | {ratio_bw:.2f}x")

    s_rounds_op = s_rounds / args.ops
    i_rounds_op = i_rounds / args.ops
    ratio_rounds = s_rounds_op / i_rounds_op if i_rounds_op > 0 else 0
    print(f"{'Rounds':<15} | {s_rounds_op:.2f} RTT/op       | {i_rounds_op:.2f} RTT/op       | {ratio_rounds:.2f}x")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
