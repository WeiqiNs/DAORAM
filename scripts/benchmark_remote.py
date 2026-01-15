"""
Benchmark script for real client-server deployment.

Usage:
  Server Machine:
    python demo/server.py
    
  Client Machine:
    python scripts/benchmark_remote.py --server-ip <SERVER_IP> --port 10000 --protocol all
"""

import os
import sys
import random
import time
import argparse
import pickle
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from daoram.dependency import crypto
from daoram.dependency.interact_server import InteractRemoteServer, InteractServer
from daoram.so.bottom_to_up_somap_fixed_cache import BottomUpSomapFixedCache
from daoram.so.top_down_somap_fixed_cache import TopDownSomapFixedCache
from daoram.omap.bplus_ods_omap import BPlusOdsOmap


class RoundCounter:
    """Wraps client to count rounds and measure bandwidth."""
    def __init__(self, client: InteractServer):
        self.client = client
        self.rounds = 0
        self.bytes_sent = 0
        self.bytes_recv = 0

    def wrap(self):
        self._orig_batch = getattr(self.client, "batch_query", None)
        self._orig_read = getattr(self.client, "read_query", None)
        self._orig_read_mul = getattr(self.client, "read_mul_query", None)
        self._orig_write = getattr(self.client, "write_query", None)
        self._orig_write_mul = getattr(self.client, "write_mul_query", None)

        def _size(obj: Any) -> int:
            try:
                return len(pickle.dumps(obj))
            except Exception:
                return 0

        # batch_query counts as 1 round (real batch over network)
        if self._orig_batch:
            def _batch(ops):
                self.rounds += 1  # batch is 1 RTT
                self.bytes_sent += _size(ops)
                res = self._orig_batch(ops)
                self.bytes_recv += _size(res)
                return res
            self.client.batch_query = _batch

        if self._orig_read:
            def _read(label, leaf):
                if not getattr(self.client, "skip_round_counting", False):
                    self.rounds += 1
                self.bytes_sent += _size({"label": label, "leaf": leaf})
                res = self._orig_read(label, leaf)
                self.bytes_recv += _size(res)
                return res
            self.client.read_query = _read

        if self._orig_read_mul:
            def _read_mul(label, leaf):
                if not getattr(self.client, "skip_round_counting", False):
                    self.rounds += 1
                self.bytes_sent += _size({"label": label, "leaf": leaf})
                res = self._orig_read_mul(label, leaf)
                self.bytes_recv += _size(res)
                return res
            self.client.read_mul_query = _read_mul

        if self._orig_write:
            def _write(label, leaf, data):
                if not getattr(self.client, "skip_round_counting", False):
                    self.rounds += 1
                self.bytes_sent += _size({"label": label, "leaf": leaf, "data": data})
                res = self._orig_write(label, leaf, data)
                self.bytes_recv += _size(res)
                return res
            self.client.write_query = _write

        if self._orig_write_mul:
            def _write_mul(label, leaf, data):
                if not getattr(self.client, "skip_round_counting", False):
                    self.rounds += 1
                self.bytes_sent += _size({"label": label, "leaf": leaf, "data": data})
                res = self._orig_write_mul(label, leaf, data)
                self.bytes_recv += _size(res)
                return res
            self.client.write_mul_query = _write_mul

    def reset(self):
        self.rounds = 0
        self.bytes_sent = 0
        self.bytes_recv = 0


def zipf_keys(n, size, alpha=1.0):
    weights = [1.0 / (i + 1) ** alpha for i in range(n)]
    total = sum(weights)
    probs = [w / total for w in weights]
    return random.choices(range(n), weights=probs, k=size)


def build_data(num_data: int, value_size: int = 16) -> Dict[int, Any]:
    """Build initial data with specified value size.
    
    :param num_data: Number of data entries
    :param value_size: Size of each value in bytes
    :return: Dictionary mapping keys to values
    """
    # Create value as bytes of specified size
    return {i: bytes(value_size) for i in range(num_data)}


def make_value(key: int, value_size: int = 16) -> bytes:
    """Create a value of specified size for a given key."""
    return bytes(value_size)


def run_bottom_up(server_ip: str, port: int, num_data: int, cache_size: int, 
                  data_size: int, keys: List[int], ops: List[str], order: int = 4,
                  key_size: int = 16, value_size: int = 16, mode: str = "mix"):
    """Run Bottom-Up protocol benchmark against remote server."""
    client = InteractRemoteServer(ip=server_ip, port=port)
    client.init_connection()
    
    counter = RoundCounter(client)
    counter.wrap()
    
    proto = BottomUpSomapFixedCache(
        num_data=num_data, cache_size=cache_size, data_size=data_size,
        client=client, use_encryption=True, aes_key=b'0'*16, order=order,
        num_key_bytes=key_size
    )
    
    print("  Uploading initial data to server...")
    setup_start = time.time()
    proto.setup(build_data(num_data, value_size))
    setup_time = time.time() - setup_start
    print(f"  Setup complete in {setup_time:.2f}s")
    
    proto.reset_peak_client_size()
    counter.reset()  # Reset after setup
    
    print("  Running operations...")
    start = time.time()
    success = 0
    
    if mode == "insert_only":
        # Insert new keys that don't exist in initial data
        base_new_key = num_data + 100000
        for i in range(len(keys)):
            if i % 10 == 0:
                sys.stdout.write(f"\r  Progress: {i}/{len(keys)}")
                sys.stdout.flush()
            try:
                new_key = base_new_key + i
                proto.access(new_key, 'write', make_value(new_key, value_size))
                success += 1
            except Exception as e:
                print(f"\n  Error on op {i}: {e}")
    else:
        for i, (key, op) in enumerate(zip(keys, ops)):
            if i % 10 == 0:
                sys.stdout.write(f"\r  Progress: {i}/{len(keys)}")
                sys.stdout.flush()
            try:
                if op == 'read':
                    proto.access(key, 'read')
                else:
                    proto.access(key, 'write', make_value(key, value_size))
                success += 1
            except Exception as e:
                print(f"\n  Error on op {i}: {e}")
            
    print(f"\r  Progress: {len(keys)}/{len(keys)} - Done!")
    
    elapsed = time.time() - start
    client.close_connection()
    
    return {
        'rounds': counter.rounds,
        'bytes_sent': counter.bytes_sent,
        'bytes_recv': counter.bytes_recv,
        'elapsed_sec': elapsed,
        'success': success,
        'max_client_size': proto._peak_client_size
    }


def run_top_down(server_ip: str, port: int, num_data: int, cache_size: int,
                 data_size: int, keys: List[int], ops: List[str], order: int = 4,
                 key_size: int = 16, value_size: int = 16, mode: str = "mix"):
    """Run Top-Down protocol benchmark against remote server."""
    client = InteractRemoteServer(ip=server_ip, port=port)
    client.init_connection()
    
    counter = RoundCounter(client)
    counter.wrap()
    
    proto = TopDownSomapFixedCache(
        num_data=num_data, cache_size=cache_size, data_size=data_size,
        client=client, use_encryption=True, aes_key=b'0'*16, order=order,
        num_key_bytes=key_size, key_size=key_size
    )
    
    print("  Uploading initial data to server...")
    setup_start = time.time()
    proto.setup([(str(k), make_value(k, value_size)) for k in range(num_data)])
    setup_time = time.time() - setup_start
    print(f"  Setup complete in {setup_time:.2f}s")
    
    proto.reset_peak_client_size()
    counter.reset()
    
    print("  Running operations...")
    start = time.time()
    success = 0
    
    if mode == "insert_only":
        # Insert new keys
        base_new_key = num_data + 100000
        for i in range(len(keys)):
            if i % 10 == 0:
                sys.stdout.write(f"\r  Progress: {i}/{len(keys)}")
                sys.stdout.flush()
            try:
                new_key = base_new_key + i
                proto.access('insert', str(new_key), value=make_value(new_key, value_size))
                success += 1
            except Exception as e:
                print(f"\n  Error on op {i}: {e}")
    else:
        for i, (key, op) in enumerate(zip(keys, ops)):
            if i % 10 == 0:
                sys.stdout.write(f"\r  Progress: {i}/{len(keys)}")
                sys.stdout.flush()
            try:
                gk = str(key)
                proto.access('search', gk)
                success += 1
            except Exception as e:
                print(f"\n  Error on op {i}: {e}")
            
    print(f"\r  Progress: {len(keys)}/{len(keys)} - Done!")
    
    elapsed = time.time() - start
    client.close_connection()
    
    return {
        'rounds': counter.rounds,
        'bytes_sent': counter.bytes_sent,
        'bytes_recv': counter.bytes_recv,
        'elapsed_sec': elapsed,
        'success': success,
        'max_client_size': proto._peak_client_size
    }


def run_baseline(server_ip: str, port: int, num_data: int, data_size: int,
                 keys: List[int], ops: List[str], order: int = 4,
                 key_size: int = 16, value_size: int = 16, mode: str = "mix"):
    """Run Baseline BPlus OMAP benchmark against remote server."""
    client = InteractRemoteServer(ip=server_ip, port=port)
    client.init_connection()
    
    counter = RoundCounter(client)
    counter.wrap()
    
    omap = BPlusOdsOmap(
        order=order, num_data=num_data, key_size=key_size, data_size=data_size,
        client=client, name="baseline", use_encryption=True, aes_key=b'0'*16,
        num_key_bytes=key_size
    )
    
    print("  Uploading initial data to server...")
    setup_start = time.time()
    storage = omap._init_ods_storage([(k, make_value(k, value_size)) for k in range(num_data)])
    client.init({omap._name: storage})
    setup_time = time.time() - setup_start
    print(f"  Setup complete in {setup_time:.2f}s")
    
    omap.reset_peak_client_size()
    counter.reset()
    
    print("  Running operations...")
    start = time.time()
    success = 0
    
    if mode == "insert_only":
        # Insert new keys
        base_new_key = num_data + 100000
        for i in range(len(keys)):
            if i % 10 == 0:
                sys.stdout.write(f"\r  Progress: {i}/{len(keys)}")
                sys.stdout.flush()
            try:
                new_key = base_new_key + i
                omap.insert(new_key, make_value(new_key, value_size))
                success += 1
            except Exception as e:
                print(f"\n  Error on op {i}: {e}")
    else:
        for i, (key, op) in enumerate(zip(keys, ops)):
            if i % 10 == 0:
                sys.stdout.write(f"\r  Progress: {i}/{len(keys)}")
                sys.stdout.flush()
            try:
                if op == 'read':
                    omap.search(key)
                else:
                    omap.insert(key, make_value(key, value_size))
                success += 1
            except Exception as e:
                print(f"\n  Error on op {i}: {e}")
            
    print(f"\r  Progress: {len(keys)}/{len(keys)} - Done!")
    
    elapsed = time.time() - start
    client.close_connection()
    
    return {
        'rounds': counter.rounds,
        'bytes_sent': counter.bytes_sent,
        'bytes_recv': counter.bytes_recv,
        'elapsed_sec': elapsed,
        'success': success,
        'max_client_size': omap._peak_client_size
    }


def print_results(name: str, result: dict, num_ops: int):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"Results: {name}")
    print(f"{'='*60}")
    print(f"  Operations:     {result['success']}/{num_ops}")
    print(f"  Total Rounds:   {result['rounds']} ({result['rounds']/num_ops:.2f}/op)")
    print(f"  Total Time:     {result['elapsed_sec']:.2f}s ({result['elapsed_sec']*1000/num_ops:.2f}ms/op)")
    print(f"  Bandwidth Sent: {result['bytes_sent']/1024:.2f} KB ({result['bytes_sent']/1024/num_ops:.2f} KB/op)")
    print(f"  Bandwidth Recv: {result['bytes_recv']/1024:.2f} KB ({result['bytes_recv']/1024/num_ops:.2f} KB/op)")
    print(f"  Max Client Stash: {result['max_client_size']} entries")


def main():
    parser = argparse.ArgumentParser(description="Remote Client-Server Benchmark")
    parser.add_argument("--server-ip", type=str, required=True, help="IP address of the server")
    parser.add_argument("--port", type=int, default=10000, help="Server port (default: 10000)")
    parser.add_argument("--ops", type=int, default=20, help="Number of operations")
    parser.add_argument("--protocol", type=str, default="all", 
                        choices=["all", "bottom_up", "top_down", "baseline"],
                        help="Which protocol to benchmark")
    parser.add_argument("--read-ratio", type=float, default=0.7, help="Ratio of read operations")
    parser.add_argument("--num-data", type=int, default=16383, help="Initial data size (default: 2^14-1)")
    parser.add_argument("--cache-size", type=int, default=1023, help="Cache size (default: 2^10-1)")
    parser.add_argument("--order", type=int, default=4, help="B+ tree order (default: 4)")
    parser.add_argument("--key-size", type=int, default=16, help="Key size in bytes (default: 16)")
    parser.add_argument("--value-size", type=int, default=16, help="Value size in bytes (default: 16)")
    parser.add_argument("--mode", type=str, default="mix", choices=["mix", "insert_only"],
                        help="Operation mode: mix (read/write) or insert_only (new keys)")
    parser.add_argument("--mock-crypto", action="store_true", help="Use mock encryption")
    args = parser.parse_args()

    if args.mock_crypto:
        print("[Mock Encryption Enabled]")
        crypto.MOCK_ENCRYPTION = True

    num_data = args.num_data
    cache_size = args.cache_size
    key_size = args.key_size
    value_size = args.value_size
    data_size = value_size  # data_size is the value size for ORAM
    total_ops = args.ops

    keys = zipf_keys(num_data, total_ops, alpha=0.8)
    ops_list = ['read' if random.random() < args.read_ratio else 'write' for _ in range(total_ops)]

    # Calculate block size (same as ORAM internal calculation)
    from daoram.dependency.helper import Data
    sample_block = Data(key=b'k' * key_size, leaf=0, value=b'v' * value_size)
    block_size = len(sample_block.dump())

    print("="*60)
    print("Remote Client-Server Benchmark")
    print("="*60)
    print(f"  Server:     {args.server_ip}:{args.port}")
    print(f"  N:          {num_data}")
    print(f"  Cache:      {cache_size}")
    print(f"  Order:      {args.order}")
    print(f"  Key Size:   {key_size} bytes")
    print(f"  Value Size: {value_size} bytes")
    print(f"  Block Size: {block_size} bytes")
    print(f"  Mode:       {args.mode}")
    print(f"  Operations: {total_ops}")
    if args.mode == "mix":
        print(f"  Read Ratio: {args.read_ratio}")
    print("="*60)

    results = {}

    if args.protocol in ["all", "bottom_up"]:
        print("\n[Bottom-Up SOMAP]")
        results['bottom_up'] = run_bottom_up(
            args.server_ip, args.port, num_data, cache_size, 
            data_size, keys, ops_list, order=args.order,
            key_size=key_size, value_size=value_size, mode=args.mode
        )
        print_results("Bottom-Up", results['bottom_up'], total_ops)

    if args.protocol in ["all", "top_down"]:
        print("\n[Top-Down SOMAP]")
        results['top_down'] = run_top_down(
            args.server_ip, args.port, num_data, cache_size,
            data_size, keys, ops_list, order=args.order,
            key_size=key_size, value_size=value_size, mode=args.mode
        )
        print_results("Top-Down", results['top_down'], total_ops)

    if args.protocol in ["all", "baseline"]:
        print("\n[Baseline BPlus OMAP]")
        results['baseline'] = run_baseline(
            args.server_ip, args.port, num_data, data_size,
            keys, ops_list, order=args.order,
            key_size=key_size, value_size=value_size, mode=args.mode
        )
        print_results("Baseline", results['baseline'], total_ops)

    # Summary comparison
    if len(results) > 1:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"{'Protocol':<15} {'RTT/op':<12} {'Time/op':<12} {'KB/op':<12}")
        print("-"*60)
        for name, r in results.items():
            print(f"{name:<15} {r['rounds']/total_ops:<12.2f} "
                  f"{r['elapsed_sec']*1000/total_ops:<12.2f}ms "
                  f"{(r['bytes_sent']+r['bytes_recv'])/1024/total_ops:<12.2f}")


if __name__ == "__main__":
    main()
