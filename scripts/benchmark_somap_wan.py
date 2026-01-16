import os
import sys
import random
import time
from typing import Any, Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import time
import pickle
import os
import sys
import random
from typing import Any, Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from daoram.dependency import crypto
from daoram.dependency.interact_server import InteractRemoteServer, InteractLocalServer, InteractServer
from daoram.so.bottom_to_up_somap_fixed_cache import BottomUpSomapFixedCache
from daoram.so.top_down_somap_fixed_cache import TopDownSomapFixedCache
from daoram.omap.bplus_ods_omap import BPlusOdsOmap

# WAN Latency Simulation
class RoundCounter:
    def __init__(self, client: InteractServer, latency: float = 0.0):
        self.client = client
        self.rounds = 0
        self.bytes_sent = 0
        self.bytes_recv = 0
        self.latency = latency

    def _simulate_latency(self):
        if self.latency > 0:
            time.sleep(self.latency)

    def wrap(self):
        # We wrap the methods to count rounds and simulate latency. 
        self._orig_batch = getattr(self.client, "batch_query", None)
        self._orig_read = getattr(self.client, "read_query", None)
        self._orig_read_mul = getattr(self.client, "read_mul_query", None)
        self._orig_write = getattr(self.client, "write_query", None)
        self._orig_write_mul = getattr(self.client, "write_mul_query", None)
        
        # Helper to size estimation (rough)
        def _size(obj: Any) -> int:
            try:
                return len(pickle.dumps(obj))
            except Exception:
                return 0

        if self._orig_batch:
            def _batch(ops):
                if not getattr(self.client, "skip_round_counting", False):
                    self.rounds += 1
                    self._simulate_latency()
                self.bytes_sent += _size({"type": "batch", "operations": ops})
                res = self._orig_batch(ops)
                self.bytes_recv += _size(res)
                return res
            self.client.batch_query = _batch

        if self._orig_read:
            def _read(label, leaf):
                if not getattr(self.client, "skip_round_counting", False): 
                    # print("R_R")
                    self.rounds += 1
                    self._simulate_latency()
                # else:
                #    print("S_R")
                self.bytes_sent += _size({"type": "r", "label": label, "leaf": leaf})
                res = self._orig_read(label, leaf)
                self.bytes_recv += _size(res)
                return res
            self.client.read_query = _read

        if self._orig_read_mul:
            def _read_mul(label, leaf):
                if not getattr(self.client, "skip_round_counting", False): 
                    self.rounds += 1
                    self._simulate_latency()
                self.bytes_sent += _size([{"type": "r", "label": l, "leaf": f} for l, f in zip(label, leaf)])
                res = self._orig_read_mul(label, leaf)
                self.bytes_recv += _size(res)
                return res
            self.client.read_mul_query = _read_mul

        if self._orig_write:
            def _write(label, leaf, data):
                if not getattr(self.client, "skip_round_counting", False):
                    self.rounds += 1
                    self._simulate_latency()
                self.bytes_sent += _size({"type": "w", "label": label, "leaf": leaf, "data": data})
                res = self._orig_write(label, leaf, data)
                self.bytes_recv += _size(res) # usually None or ack
                return res
            self.client.write_query = _write
            
        if self._orig_write_mul:
            def _write_mul(label, leaf, data):
                if not getattr(self.client, "skip_round_counting", False):
                    self.rounds += 1
                    self._simulate_latency()
                payload = [{"type": "w", "label": label[i], "leaf": leaf[i], "data": data[i]} for i in range(len(label))]
                self.bytes_sent += _size(payload)
                res = self._orig_write_mul(label, leaf, data)
                self.bytes_recv += _size(res)
                return res
            self.client.write_mul_query = _write_mul

def zipf_keys(n, size, alpha=1.0):
    weights = [1.0 / (i + 1) ** alpha for i in range(n)]
    total = sum(weights)
    probs = [w / total for w in weights]
    return random.choices(range(n), weights=probs, k=size)

def build_data(num_data: int) -> Dict[int, Any]:
    return {i: [i, i] for i in range(num_data)}

def run_bottom_up(num_data: int, cache_size: int, data_size: int, keys: List[int], ops: List[str], latency: float = 0.0, mode: str = "mix"):
    # Connect to Remote Server
    client = InteractLocalServer()
    try: client.init_connection()
    except Exception: pass
    
    counter = RoundCounter(client, latency=latency)
    counter.wrap()
    
    proto = BottomUpSomapFixedCache(num_data=num_data, cache_size=cache_size, data_size=data_size,
                                    client=client, use_encryption=True, aes_key=b'0'*16)
    
    print("Setup Protocol (Upload Initial State)...")
    proto.setup(build_data(num_data))
    proto.reset_peak_client_size()
    
    if crypto.MOCK_ENCRYPTION:
        print("Enabling Simulated CPU Cost for operations...")
        crypto.SIMULATE_CPU_COST = True

    start = time.time()
    counter.rounds = 0
    counter.bytes_sent = 0
    counter.bytes_recv = 0
    counter.rounds = 0
    counter.bytes_sent = 0
    counter.bytes_recv = 0
    print("Start Operations...")
    
    if mode == "insert_only":
        base_new_key = num_data + 200000
        for i in range(len(keys)): # len(keys) is ops count
           if i % 10 == 0:
               sys.stdout.write(f"\rProgress: {i}/{len(keys)}")
               sys.stdout.flush()
           new_key = base_new_key + i
           # Insert for BottomUp is access with new key (Write)
           proto.access(new_key, 'write', [new_key, new_key])
           
    else: # mix or search_only (uses provided keys/ops)
        for i, (key, op) in enumerate(zip(keys, ops)):
            if i % 10 == 0:
                sys.stdout.write(f"\rProgress: {i}/{len(keys)}")
                sys.stdout.flush()
            if op == 'read':
                proto.access(key, 'read')
            else:
                proto.access(key, 'write', [key, key + 1])
                
    print("")

    if crypto.MOCK_ENCRYPTION:
        crypto.SIMULATE_CPU_COST = False
            
    client.close_connection()
            
    return {
        'rounds': counter.rounds,
        'bytes_sent': counter.bytes_sent,
        'bytes_recv': counter.bytes_recv,
        'elapsed_sec': time.time() - start,
        'max_client_size': proto._peak_client_size
    }

def run_top_down(num_data: int, cache_size: int, data_size: int, keys: List[int], ops: List[str], latency: float = 0.0, mode: str = "mix"):
    # Connect to Remote Server
    client = InteractLocalServer()
    try: client.init_connection()
    except Exception: pass
        
    counter = RoundCounter(client, latency=latency)
    counter.wrap()

    proto = TopDownSomapFixedCache(num_data=num_data, cache_size=cache_size, data_size=data_size,
                                   client=client, use_encryption=True, aes_key=b'0'*16)
    
    print("Setup Protocol (Upload Initial State)...")
    # Use string keys consistently
    proto.setup([(str(k), [k, k]) for k in range(num_data)])
    proto.reset_peak_client_size()
    
    if crypto.MOCK_ENCRYPTION:
        print("Enabling Simulated CPU Cost for operations...")
        crypto.SIMULATE_CPU_COST = True

    start = time.time()
    counter.rounds = 0
    counter.bytes_sent = 0
    counter.bytes_recv = 0
    counter.rounds = 0
    counter.bytes_sent = 0
    counter.bytes_recv = 0
    print("Start Operations...")
    
    if mode == "insert_only":
         base_new_key = num_data + 200000
         for i in range(len(keys)):
            if i % 10 == 0:
                sys.stdout.write(f"\rProgress: {i}/{len(keys)}")
                sys.stdout.flush()
            new_key = base_new_key + i
            gk = str(new_key)
            try:
                proto.access('insert', gk, value=[new_key, new_key])
            except (KeyError, MemoryError):
                 continue
    else:
        success_count = 0
        for i, (key, op) in enumerate(zip(keys, ops)):
            if i % 10 == 0:
                sys.stdout.write(f"\rProgress: {i}/{len(keys)}")
                sys.stdout.flush()
            gk = str(key)
            try:
                # Treats both read and write as 'search' (update/read) for existing keys
                proto.access('search', gk)
                success_count += 1
            except (KeyError, MemoryError):
                 continue
        print(f"\nSuccessful ops: {success_count}/{len(keys)}")
                 
    print("")

    if crypto.MOCK_ENCRYPTION:
        crypto.SIMULATE_CPU_COST = False

    client.close_connection()
    return {
        'rounds': counter.rounds,
        'bytes_sent': counter.bytes_sent,
        'bytes_recv': counter.bytes_recv,
        'elapsed_sec': time.time() - start,
        'max_client_size': proto._peak_client_size
    }

def run_baseline_omap(num_data: int, data_size: int, keys: List[int], ops: List[str], latency: float = 0.0, mode: str = "mix"):
    # Connect to Remote Server
    client = InteractLocalServer()
    try: client.init_connection()
    except Exception: pass
    
    counter = RoundCounter(client, latency=latency)
    counter.wrap()

    omap = BPlusOdsOmap(order=4, num_data=num_data, key_size=16, data_size=data_size,
                        client=client, name="baseline", use_encryption=True, aes_key=b'0'*16)
    
    print("Setup Protocol (Upload Initial State)...")
    storage = omap._init_ods_storage([(k, [k, k]) for k in range(num_data)])
    client.init({omap._name: storage})
    omap.reset_peak_client_size()

    if crypto.MOCK_ENCRYPTION:
        print("Enabling Simulated CPU Cost for operations...")
        crypto.SIMULATE_CPU_COST = True

    start = time.time()
    counter.rounds = 0
    counter.bytes_sent = 0
    counter.bytes_recv = 0
    counter.rounds = 0
    counter.bytes_sent = 0
    counter.bytes_recv = 0
    print("Start Operations...")
    
    if mode == "insert_only":
        base_new_key = num_data + 200000
        for i in range(len(keys)):
           if i % 10 == 0:
               sys.stdout.write(f"\rProgress: {i}/{len(keys)}")
               sys.stdout.flush()
           new_key = base_new_key + i
           omap.insert(new_key, [new_key, new_key])
           
    else:
        for i, (key, op) in enumerate(zip(keys, ops)):
            if i % 10 == 0:
                sys.stdout.write(f"\rProgress: {i}/{len(keys)}")
                sys.stdout.flush()
            if op == 'read':
                omap.search(key)
            else:
                # For baseline, insert acts as update if key exists
                omap.insert(key, [key, key + 1])
                
    print("")

    if crypto.MOCK_ENCRYPTION:
        crypto.SIMULATE_CPU_COST = False

    client.close_connection()
    return {
        'rounds': counter.rounds,
        'bytes_sent': counter.bytes_sent,
        'bytes_recv': counter.bytes_recv,
        'elapsed_sec': time.time() - start,
        'max_client_size': omap._peak_client_size
    }

def main():
    parser = argparse.ArgumentParser(description="WAN Benchmark")
    parser.add_argument("--latency", type=float, default=0.00, help="Simulated Network Latency (one-way? No, Round Trip Delay approx) per round in seconds.")
    parser.add_argument("--ops", type=int, default=20, help="Number of operations")
    parser.add_argument("--protocol", type=str, default="all", choices=["all", "bottom_up", "top_down", "baseline", "top_down_insert", "bottom_up_insert", "baseline_insert"], help="Which protocol to benchmark")
    parser.add_argument("--read-ratio", type=float, default=0.7, help="Ratio of read operations (0.0 to 1.0)")
    parser.add_argument("--mock-crypto", action="store_true", help="Enable fast mock encryption for initialization (preserves size, skips CPU cost)")
    args = parser.parse_args()

    if args.mock_crypto:
        print("[Using Mock Encryption] Fast init enabled. CPU cost for encryption is removed, but bandwidth overhead is preserved.")
        crypto.MOCK_ENCRYPTION = True

    num_data = (2 ** 14)-1  # 16383
    cache_size = (2 ** 10)-1 # 1023
    data_size = 16
    total_ops = args.ops
    read_ratio = args.read_ratio

    keys = zipf_keys(num_data, total_ops, alpha=0.8)
    ops = ['read' if random.random() < read_ratio else 'write' for _ in range(total_ops)]

    print(f"Client-Server Benchmark Config: N={num_data}, cache={cache_size}, ops={total_ops}, Latency={args.latency}s")

    if args.protocol == "all" or args.protocol == "bottom_up":
        print("\n[运行] Bottom-Up (WAN Mode) - Mixed/Search")
        r1 = run_bottom_up(num_data, cache_size, data_size, keys, ops, latency=args.latency)
        print(f"Result BottomUp: {r1}")
        
    if args.protocol == "all" or args.protocol == "top_down":
        print("\n[运行] Top-Down (WAN Mode) - Search/Update")
        r2 = run_top_down(num_data, cache_size, data_size, keys, ops, latency=args.latency, mode="mix")
        print(f"Result TopDown: {r2}")
        
    if args.protocol == "top_down_insert":
        print("\n[运行] Top-Down (WAN Mode) - Insert Only")
        r2i = run_top_down(num_data, cache_size, data_size, keys, ops, latency=args.latency, mode="insert_only")
        print(f"Result TopDown Insert: {r2i}")

    if args.protocol == "bottom_up_insert":
        print("\n[运行] Bottom-Up (WAN Mode) - Insert Only")
        r1i = run_bottom_up(num_data, cache_size, data_size, keys, ops, latency=args.latency, mode="insert_only")
        print(f"Result BottomUp Insert: {r1i}")

    if args.protocol == "baseline_insert":
        print("\n[运行] Baseline (WAN Mode) - Insert Only")
        r3i = run_baseline_omap(num_data, data_size, keys, ops, latency=args.latency, mode="insert_only")
        print(f"Result Baseline Insert: {r3i}")

    if args.protocol == "all" or args.protocol == "baseline":
        print("\n[运行] Baseline BPlusOdsOmap (WAN Mode) - Mixed/Search")
        r3 = run_baseline_omap(num_data, data_size, keys, ops, latency=args.latency, mode="mix")
        print(f"Result Baseline: {r3}")

if __name__ == "__main__":
    main()

