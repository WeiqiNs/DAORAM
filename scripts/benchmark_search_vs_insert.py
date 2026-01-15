"""
Benchmark: Search vs Insert performance across three protocols
"""
import os
import sys
import random
import time
import pickle
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from daoram.dependency import crypto
from daoram.dependency.interact_server import InteractLocalServer, InteractServer
from daoram.so.bottom_to_up_somap_fixed_cache import BottomUpSomapFixedCache
from daoram.so.top_down_somap_fixed_cache import TopDownSomapFixedCache
from daoram.omap.bplus_ods_omap import BPlusOdsOmap


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

        # batch_query itself doesn't count as a round
        # the internal read/write calls will be counted
        if self._orig_batch:
            def _batch(ops):
                self.bytes_sent += _size(ops)
                res = self._orig_batch(ops)
                self.bytes_recv += _size(res)
                return res
            self.client.batch_query = _batch

        if self._orig_read:
            def _read(label, leaf):
                if not getattr(self.client, "skip_round_counting", False):
                    self.rounds += 1
                    self._simulate_latency()
                self.bytes_sent += _size({"label": label, "leaf": leaf})
                res = self._orig_read(label, leaf)
                self.bytes_recv += _size(res)
                return res
            self.client.read_query = _read

        if self._orig_read_mul:
            def _read_mul(label, leaf):
                if not getattr(self.client, "skip_round_counting", False):
                    self.rounds += 1
                    self._simulate_latency()
                self.bytes_sent += _size({"label": label, "leaf": leaf})
                res = self._orig_read_mul(label, leaf)
                self.bytes_recv += _size(res)
                return res
            self.client.read_mul_query = _read_mul

        if self._orig_write:
            def _write(label, leaf, data):
                if not getattr(self.client, "skip_round_counting", False):
                    self.rounds += 1
                    self._simulate_latency()
                self.bytes_sent += _size({"label": label, "leaf": leaf, "data": data})
                res = self._orig_write(label, leaf, data)
                self.bytes_recv += _size(res)
                return res
            self.client.write_query = _write

        if self._orig_write_mul:
            def _write_mul(label, leaf, data):
                if not getattr(self.client, "skip_round_counting", False):
                    self.rounds += 1
                    self._simulate_latency()
                self.bytes_sent += _size({"label": label, "leaf": leaf, "data": data})
                res = self._orig_write_mul(label, leaf, data)
                self.bytes_recv += _size(res)
                return res
            self.client.write_mul_query = _write_mul


def test_bottom_up(num_data: int, cache_size: int, data_size: int, num_ops: int, op_type: str, latency: float = 0.0):
    """Test Bottom-Up protocol"""
    client = InteractLocalServer()
    counter = RoundCounter(client, latency=latency)
    counter.wrap()

    proto = BottomUpSomapFixedCache(
        num_data=num_data, cache_size=cache_size, data_size=data_size,
        client=client, use_encryption=True, aes_key=b'0'*16
    )

    # Setup with initial data
    proto.setup({i: [i, i] for i in range(num_data)})
    proto.reset_peak_client_size()
    counter.rounds = 0
    counter.bytes_sent = 0
    counter.bytes_recv = 0

    start = time.time()
    success = 0

    if op_type == "search":
        # Search existing keys
        keys = [random.randint(0, num_data - 1) for _ in range(num_ops)]
        for key in keys:
            try:
                proto.access(key, 'read')
                success += 1
            except (KeyError, MemoryError):
                pass
    else:  # insert
        # Insert new keys
        base_key = num_data + 100000
        for i in range(num_ops):
            new_key = base_key + i
            try:
                proto.access(new_key, 'write', [new_key, new_key])
                success += 1
            except (KeyError, MemoryError):
                pass

    elapsed = time.time() - start
    return {
        'rounds': counter.rounds,
        'rounds_per_op': counter.rounds / num_ops if num_ops > 0 else 0,
        'bytes_sent': counter.bytes_sent,
        'bytes_recv': counter.bytes_recv,
        'elapsed_sec': elapsed,
        'success': success,
        'total': num_ops,
        'max_client_size': proto._peak_client_size
    }


def test_top_down(num_data: int, cache_size: int, data_size: int, num_ops: int, op_type: str, latency: float = 0.0):
    """Test Top-Down protocol"""
    client = InteractLocalServer()
    counter = RoundCounter(client, latency=latency)
    counter.wrap()

    proto = TopDownSomapFixedCache(
        num_data=num_data, cache_size=cache_size, data_size=data_size,
        client=client, use_encryption=True, aes_key=b'0'*16
    )

    # Setup with initial data (string keys)
    proto.setup([(str(k), [k, k]) for k in range(num_data)])
    proto.reset_peak_client_size()
    counter.rounds = 0
    counter.bytes_sent = 0
    counter.bytes_recv = 0

    start = time.time()
    success = 0

    if op_type == "search":
        # Search existing keys
        keys = [random.randint(0, num_data - 1) for _ in range(num_ops)]
        for key in keys:
            try:
                proto.access('search', str(key))
                success += 1
            except (KeyError, MemoryError):
                pass
    else:  # insert
        # Insert new keys
        base_key = num_data + 100000
        for i in range(num_ops):
            new_key = base_key + i
            try:
                proto.access('insert', str(new_key), value=[new_key, new_key])
                success += 1
            except (KeyError, MemoryError):
                pass

    elapsed = time.time() - start
    return {
        'rounds': counter.rounds,
        'rounds_per_op': counter.rounds / num_ops if num_ops > 0 else 0,
        'bytes_sent': counter.bytes_sent,
        'bytes_recv': counter.bytes_recv,
        'elapsed_sec': elapsed,
        'success': success,
        'total': num_ops,
        'max_client_size': proto._peak_client_size
    }


def test_baseline(num_data: int, data_size: int, num_ops: int, op_type: str, latency: float = 0.0):
    """Test Baseline BPlusOdsOmap protocol"""
    client = InteractLocalServer()
    counter = RoundCounter(client, latency=latency)
    counter.wrap()

    proto = BPlusOdsOmap(
        order=4, num_data=num_data, key_size=16, data_size=data_size,
        client=client, use_encryption=True, aes_key=b'0'*16
    )

    # Setup
    storage = proto._init_ods_storage([(k, [k, k]) for k in range(num_data)])
    client.init({'bplus': storage})
    counter.rounds = 0
    counter.bytes_sent = 0
    counter.bytes_recv = 0

    start = time.time()
    success = 0

    if op_type == "search":
        # Search existing keys
        keys = [random.randint(0, num_data - 1) for _ in range(num_ops)]
        for key in keys:
            try:
                proto.search(key)
                success += 1
            except (KeyError, MemoryError):
                pass
    else:  # insert
        # Insert new keys
        base_key = num_data + 100000
        for i in range(num_ops):
            new_key = base_key + i
            try:
                proto.insert(new_key, [new_key, new_key])
                success += 1
            except (KeyError, MemoryError):
                pass

    elapsed = time.time() - start
    
    # Get max stash size for baseline
    max_stash = proto._peak_client_size if hasattr(proto, '_peak_client_size') else len(proto._stash)
    
    return {
        'rounds': counter.rounds,
        'rounds_per_op': counter.rounds / num_ops if num_ops > 0 else 0,
        'bytes_sent': counter.bytes_sent,
        'bytes_recv': counter.bytes_recv,
        'elapsed_sec': elapsed,
        'success': success,
        'total': num_ops,
        'max_client_size': max_stash
    }


def main():
    # Configuration
    NUM_DATA = 16383
    CACHE_SIZE = 1023
    DATA_SIZE = 16
    NUM_OPS = 10  # 减少操作数以加快测试（有延迟时）
    LATENCY = 0.05  # 50ms RTT 模拟 WAN 延迟

    print(f"=" * 70)
    print(f"Benchmark: Search vs Insert")
    print(f"Config: N={NUM_DATA}, cache={CACHE_SIZE}, ops={NUM_OPS}, latency={LATENCY*1000:.0f}ms")
    print(f"=" * 70)

    results = {}

    # Test Search
    print("\n" + "=" * 70)
    print("SEARCH Operations")
    print("=" * 70)

    print("\n[Bottom-Up] Search...")
    results['bottom_up_search'] = test_bottom_up(NUM_DATA, CACHE_SIZE, DATA_SIZE, NUM_OPS, "search", latency=LATENCY)
    print(f"  Rounds: {results['bottom_up_search']['rounds']} ({results['bottom_up_search']['rounds_per_op']:.2f}/op)")
    print(f"  Success: {results['bottom_up_search']['success']}/{results['bottom_up_search']['total']}")

    print("\n[Top-Down] Search...")
    results['top_down_search'] = test_top_down(NUM_DATA, CACHE_SIZE, DATA_SIZE, NUM_OPS, "search", latency=LATENCY)
    print(f"  Rounds: {results['top_down_search']['rounds']} ({results['top_down_search']['rounds_per_op']:.2f}/op)")
    print(f"  Success: {results['top_down_search']['success']}/{results['top_down_search']['total']}")

    print("\n[Baseline] Search...")
    results['baseline_search'] = test_baseline(NUM_DATA, DATA_SIZE, NUM_OPS, "search", latency=LATENCY)
    print(f"  Rounds: {results['baseline_search']['rounds']} ({results['baseline_search']['rounds_per_op']:.2f}/op)")
    print(f"  Success: {results['baseline_search']['success']}/{results['baseline_search']['total']}")

    # Test Insert
    print("\n" + "=" * 70)
    print("INSERT Operations")
    print("=" * 70)

    print("\n[Bottom-Up] Insert...")
    results['bottom_up_insert'] = test_bottom_up(NUM_DATA, CACHE_SIZE, DATA_SIZE, NUM_OPS, "insert", latency=LATENCY)
    print(f"  Rounds: {results['bottom_up_insert']['rounds']} ({results['bottom_up_insert']['rounds_per_op']:.2f}/op)")
    print(f"  Success: {results['bottom_up_insert']['success']}/{results['bottom_up_insert']['total']}")

    print("\n[Top-Down] Insert...")
    results['top_down_insert'] = test_top_down(NUM_DATA, CACHE_SIZE, DATA_SIZE, NUM_OPS, "insert", latency=LATENCY)
    print(f"  Rounds: {results['top_down_insert']['rounds']} ({results['top_down_insert']['rounds_per_op']:.2f}/op)")
    print(f"  Success: {results['top_down_insert']['success']}/{results['top_down_insert']['total']}")

    print("\n[Baseline] Insert...")
    results['baseline_insert'] = test_baseline(NUM_DATA, DATA_SIZE, NUM_OPS, "insert", latency=LATENCY)
    print(f"  Rounds: {results['baseline_insert']['rounds']} ({results['baseline_insert']['rounds_per_op']:.2f}/op)")
    print(f"  Success: {results['baseline_insert']['success']}/{results['baseline_insert']['total']}")

    # Summary Table
    print("\n" + "=" * 70)
    print("SUMMARY (Rounds per Operation)")
    print("=" * 70)
    print(f"{'Protocol':<15} {'Search RTT/op':<15} {'Insert RTT/op':<15}")
    print("-" * 45)
    print(f"{'Bottom-Up':<15} {results['bottom_up_search']['rounds_per_op']:<15.2f} {results['bottom_up_insert']['rounds_per_op']:<15.2f}")
    print(f"{'Top-Down':<15} {results['top_down_search']['rounds_per_op']:<15.2f} {results['top_down_insert']['rounds_per_op']:<15.2f}")
    print(f"{'Baseline':<15} {results['baseline_search']['rounds_per_op']:<15.2f} {results['baseline_insert']['rounds_per_op']:<15.2f}")

    # Bandwidth Summary
    print("\n" + "=" * 70)
    print("BANDWIDTH (KB per Operation)")
    print("=" * 70)
    print(f"{'Protocol':<15} {'Search Send':<12} {'Search Recv':<12} {'Insert Send':<12} {'Insert Recv':<12}")
    print("-" * 63)
    for name, prefix in [('Bottom-Up', 'bottom_up'), ('Top-Down', 'top_down'), ('Baseline', 'baseline')]:
        s_send = results[f'{prefix}_search']['bytes_sent'] / NUM_OPS / 1024
        s_recv = results[f'{prefix}_search']['bytes_recv'] / NUM_OPS / 1024
        i_send = results[f'{prefix}_insert']['bytes_sent'] / NUM_OPS / 1024
        i_recv = results[f'{prefix}_insert']['bytes_recv'] / NUM_OPS / 1024
        print(f"{name:<15} {s_send:<12.2f} {s_recv:<12.2f} {i_send:<12.2f} {i_recv:<12.2f}")

    # Time Summary
    print("\n" + "=" * 70)
    print("PROCESSING TIME (ms per Operation)")
    print("=" * 70)
    print(f"{'Protocol':<15} {'Search ms/op':<15} {'Insert ms/op':<15}")
    print("-" * 45)
    for name, prefix in [('Bottom-Up', 'bottom_up'), ('Top-Down', 'top_down'), ('Baseline', 'baseline')]:
        s_time = results[f'{prefix}_search']['elapsed_sec'] / NUM_OPS * 1000
        i_time = results[f'{prefix}_insert']['elapsed_sec'] / NUM_OPS * 1000
        print(f"{name:<15} {s_time:<15.2f} {i_time:<15.2f}")

    # Client Storage Summary
    print("\n" + "=" * 70)
    print("CLIENT-SIDE STORAGE (max stash entries)")
    print("=" * 70)
    print(f"{'Protocol':<15} {'Search':<15} {'Insert':<15}")
    print("-" * 45)
    for name, prefix in [('Bottom-Up', 'bottom_up'), ('Top-Down', 'top_down'), ('Baseline', 'baseline')]:
        s_storage = results[f'{prefix}_search'].get('max_client_size', 'N/A')
        i_storage = results[f'{prefix}_insert'].get('max_client_size', 'N/A')
        print(f"{name:<15} {s_storage:<15} {i_storage:<15}")


if __name__ == "__main__":
    main()
