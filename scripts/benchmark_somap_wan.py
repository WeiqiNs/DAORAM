#!/usr/bin/env python3
"""
Benchmark SOMAP fixed-cache variants and baseline OMAP in client/server mode.
- Supports local (in-memory) or remote socket server.
- Counts rounds as RPC calls; bytes as pickle size of payloads.

Usage:
  python scripts/benchmark_somap_wan.py [--remote ip port] [--ops 200] [--read-ratio 0.7]

For remote mode, start a server compatible with InteractServer on the given ip/port.
"""
import argparse
import os
import pickle
import random
import sys
import time
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from daoram.dependency.interact_server import InteractLocalServer, InteractServer  # type: ignore
from daoram.so.bottom_to_up_somap_fixed_cache import BottomUpSomapFixedCache  # type: ignore
from daoram.so.top_down_somap_fixed_cache import TopDownSomapFixedCache  # type: ignore
from daoram.omap.bplus_ods_omap import BPlusOdsOmap  # type: ignore


class RoundCounter:
    def __init__(self, client):
        self.client = client
        self.rounds = 0
        self.bytes_sent = 0
        self.bytes_recv = 0
        self._patched = False

    def _size(self, obj: Any) -> int:
        try:
            return len(pickle.dumps(obj))
        except Exception:
            return 0

    def wrap(self):
        if self._patched:
            return
        self._patched = True
        # wrap client RPCs - reads and batch count as rounds, writes don't (batched with reads)
        for name, ctor in {
            "batch_query": lambda fn: self._wrap_rpc(fn, lambda ops: {"type": "batch", "operations": ops}, count_round=True),
            "read_query": lambda fn: self._wrap_rpc(fn, lambda label, leaf: {"type": "r", "label": label, "leaf": leaf}, count_round=True),
            "read_mul_query": lambda fn: self._wrap_rpc(fn, lambda label, leaf: [
                {"type": "r", "label": label[i], "leaf": leaf[i]} for i in range(len(label))
            ], count_round=True),
            "write_query": lambda fn: self._wrap_rpc(fn, lambda label, leaf, data: {"type": "w", "label": label, "leaf": leaf, "data": data}, count_round=False),
            "write_mul_query": lambda fn: self._wrap_rpc(fn, lambda label, leaf, data: [
                {"type": "w", "label": label[i], "leaf": leaf[i], "data": data[i]} for i in range(len(label))
            ], count_round=False),
        }.items():
            orig = getattr(self.client, name, None)
            if orig:
                setattr(self.client, name, ctor(orig))

    def _wrap_rpc(self, fn, payload_builder, count_round=True):
        def inner(*args, **kwargs):
            if count_round:
                self.rounds += 1
            try:
                payload = payload_builder(*args, **kwargs)
                self.bytes_sent += self._size(payload)
            except Exception:
                pass
            resp = fn(*args, **kwargs)
            self.bytes_recv += self._size(resp)
            return resp
        return inner


def zipf_keys(n: int, size: int, alpha: float = 0.8) -> List[int]:
    weights = [1 / ((i + 1) ** alpha) for i in range(n)]
    total = sum(weights)
    probs = [w / total for w in weights]
    return random.choices(range(n), weights=probs, k=size)


def build_data(num_data: int) -> Dict[int, Any]:
    return {i: [i, i] for i in range(num_data)}


def client_sizes_bottom_up(proto: BottomUpSomapFixedCache) -> int:
    return len(proto._Ow._stash) + len(proto._Or._stash) + len(proto._Ds._stash) + proto._Qw_len + proto._Qr_len


def client_sizes_top_down(proto: TopDownSomapFixedCache) -> int:
    return len(proto._Ow._stash) + len(proto._Or._stash) + len(proto._tree_stash) + proto._Qw_len + proto._Qr_len


def run_bottom_up(client, num_data, cache_size, data_size, keys, ops, counter):
    proto = BottomUpSomapFixedCache(num_data=num_data, cache_size=cache_size, data_size=data_size,
                                    client=client, use_encryption=False)
    proto.setup(build_data(num_data))
    proto.reset_peak_client_size()
    max_size = 0
    start = time.time()
    for key, op in zip(keys, ops):
        if op == 'read':
            proto.access(key, 'read')
        else:
            proto.access(key, 'write', [key, key + 1])
        max_size = max(max_size, client_sizes_bottom_up(proto))
    return {
        'rounds': counter.rounds,
        'bytes_sent': counter.bytes_sent,
        'bytes_recv': counter.bytes_recv,
        'max_client_size_external': max_size,
        'max_client_size_internal': max(proto._peak_client_size, max_size),
        'elapsed_sec': time.time() - start
    }


def run_top_down(client, num_data, cache_size, data_size, keys, ops, counter):
    proto = TopDownSomapFixedCache(num_data=num_data, cache_size=cache_size, data_size=data_size,
                                   client=client, use_encryption=False)
    proto.setup([(k, [k, k]) for k in range(num_data)])
    proto.reset_peak_client_size()
    max_size = 0
    start = time.time()
    for key, op in zip(keys, ops):
        gk = str(key)
        try:
            if op == 'read':
                proto.access('search', gk)
            else:
                proto.access('insert', gk, [key, key + 1])
            max_size = max(max_size, client_sizes_top_down(proto))
        except (KeyError, MemoryError):
            continue
    return {
        'rounds': counter.rounds,
        'bytes_sent': counter.bytes_sent,
        'bytes_recv': counter.bytes_recv,
        'max_client_size_external': max_size,
        'max_client_size_internal': max(proto._peak_client_size, max_size),
        'elapsed_sec': time.time() - start
    }


def run_baseline_omap(client, num_data, data_size, keys, ops, counter):
    omap = BPlusOdsOmap(order=4, num_data=num_data, key_size=16, data_size=data_size,
                        client=client, name="baseline", use_encryption=False)
    storage = omap._init_ods_storage([(k, [k, k]) for k in range(num_data)])
    client.init({omap._name: storage})
    omap.reset_peak_client_size()
    max_size = 0
    start = time.time()
    for key, op in zip(keys, ops):
        if op == 'read':
            omap.search(key)
        else:
            omap.insert(key, [key, key + 1])
        max_size = max(max_size, len(omap._stash))
    return {
        'rounds': counter.rounds,
        'bytes_sent': counter.bytes_sent,
        'bytes_recv': counter.bytes_recv,
        'max_client_size_external': max_size,
        'max_client_size_internal': max(omap._peak_client_size, max_size),
        'elapsed_sec': time.time() - start
    }


def make_client(remote_ip: str = None, remote_port: int = None):
    if remote_ip:
        c = InteractServer(ip=remote_ip, port=remote_port)
        c.init_connection()
    else:
        c = InteractLocalServer()
    return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote', nargs=2, metavar=('IP', 'PORT'), help='Use remote InteractServer at IP PORT')
    parser.add_argument('--ops', type=int, default=200)
    parser.add_argument('--read-ratio', type=float, default=0.7)
    parser.add_argument('--num-data', type=int, default=2 ** 14)
    parser.add_argument('--cache-size', type=int, default=2 ** 10)
    parser.add_argument('--data-size', type=int, default=16)
    args = parser.parse_args()

    num_data = args.num_data
    cache_size = args.cache_size
    data_size = args.data_size
    total_ops = args.ops
    read_ratio = args.read_ratio

    keys = zipf_keys(num_data, total_ops, alpha=0.8)
    ops = ['read' if random.random() < read_ratio else 'write' for _ in range(total_ops)]

    remote_ip, remote_port = (args.remote[0], int(args.remote[1])) if args.remote else (None, None)
    mode = 'remote' if remote_ip else 'local'
    print(f"模式: {mode}, N={num_data}, cache={cache_size}, ops={total_ops}, read_ratio={read_ratio}")

    print("[运行] bottom_up_somap_fixed_cache")
    client = make_client(remote_ip, remote_port)
    counter = RoundCounter(client)
    counter.wrap()
    r1 = run_bottom_up(client, num_data, cache_size, data_size, keys, ops, counter)
    print(r1)

    print("[运行] top_down_somap_fixed_cache")
    client = make_client(remote_ip, remote_port)
    counter = RoundCounter(client)
    counter.wrap()
    r2 = run_top_down(client, num_data, cache_size, data_size, keys, ops, counter)
    print(r2)

    print("[运行] baseline bplus_ods_omap")
    client = make_client(remote_ip, remote_port)
    counter = RoundCounter(client)
    counter.wrap()
    r3 = run_baseline_omap(client, num_data, data_size, keys, ops, counter)
    print(r3)

    if remote_ip:
        client.close_connection()


if __name__ == "__main__":
    main()
