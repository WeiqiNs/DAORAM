import os
import sys
import random
import time
from typing import Any, Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


import pickle
from daoram.dependency.binary_tree import BinaryTree
from daoram.dependency.interact_server import InteractLocalServer, InteractServer
from daoram.so.bottom_to_up_somap_fixed_cache import BottomUpSomapFixedCache
from daoram.so.top_down_somap_fixed_cache import TopDownSomapFixedCache
from daoram.omap.bplus_ods_omap import BPlusOdsOmap


WAN_LATENCY_SEC = float(os.environ.get("WAN_LATENCY_SEC", "0.05"))



# 用于真实通信的统计辅助
class RoundCounter:
    def __init__(self, client: InteractServer):
        self.client = client
        self.rounds = 0
        self.bytes_sent = 0
        self.bytes_recv = 0
        self._tree_patched = False

    def wrap(self):
        # 本地模式没有真实网络，这里以序列化大小估算通信量，并把每次 RPC 计为一轮
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

        if self._orig_batch:
            def _batch(ops):
                self.rounds += 1
                self.bytes_sent += _size({"type": "batch", "operations": ops})
                resp = self._orig_batch(ops)
                self.bytes_recv += _size(resp)
                return resp
            self.client.batch_query = _batch

        if self._orig_read:
            def _read(label, leaf):
                self.rounds += 1
                self.bytes_sent += _size({"type": "r", "label": label, "leaf": leaf})
                resp = self._orig_read(label, leaf)
                self.bytes_recv += _size(resp)
                return resp
            self.client.read_query = _read

        if self._orig_read_mul:
            def _read_mul(label, leaf):
                self.rounds += 1
                self.bytes_sent += _size([{"type": "r", "label": label[i], "leaf": leaf[i]} for i in range(len(label))])
                resp = self._orig_read_mul(label, leaf)
                self.bytes_recv += _size(resp)
                return resp
            self.client.read_mul_query = _read_mul

        if self._orig_write:
            def _write(label, leaf, data):
                self.rounds += 1
                self.bytes_sent += _size({"type": "w", "label": label, "leaf": leaf, "data": data})
                resp = self._orig_write(label, leaf, data)
                self.bytes_recv += _size(None)
                return resp
            self.client.write_query = _write

        if self._orig_write_mul:
            def _write_mul(label, leaf, data):
                self.rounds += 1
                self.bytes_sent += _size([{"type": "w", "label": label[i], "leaf": leaf[i], "data": data[i]} for i in range(len(label))])
                resp = self._orig_write_mul(label, leaf, data)
                self.bytes_recv += _size(None)
                return resp
            self.client.write_mul_query = _write_mul

        # 进一步细粒度统计：BinaryTree 路径读写也计入轮次/字节（本地模式模拟 WAN 轮）。
        if not self._tree_patched:
            self._orig_bt_read = getattr(BinaryTree, "read_path", None)
            self._orig_bt_write = getattr(BinaryTree, "write_path", None)

            def _bt_read(this, leaf):
                self.rounds += 1
                self.bytes_sent += _size({"type": "bt_r", "leaf": leaf})
                resp = self._orig_bt_read(this, leaf)
                self.bytes_recv += _size(resp)
                return resp

            def _bt_write(this, leaf, data):
                self.rounds += 1
                self.bytes_sent += _size({"type": "bt_w", "leaf": leaf, "data": data})
                resp = self._orig_bt_write(this, leaf, data)
                self.bytes_recv += _size(None)
                return resp

            if self._orig_bt_read:
                BinaryTree.read_path = _bt_read
            if self._orig_bt_write:
                BinaryTree.write_path = _bt_write
            self._tree_patched = True



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


def run_bottom_up(num_data: int, cache_size: int, data_size: int, keys: List[int], ops: List[str]):
    client = InteractLocalServer()
    counter = RoundCounter(client)
    counter.wrap()
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


def run_top_down(num_data: int, cache_size: int, data_size: int, keys: List[int], ops: List[str]):
    client = InteractLocalServer()
    counter = RoundCounter(client)
    counter.wrap()
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


def run_baseline_omap(num_data: int, data_size: int, keys: List[int], ops: List[str]):
    client = InteractLocalServer()
    counter = RoundCounter(client)
    counter.wrap()
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


def main():
    num_data = 2 ** 14
    cache_size = 2 ** 10
    data_size = 16
    total_ops = 200
    read_ratio = 0.7

    keys = zipf_keys(num_data, total_ops, alpha=0.8)
    ops = ['read' if random.random() < read_ratio else 'write' for _ in range(total_ops)]

    print(f"配置: N={num_data}, cache={cache_size}, ops={total_ops}, read_ratio={read_ratio}")

    print("[运行] bottom_up_somap_fixed_cache")
    r1 = run_bottom_up(num_data, cache_size, data_size, keys, ops)
    print(r1)

    print("[运行] top_down_somap_fixed_cache")
    r2 = run_top_down(num_data, cache_size, data_size, keys, ops)
    print(r2)

    print("[运行] baseline bplus_ods_omap")
    r3 = run_baseline_omap(num_data, data_size, keys, ops)
    print(r3)


if __name__ == "__main__":
    main()
