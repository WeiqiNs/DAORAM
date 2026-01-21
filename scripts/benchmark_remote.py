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
    def __init__(
        self,
        client: InteractServer,
        sim_latency_ms: float = 0.0,
        simulate_padding: bool = False,
        sim_bucket_size: int = None,
        sim_block_size: int = None,
        sim_crypto_mbps: float = 200.0,
    ):
        self.client = client
        self.rounds = 0
        self.bytes_sent = 0
        self.bytes_recv = 0
        self.raw_bytes_sent = 0
        self.raw_bytes_recv = 0
        self.comm_time = 0.0
        self.sim_latency_sec = (sim_latency_ms * 2) / 1000.0 if sim_latency_ms > 0 else 0.0
        self.simulate_padding = simulate_padding
        self.sim_bucket_size = sim_bucket_size
        self.sim_block_size = sim_block_size
        self.sim_crypto_mbps = sim_crypto_mbps
        self.sim_proc_time = 0.0
        self.total_read_ops = 0
        self.total_write_ops = 0
        self.total_read_buckets = 0
        self.total_write_buckets = 0

    def _count_buckets(self, buckets) -> int:
        if not buckets or not isinstance(buckets, list):
            return 0
        if len(buckets) == 0:
            return 0
        if isinstance(buckets[0], list) and len(buckets[0]) > 0 and isinstance(buckets[0][0], list):
            # List[Buckets]
            return sum(len(path) for path in buckets)
        # Buckets
        return len(buckets)

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
                sent = _size(ops)
                self.bytes_sent += sent
                self.raw_bytes_sent += sent
                start = time.time()
                if self.sim_latency_sec > 0:
                    time.sleep(self.sim_latency_sec)
                res = self._orig_batch(ops)
                self.comm_time += (time.time() - start)
                recv = _size(res)
                self.bytes_recv += recv
                self.raw_bytes_recv += recv
                # Simulate padding for batch operations
                self._simulate_batch_padding(ops, res, sent, recv)
                return res
            self.client.batch_query = _batch

        if self._orig_read:
            def _read(label, leaf):
                if not getattr(self.client, "skip_round_counting", False):
                    self.rounds += 1
                sent = _size({"label": label, "leaf": leaf})
                self.bytes_sent += sent
                self.raw_bytes_sent += sent
                start = time.time()
                if self.sim_latency_sec > 0 and not getattr(self.client, "skip_round_counting", False):
                     time.sleep(self.sim_latency_sec)
                res = self._orig_read(label, leaf)
                self.comm_time += (time.time() - start)
                recv = _size(res)
                self.bytes_recv += recv
                self.raw_bytes_recv += recv
                self._simulate_padding_recv(res, recv)
                return res
            self.client.read_query = _read

        if self._orig_read_mul:
            def _read_mul(label, leaf):
                if not getattr(self.client, "skip_round_counting", False):
                    self.rounds += 1
                sent = _size({"label": label, "leaf": leaf})
                self.bytes_sent += sent
                self.raw_bytes_sent += sent
                start = time.time()
                if self.sim_latency_sec > 0 and not getattr(self.client, "skip_round_counting", False): 
                    time.sleep(self.sim_latency_sec)
                res = self._orig_read_mul(label, leaf)
                self.comm_time += (time.time() - start)
                recv = _size(res)
                self.bytes_recv += recv
                self.raw_bytes_recv += recv
                self._simulate_padding_recv(res, recv)
                return res
            self.client.read_mul_query = _read_mul

        if self._orig_write:
            def _write(label, leaf, data):
                if not getattr(self.client, "skip_round_counting", False):
                    self.rounds += 1
                sent = _size({"label": label, "leaf": leaf, "data": data})
                self.bytes_sent += sent
                self.raw_bytes_sent += sent
                start = time.time()
                if self.sim_latency_sec > 0 and not getattr(self.client, "skip_round_counting", False):
                    time.sleep(self.sim_latency_sec)
                res = self._orig_write(label, leaf, data)
                self.comm_time += (time.time() - start)
                recv = _size(res)
                self.bytes_recv += recv
                self.raw_bytes_recv += recv
                data_size = _size(data)
                self._simulate_padding_send(data, data_size)
                return res
            self.client.write_query = _write

        if self._orig_write_mul:
            def _write_mul(label, leaf, data):
                if not getattr(self.client, "skip_round_counting", False):
                    self.rounds += 1
                sent = _size({"label": label, "leaf": leaf, "data": data})
                self.bytes_sent += sent
                self.raw_bytes_sent += sent
                start = time.time()
                if self.sim_latency_sec > 0 and not getattr(self.client, "skip_round_counting", False):
                    time.sleep(self.sim_latency_sec)
                res = self._orig_write_mul(label, leaf, data)
                self.comm_time += (time.time() - start)
                recv = _size(res)
                self.bytes_recv += recv
                self.raw_bytes_recv += recv
                data_size = _size(data)
                self._simulate_padding_send(data, data_size)
                return res
            self.client.write_mul_query = _write_mul

    def _simulate_padding_recv(self, buckets, raw_bytes: int = 0):
        """Simulate full bucket padding for received data.
        
        In real ORAM, each bucket is always padded to bucket_size blocks,
        and each block is padded to block_size bytes. So total recv should be:
        num_buckets × bucket_size × block_size
        
        We replace the actual pickle size with this theoretical full size.
        
        Note: buckets can be either:
        - Buckets (List[Bucket]) from read_query
        - List[Buckets] from read_mul_query
        """
        if not self.simulate_padding or self.sim_bucket_size is None or self.sim_block_size is None:
            return
        try:
            # Detect if this is List[Buckets] (from read_mul_query) or Buckets (from read_query)
            # List[Buckets] = List[List[Bucket]], Buckets = List[Bucket]
            # Check if first element is a list (Bucket) or a list of lists (Buckets)
            if len(buckets) == 0:
                return
            num_buckets = self._count_buckets(buckets)
            if num_buckets > 0:
                self.total_read_ops += 1
                self.total_read_buckets += num_buckets
            
            full_size = num_buckets * self.sim_bucket_size * self.sim_block_size
            # Replace actual pickle size with full padded size
            extra_bytes = full_size - raw_bytes
            if extra_bytes > 0:
                self.bytes_recv += extra_bytes
                self.sim_proc_time += extra_bytes / (self.sim_crypto_mbps * 1024 * 1024)
        except Exception:
            return

    def _simulate_padding_send(self, buckets, raw_bytes: int = 0):
        """Simulate full bucket padding for sent data.
        
        In real ORAM, each bucket is always padded to bucket_size blocks,
        and each block is padded to block_size bytes. So total send should be:
        num_buckets × bucket_size × block_size
        
        We replace the actual pickle size with this theoretical full size.
        
        Note: buckets can be either:
        - Buckets (List[Bucket]) from write_query
        - List[Buckets] from write_mul_query
        """
        if not self.simulate_padding or self.sim_bucket_size is None or self.sim_block_size is None:
            return
        try:
            # Detect if this is List[Buckets] or Buckets
            if len(buckets) == 0:
                return
            num_buckets = self._count_buckets(buckets)
            if num_buckets > 0:
                self.total_write_ops += 1
                self.total_write_buckets += num_buckets
            
            full_size = num_buckets * self.sim_bucket_size * self.sim_block_size
            # Replace actual pickle size with full padded size
            extra_bytes = full_size - raw_bytes
            if extra_bytes > 0:
                self.bytes_sent += extra_bytes
                self.sim_proc_time += extra_bytes / (self.sim_crypto_mbps * 1024 * 1024)
        except Exception:
            return

    def _simulate_batch_padding(self, ops, results, raw_sent: int, raw_recv: int):
        """Simulate full bucket padding for batch operations.
        
        Batch operations can include:
        - read: Returns Buckets, needs recv padding
        - write: Sends Buckets in op['data'], needs send padding
        - list_insert/list_pop/list_get/list_update: Small metadata, no padding needed
        
        We calculate padded size for all read/write ops and add the difference from raw.
        """
        if not self.simulate_padding or self.sim_bucket_size is None or self.sim_block_size is None:
            return
        
        try:
            def _size(obj: Any) -> int:
                try:
                    return len(pickle.dumps(obj))
                except Exception:
                    return 0

            extra_send = 0
            for op in ops:
                if op.get('op') == 'write' and 'data' in op:
                    data = op['data']
                    num_buckets = self._count_buckets(data)
                    if num_buckets > 0:
                        self.total_write_ops += 1
                        self.total_write_buckets += num_buckets
                        full_send = num_buckets * self.sim_bucket_size * self.sim_block_size
                        raw_send = _size(data)
                        if full_send > raw_send:
                            extra_send += (full_send - raw_send)

            extra_recv = 0
            for i, op in enumerate(ops):
                if op.get('op') == 'read' and i < len(results):
                    res = results[i]
                    num_buckets = self._count_buckets(res)
                    if num_buckets > 0:
                        self.total_read_ops += 1
                        self.total_read_buckets += num_buckets
                        full_recv = num_buckets * self.sim_bucket_size * self.sim_block_size
                        raw_recv_local = _size(res)
                        if full_recv > raw_recv_local:
                            extra_recv += (full_recv - raw_recv_local)

            if extra_send > 0:
                self.bytes_sent += extra_send
                self.sim_proc_time += extra_send / (self.sim_crypto_mbps * 1024 * 1024)

            if extra_recv > 0:
                self.bytes_recv += extra_recv
                self.sim_proc_time += extra_recv / (self.sim_crypto_mbps * 1024 * 1024)
                
        except Exception:
            return

    def reset(self):
        self.rounds = 0
        self.bytes_sent = 0
        self.bytes_recv = 0
        self.raw_bytes_sent = 0
        self.raw_bytes_recv = 0
        self.sim_proc_time = 0.0
        self.total_read_ops = 0
        self.total_write_ops = 0
        self.total_read_buckets = 0
        self.total_write_buckets = 0


def calc_sim_real_crypto_time(counter: RoundCounter, sim_crypto_mbps: float) -> float:
    if sim_crypto_mbps <= 0:
        return 0.0
    total = counter.raw_bytes_sent + counter.raw_bytes_recv
    return total / (sim_crypto_mbps * 1024 * 1024)


def zipf_keys(n, size, alpha=1.0):
    weights = [1.0 / (i + 1) ** alpha for i in range(n)]
    total = sum(weights)
    probs = [w / total for w in weights]
    return random.choices(range(n), weights=probs, k=size)


def calc_block_size(key_size: int, value_size: int, encrypted: bool = True) -> int:
    """Calculate the block size used in ORAM.
    
    :param encrypted: If True, include AES encryption overhead (IV + padding)
    """
    from daoram.dependency.helper import Data
    sample = Data(key=b'k' * key_size, leaf=0, value=b'v' * value_size)
    raw_size = len(sample.dump())
    
    if encrypted:
        # AES encryption adds: IV (16 bytes) + PKCS7 padding (up to 16 bytes)
        # Encrypted size = 16 (IV) + ceil(raw_size / 16) * 16
        import math
        padded_size = math.ceil(raw_size / 16) * 16
        return 16 + padded_size  # IV + padded data
    return raw_size


def calc_client_storage_bottom_up(proto, key_size: int, value_size: int) -> dict:
    """Calculate detailed client storage for Bottom-Up SOMAP.
    
    Client stores:
    - Stash for O_W, O_R, D_S (ORAM stashes)
    - Local buffers
    - Pending queues
    - AES key and metadata
    
    Note: No pos_map on client - position mapping is handled by B+ tree structure
    """
    block_size = calc_block_size(key_size, value_size)
    
    # Stash entries (O_W + O_R + D_S)
    ow_stash = len(proto._Ow._stash) if proto._Ow else 0
    or_stash = len(proto._Or._stash) if proto._Or else 0
    ds_stash = len(getattr(proto._Ds, "_stash", [])) if proto._Ds else 0
    
    # Local buffers
    ow_local = len(getattr(proto._Ow, "_local", [])) if proto._Ow else 0
    or_local = len(getattr(proto._Or, "_local", [])) if proto._Or else 0
    
    # Pending operations
    pending_qr = len(proto._pending_qr_inserts) if hasattr(proto, '_pending_qr_inserts') else 0
    
    total_entries = ow_stash + or_stash + ds_stash + ow_local + or_local + pending_qr
    stash_bytes = total_entries * block_size
    
    # Metadata: AES key (16 bytes), tree root info, parameters
    metadata_bytes = 128  # Fixed overhead
    
    total_bytes = stash_bytes + metadata_bytes
    
    return {
        'stash_entries': total_entries,
        'stash_bytes': stash_bytes,
        'metadata_bytes': metadata_bytes,
        'total_bytes': total_bytes,
        'block_size': block_size
    }


def calc_client_storage_top_down(proto, key_size: int, value_size: int) -> dict:
    """Calculate detailed client storage for Top-Down SOMAP.
    
    Client stores:
    - Stash for O_W, O_R (ORAM stashes)
    - Tree stash (for B+ tree nodes)
    - Local buffers
    - Pending operations
    - AES key and metadata
    """
    block_size = calc_block_size(key_size, value_size)
    
    # Stash entries
    ow_stash = len(proto._Ow._stash) if proto._Ow else 0
    or_stash = len(proto._Or._stash) if proto._Or else 0
    tree_stash = len(proto._tree_stash) if hasattr(proto, '_tree_stash') else 0
    
    # Local buffers  
    ow_local = len(getattr(proto._Ow, "_local", [])) if proto._Ow else 0
    or_local = len(getattr(proto._Or, "_local", [])) if proto._Or else 0
    
    # Pending operations
    pending_qr = len(proto._pending_qr_inserts) if hasattr(proto, '_pending_qr_inserts') else 0
    
    total_entries = ow_stash + or_stash + tree_stash + ow_local + or_local + pending_qr
    stash_bytes = total_entries * block_size
    
    # Metadata
    metadata_bytes = 128
    
    total_bytes = stash_bytes + metadata_bytes
    
    return {
        'stash_entries': total_entries,
        'stash_bytes': stash_bytes,
        'metadata_bytes': metadata_bytes,
        'total_bytes': total_bytes,
        'block_size': block_size
    }


def calc_client_storage_baseline(omap, key_size: int, value_size: int) -> dict:
    """Calculate detailed client storage for Baseline BPlus OMAP.
    
    Client stores:
    - Stash (ORAM stash)
    - Local buffer
    - AES key and metadata
    """
    block_size = calc_block_size(key_size, value_size)
    
    # Stash and local
    stash_entries = len(omap._stash) if hasattr(omap, '_stash') else 0
    local_entries = len(omap._local) if hasattr(omap, '_local') else 0
    total_entries = stash_entries + local_entries
    stash_bytes = total_entries * block_size
    
    # Metadata
    metadata_bytes = 128
    
    total_bytes = stash_bytes + metadata_bytes
    
    return {
        'stash_entries': total_entries,
        'stash_bytes': stash_bytes,
        'metadata_bytes': metadata_bytes,
        'total_bytes': total_bytes,
        'block_size': block_size
    }


def calc_server_storage_bottom_up(proto, key_size: int, value_size: int, num_data: int, cache_size: int) -> dict:
    """Calculate server storage for Bottom-Up SOMAP.
    
    Server stores:
    - O_W: ORAM tree for write operations (size = cache_size)
    - O_R: ORAM tree for read operations (size = cache_size)
    - D_S: Static ORAM for initial data (size = num_data)
    - Q_W, Q_R: Queues
    """
    block_size = calc_block_size(key_size, value_size)
    bucket_size = 4  # Default bucket size
    
    # O_W and O_R: cache_size entries
    import math
    ow_level = int(math.ceil(math.log(cache_size + 1, 2))) + 1
    ow_num_buckets = pow(2, ow_level) - 1
    ow_bytes = ow_num_buckets * bucket_size * block_size
    
    or_level = int(math.ceil(math.log(cache_size + 1, 2))) + 1  
    or_num_buckets = pow(2, or_level) - 1
    or_bytes = or_num_buckets * bucket_size * block_size
    
    # D_S: num_data entries (Static ORAM)
    ds_level = int(math.ceil(math.log(num_data + 1, 2))) + 1
    ds_num_buckets = pow(2, ds_level) - 1
    ds_bytes = ds_num_buckets * bucket_size * block_size
    
    # Q_W, Q_R: queue storage (estimate)
    qw_bytes = cache_size * block_size  # Max queue size
    qr_bytes = cache_size * block_size
    
    total_bytes = ow_bytes + or_bytes + ds_bytes + qw_bytes + qr_bytes
    
    return {
        'ow_bytes': ow_bytes,
        'or_bytes': or_bytes,
        'ds_bytes': ds_bytes,
        'queue_bytes': qw_bytes + qr_bytes,
        'total_bytes': total_bytes
    }


def calc_server_storage_top_down(proto, key_size: int, value_size: int, num_data: int, cache_size: int) -> dict:
    """Calculate server storage for Top-Down SOMAP.
    
    Server stores:
    - O_W: ORAM tree for write operations (size = cache_size)
    - O_R: ORAM tree for read operations (size = cache_size)
    - O_B: B+ tree stored in ORAM
    - Q_W, Q_R: Queues
    """
    block_size = calc_block_size(key_size, value_size)
    bucket_size = 4
    
    import math
    # O_W and O_R
    ow_level = int(math.ceil(math.log(cache_size + 1, 2))) + 1
    ow_num_buckets = pow(2, ow_level) - 1
    ow_bytes = ow_num_buckets * bucket_size * block_size
    
    or_level = int(math.ceil(math.log(cache_size + 1, 2))) + 1
    or_num_buckets = pow(2, or_level) - 1
    or_bytes = or_num_buckets * bucket_size * block_size
    
    # O_B: B+ tree ORAM (num_data entries)
    ob_level = int(math.ceil(math.log(num_data + 1, 2))) + 1
    ob_num_buckets = pow(2, ob_level) - 1
    ob_bytes = ob_num_buckets * bucket_size * block_size
    
    # Q_W, Q_R: queue storage
    qw_bytes = cache_size * block_size
    qr_bytes = cache_size * block_size
    
    total_bytes = ow_bytes + or_bytes + ob_bytes + qw_bytes + qr_bytes
    
    return {
        'ow_bytes': ow_bytes,
        'or_bytes': or_bytes,
        'ob_bytes': ob_bytes,
        'queue_bytes': qw_bytes + qr_bytes,
        'total_bytes': total_bytes
    }


def calc_server_storage_baseline(omap, key_size: int, value_size: int, num_data: int) -> dict:
    """Calculate server storage for Baseline BPlus OMAP.
    
    Server stores:
    - Single ORAM tree (size = num_data)
    """
    block_size = calc_block_size(key_size, value_size)
    bucket_size = 4
    
    import math
    level = int(math.ceil(math.log(num_data + 1, 2))) + 1
    num_buckets = pow(2, level) - 1
    oram_bytes = num_buckets * bucket_size * block_size
    
    total_bytes = oram_bytes
    
    return {
        'oram_bytes': oram_bytes,
        'total_bytes': total_bytes
    }


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
                  key_size: int = 16, value_size: int = 16, mode: str = "mix",
                  load_storage: str = None, latency_ms: float = 0.0,
                  simulate_init: bool = False, sim_crypto_mbps: float = 200.0,
                  force_reset_caches: bool = False):
    """Run Bottom-Up protocol benchmark against remote server."""
    client = InteractRemoteServer(ip=server_ip, port=port)
    client.init_connection()
    
    enc_block_size = calc_block_size(key_size, value_size, encrypted=True)
    counter = RoundCounter(
        client,
        sim_latency_ms=latency_ms,
        simulate_padding=simulate_init,
        sim_bucket_size=None,
        sim_block_size=enc_block_size,
        sim_crypto_mbps=sim_crypto_mbps,
    )
    counter.wrap()
    
    proto = BottomUpSomapFixedCache(
        num_data=num_data, cache_size=cache_size, data_size=data_size,
        client=client, use_encryption=not simulate_init, aes_key=b'0'*16, order=order,
        num_key_bytes=key_size
    )
    if simulate_init:
        counter.sim_bucket_size = proto._bucket_size
    
    if load_storage:
        # Load one or multiple files
        files = [f.strip() for f in load_storage.split(',')]
        print(f"  Loading pre-built storage from server file(s): {files}")
        setup_start = time.time()
        for i, filename in enumerate(files):
            # For the first file, we might assume it replaces storage.
            # But our updated server load_storage always UPDATES/MERGES.
            # So if we want clean slate, we depend on server restart or user logic.
            # Usually the first file is the big D_S base.
            print(f"    - Loading {filename} ...")
            client.load_storage(filename)
        
        # BottomUpSomap structure:
        # - D_S (StaticOram): stateless on client (except params)
        # - O_W, O_R: initially empty
        # - Q_W, Q_R: initially empty
        # So we just need to ensure the server side is loaded. 
        # CAUTION: The server file must have been built with SAME num_data/value_size
        
        # NOTE: If we loaded cache structures from file (target=cache_only),
        # we don't need 'force_reset_caches=True' (which wipes them).
        # We assume the user is smart: 
        #   If they loaded a 'cache config' file, they want THAT config.
        #   So we just do local hydration.
        # However, we must ensure the local params (cache_size) match what was loaded.
        
        # Auto-detect ds_only: if file ends with _ds.pkl, force reset caches
        auto_force_reset = any('_ds.pkl' in f for f in files) or force_reset_caches
        proto.restore_client_state(force_reset_caches=auto_force_reset)
        if auto_force_reset:
            print(f"  [Note] Detected ds_only file or --force-reset-caches; initializing empty caches on server.")
        print(f"  Storage loaded in {time.time() - setup_start:.2f}s")
        
        # When loading a pre-built storage which contains a FULL structure,
        # but we want to simulate starting with EMPTY cache (O_W/O_R) but FULL D_S.
        # Actually prebuild_server.py generates a structure with D_S full and O_W/O_R empty.
        # So loading it brings us exactly to the state "after setup() completes".
        # This is perfect.
    else:
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
    
    # Calculate detailed client storage
    client_storage = calc_client_storage_bottom_up(proto, key_size, value_size)
    
    # Calculate server storage
    server_storage = calc_server_storage_bottom_up(proto, key_size, value_size, num_data, cache_size)
    
    client.close_connection()
    
    sim_real_time = calc_sim_real_crypto_time(counter, sim_crypto_mbps) if simulate_init else 0.0
    return {
        'rounds': counter.rounds,
        'bytes_sent': counter.bytes_sent,
        'bytes_recv': counter.bytes_recv,
        'elapsed_sec': elapsed + counter.sim_proc_time + sim_real_time,
        'comm_time': counter.comm_time,
        'success': success,
        'read_ops': counter.total_read_ops,
        'write_ops': counter.total_write_ops,
        'read_buckets': counter.total_read_buckets,
        'write_buckets': counter.total_write_buckets,
        'max_client_size': proto._peak_client_size,
        'client_storage': client_storage,
        'server_storage': server_storage
    }


def run_top_down(server_ip: str, port: int, num_data: int, cache_size: int,
                 data_size: int, keys: List[int], ops: List[str], order: int = 4,
                 key_size: int = 16, value_size: int = 16, mode: str = "mix",
                 load_storage: str = None, latency_ms: float = 0.0,
                 simulate_init: bool = False, sim_crypto_mbps: float = 200.0,
                 force_reset_caches: bool = False):
    """Run Top-Down protocol benchmark against remote server."""
    client = InteractRemoteServer(ip=server_ip, port=port)
    client.init_connection()
    
    enc_block_size = calc_block_size(key_size, value_size, encrypted=True)
    counter = RoundCounter(
        client,
        sim_latency_ms=latency_ms,
        simulate_padding=simulate_init,
        sim_bucket_size=None,
        sim_block_size=enc_block_size,
        sim_crypto_mbps=sim_crypto_mbps,
    )
    counter.wrap()
    
    proto = TopDownSomapFixedCache(
        num_data=num_data, cache_size=cache_size, data_size=data_size,
        client=client, use_encryption=not simulate_init, aes_key=b'0'*16, order=order,
        num_key_bytes=key_size, key_size=key_size
    )
    if simulate_init:
        counter.sim_bucket_size = proto._bucket_size
    
    if load_storage:
        # Load one or multiple files
        files = [f.strip() for f in load_storage.split(',')]
        print(f"  Loading pre-built storage from server file(s): {files}")
        setup_start = time.time()
        for i, filename in enumerate(files):
            print(f"    - Loading {filename} ...")
            client.load_storage(filename)
        
        # Determine if we need to force reset caches based on what we loaded.
        # If we loaded a 'cache' file, we assume it matches the desired config (cache_size).
        # But if we just loaded a 'base' file (ds_only) and want to use a specific cache size, 
        # we might need to reset.
        # For simplicity, similar to Bottom-Up, if the user explicitly provided cache files that match params,
        # they probably don't need reset.
        # But to be safe, let's allow force_reset via restore_client_state logic if needed.
        # Here we assume if they load files, they want to use them. 
        # But wait, restore_client_state also initializes local objects.
        
        # Logic: Always call restore. If user wants fresh cache (different from file or file is empty cache),
        # they might need a way to signal.
        # Currently we follow Bottom-Up logic: Load what is given. 
        # But note: Top-Down "forced reset" requires sending an init with empty structures.
        # If we just loaded "cache_1023.pkl", we do NOT want to reset it.
        # If we loaded "base.pkl" only, we DO want to reset/init cache to empty.
        
        # Heuristic: If filename contains "_ds.pkl", assume we loaded ds_only, so reset caches.
        # Also respect explicit --force-reset-caches flag.
        auto_force_reset = any('_ds.pkl' in f for f in files) or force_reset_caches
        
        proto.restore_client_state(force_reset_caches=auto_force_reset)
        if auto_force_reset:
            print(f"  [Note] Detected ds_only file or --force-reset-caches; initializing empty caches on server.")
        print(f"  Storage loaded in {time.time() - setup_start:.2f}s")

    else:
        print("  Uploading initial data to server...")
        setup_start = time.time()
        # Note: top down setup expects list of (key, value)
        # But keys are ints in our benchmark. 
        # proto.setup converts them? 
        # Looking at TopDownSomap.setup: data_map = Helper.hash_data_to_map... data is key, value
        # Existing code used str(k). Let's stick to int k to match BottomUp if possible?
        # No, existing call was: proto.setup([(str(k), make_value(k, value_size)) for k in range(num_data)])
        # Wait, if we use str(k), the key type is str.
        # TopDownSomapFixedCache uses: group_index = Helper.hash_data_to_leaf(..., data=key, ...)
        # If key is int, Prf handles it? data=key.
        # Prf.digest takes bytes. Helper.hash_data_to_leaf: data can be anything dumpable?
        # Let's check Helper.
        # For now, keep existing behavior: strictly follow what was there.
        # But BottomUp uses int keys. Using str keys might make Prf results different if not consistent.
        # But let's respect existing: keys: List[int]. But setup passed str(k).
        # This implies keys in 'ops' loop (zip(keys, ops)) which are ints from zipf_keys 
        # might mismatch if setup used str keys?
        # Yes, big risk.
        # Let's check `zipf_keys` return type. It returns ints.
        # If setup uses str(k), and access uses int k, TopDown won't find items!
        # Let's fix this standardization to INT while we are here, or carefully check.
        # Actually TopDownSomap setup (line 527) in my read was: 
        #   proto.setup([(str(k), make_value(k, value_size)) for k in range(num_data)])
        # If I change it, I might break it if underlying expects str.
        # BottomUp uses int.
        # Let's invoke access with int.
        
        # Recommendation: Use int for both to be consistent. 
        # I will change setup to use int k.
        proto.setup([(k, make_value(k, value_size)) for k in range(num_data)])
        
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
                # access(op, key, value) for TopDown
                proto.access('insert', new_key, value=make_value(new_key, value_size))
                success += 1
            except Exception as e:
                print(f"\n  Error on op {i}: {e}")
    else:
        for i, (key, op) in enumerate(zip(keys, ops)):
            if i % 10 == 0:
                sys.stdout.write(f"\r  Progress: {i}/{len(keys)}")
                sys.stdout.flush()
            try:
                actual_op = 'search' if op == 'read' else 'insert'
                val = make_value(key, value_size) if actual_op == 'insert' else None
                # access(op, key, value) for TopDown
                proto.access(actual_op, key, val)
                success += 1
            except Exception as e:
                print(f"\n  Error on op {i}: {e}")
            
    print(f"\r  Progress: {len(keys)}/{len(keys)} - Done!")
    
    elapsed = time.time() - start
    
    # Calculate detailed client storage
    client_storage = calc_client_storage_top_down(proto, key_size, value_size)
    
    # Calculate server storage
    server_storage = calc_server_storage_top_down(proto, key_size, value_size, num_data, cache_size)
    
    client.close_connection()
    
    sim_real_time = calc_sim_real_crypto_time(counter, sim_crypto_mbps) if simulate_init else 0.0
    return {
        'rounds': counter.rounds,
        'bytes_sent': counter.bytes_sent,
        'bytes_recv': counter.bytes_recv,
        'elapsed_sec': elapsed + counter.sim_proc_time + sim_real_time,
        'comm_time': counter.comm_time,
        'success': success,
        'max_client_size': proto._peak_client_size,
        'client_storage': client_storage,
        'server_storage': server_storage
    }


def run_baseline(server_ip: str, port: int, num_data: int, data_size: int,
                 keys: List[int], ops: List[str], order: int = 4,
                 key_size: int = 16, value_size: int = 16, mode: str = "mix",
                 load_storage: str = None, latency_ms: float = 0.0,
                 simulate_init: bool = False, sim_crypto_mbps: float = 200.0):
    """Run Baseline BPlus OMAP benchmark against remote server."""
    client = InteractRemoteServer(ip=server_ip, port=port)
    client.init_connection()
    
    enc_block_size = calc_block_size(key_size, value_size, encrypted=True)
    counter = RoundCounter(
        client,
        sim_latency_ms=latency_ms,
        simulate_padding=simulate_init,
        sim_bucket_size=None,
        sim_block_size=enc_block_size,
        sim_crypto_mbps=sim_crypto_mbps,
    )
    counter.wrap()
    
    omap = BPlusOdsOmap(
        order=order, num_data=num_data, key_size=key_size, data_size=data_size,
        client=client, name="baseline", use_encryption=not simulate_init, aes_key=b'0'*16,
        num_key_bytes=key_size
    )
    if simulate_init:
        counter.sim_bucket_size = omap._bucket_size
    
    if load_storage:
        print(f"  Loading storage from server: {load_storage} (skipping data upload)...")
        t_start = time.time()
        client.load_storage(load_storage)
        # IMPORTANT: Restore the B+ Tree Root pointer from the loaded metadata
        omap.restore_client_state()
        print(f"  Storage loaded & Client State restored in {time.time() - t_start:.2f}s")
    else:
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
    
    # Calculate detailed client storage
    client_storage = calc_client_storage_baseline(omap, key_size, value_size)
    
    # Calculate server storage
    server_storage = calc_server_storage_baseline(omap, key_size, value_size, num_data)
    
    client.close_connection()
    
    sim_real_time = calc_sim_real_crypto_time(counter, sim_crypto_mbps) if simulate_init else 0.0
    return {
        'rounds': counter.rounds,
        'bytes_sent': counter.bytes_sent,
        'bytes_recv': counter.bytes_recv,
        'elapsed_sec': elapsed + counter.sim_proc_time + sim_real_time,
        'comm_time': counter.comm_time,
        'success': success,
        'max_client_size': omap._peak_client_size,
        'client_storage': client_storage,
        'server_storage': server_storage
    }


def print_results(name: str, result: dict, num_ops: int):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"Results: {name}")
    print(f"{'='*60}")
    print(f"  Operations:     {result['success']}/{num_ops}")
    print(f"  Total Rounds:   {result['rounds']} ({result['rounds']/num_ops:.2f}/op)")
    
    total_ms = result['elapsed_sec']*1000/num_ops
    comm_ms = result['comm_time']*1000/num_ops
    proc_ms = total_ms - comm_ms
    
    print(f"  Total Time:     {result['elapsed_sec']:.2f}s ({total_ms:.2f}ms/op)")
    print(f"    - Processing: {proc_ms:.2f}ms/op ({(proc_ms/total_ms)*100:.1f}%)")
    print(f"    - Commun.:    {comm_ms:.2f}ms/op ({(comm_ms/total_ms)*100:.1f}%)")
    
    print(f"  Bandwidth Sent: {result['bytes_sent']/1024:.2f} KB ({result['bytes_sent']/1024/num_ops:.2f} KB/op)")
    print(f"  Bandwidth Recv: {result['bytes_recv']/1024:.2f} KB ({result['bytes_recv']/1024/num_ops:.2f} KB/op)")

    if 'read_buckets' in result or 'write_buckets' in result:
        read_ops = result.get('read_ops', 0)
        write_ops = result.get('write_ops', 0)
        read_buckets = result.get('read_buckets', 0)
        write_buckets = result.get('write_buckets', 0)
        read_avg = (read_buckets / read_ops) if read_ops else 0.0
        write_avg = (write_buckets / write_ops) if write_ops else 0.0
        print(f"  Read Ops/Buckets:  {read_ops} ops, {read_buckets} buckets ({read_avg:.2f}/op)")
        print(f"  Write Ops/Buckets: {write_ops} ops, {write_buckets} buckets ({write_avg:.2f}/op)")
    
    # Client storage details
    cs = result.get('client_storage', {})
    if cs:
        print(f"  --- Client Storage ---")
        print(f"  Peak Stash Entries: {result['max_client_size']}")
        print(f"  Current Stash:  {cs.get('stash_entries', 0)} entries ({cs.get('stash_bytes', 0)/1024:.2f} KB)")
        print(f"  Metadata:       {cs.get('metadata_bytes', 0)/1024:.2f} KB")
        print(f"  Total Client:   {cs.get('total_bytes', 0)/1024:.2f} KB")
    else:
        print(f"  Max Client Stash: {result['max_client_size']} entries")
    
    # Server storage details
    ss = result.get('server_storage', {})
    if ss:
        print(f"  --- Server Storage ---")
        print(f"  Total Server:   {ss.get('total_bytes', 0)/1024/1024:.2f} MB")


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
    parser.add_argument("--simulate-init", action="store_true",
                        help="Simulate init/padding: disable encryption/padding and add simulated cost")
    parser.add_argument("--sim-crypto-mbps", type=float, default=200.0,
                        help="Simulated crypto throughput in MB/s (default: 200)")
    parser.add_argument("--load-storage", type=str, default=None, help="Path to pre-built storage on server to load")
    parser.add_argument("--latency", type=float, default=0.0, help="Simulated network latency (one-way) in ms")
    parser.add_argument("--force-reset-caches", action="store_true",
                        help="Force reset caches (O_W/O_R/Q_W/Q_R) after loading storage. Required when loading ds_only files.")
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
    
    # Load storage mode check
    if args.load_storage and args.protocol not in ['bottom_up', 'top_down', 'all']:
        print("[Warning] --load-storage is optimized for bottom_up and top_down. Other protocols may need fresh init.")

    keys = zipf_keys(num_data, total_ops, alpha=0.8)
    ops_list = ['read' if random.random() < args.read_ratio else 'write' for _ in range(total_ops)]

    # Calculate block sizes
    raw_block_size = calc_block_size(key_size, value_size, encrypted=False)
    enc_block_size = calc_block_size(key_size, value_size, encrypted=True)

    print("="*60)
    print("Remote Client-Server Benchmark")
    print("="*60)
    print(f"  Server:     {args.server_ip}:{args.port}")
    print(f"  N:          {num_data}")
    print(f"  Cache:      {cache_size}")
    print(f"  Order:      {args.order}")
    print(f"  Key Size:   {key_size} bytes")
    print(f"  Value Size: {value_size} bytes")
    print(f"  Block Size: {raw_block_size} bytes (raw), {enc_block_size} bytes (encrypted)")
    print(f"  Mode:       {args.mode}")
    print(f"  Operations: {total_ops}")
    if args.simulate_init:
        print(f"  Sim Init:   ON (crypto {args.sim_crypto_mbps} MB/s)")
    if args.mode == "mix":
        print(f"  Read Ratio: {args.read_ratio}")
    print("="*60)

    results = {}

    if args.protocol in ["all", "bottom_up"]:
        print("\n[Bottom-Up SOMAP]")
        results['bottom_up'] = run_bottom_up(
            args.server_ip, args.port, num_data, cache_size, 
            data_size, keys, ops_list, order=args.order,
            key_size=key_size, value_size=value_size, mode=args.mode,
            load_storage=args.load_storage, latency_ms=args.latency,
            simulate_init=args.simulate_init, sim_crypto_mbps=args.sim_crypto_mbps,
            force_reset_caches=args.force_reset_caches
        )
        print_results("Bottom-Up", results['bottom_up'], total_ops)
        time.sleep(1)

    if args.protocol in ["all", "top_down"]:
        print("\n[Top-Down SOMAP]")
        results['top_down'] = run_top_down(
            args.server_ip, args.port, num_data, cache_size,
            data_size, keys, ops_list, order=args.order,
            key_size=key_size, value_size=value_size, mode=args.mode,
            load_storage=args.load_storage, latency_ms=args.latency,
            simulate_init=args.simulate_init, sim_crypto_mbps=args.sim_crypto_mbps,
            force_reset_caches=args.force_reset_caches
        )
        print_results("Top-Down", results['top_down'], total_ops)
        time.sleep(1)

    if args.protocol in ["all", "baseline"]:
        print("\n[Baseline BPlus OMAP]")
        results['baseline'] = run_baseline(
            args.server_ip, args.port, num_data, data_size,
            keys, ops_list, order=args.order,
            key_size=key_size, value_size=value_size, mode=args.mode,
            load_storage=args.load_storage, latency_ms=args.latency,
            simulate_init=args.simulate_init, sim_crypto_mbps=args.sim_crypto_mbps
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
