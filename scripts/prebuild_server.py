"""
Script to pre-build server storage for SOMAP/ORAM and save to disk.

Usage:
    python scripts/prebuild_server.py --num-data 262143 --value-size 4096 --output omap_N18_V4096.pkl
"""

import os
import sys
import pickle
import argparse
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from daoram.dependency import crypto
from daoram.dependency.interact_server import InteractLocalServer
from daoram.so.bottom_to_up_somap_fixed_cache import BottomUpSomapFixedCache
from daoram.oram.static_oram import StaticOram

def make_value(value_size: int = 16) -> bytes:
    return bytes(value_size)

def main():
    parser = argparse.ArgumentParser(description="Pre-build server storage")
    parser.add_argument("--num-data", type=int, required=True, help="Number of data entries")
    parser.add_argument("--value-size", type=int, default=4096, help="Value size in bytes")
    parser.add_argument("--order", type=int, default=4, help="Tree order")
    parser.add_argument("--key-size", type=int, default=16, help="Key size")
    parser.add_argument("--cache-size", type=int, default=1023, help="Cache size (for structure initialization only)")
    parser.add_argument("--output", type=str, required=True, help="Output pickle filename")
    parser.add_argument("--mock-crypto", action="store_true", help="Use mock encryption")
    parser.add_argument("--target", type=str, default="full", choices=["full", "ds_only", "cache_only"],
                        help="What components to save: full (all), ds_only (StaticORAM), cache_only (OW/OR/Queues)")
    parser.add_argument("--protocol", type=str, default="bottom_up", choices=["bottom_up", "top_down", "baseline"],
                        help="Protocol to build storage for")
    
    args = parser.parse_args()

    if args.mock_crypto:
        print("[Mock Encryption Enabled]")
        crypto.MOCK_ENCRYPTION = True

    print(f"Pre-building storage: N={args.num_data}, V={args.value_size}, Target={args.target} -> {args.output}")
    
    # We use InteractLocalServer to capture the storage in memory
    server_side = InteractLocalServer()
    
    data_size = args.value_size
    num_data = args.num_data
    
    # Initialize initial data
    initial_data = {i: make_value(data_size) for i in range(num_data)}
    
    # 1. Initialize Protocol Structure
    print(f"Initializing {args.protocol} (generating trees)...")
    start = time.time()
    
    if args.protocol == "bottom_up":
        proto = BottomUpSomapFixedCache(
            num_data=num_data, 
            cache_size=args.cache_size, 
            data_size=data_size,
            client=server_side, 
            use_encryption=True, 
            aes_key=b'0'*16, 
            order=args.order,
            num_key_bytes=args.key_size
        )
        # Setup triggers the heavy lifting
        proto.setup(initial_data)
        
    elif args.protocol == "top_down":
        from daoram.so.top_down_somap_fixed_cache import TopDownSomapFixedCache
        proto = TopDownSomapFixedCache(
            num_data=num_data,
            cache_size=args.cache_size,
            data_size=data_size,
            client=server_side,
            use_encryption=True,
            aes_key=b'0'*16,
            key_size=args.key_size,
            order=args.order,
            num_key_bytes=args.key_size
        )
        proto.setup([(k, v) for k, v in initial_data.items()])
        
    elif args.protocol == "baseline":
        from daoram.omap.bplus_ods_omap import BPlusOdsOmap
        # Baseline BPlus OMAP
        proto = BPlusOdsOmap(
            order=args.order, num_data=num_data, key_size=args.key_size, data_size=data_size,
            client=server_side, name="baseline", use_encryption=True, aes_key=b'0'*16,
            num_key_bytes=args.key_size
        )
        # Manually init storage and upload
        storage = proto._init_ods_storage([(k, v) for k, v in initial_data.items()])
        server_side.init({proto._name: storage})
        
    print(f"Build complete in {time.time() - start:.2f}s")
    
    # --- Metadata Injection (Roots) ---
    # We inject the Root (ID, Path) into storage so client can restore it.
    storage_dict = server_side._InteractLocalServer__storage
    
    def save_root(obj, name_suffix=""):
        if hasattr(obj, 'root') and obj.root is not None:
             key = f"{obj._name}_root" 
             storage_dict[key] = obj.root
             print(f"  Saved metadata: {key} -> {obj.root}")

    if args.protocol == "bottom_up":
        save_root(proto._Ow)
        save_root(proto._Or)
    elif args.protocol == "top_down":
        save_root(proto._Ow)
        save_root(proto._Or)
        save_root(proto._Ob)
    elif args.protocol == "baseline":
        save_root(proto)

    # Filter storage based on target
    filtered_storage = {}
    if args.target == "full":
        filtered_storage = storage_dict
    else:
        for k, v in storage_dict.items():
            if args.protocol == "bottom_up":
                if "_D_S" in k:
                    if args.target == "ds_only":
                        filtered_storage[k] = v
                elif "_O_W" in k or "_O_R" in k or "_Q_W" in k or "_Q_R" in k:
                     if args.target == "cache_only":
                        filtered_storage[k] = v
            elif args.protocol == "top_down":
                if "_Tree" in k or k == "DB":
                    if args.target == "ds_only": # Reuse ds_only for Base (Tree+DB)
                        filtered_storage[k] = v
                elif "_O_W" in k or "_O_R" in k or "_O_B" in k or "_Q_W" in k or "_Q_R" in k:
                    if args.target == "cache_only":
                        filtered_storage[k] = v
            elif args.protocol == "baseline":
                 filtered_storage[k] = v

    if not filtered_storage:
        print("[Warning] No components found matching the target! Stored keys:", storage_dict.keys())

    print(f"Saving to {args.output} ...")
    with open(args.output, 'wb') as f:
        pickle.dump(filtered_storage, f)
    
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Done. File size: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
