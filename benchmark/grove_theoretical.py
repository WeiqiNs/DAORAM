"""
Theoretical bandwidth calculator for Grove operations.

This script calculates theoretical bandwidth without creating the actual database,
allowing analysis of very large configurations (millions/billions of vertices).

It also supports a sparse benchmark mode that only stores accessed buckets,
enabling actual timing measurements with encryption for large configurations.

Usage:
    # Calculate theoretical bandwidth for 1M vertices with 4KB data
    python benchmark/grove_theoretical.py --num_data 1000000 --data_size 4096

    # Calculate for different max_deg values
    python benchmark/grove_theoretical.py --num_data 1000000 --max_deg 10

    # Run sparse benchmark with encryption (actual timing, simulated storage)
    python benchmark/grove_theoretical.py --num_data 1000000 --data_size 4096 --sparse --num_ops 20

    # Compare multiple configurations
    python benchmark/grove_theoretical.py --compare

Parameters:
    --num_data: Number of vertices (default: 1024)
    --max_deg: Maximum degree per vertex (default: 5)
    --data_size: Size of vertex data in bytes (default: 64)
    --bucket_size: ORAM bucket size (default: 4)
    --num_ops: Number of operations for sparse benchmark (default: 0, theoretical only)
    --sparse: Enable sparse benchmark mode for actual timing
    --encryption: Enable encryption in sparse mode
    --compare: Run comparison across multiple configurations
"""

import argparse
import math
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TheoreticalBandwidth:
    """Theoretical bandwidth for an operation."""
    operation: str
    paths_read: int
    paths_written: int
    buckets_per_path: int
    bucket_size: int
    block_size: int

    @property
    def bytes_read(self) -> int:
        return self.paths_read * self.buckets_per_path * self.bucket_size * self.block_size

    @property
    def bytes_written(self) -> int:
        return self.paths_written * self.buckets_per_path * self.bucket_size * self.block_size

    @property
    def total_bytes(self) -> int:
        return self.bytes_read + self.bytes_written

    @property
    def total_kb(self) -> float:
        return self.total_bytes / 1024

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)


@dataclass
class GroveConfig:
    """Configuration for Grove bandwidth calculation."""
    num_data: int
    max_deg: int
    data_size: int
    bucket_size: int = 4
    key_size: int = 16

    @property
    def tree_level(self) -> int:
        """Number of levels in the ORAM tree."""
        return int(math.ceil(math.log2(self.num_data))) + 1

    @property
    def leaf_range(self) -> int:
        """Number of leaves in the tree."""
        return pow(2, self.tree_level - 1)

    @property
    def block_size(self) -> int:
        """Estimated size of each block after serialization."""
        # Block contains: key, leaf, value (vertex_data, adjacency_dict, pos_leaf)
        # Adjacency dict: {neighbor_key: graph_leaf} with up to max_deg entries
        adjacency_overhead = self.max_deg * (self.key_size + 8)  # key + int
        # pos_leaf is an int (8 bytes)
        # Some pickle overhead
        pickle_overhead = 100
        return self.data_size + adjacency_overhead + pickle_overhead

    @property
    def meta_bucket_size(self) -> int:
        """Bucket size for meta ORAMs (from theoretical bound calculation)."""
        # Simplified version - in practice this is computed by find_bound()
        # Using approximation based on typical values
        return max(10, int(math.log2(self.num_data) * 2))


class TheoreticalCalculator:
    """Calculate theoretical bandwidth for Grove operations."""

    def __init__(self, config: GroveConfig):
        self.config = config

    def lookup(self) -> TheoreticalBandwidth:
        """
        Lookup operation:
        - PosMap ORAM: ~log(n) paths for AVL traversal
        - Graph ORAM: 1 path
        - Graph Meta: max_deg + 1 paths (RL paths + vertex path)
        """
        posmap_paths = self.config.tree_level  # AVL traversal depth
        graph_paths = 1
        meta_paths = self.config.max_deg + 1

        total_read = posmap_paths + graph_paths + meta_paths
        total_write = total_read  # Write back same paths

        return TheoreticalBandwidth(
            operation="lookup",
            paths_read=total_read,
            paths_written=total_write,
            buckets_per_path=self.config.tree_level,
            bucket_size=self.config.bucket_size,
            block_size=self.config.block_size,
        )

    def insert(self) -> TheoreticalBandwidth:
        """
        Insert operation:
        - PosMap ORAM: ~log(n) paths for AVL insert
        - PosMap ORAM: ~log(n) * max_deg paths for neighbor lookup
        - Graph ORAM: max_deg + 1 paths
        - Graph Meta: max_deg^2 paths
        """
        posmap_insert = self.config.tree_level
        posmap_lookup = self.config.tree_level * self.config.max_deg
        graph_paths = self.config.max_deg + 1
        meta_paths = self.config.max_deg * self.config.max_deg

        total_read = posmap_insert + posmap_lookup + graph_paths + meta_paths
        total_write = total_read

        return TheoreticalBandwidth(
            operation="insert",
            paths_read=total_read,
            paths_written=total_write,
            buckets_per_path=self.config.tree_level,
            bucket_size=self.config.bucket_size,
            block_size=self.config.block_size,
        )

    def delete(self) -> TheoreticalBandwidth:
        """
        Delete operation:
        - PosMap ORAM: ~log(n) paths for delete
        - Graph ORAM: max_deg paths (oblivious reads)
        - Graph Meta: max_deg paths
        """
        posmap_paths = self.config.tree_level
        graph_paths = self.config.max_deg
        meta_paths = self.config.max_deg

        total_read = posmap_paths + graph_paths + meta_paths
        total_write = total_read

        return TheoreticalBandwidth(
            operation="delete",
            paths_read=total_read,
            paths_written=total_write,
            buckets_per_path=self.config.tree_level,
            bucket_size=self.config.bucket_size,
            block_size=self.config.block_size,
        )

    def neighbor(self) -> TheoreticalBandwidth:
        """
        Neighbor query:
        - lookup() for center vertex
        - Graph ORAM: max_deg paths for neighbors
        - Graph Meta: max_deg^2 paths
        """
        lookup_bw = self.lookup()

        graph_paths = self.config.max_deg
        meta_paths = self.config.max_deg * self.config.max_deg

        additional_read = graph_paths + meta_paths
        additional_write = additional_read

        return TheoreticalBandwidth(
            operation="neighbor",
            paths_read=lookup_bw.paths_read + additional_read,
            paths_written=lookup_bw.paths_written + additional_write,
            buckets_per_path=self.config.tree_level,
            bucket_size=self.config.bucket_size,
            block_size=self.config.block_size,
        )

    def all_operations(self) -> Dict[str, TheoreticalBandwidth]:
        """Calculate bandwidth for all operations."""
        return {
            "lookup": self.lookup(),
            "insert": self.insert(),
            "delete": self.delete(),
            "neighbor": self.neighbor(),
        }


class SparseStorage:
    """Sparse storage that only stores accessed buckets."""

    def __init__(self, bucket_size: int, block_size: int, encryption: bool = False):
        self.bucket_size = bucket_size
        self.block_size = block_size
        self.encryption = encryption
        self._data: Dict[int, List[bytes]] = {}
        self._encryptor = None

        if encryption:
            from daoram.dependency.crypto import AesGcm
            self._encryptor = AesGcm()

    def read_row(self, index: int) -> List[bytes]:
        """Read a bucket, returning empty list if not stored."""
        if index in self._data:
            data = self._data[index]
            if self.encryption and self._encryptor:
                return [self._encryptor.dec(block) for block in data]
            return data
        # Return dummy data for unstored buckets
        return [b'\x00' * self.block_size for _ in range(self.bucket_size)]

    def write_row(self, index: int, data: List[bytes]) -> None:
        """Write a bucket."""
        if self.encryption and self._encryptor:
            data = [self._encryptor.enc(block) for block in data]
        self._data[index] = data

    def get_bandwidth(self, indices: List[int]) -> int:
        """Calculate bandwidth for accessing given indices."""
        return len(indices) * self.bucket_size * self.block_size


class SparseBenchmark:
    """Benchmark using sparse storage for large configurations."""

    def __init__(self, config: GroveConfig, encryption: bool = False):
        self.config = config
        self.encryption = encryption
        self.calculator = TheoreticalCalculator(config)

    def simulate_operation(self, op_name: str, num_ops: int = 1) -> Dict[str, float]:
        """
        Simulate operation timing with sparse storage.

        This measures:
        - Encryption/decryption time (if enabled)
        - Serialization overhead
        - Theoretical bandwidth
        """
        bw = getattr(self.calculator, op_name)()

        # Create sparse storage
        storage = SparseStorage(
            bucket_size=self.config.bucket_size,
            block_size=self.config.block_size,
            encryption=self.encryption,
        )

        # Simulate reads and writes
        times = []
        for _ in range(num_ops):
            start = time.perf_counter()

            # Simulate path reads
            for i in range(bw.paths_read):
                path_indices = list(range(self.config.tree_level))
                for idx in path_indices:
                    data = storage.read_row(idx + i * self.config.tree_level)

            # Simulate path writes
            dummy_block = b'x' * self.config.block_size
            for i in range(bw.paths_written):
                path_indices = list(range(self.config.tree_level))
                for idx in path_indices:
                    storage.write_row(
                        idx + i * self.config.tree_level,
                        [dummy_block] * self.config.bucket_size
                    )

            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = sum(times) / len(times) if times else 0

        return {
            "operation": op_name,
            "avg_time_ms": avg_time,
            "bandwidth_kb": bw.total_kb,
            "bandwidth_mb": bw.total_mb,
            "paths_read": bw.paths_read,
            "paths_written": bw.paths_written,
        }


def print_theoretical(config: GroveConfig) -> None:
    """Print theoretical bandwidth analysis."""
    calc = TheoreticalCalculator(config)
    results = calc.all_operations()

    print("\n" + "=" * 80)
    print("GROVE THEORETICAL BANDWIDTH ANALYSIS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  num_data: {config.num_data:,} vertices")
    print(f"  max_deg: {config.max_deg}")
    print(f"  data_size: {config.data_size} bytes ({config.data_size/1024:.1f} KB)")
    print(f"  bucket_size: {config.bucket_size}")
    print(f"  tree_level: {config.tree_level}")
    print(f"  leaf_range: {config.leaf_range:,}")
    print(f"  est_block_size: {config.block_size} bytes")

    print("\n" + "-" * 80)
    print(f"{'Operation':<12} {'Paths R':>10} {'Paths W':>10} {'BW Read':>12} {'BW Write':>12} {'Total':>12}")
    print("-" * 80)

    for name, bw in results.items():
        print(f"{name:<12} {bw.paths_read:>10} {bw.paths_written:>10} "
              f"{bw.bytes_read/1024:>10.1f} KB {bw.bytes_written/1024:>10.1f} KB "
              f"{bw.total_kb:>10.1f} KB")

    print("-" * 80)

    # Show scaling info
    print(f"\nScaling Analysis:")
    print(f"  Tree depth scales as: O(log n) = {config.tree_level} levels for n={config.num_data:,}")
    print(f"  Lookup bandwidth: O(log n * max_deg * bucket_size * block_size)")
    print(f"  Insert bandwidth: O(log n * max_deg^2 * bucket_size * block_size)")


def print_sparse_benchmark(config: GroveConfig, num_ops: int, encryption: bool) -> None:
    """Run and print sparse benchmark results."""
    benchmark = SparseBenchmark(config, encryption=encryption)

    print("\n" + "=" * 80)
    print("GROVE SPARSE BENCHMARK (Simulated Storage)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  num_data: {config.num_data:,} vertices")
    print(f"  max_deg: {config.max_deg}")
    print(f"  data_size: {config.data_size} bytes ({config.data_size/1024:.1f} KB)")
    print(f"  encryption: {encryption}")
    print(f"  num_ops: {num_ops}")

    print("\n" + "-" * 80)
    print(f"{'Operation':<12} {'Avg Time (ms)':>14} {'Bandwidth (KB)':>16} {'Bandwidth (MB)':>16}")
    print("-" * 80)

    for op in ["lookup", "insert", "delete", "neighbor"]:
        result = benchmark.simulate_operation(op, num_ops)
        print(f"{result['operation']:<12} {result['avg_time_ms']:>14.3f} "
              f"{result['bandwidth_kb']:>16.2f} {result['bandwidth_mb']:>16.4f}")

    print("-" * 80)


def print_comparison() -> None:
    """Print comparison across multiple configurations."""
    configs = [
        GroveConfig(num_data=1024, max_deg=5, data_size=64),
        GroveConfig(num_data=1024, max_deg=5, data_size=4096),
        GroveConfig(num_data=10000, max_deg=5, data_size=4096),
        GroveConfig(num_data=100000, max_deg=5, data_size=4096),
        GroveConfig(num_data=1000000, max_deg=5, data_size=4096),
        GroveConfig(num_data=1000000, max_deg=10, data_size=4096),
        GroveConfig(num_data=10000000, max_deg=5, data_size=4096),
    ]

    print("\n" + "=" * 100)
    print("GROVE BANDWIDTH COMPARISON")
    print("=" * 100)
    print(f"\n{'num_data':>12} {'max_deg':>8} {'data_size':>10} {'tree_lvl':>9} "
          f"{'lookup':>12} {'insert':>12} {'neighbor':>12}")
    print("-" * 100)

    for config in configs:
        calc = TheoreticalCalculator(config)
        lookup_bw = calc.lookup()
        insert_bw = calc.insert()
        neighbor_bw = calc.neighbor()

        print(f"{config.num_data:>12,} {config.max_deg:>8} {config.data_size:>10} "
              f"{config.tree_level:>9} "
              f"{lookup_bw.total_kb:>10.1f} KB "
              f"{insert_bw.total_kb:>10.1f} KB "
              f"{neighbor_bw.total_kb:>10.1f} KB")

    print("-" * 100)


def main():
    parser = argparse.ArgumentParser(description="Grove theoretical bandwidth calculator")
    parser.add_argument("--num_data", type=int, default=1024,
                        help="Number of vertices (default: 1024)")
    parser.add_argument("--max_deg", type=int, default=5,
                        help="Maximum degree per vertex (default: 5)")
    parser.add_argument("--data_size", type=int, default=64,
                        help="Size of vertex data in bytes (default: 64)")
    parser.add_argument("--bucket_size", type=int, default=4,
                        help="ORAM bucket size (default: 4)")
    parser.add_argument("--num_ops", type=int, default=0,
                        help="Number of ops for sparse benchmark (default: 0, theoretical only)")
    parser.add_argument("--sparse", action="store_true",
                        help="Run sparse benchmark with simulated storage")
    parser.add_argument("--encryption", action="store_true",
                        help="Enable encryption in sparse benchmark")
    parser.add_argument("--compare", action="store_true",
                        help="Run comparison across multiple configurations")

    args = parser.parse_args()

    if args.compare:
        print_comparison()
        return

    config = GroveConfig(
        num_data=args.num_data,
        max_deg=args.max_deg,
        data_size=args.data_size,
        bucket_size=args.bucket_size,
    )

    # Always show theoretical analysis
    print_theoretical(config)

    # Optionally run sparse benchmark
    if args.sparse and args.num_ops > 0:
        print_sparse_benchmark(config, args.num_ops, args.encryption)


if __name__ == "__main__":
    main()
