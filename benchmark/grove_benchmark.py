"""
Benchmark for measuring Grove client time and bandwidth.

This script measures:
1. Client-side computation time (excluding simulated network latency)
2. Total bandwidth (bytes read + bytes written via pickle serialization)

Usage:
    # Run with default parameters
    python benchmark/grove_benchmark.py

    # Run with custom parameters
    python benchmark/grove_benchmark.py --num_data 2048 --max_deg 10 --num_ops 50

    # Run specific operations only
    python benchmark/grove_benchmark.py --operations insert lookup delete

    # Output results to CSV
    python benchmark/grove_benchmark.py --output results.csv

    # Run with different graph configurations
    python benchmark/grove_benchmark.py --graph_type star    # Star graph
    python benchmark/grove_benchmark.py --graph_type chain   # Chain graph
    python benchmark/grove_benchmark.py --graph_type random  # Random graph

Parameters:
    --num_data: Number of vertices the Grove can store (default: 1024)
    --max_deg: Maximum degree per vertex (default: 5)
    --data_size: Size of vertex data in bytes (default: 64, use 4096 for 4KB)
    --num_ops: Number of operations to benchmark (default: 100)
    --operations: Operations to benchmark (default: all)
    --graph_type: Type of graph to construct (star, chain, random, complete)
    --output: Output file for results (CSV format)
    --warmup: Number of warmup operations before measurement (default: 10)
    --seed: Random seed for reproducibility (default: None)
"""

import argparse
import csv
import random
import secrets
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from daoram.dependency import InteractLocalServer
from daoram.graph.grove import Grove


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    operation: str
    client_time_ms: float
    bytes_read: int
    bytes_written: int
    total_bandwidth: int

    @property
    def bandwidth_kb(self) -> float:
        return self.total_bandwidth / 1024


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results for an operation type."""
    operation: str
    num_samples: int
    avg_client_time_ms: float
    std_client_time_ms: float
    min_client_time_ms: float
    max_client_time_ms: float
    avg_bandwidth_bytes: float
    std_bandwidth_bytes: float
    avg_bytes_read: float
    avg_bytes_written: float

    @property
    def avg_bandwidth_kb(self) -> float:
        return self.avg_bandwidth_bytes / 1024


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark."""
    num_data: int = 1024
    max_deg: int = 5
    num_ops: int = 100
    warmup: int = 10
    graph_type: str = "random"
    operations: List[str] = field(default_factory=lambda: ["insert", "lookup", "delete", "neighbor"])
    seed: Optional[int] = None
    key_size: int = 16
    data_size: int = 64
    bucket_size: int = 4
    stash_scale: int = 20


class GroveBenchmark:
    """Benchmark runner for Grove operations."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.client = InteractLocalServer()
        self.grove: Optional[Grove] = None
        self.metrics: Dict[str, List[OperationMetrics]] = {
            "insert": [],
            "lookup": [],
            "delete": [],
            "neighbor": [],
            "t_hop": [],
            "t_traversal": [],
        }
        self.inserted_keys: List[int] = []

        if config.seed is not None:
            random.seed(config.seed)

    def _make_vertex_data(self, key: int) -> bytes:
        """Generate vertex data of the configured size."""
        # Create data with key identifier + padding to reach data_size
        prefix = f"v_{key}_".encode()
        if self.config.data_size <= len(prefix):
            return prefix[:self.config.data_size]
        padding_size = self.config.data_size - len(prefix)
        return prefix + b'x' * padding_size

    def setup(self) -> None:
        """Initialize Grove instance."""
        self.grove = Grove(
            max_deg=self.config.max_deg,
            num_opr=self.config.num_ops * 2,  # Extra capacity for ops
            num_data=self.config.num_data,
            key_size=self.config.key_size,
            data_size=self.config.data_size,
            client=self.client,
            encryptor=None,
            stash_scale=self.config.stash_scale,
            bucket_size=self.config.bucket_size,
        )
        self.grove._pos_omap.init_server_storage()
        self.grove._graph_oram.init_server_storage()
        self.grove._graph_meta.init_server_storage()
        self.grove._pos_meta.init_server_storage()
        self.inserted_keys = []

    def _measure_operation(self, operation: str, func, *args, **kwargs) -> OperationMetrics:
        """Measure a single operation's time and bandwidth."""
        self.client.reset_bandwidth()

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        bytes_read, bytes_written = self.client.get_bandwidth()
        client_time_ms = (end_time - start_time) * 1000

        return OperationMetrics(
            operation=operation,
            client_time_ms=client_time_ms,
            bytes_read=bytes_read,
            bytes_written=bytes_written,
            total_bandwidth=bytes_read + bytes_written,
        )

    def _build_initial_graph(self, num_vertices: int) -> None:
        """Build initial graph based on configuration."""
        graph_type = self.config.graph_type
        max_deg = self.config.max_deg

        if graph_type == "star":
            # Star graph: vertex 0 at center, others connected to it
            self.grove.insert(vertex=(0, self._make_vertex_data(0), {}))
            self.inserted_keys.append(0)
            for i in range(1, min(num_vertices, max_deg + 1)):
                self.grove.insert(vertex=(i, self._make_vertex_data(i), {0: None}))
                self.inserted_keys.append(i)

        elif graph_type == "chain":
            # Chain graph: 0-1-2-3-...
            self.grove.insert(vertex=(0, self._make_vertex_data(0), {}))
            self.inserted_keys.append(0)
            for i in range(1, num_vertices):
                self.grove.insert(vertex=(i, self._make_vertex_data(i), {i - 1: None}))
                self.inserted_keys.append(i)

        elif graph_type == "complete":
            # Complete graph (up to max_deg connections)
            for i in range(num_vertices):
                neighbors = {}
                for j in range(max(0, i - max_deg), i):
                    neighbors[j] = None
                self.grove.insert(vertex=(i, self._make_vertex_data(i), neighbors))
                self.inserted_keys.append(i)

        else:  # random
            # Random graph with random neighbor connections
            self.grove.insert(vertex=(0, self._make_vertex_data(0), {}))
            self.inserted_keys.append(0)
            for i in range(1, num_vertices):
                # Connect to random subset of existing vertices
                num_neighbors = min(random.randint(1, max_deg), i)
                neighbor_keys = random.sample(self.inserted_keys, num_neighbors)
                neighbors = {k: None for k in neighbor_keys}
                self.grove.insert(vertex=(i, self._make_vertex_data(i), neighbors))
                self.inserted_keys.append(i)

    def benchmark_insert(self, num_ops: int, warmup: int = 0) -> BenchmarkResult:
        """Benchmark insert operations."""
        # Reset and build small initial graph
        self.setup()
        initial_size = min(10, self.config.num_data // 10)
        self._build_initial_graph(initial_size)

        next_key = max(self.inserted_keys) + 1

        # Warmup
        for _ in range(warmup):
            neighbors = {}
            if self.inserted_keys:
                num_neighbors = min(random.randint(0, self.config.max_deg), len(self.inserted_keys))
                if num_neighbors > 0:
                    neighbor_keys = random.sample(self.inserted_keys, num_neighbors)
                    neighbors = {k: None for k in neighbor_keys}
            self.grove.insert(vertex=(next_key, self._make_vertex_data(next_key), neighbors))
            self.inserted_keys.append(next_key)
            next_key += 1

        # Measure
        for i in range(num_ops):
            neighbors = {}
            if self.inserted_keys:
                num_neighbors = min(random.randint(0, self.config.max_deg), len(self.inserted_keys))
                if num_neighbors > 0:
                    neighbor_keys = random.sample(self.inserted_keys, num_neighbors)
                    neighbors = {k: None for k in neighbor_keys}

            metrics = self._measure_operation(
                "insert",
                self.grove.insert,
                vertex=(next_key, self._make_vertex_data(next_key), neighbors)
            )
            self.metrics["insert"].append(metrics)
            self.inserted_keys.append(next_key)
            next_key += 1

        return self._aggregate_metrics("insert")

    def benchmark_lookup(self, num_ops: int, warmup: int = 0) -> BenchmarkResult:
        """Benchmark lookup operations."""
        # Reset and build graph
        self.setup()
        graph_size = min(100, self.config.num_data // 2)
        self._build_initial_graph(graph_size)

        if not self.inserted_keys:
            raise ValueError("No vertices to lookup")

        # Warmup
        for _ in range(warmup):
            key = random.choice(self.inserted_keys)
            self.grove.lookup(keys=[key])

        # Measure
        for _ in range(num_ops):
            key = random.choice(self.inserted_keys)
            metrics = self._measure_operation(
                "lookup",
                self.grove.lookup,
                keys=[key]
            )
            self.metrics["lookup"].append(metrics)

        return self._aggregate_metrics("lookup")

    def benchmark_delete(self, num_ops: int, warmup: int = 0) -> BenchmarkResult:
        """Benchmark delete operations."""
        # Reset and build graph with enough vertices
        self.setup()
        graph_size = min(num_ops + warmup + 50, self.config.num_data // 2)
        self._build_initial_graph(graph_size)

        if len(self.inserted_keys) < num_ops + warmup:
            raise ValueError(f"Not enough vertices ({len(self.inserted_keys)}) to delete {num_ops + warmup}")

        # Shuffle keys for random deletion order
        keys_to_delete = self.inserted_keys.copy()
        random.shuffle(keys_to_delete)

        # Warmup
        for i in range(warmup):
            key = keys_to_delete.pop()
            self.grove.delete(key=key)
            self.inserted_keys.remove(key)

        # Measure
        for i in range(num_ops):
            if not keys_to_delete:
                break
            key = keys_to_delete.pop()
            metrics = self._measure_operation(
                "delete",
                self.grove.delete,
                key=key
            )
            self.metrics["delete"].append(metrics)
            self.inserted_keys.remove(key)

        return self._aggregate_metrics("delete")

    def benchmark_neighbor(self, num_ops: int, warmup: int = 0) -> BenchmarkResult:
        """Benchmark neighbor query operations."""
        # Reset and build graph
        self.setup()
        graph_size = min(100, self.config.num_data // 2)
        self._build_initial_graph(graph_size)

        if not self.inserted_keys:
            raise ValueError("No vertices for neighbor query")

        # Warmup
        for _ in range(warmup):
            key = random.choice(self.inserted_keys)
            self.grove.neighbor(keys=[key])

        # Measure
        for _ in range(num_ops):
            key = random.choice(self.inserted_keys)
            metrics = self._measure_operation(
                "neighbor",
                self.grove.neighbor,
                keys=[key]
            )
            self.metrics["neighbor"].append(metrics)

        return self._aggregate_metrics("neighbor")

    def benchmark_t_hop(self, num_ops: int, num_hop: int = 2, warmup: int = 0) -> BenchmarkResult:
        """Benchmark t-hop query operations."""
        # Reset and build graph
        self.setup()
        graph_size = min(100, self.config.num_data // 2)
        self._build_initial_graph(graph_size)

        if not self.inserted_keys:
            raise ValueError("No vertices for t-hop query")

        # Warmup
        for _ in range(warmup):
            key = random.choice(self.inserted_keys)
            self.grove.t_hop(key=key, num_hop=num_hop)

        # Measure
        for _ in range(num_ops):
            key = random.choice(self.inserted_keys)
            metrics = self._measure_operation(
                "t_hop",
                self.grove.t_hop,
                key=key,
                num_hop=num_hop
            )
            self.metrics["t_hop"].append(metrics)

        return self._aggregate_metrics("t_hop")

    def benchmark_t_traversal(self, num_ops: int, num_hop: int = 3, warmup: int = 0) -> BenchmarkResult:
        """Benchmark t-traversal operations."""
        # Reset and build graph
        self.setup()
        graph_size = min(100, self.config.num_data // 2)
        self._build_initial_graph(graph_size)

        if not self.inserted_keys:
            raise ValueError("No vertices for t-traversal")

        # Warmup
        for _ in range(warmup):
            key = random.choice(self.inserted_keys)
            self.grove.t_traversal(key=key, num_hop=num_hop)

        # Measure
        for _ in range(num_ops):
            key = random.choice(self.inserted_keys)
            metrics = self._measure_operation(
                "t_traversal",
                self.grove.t_traversal,
                key=key,
                num_hop=num_hop
            )
            self.metrics["t_traversal"].append(metrics)

        return self._aggregate_metrics("t_traversal")

    def _aggregate_metrics(self, operation: str) -> BenchmarkResult:
        """Aggregate metrics for an operation type."""
        op_metrics = self.metrics[operation]
        if not op_metrics:
            return BenchmarkResult(
                operation=operation,
                num_samples=0,
                avg_client_time_ms=0,
                std_client_time_ms=0,
                min_client_time_ms=0,
                max_client_time_ms=0,
                avg_bandwidth_bytes=0,
                std_bandwidth_bytes=0,
                avg_bytes_read=0,
                avg_bytes_written=0,
            )

        times = [m.client_time_ms for m in op_metrics]
        bandwidths = [m.total_bandwidth for m in op_metrics]
        bytes_read = [m.bytes_read for m in op_metrics]
        bytes_written = [m.bytes_written for m in op_metrics]

        return BenchmarkResult(
            operation=operation,
            num_samples=len(op_metrics),
            avg_client_time_ms=statistics.mean(times),
            std_client_time_ms=statistics.stdev(times) if len(times) > 1 else 0,
            min_client_time_ms=min(times),
            max_client_time_ms=max(times),
            avg_bandwidth_bytes=statistics.mean(bandwidths),
            std_bandwidth_bytes=statistics.stdev(bandwidths) if len(bandwidths) > 1 else 0,
            avg_bytes_read=statistics.mean(bytes_read),
            avg_bytes_written=statistics.mean(bytes_written),
        )

    def run_all(self) -> Dict[str, BenchmarkResult]:
        """Run all configured benchmarks."""
        results = {}

        operations = self.config.operations
        num_ops = self.config.num_ops
        warmup = self.config.warmup

        if "insert" in operations:
            print(f"Benchmarking insert ({num_ops} ops, {warmup} warmup)...")
            results["insert"] = self.benchmark_insert(num_ops, warmup)

        if "lookup" in operations:
            print(f"Benchmarking lookup ({num_ops} ops, {warmup} warmup)...")
            results["lookup"] = self.benchmark_lookup(num_ops, warmup)

        if "delete" in operations:
            print(f"Benchmarking delete ({num_ops} ops, {warmup} warmup)...")
            results["delete"] = self.benchmark_delete(num_ops, warmup)

        if "neighbor" in operations:
            print(f"Benchmarking neighbor ({num_ops} ops, {warmup} warmup)...")
            results["neighbor"] = self.benchmark_neighbor(num_ops, warmup)

        if "t_hop" in operations:
            print(f"Benchmarking t_hop ({num_ops} ops, {warmup} warmup)...")
            results["t_hop"] = self.benchmark_t_hop(num_ops, num_hop=2, warmup=warmup)

        if "t_traversal" in operations:
            print(f"Benchmarking t_traversal ({num_ops} ops, {warmup} warmup)...")
            results["t_traversal"] = self.benchmark_t_traversal(num_ops, num_hop=3, warmup=warmup)

        return results


def print_results(results: Dict[str, BenchmarkResult], config: BenchmarkConfig) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print("GROVE BENCHMARK RESULTS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  num_data: {config.num_data}")
    print(f"  max_deg: {config.max_deg}")
    print(f"  data_size: {config.data_size} bytes ({config.data_size/1024:.1f} KB)")
    print(f"  num_ops: {config.num_ops}")
    print(f"  warmup: {config.warmup}")
    print(f"  graph_type: {config.graph_type}")
    print(f"  bucket_size: {config.bucket_size}")
    print(f"  stash_scale: {config.stash_scale}")

    print("\n" + "-" * 80)
    print(f"{'Operation':<15} {'Samples':>8} {'Avg Time (ms)':>14} {'Std (ms)':>10} "
          f"{'Avg BW (KB)':>12} {'Read (KB)':>10} {'Write (KB)':>11}")
    print("-" * 80)

    for op_name, result in results.items():
        if result.num_samples > 0:
            print(f"{result.operation:<15} {result.num_samples:>8} "
                  f"{result.avg_client_time_ms:>14.3f} {result.std_client_time_ms:>10.3f} "
                  f"{result.avg_bandwidth_kb:>12.2f} {result.avg_bytes_read/1024:>10.2f} "
                  f"{result.avg_bytes_written/1024:>11.2f}")

    print("-" * 80)

    # Summary statistics
    total_time = sum(r.avg_client_time_ms * r.num_samples for r in results.values() if r.num_samples > 0)
    total_ops = sum(r.num_samples for r in results.values())
    total_bandwidth = sum(r.avg_bandwidth_bytes * r.num_samples for r in results.values() if r.num_samples > 0)

    if total_ops > 0:
        print(f"\nSummary:")
        print(f"  Total operations: {total_ops}")
        print(f"  Total client time: {total_time:.2f} ms ({total_time/1000:.2f} s)")
        print(f"  Total bandwidth: {total_bandwidth/1024:.2f} KB ({total_bandwidth/1024/1024:.2f} MB)")
        print(f"  Avg time per operation: {total_time/total_ops:.3f} ms")
        print(f"  Avg bandwidth per operation: {total_bandwidth/total_ops/1024:.2f} KB")


def save_results_csv(results: Dict[str, BenchmarkResult], config: BenchmarkConfig,
                     output_path: str) -> None:
    """Save benchmark results to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write config header
        writer.writerow(["# Configuration"])
        writer.writerow(["num_data", config.num_data])
        writer.writerow(["max_deg", config.max_deg])
        writer.writerow(["data_size", config.data_size])
        writer.writerow(["num_ops", config.num_ops])
        writer.writerow(["warmup", config.warmup])
        writer.writerow(["graph_type", config.graph_type])
        writer.writerow([])

        # Write results header
        writer.writerow([
            "operation", "num_samples",
            "avg_client_time_ms", "std_client_time_ms", "min_client_time_ms", "max_client_time_ms",
            "avg_bandwidth_bytes", "std_bandwidth_bytes", "avg_bytes_read", "avg_bytes_written"
        ])

        # Write results
        for result in results.values():
            writer.writerow([
                result.operation, result.num_samples,
                result.avg_client_time_ms, result.std_client_time_ms,
                result.min_client_time_ms, result.max_client_time_ms,
                result.avg_bandwidth_bytes, result.std_bandwidth_bytes,
                result.avg_bytes_read, result.avg_bytes_written
            ])

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Grove client time and bandwidth")
    parser.add_argument("--num_data", type=int, default=1024,
                        help="Number of vertices Grove can store (default: 1024)")
    parser.add_argument("--max_deg", type=int, default=5,
                        help="Maximum degree per vertex (default: 5)")
    parser.add_argument("--num_ops", type=int, default=100,
                        help="Number of operations to benchmark (default: 100)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup operations (default: 10)")
    parser.add_argument("--operations", nargs="+",
                        default=["insert", "lookup", "delete", "neighbor"],
                        choices=["insert", "lookup", "delete", "neighbor", "t_hop", "t_traversal"],
                        help="Operations to benchmark (default: insert lookup delete neighbor)")
    parser.add_argument("--graph_type", type=str, default="random",
                        choices=["star", "chain", "random", "complete"],
                        help="Type of graph to construct (default: random)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (CSV format)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--bucket_size", type=int, default=4,
                        help="ORAM bucket size (default: 4)")
    parser.add_argument("--stash_scale", type=int, default=20,
                        help="ORAM stash scale (default: 20)")
    parser.add_argument("--data_size", type=int, default=64,
                        help="Size of vertex data in bytes (default: 64, use 4096 for 4KB)")

    args = parser.parse_args()

    config = BenchmarkConfig(
        num_data=args.num_data,
        max_deg=args.max_deg,
        num_ops=args.num_ops,
        warmup=args.warmup,
        graph_type=args.graph_type,
        operations=args.operations,
        seed=args.seed,
        bucket_size=args.bucket_size,
        stash_scale=args.stash_scale,
        data_size=args.data_size,
    )

    print("Starting Grove benchmark...")
    print(f"Configuration: num_data={config.num_data}, max_deg={config.max_deg}, "
          f"data_size={config.data_size}B, num_ops={config.num_ops}, graph_type={config.graph_type}")

    benchmark = GroveBenchmark(config)
    results = benchmark.run_all()

    print_results(results, config)

    if args.output:
        save_results_csv(results, config, args.output)


if __name__ == "__main__":
    main()
