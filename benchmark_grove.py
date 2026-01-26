#!/usr/bin/env python3
"""
Grove Efficiency Benchmark Tool

Usage:
    python benchmark_grove.py --op lookup --num_vertices 1024 --num_edges 2048 --max_deg 10 --runs 100
"""

import argparse
import math
import pickle
import random
import secrets
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, '/Users/xinle/Desktop/Grove')

from daoram.dependency import Data, InteractServer
from daoram.dependency.binary_tree import BinaryTree
from daoram.dependency.types import ExecuteResult
from daoram.graph.grove import Grove


@dataclass
class BenchmarkResult:
    """Result of a single operation benchmark."""
    total_time_ms: float
    network_time_ms: float
    compute_time_ms: float
    num_rounds: int
    bytes_sent: int
    bytes_received: int


class WANSimulatedServer(InteractServer):
    """Local server with simulated WAN latency."""

    def __init__(self, latency_ms: float = 0.0):
        super().__init__()
        self._storage = {}
        self._latency_ms = latency_ms
        self._round_count = 0

    def init_connection(self, client=None):
        pass

    def close_connection(self):
        pass

    def init_storage(self, storage):
        self._storage.update(storage)

    def _get_tree(self, label: str) -> BinaryTree:
        if label not in self._storage:
            raise KeyError(f"Label {label} not in storage")
        return self._storage[label]

    def _get_queue(self, label: str) -> List:
        if label not in self._storage:
            raise KeyError(f"Label {label} not in storage")
        return self._storage[label]

    def execute(self) -> ExecuteResult:
        """Execute with simulated latency."""
        self._round_count += 1
        
        # Simulate network latency (round trip)
        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000.0)

        try:
            results = {}
            
            # Track bandwidth
            request = (
                self._read_paths, self._read_buckets, self._read_blocks, self._read_queues,
                self._write_paths, self._write_buckets, self._write_blocks, self._write_queues,
            )
            self._bytes_written += len(pickle.dumps(request))

            # Writes
            for label, data in self._write_paths.items():
                self._get_tree(label).write_path(data)
            for label, data in self._write_buckets.items():
                self._get_tree(label).write_bucket(data)
            for label, data in self._write_blocks.items():
                self._get_tree(label).write_block(data)
            for label, operations in self._write_queues.items():
                queue = self._get_queue(label)
                for op, value in operations:
                    if op == "push_front":
                        queue.insert(0, value)
                    elif op == "push_back":
                        queue.append(value)
                    elif op == "pop_front":
                        results[label] = queue.pop(0)
                    elif op == "pop_back":
                        results[label] = queue.pop()
                    elif isinstance(op, tuple) and op[0] == "set":
                        queue[op[1]] = value

            # Reads
            for label, leaves in self._read_paths.items():
                results[label] = self._get_tree(label).read_path(list(set(leaves)))
            for label, keys in self._read_buckets.items():
                results[label] = self._get_tree(label).read_bucket(list(set(keys)))
            for label, keys in self._read_blocks.items():
                results[label] = self._get_tree(label).read_block(list(set(keys)))
            for label, indices in self._read_queues.items():
                queue = self._get_queue(label)
                if indices is None:
                    results[label] = list(queue)
                else:
                    results[label] = {i: queue[i] for i in set(indices)}

            result = ExecuteResult(success=True, results=results)
            self._bytes_read += len(pickle.dumps(result))
            return result

        except Exception as e:
            return ExecuteResult(success=False, error=str(e))
        finally:
            self.clear_queries()

    def reset_round_count(self):
        self._round_count = 0

    def get_round_count(self):
        return self._round_count


def generate_random_graph(num_vertices: int, num_edges: int, max_deg: int) -> Dict[int, List[int]]:
    """Generate a random graph with given constraints."""
    adjacency = {i: [] for i in range(1, num_vertices + 1)}
    edges_added = 0
    attempts = 0
    max_attempts = num_edges * 10

    while edges_added < num_edges and attempts < max_attempts:
        attempts += 1
        u = random.randint(1, num_vertices)
        v = random.randint(1, num_vertices)
        if u == v:
            continue
        if v in adjacency[u]:
            continue
        if len(adjacency[u]) >= max_deg or len(adjacency[v]) >= max_deg:
            continue
        adjacency[u].append(v)
        adjacency[v].append(u)
        edges_added += 1

    return adjacency


def direct_initialize_grove(grove: Grove, adjacency: Dict[int, List[int]], 
                            value_size: int = 64) -> None:
    """Directly initialize Grove ORAMs without using insert()."""
    num_vertices = len(adjacency)
    leaf_range = grove._leaf_range

    # Assign random graph_leaf and pos_leaf for each vertex
    vertex_graph_leaves = {v: secrets.randbelow(leaf_range) for v in adjacency}
    vertex_pos_leaves = {v: secrets.randbelow(leaf_range) for v in adjacency}

    # Build adjacency_dict with graph_leaf for each vertex
    vertex_data = {}
    for v, neighbors in adjacency.items():
        adj_dict = {n: vertex_graph_leaves[n] for n in neighbors}
        v_data = "x" * value_size  # Dummy vertex data
        pos_leaf = vertex_pos_leaves[v]
        vertex_data[v] = Data(
            key=v,
            leaf=vertex_graph_leaves[v],
            value=(v_data, adj_dict, pos_leaf)
        )

    # Initialize Graph ORAM tree
    grove._graph_oram.init_server_storage()
    for v, data in vertex_data.items():
        grove._graph_oram.stash.append(data)
    # Evict to tree
    all_leaves = list(set(vertex_graph_leaves.values()))
    for leaf in all_leaves:
        evicted = grove._graph_oram._evict_stash(leaves=[leaf])
        grove._client.add_write_path(label=grove._graph_oram._name, data=evicted)
    grove._client.execute()
    grove._graph_oram.stash.clear()

    # Initialize PosMap ORAM (AVL tree) - simplified direct insertion
    grove._pos_omap.init_server_storage()
    
    # Build AVL nodes
    from daoram.omap.tree_node import AVLNodeData
    sorted_keys = sorted(adjacency.keys())
    
    def build_balanced_avl(keys, parent_leaf=None):
        if not keys:
            return None, None
        mid = len(keys) // 2
        key = keys[mid]
        node_leaf = vertex_pos_leaves[key]
        
        left_keys = keys[:mid]
        right_keys = keys[mid+1:]
        
        l_key, l_leaf = build_balanced_avl(left_keys, node_leaf)
        r_key, r_leaf = build_balanced_avl(right_keys, node_leaf)
        
        height = 1 + max(
            (len(left_keys) > 0) and int(math.log2(len(left_keys) + 1)) or 0,
            (len(right_keys) > 0) and int(math.log2(len(right_keys) + 1)) or 0
        )
        
        node_data = AVLNodeData(
            value=vertex_graph_leaves[key],
            height=height,
            l_key=l_key, l_leaf=l_leaf if l_leaf else 0,
            r_key=r_key, r_leaf=r_leaf if r_leaf else 0
        )
        
        data = Data(key=key, leaf=node_leaf, value=node_data)
        grove._pos_omap._stash.append(data)
        
        return key, node_leaf

    root_key, root_leaf = build_balanced_avl(sorted_keys)
    grove._pos_omap.root = (root_key, root_leaf)
    
    # Evict to tree
    all_pos_leaves = list(set(vertex_pos_leaves.values()))
    for leaf in all_pos_leaves:
        evicted = grove._pos_omap._evict_stash(leaves=[leaf])
        grove._client.add_write_path(label=grove._pos_omap._name, data=evicted)
    grove._client.execute()
    grove._pos_omap._stash.clear()

    # Initialize meta ORAMs (empty)
    grove._graph_meta.init_server_storage()
    grove._pos_meta.init_server_storage()


def run_benchmark(grove: Grove, client: WANSimulatedServer, op: str, 
                  keys: List[int], runs: int) -> List[BenchmarkResult]:
    """Run benchmark for specified operation."""
    results = []
    
    for i in range(runs):
        # Reset counters
        client.reset_bandwidth()
        client.reset_round_count()
        
        key = keys[i % len(keys)]
        
        start_time = time.perf_counter()
        
        try:
            if op == "lookup":
                grove.lookup([key])
            elif op == "neighbor":
                grove.neighbor([key])
            elif op == "insert":
                # Insert a new vertex with random neighbors
                new_key = 100000 + i
                neighbors = random.sample(keys[:min(10, len(keys))], min(2, len(keys)))
                grove.insert((new_key, f"v{new_key}", {n: 0 for n in neighbors}))
            elif op == "delete":
                grove.delete(key)
            elif op == "t_hop":
                grove.t_hop(key, num_hop=2)
        except Exception as e:
            print(f"  Run {i+1} failed: {e}")
            continue
        
        end_time = time.perf_counter()
        
        total_time = (end_time - start_time) * 1000
        bytes_read, bytes_written = client.get_bandwidth()
        rounds = client.get_round_count()
        network_time = rounds * client._latency_ms
        compute_time = total_time - network_time
        
        results.append(BenchmarkResult(
            total_time_ms=total_time,
            network_time_ms=network_time,
            compute_time_ms=compute_time,
            num_rounds=rounds,
            bytes_sent=bytes_written,
            bytes_received=bytes_read
        ))
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{runs} runs")
    
    return results


def print_results(results: List[BenchmarkResult], op: str):
    """Print benchmark statistics."""
    if not results:
        print("No results to display")
        return
    
    n = len(results)
    avg_total = sum(r.total_time_ms for r in results) / n
    avg_network = sum(r.network_time_ms for r in results) / n
    avg_compute = sum(r.compute_time_ms for r in results) / n
    avg_rounds = sum(r.num_rounds for r in results) / n
    avg_sent = sum(r.bytes_sent for r in results) / n
    avg_recv = sum(r.bytes_received for r in results) / n
    
    print(f"\n{'='*60}")
    print(f"Benchmark Results for: {op.upper()}")
    print(f"{'='*60}")
    print(f"Number of runs: {n}")
    print(f"\nTiming (ms):")
    print(f"  Total time:    {avg_total:10.2f} ms (avg)")
    print(f"  Network time:  {avg_network:10.2f} ms (avg)")
    print(f"  Compute time:  {avg_compute:10.2f} ms (avg)")
    print(f"\nCommunication:")
    print(f"  Rounds:        {avg_rounds:10.1f} (avg)")
    print(f"  Bytes sent:    {avg_sent:10.0f} bytes (avg)")
    print(f"  Bytes recv:    {avg_recv:10.0f} bytes (avg)")
    print(f"  Total traffic: {(avg_sent + avg_recv):10.0f} bytes (avg)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Grove Efficiency Benchmark")
    parser.add_argument("--op", type=str, default="lookup",
                        choices=["lookup", "neighbor", "insert", "delete", "t_hop"],
                        help="Operation to benchmark")
    parser.add_argument("--num_vertices", type=int, default=1024,
                        help="Number of vertices in graph")
    parser.add_argument("--num_edges", type=int, default=2048,
                        help="Number of edges in graph")
    parser.add_argument("--max_deg", type=int, default=10,
                        help="Maximum degree of vertices")
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of runs for benchmark")
    parser.add_argument("--latency", type=float, default=0.0,
                        help="Simulated WAN latency in ms (round trip)")
    parser.add_argument("--key_size", type=int, default=8,
                        help="Size of keys in bytes")
    parser.add_argument("--value_size", type=int, default=64,
                        help="Size of vertex data in bytes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Operation: {args.op}")
    print(f"  Vertices: {args.num_vertices}, Edges: {args.num_edges}, Max degree: {args.max_deg}")
    print(f"  Runs: {args.runs}, Latency: {args.latency}ms")
    print(f"  Key size: {args.key_size}, Value size: {args.value_size}")
    
    random.seed(args.seed)
    
    print("\nGenerating random graph...")
    adjacency = generate_random_graph(args.num_vertices, args.num_edges, args.max_deg)
    actual_edges = sum(len(n) for n in adjacency.values()) // 2
    print(f"  Generated graph with {len(adjacency)} vertices and {actual_edges} edges")
    
    print("\nInitializing Grove...")
    client = WANSimulatedServer(latency_ms=args.latency)
    grove = Grove(
        client=client,
        num_data=args.num_vertices * 2,
        data_size=args.value_size,
        max_deg=args.max_deg,
        num_opr=args.runs * 2,
        key_size=args.key_size,
        encryptor=None,
        stash_scale=20
    )
    
    print("  Directly initializing ORAMs...")
    direct_initialize_grove(grove, adjacency, args.value_size)
    print("  Initialization complete")
    
    # Prepare test keys
    all_keys = list(adjacency.keys())
    test_keys = random.sample(all_keys, min(args.runs, len(all_keys)))
    
    print(f"\nRunning {args.op} benchmark ({args.runs} runs)...")
    results = run_benchmark(grove, client, args.op, test_keys, args.runs)
    
    print_results(results, args.op)


if __name__ == "__main__":
    main()
