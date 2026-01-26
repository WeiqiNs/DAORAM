#!/usr/bin/env python3
"""
Grove Efficiency Benchmark Tool

Usage:
    python benchmark_grove.py --operation lookup --num_vertices 1024 --num_edges 2048 --max_deg 10 --trials 100
"""

import argparse
import math
import random
import secrets
import time
import pickle
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from daoram.dependency import InteractServer, Data, BinaryTree
from daoram.dependency.types import ExecuteResult
from daoram.graph.grove import Grove


@dataclass
class BenchmarkResult:
    """Stores benchmark metrics for a single operation."""
    total_time_ms: float
    interaction_time_ms: float
    client_compute_time_ms: float
    num_rounds: int
    bytes_sent: int
    bytes_received: int


class WANSimulatedServer(InteractServer):
    """InteractServer with simulated WAN latency."""
    
    def __init__(self, latency_ms: float = 0):
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
    
    def get_round_count(self) -> int:
        return self._round_count
    
    def reset_round_count(self):
        self._round_count = 0
    
    def execute(self) -> ExecuteResult:
        """Execute with simulated latency."""
        # Count this as one round
        self._round_count += 1
        
        # Simulate WAN latency (round-trip)
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
            
            # Execute writes
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
            
            # Execute reads
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


def generate_random_graph(num_vertices: int, num_edges: int, max_deg: int, 
                          key_size: int, value_size: int) -> Dict[int, Tuple[bytes, Dict[int, None]]]:
    """
    Generate a random graph.
    Returns: {vertex_key: (vertex_data, {neighbor_key: None, ...})}
    """
    graph = {}
    
    # Create vertices with random data
    for v in range(1, num_vertices + 1):
        vertex_data = secrets.token_bytes(value_size)
        graph[v] = [vertex_data, {}]  # [data, adjacency_dict]
    
    # Add random edges (undirected)
    edges_added = 0
    attempts = 0
    max_attempts = num_edges * 10
    
    while edges_added < num_edges and attempts < max_attempts:
        attempts += 1
        u = random.randint(1, num_vertices)
        v = random.randint(1, num_vertices)
        
        if u == v:
            continue
        if len(graph[u][1]) >= max_deg or len(graph[v][1]) >= max_deg:
            continue
        if v in graph[u][1]:
            continue
        
        # Add edge (will set graph_leaf later during initialization)
        graph[u][1][v] = None
        graph[v][1][u] = None
        edges_added += 1
    
    print(f"Generated graph: {num_vertices} vertices, {edges_added} edges")
    return graph


def initialize_grove_directly(client: WANSimulatedServer, graph: Dict, 
                              num_data: int, max_deg: int, key_size: int, 
                              data_size: int, num_opr: int) -> Grove:
    """
    Initialize Grove by directly inserting data into ORAM trees.
    This simulates client-side initialization without using Grove's insert method.
    """
    # Create Grove instance
    grove = Grove(
        client=client,
        num_data=num_data,
        data_size=data_size,
        max_deg=max_deg,
        num_opr=num_opr,
        key_size=key_size,
        encryptor=None,
        stash_scale=20
    )
    
    level = grove._level
    leaf_range = grove._leaf_range
    
    # Assign random graph_leaf and pos_leaf for each vertex
    vertex_graph_leaves = {}
    vertex_pos_leaves = {}
    for v in graph:
        vertex_graph_leaves[v] = secrets.randbelow(leaf_range)
        vertex_pos_leaves[v] = secrets.randbelow(leaf_range)
    
    # Update adjacency_dict with actual graph_leaf values
    for v, (data, adj) in graph.items():
        for neighbor in adj:
            adj[neighbor] = vertex_graph_leaves[neighbor]
    
    # Initialize Graph ORAM storage
    graph_tree = BinaryTree(level=level, bucket_size=grove._graph_oram._bucket_size)
    for v, (data, adj) in graph.items():
        gl = vertex_graph_leaves[v]
        pl = vertex_pos_leaves[v]
        # Value format: (vertex_data, adjacency_dict, pos_leaf)
        vertex_value = (data, adj, pl)
        graph_tree.fill_data_to_storage(Data(key=v, leaf=gl, value=vertex_value))
    
    # Initialize Graph Meta ORAM storage (empty initially)
    graph_meta_tree = BinaryTree(level=level, bucket_size=grove._graph_meta._bucket_size)
    
    # Initialize PosMap ORAM (AVL tree) - we need to build the AVL structure
    # For simplicity, we'll use the AVLOmapCached's initialization
    # First, build the AVL tree structure
    from daoram.omap.tree_utils.avl_node_value import AVLNodeValue
    
    pos_tree = BinaryTree(level=level, bucket_size=grove._pos_omap._bucket_size)
    pos_meta_tree = BinaryTree(level=level, bucket_size=grove._pos_omap._meta._bucket_size)
    
    # Build a simple balanced BST for PosMap
    sorted_keys = sorted(graph.keys())
    
    def build_avl_nodes(keys, parent_key=None):
        """Build AVL nodes recursively."""
        if not keys:
            return None
        
        mid = len(keys) // 2
        key = keys[mid]
        left_keys = keys[:mid]
        right_keys = keys[mid+1:]
        
        # Create AVL node
        pos_leaf = vertex_pos_leaves[key]
        graph_leaf = vertex_graph_leaves[key]  # This is the value stored in PosMap
        
        # Get children info
        left_child = build_avl_nodes(left_keys, key)
        right_child = build_avl_nodes(right_keys, key)
        
        l_key = left_child[0] if left_child else None
        l_leaf = left_child[1] if left_child else None
        l_height = left_child[2] if left_child else 0
        
        r_key = right_child[0] if right_child else None
        r_leaf = right_child[1] if right_child else None
        r_height = right_child[2] if right_child else 0
        
        height = max(l_height, r_height) + 1
        
        node_value = AVLNodeValue(
            l_key=l_key, l_leaf=l_leaf,
            r_key=r_key, r_leaf=r_leaf,
            height=height, value=graph_leaf
        )
        
        pos_tree.fill_data_to_storage(Data(key=key, leaf=pos_leaf, value=node_value))
        
        return (key, pos_leaf, height)
    
    root_info = build_avl_nodes(sorted_keys)
    if root_info:
        grove._pos_omap.root = (root_info[0], root_info[1])
    
    # Register storage with client
    client.init_storage({
        grove._graph_oram._name: graph_tree,
        grove._graph_meta._name: graph_meta_tree,
        grove._pos_omap._name: pos_tree,
        grove._pos_omap._meta._name: pos_meta_tree,
    })
    
    return grove


def run_benchmark(grove: Grove, client: WANSimulatedServer, 
                  operation: str, graph: Dict, trials: int) -> List[BenchmarkResult]:
    """Run benchmark for specified operation."""
    results = []
    vertex_keys = list(graph.keys())
    
    for trial in range(trials):
        # Reset counters
        client.reset_bandwidth()
        client.reset_round_count()
        
        # Select random target(s)
        if operation in ["lookup", "neighbor", "delete"]:
            target = random.choice(vertex_keys)
        elif operation == "insert":
            # For insert, create a new vertex
            new_key = max(vertex_keys) + trial + 1
            neighbors = random.sample(vertex_keys, min(3, len(vertex_keys)))
            target = (new_key, secrets.token_bytes(64), {n: None for n in neighbors})
        elif operation == "t_hop":
            target = random.choice(vertex_keys)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Measure time
        start_time = time.perf_counter()
        interaction_start = start_time
        
        try:
            if operation == "lookup":
                grove.lookup([target])
            elif operation == "neighbor":
                grove.neighbor([target])
            elif operation == "insert":
                grove.insert(target)
            elif operation == "delete":
                grove.delete(target)
                # Remove from our tracking
                if target in vertex_keys:
                    vertex_keys.remove(target)
            elif operation == "t_hop":
                grove.t_hop(target, num_hop=2)
        except Exception as e:
            print(f"Trial {trial} failed: {e}")
            continue
        
        end_time = time.perf_counter()
        
        # Calculate metrics
        total_time_ms = (end_time - start_time) * 1000
        bytes_read, bytes_written = client.get_bandwidth()
        num_rounds = client.get_round_count()
        
        # Estimate interaction time (latency * rounds)
        interaction_time_ms = client._latency_ms * num_rounds
        client_compute_time_ms = total_time_ms - interaction_time_ms
        
        results.append(BenchmarkResult(
            total_time_ms=total_time_ms,
            interaction_time_ms=interaction_time_ms,
            client_compute_time_ms=client_compute_time_ms,
            num_rounds=num_rounds,
            bytes_sent=bytes_written,
            bytes_received=bytes_read
        ))
        
        if (trial + 1) % 10 == 0:
            print(f"  Completed {trial + 1}/{trials} trials")
    
    return results


def print_results(operation: str, results: List[BenchmarkResult]):
    """Print benchmark results summary."""
    if not results:
        print("No results to display")
        return
    
    n = len(results)
    
    avg_total = sum(r.total_time_ms for r in results) / n
    avg_interact = sum(r.interaction_time_ms for r in results) / n
    avg_compute = sum(r.client_compute_time_ms for r in results) / n
    avg_rounds = sum(r.num_rounds for r in results) / n
    avg_sent = sum(r.bytes_sent for r in results) / n
    avg_recv = sum(r.bytes_received for r in results) / n
    
    print(f"\n{'='*60}")
    print(f"Benchmark Results: {operation.upper()}")
    print(f"{'='*60}")
    print(f"Trials: {n}")
    print(f"")
    print(f"Time Metrics (ms):")
    print(f"  Total Time:       {avg_total:10.2f} ms")
    print(f"  Interaction Time: {avg_interact:10.2f} ms")
    print(f"  Client Compute:   {avg_compute:10.2f} ms")
    print(f"")
    print(f"Communication Metrics:")
    print(f"  Rounds:           {avg_rounds:10.1f}")
    print(f"  Bytes Sent:       {avg_sent:10.0f} bytes ({avg_sent/1024:.2f} KB)")
    print(f"  Bytes Received:   {avg_recv:10.0f} bytes ({avg_recv/1024:.2f} KB)")
    print(f"  Total Bandwidth:  {(avg_sent+avg_recv):10.0f} bytes ({(avg_sent+avg_recv)/1024:.2f} KB)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Grove Efficiency Benchmark")
    parser.add_argument("--operation", type=str, required=True,
                        choices=["lookup", "neighbor", "insert", "delete", "t_hop"],
                        help="Operation to benchmark")
    parser.add_argument("--num_vertices", type=int, default=1024,
                        help="Number of vertices in graph")
    parser.add_argument("--num_edges", type=int, default=2048,
                        help="Number of edges in graph")
    parser.add_argument("--max_deg", type=int, default=10,
                        help="Maximum degree per vertex")
    parser.add_argument("--trials", type=int, default=100,
                        help="Number of trials to run")
    parser.add_argument("--latency_ms", type=float, default=0,
                        help="Simulated WAN latency in milliseconds")
    parser.add_argument("--key_size", type=int, default=8,
                        help="Size of keys in bytes")
    parser.add_argument("--value_size", type=int, default=64,
                        help="Size of vertex data in bytes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    print(f"Grove Benchmark Configuration:")
    print(f"  Operation:     {args.operation}")
    print(f"  Vertices:      {args.num_vertices}")
    print(f"  Edges:         {args.num_edges}")
    print(f"  Max Degree:    {args.max_deg}")
    print(f"  Trials:        {args.trials}")
    print(f"  WAN Latency:   {args.latency_ms} ms")
    print(f"  Key Size:      {args.key_size} bytes")
    print(f"  Value Size:    {args.value_size} bytes")
    print(f"  Random Seed:   {args.seed}")
    print()
    
    # Set random seed
    random.seed(args.seed)
    
    # Generate graph
    print("Generating random graph...")
    graph = generate_random_graph(
        num_vertices=args.num_vertices,
        num_edges=args.num_edges,
        max_deg=args.max_deg,
        key_size=args.key_size,
        value_size=args.value_size
    )
    
    # Create client with WAN simulation
    print("Initializing Grove...")
    client = WANSimulatedServer(latency_ms=args.latency_ms)
    
    # Calculate num_data (next power of 2)
    num_data = 2 ** math.ceil(math.log2(args.num_vertices + args.trials + 100))
    
    grove = initialize_grove_directly(
        client=client,
        graph=graph,
        num_data=num_data,
        max_deg=args.max_deg,
        key_size=args.key_size,
        data_size=args.value_size,
        num_opr=args.trials * 10
    )
    
    # Run benchmark
    print(f"\nRunning {args.operation} benchmark ({args.trials} trials)...")
    results = run_benchmark(grove, client, args.operation, graph, args.trials)
    
    # Print results
    print_results(args.operation, results)


if __name__ == "__main__":
    main()
