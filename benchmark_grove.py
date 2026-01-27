#!/usr/bin/env python3
"""
Grove Efficiency Benchmark
测试 Grove 各操作的效率指标
"""

import sys
import time
import random
import secrets
import pickle
import argparse
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

from daoram.dependency import InteractServer, Data, BinaryTree
from daoram.dependency.types import ExecuteResult
from daoram.graph.grove import Grove


@dataclass
class BenchmarkResult:
    """单次操作的测试结果"""
    total_time_ms: float
    client_compute_ms: float
    network_time_ms: float
    num_rounds: int
    bytes_sent: int
    bytes_received: int


class WANSimulatedServer(InteractServer):
    """模拟 WAN 延迟的本地服务器"""
    
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
    
    def reset_round_count(self):
        self._round_count = 0
    
    def get_round_count(self) -> int:
        return self._round_count
    
    def _get_tree(self, label: str) -> BinaryTree:
        if label not in self._storage:
            raise KeyError(f"Label {label} not in storage")
        return self._storage[label]
    
    def _get_queue(self, label: str) -> List:
        if label not in self._storage:
            raise KeyError(f"Label {label} not in storage")
        return self._storage[label]
    
    def execute(self) -> ExecuteResult:
        self._round_count += 1
        
        # 模拟网络延迟
        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000.0)
        
        try:
            results = {}
            
            # 计算发送的字节数
            request = (self._read_paths, self._read_buckets, self._read_blocks, 
                      self._read_queues, self._write_paths, self._write_buckets,
                      self._write_blocks, self._write_queues)
            self._bytes_written += len(pickle.dumps(request))
            
            # 执行写操作
            for label, data in self._write_paths.items():
                self._get_tree(label).write_path(data)
            for label, data in self._write_buckets.items():
                self._get_tree(label).write_bucket(data)
            for label, data in self._write_blocks.items():
                self._get_tree(label).write_block(data)
            for label, operations in self._write_queues.items():
                queue = self._get_queue(label)
                for op, value in operations:
                    if op == "push_front": queue.insert(0, value)
                    elif op == "push_back": queue.append(value)
                    elif op == "pop_front": results[label] = queue.pop(0)
                    elif op == "pop_back": results[label] = queue.pop()
                    elif isinstance(op, tuple) and op[0] == "set": queue[op[1]] = value
            
            # 执行读操作
            for label, leaves in self._read_paths.items():
                results[label] = self._get_tree(label).read_path(list(set(leaves)))
            for label, keys in self._read_buckets.items():
                results[label] = self._get_tree(label).read_bucket(list(set(keys)))
            for label, keys in self._read_blocks.items():
                results[label] = self._get_tree(label).read_block(list(set(keys)))
            for label, indices in self._read_queues.items():
                queue = self._get_queue(label)
                if indices is None: results[label] = list(queue)
                else: results[label] = {i: queue[i] for i in set(indices)}
            
            result = ExecuteResult(success=True, results=results)
            self._bytes_read += len(pickle.dumps(result))
            return result
        except Exception as e:
            return ExecuteResult(success=False, error=str(e))
        finally:
            self.clear_queries()


def generate_random_graph(num_vertices: int, num_edges: int, max_degree: int) -> Dict[int, List[int]]:
    """生成随机图"""
    adj = {i: [] for i in range(1, num_vertices + 1)}
    edges_added = 0
    attempts = 0
    max_attempts = num_edges * 10
    
    while edges_added < num_edges and attempts < max_attempts:
        u = random.randint(1, num_vertices)
        v = random.randint(1, num_vertices)
        if u != v and v not in adj[u] and len(adj[u]) < max_degree and len(adj[v]) < max_degree:
            adj[u].append(v)
            adj[v].append(u)
            edges_added += 1
        attempts += 1
    
    return adj


def init_grove_directly(grove: Grove, adj: Dict[int, List[int]], value_size: int):
    """直接初始化 Grove（不使用 insert，模拟 ORAM 初始化）"""
    num_vertices = len(adj)
    leaf_range = grove._leaf_range
    
    # 为每个 vertex 分配 graph_leaf 和 pos_leaf
    graph_leaves = {k: secrets.randbelow(leaf_range) for k in adj.keys()}
    pos_leaves = {k: secrets.randbelow(leaf_range) for k in adj.keys()}
    
    # 构建 Graph ORAM 数据
    graph_data = []
    for vertex_key, neighbors in adj.items():
        adjacency_dict = {n: graph_leaves[n] for n in neighbors}
        vertex_data = "v" * value_size  # value_size 字节的数据
        pos_leaf = pos_leaves[vertex_key]
        graph_data.append(Data(
            key=vertex_key,
            leaf=graph_leaves[vertex_key],
            value=(vertex_data, adjacency_dict, pos_leaf)
        ))
    
    # 初始化 Graph ORAM
    grove._graph_oram.init_server_storage()
    for data in graph_data:
        grove._graph_oram.stash.append(data)
    # Evict to tree
    evicted = grove._graph_oram._evict_stash(leaves=list(range(leaf_range)))
    grove._graph_oram._tree.write_path(evicted)
    grove._graph_oram.stash.clear()
    
    # 初始化 PosMap ORAM (AVL)
    grove._pos_omap.init_server_storage()
    for vertex_key in adj.keys():
        grove._pos_omap.insert(key=vertex_key, value=graph_leaves[vertex_key])
    
    # 初始化 meta ORAMs
    grove._graph_meta.init_server_storage()
    grove._pos_meta.init_server_storage()


def benchmark_operation(grove: Grove, client: WANSimulatedServer, 
                       operation: str, adj: Dict[int, List[int]], 
                       num_ops: int, value_size: int) -> List[BenchmarkResult]:
    """测试单个操作类型"""
    results = []
    vertices = list(adj.keys())
    next_vertex_id = max(vertices) + 1
    
    for i in range(num_ops):
        # 重置计数器
        client.reset_round_count()
        client.reset_bandwidth()
        
        start_time = time.perf_counter()
        
        if operation == "lookup":
            key = random.choice(vertices)
            grove.lookup([key])
        
        elif operation == "insert":
            # 插入新节点，连接到随机已有节点
            neighbors = random.sample(vertices, min(2, len(vertices)))
            vertex_data = "v" * value_size
            grove.insert((next_vertex_id, vertex_data, neighbors))
            vertices.append(next_vertex_id)
            adj[next_vertex_id] = neighbors
            for n in neighbors:
                adj[n].append(next_vertex_id)
            next_vertex_id += 1
        
        elif operation == "delete":
            if len(vertices) > 10:
                key = random.choice(vertices[10:])  # 保留前10个节点
                grove.delete(key)
                vertices.remove(key)
                for n in adj.get(key, []):
                    if key in adj.get(n, []):
                        adj[n].remove(key)
                del adj[key]
        
        elif operation == "neighbor":
            key = random.choice(vertices)
            grove.neighbor([key])
        
        elif operation == "t_hop":
            key = random.choice(vertices)
            grove.t_hop(key, num_hop=2)
        
        end_time = time.perf_counter()
        
        total_ms = (end_time - start_time) * 1000
        bytes_sent, bytes_recv = client.get_bandwidth()
        num_rounds = client.get_round_count()
        network_ms = num_rounds * client._latency_ms
        client_ms = total_ms - network_ms
        
        results.append(BenchmarkResult(
            total_time_ms=total_ms,
            client_compute_ms=client_ms,
            network_time_ms=network_ms,
            num_rounds=num_rounds,
            bytes_sent=bytes_sent,
            bytes_received=bytes_recv
        ))
    
    return results


def print_results(operation: str, results: List[BenchmarkResult]):
    """打印结果统计"""
    n = len(results)
    
    avg_total = sum(r.total_time_ms for r in results) / n
    avg_client = sum(r.client_compute_ms for r in results) / n
    avg_network = sum(r.network_time_ms for r in results) / n
    avg_rounds = sum(r.num_rounds for r in results) / n
    avg_sent = sum(r.bytes_sent for r in results) / n
    avg_recv = sum(r.bytes_received for r in results) / n
    
    print(f"\n{'='*60}")
    print(f"Operation: {operation.upper()} ({n} trials)")
    print(f"{'='*60}")
    print(f"  Total Time:      {avg_total:10.2f} ms")
    print(f"  Client Compute:  {avg_client:10.2f} ms")
    print(f"  Network Time:    {avg_network:10.2f} ms")
    print(f"  Rounds:          {avg_rounds:10.1f}")
    print(f"  Bytes Sent:      {avg_sent:10.0f} bytes")
    print(f"  Bytes Received:  {avg_recv:10.0f} bytes")
    print(f"  Total Bandwidth: {(avg_sent + avg_recv):10.0f} bytes")


def main():
    parser = argparse.ArgumentParser(description='Grove Efficiency Benchmark')
    parser.add_argument('--vertices', type=int, default=1024, help='Number of vertices')
    parser.add_argument('--edges', type=int, default=2048, help='Number of edges')
    parser.add_argument('--max-degree', type=int, default=10, help='Maximum degree')
    parser.add_argument('--latency', type=float, default=0, help='WAN latency in ms')
    parser.add_argument('--num-ops', type=int, default=100, help='Number of operations')
    parser.add_argument('--value-size', type=int, default=256, help='Value size in bytes')
    parser.add_argument('--key-size', type=int, default=8, help='Key size in bytes')
    parser.add_argument('--operation', type=str, default='all',
                       choices=['lookup', 'insert', 'delete', 'neighbor', 't_hop', 'all'],
                       help='Operation to benchmark')
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Vertices: {args.vertices}, Edges: {args.edges}, Max Degree: {args.max_degree}")
    print(f"  Latency: {args.latency}ms, Operations: {args.num_ops}")
    print(f"  Key Size: {args.key_size}, Value Size: {args.value_size}")
    
    # 生成图
    print("\nGenerating random graph...")
    adj = generate_random_graph(args.vertices, args.edges, args.max_degree)
    
    operations = ['lookup', 'insert', 'delete', 'neighbor', 't_hop'] if args.operation == 'all' else [args.operation]
    
    for op in operations:
        print(f"\nInitializing Grove for {op}...")
        client = WANSimulatedServer(latency_ms=args.latency)
        grove = Grove(
            client=client,
            num_data=args.vertices * 2,
            data_size=args.value_size,
            max_deg=args.max_degree,
            num_opr=args.num_ops * 10,
            key_size=args.key_size,
            encryptor=None,
            stash_scale=20
        )
        
        # 直接初始化
        init_grove_directly(grove, dict(adj), args.value_size)
        
        print(f"Running {op} benchmark...")
        results = benchmark_operation(grove, client, op, dict(adj), args.num_ops, args.value_size)
        print_results(op, results)


if __name__ == "__main__":
    main()
