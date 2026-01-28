"""
Benchmark script for simplified Grove Graph ORAM.

This script creates a simplified version of Grove with:
1. Only Graph ORAM (no PosMap ORAM)
2. Local position map (vertex_id -> position)
3. Internal duplications for neighbor position notifications
4. Configurable meta bucket size
5. Direct initialization (not insertion)
6. Tracking of meta ORAM stash size over time

Usage:
    Run directly in PyCharm or command line:
    python benchmark_grove.py
    
    Or with PyPy for ~5-10x speedup:
    pypy3 benchmark_grove.py
"""

import math
import random
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple




# ============================================================================
# Minimal Data and BinaryTree (no external dependencies)
# ============================================================================

@dataclass
class Data:
    """Simple data block for ORAM."""
    key: Any = None
    leaf: int = None
    value: Any = None


class BinaryTree:
    """Minimal BinaryTree implementation for ORAM."""
    
    def __init__(self, num_data: int, bucket_size: int, data_size: int = None):
        self._num_data = num_data
        self._bucket_size = bucket_size
        self._level = int(math.ceil(math.log(num_data, 2))) + 1
        self._size = pow(2, self._level) - 1
        self._start_leaf = pow(2, self._level - 1) - 1
        # Storage: list of buckets, each bucket is a list of Data
        self._storage: List[Optional[List[Data]]] = [None] * self._size
    
    @property
    def start_leaf(self) -> int:
        return self._start_leaf
    
    @property
    def storage(self) -> List:
        return self._storage
    
    @staticmethod
    def get_parent_index(index: int) -> int:
        return int(math.ceil(index / 2)) - 1
    
    @staticmethod
    def get_path_indices(index: int) -> List[int]:
        """Get indices from leaf to root."""
        path = []
        while index >= 0:
            path.append(index)
            index = BinaryTree.get_parent_index(index)
        return path
    
    def fill_data_to_storage_leaf(self, data: Data) -> bool:
        """Insert data into the tree at the appropriate leaf position."""
        leaf_index = self._start_leaf + data.leaf
        path_indices = self.get_path_indices(leaf_index)
        
        for idx in path_indices:
            if self._storage[idx] is None:
                self._storage[idx] = []
            if len(self._storage[idx]) < self._bucket_size:
                self._storage[idx].append(data)
                return True
        return False
    
    @staticmethod
    def get_cross_index(leaf_one: int, leaf_two: int, level: int) -> int:
        """Find the lowest index where two leaf paths cross."""
        start_leaf = pow(2, level - 1) - 1
        idx1 = leaf_one + start_leaf
        idx2 = leaf_two + start_leaf
        
        while idx1 != idx2:
            idx1 = int(math.ceil(idx1 / 2)) - 1
            idx2 = int(math.ceil(idx2 / 2)) - 1
        
        return idx1
    
    @staticmethod
    def fill_data_to_path(data: Data, path: Dict[int, List], leaves: List[int], 
                          level: int, bucket_size: int) -> bool:
        """
        Fill data to the lowest possible bucket in a PathData dict.
        Matches Grove's original eviction logic exactly.
        """
        if data.key is None:
            return True  # Dummy data, always "inserted"
        
        # Get the lowest crossed index for each leaf path
        indices = [BinaryTree.get_cross_index(data.leaf, leaf, level) for leaf in leaves]
        max_index = max(indices)
        
        # Go backwards from bottom to up
        while max_index >= 0:
            if max_index in path and len(path[max_index]) < bucket_size:
                path[max_index].append(data)
                return True
            else:
                max_index = BinaryTree.get_parent_index(max_index)
        
        return False


# ============================================================================
# Query Distribution (Black Box - Replace with your own implementation)
# ============================================================================

class QueryDistribution(ABC):
    """Abstract base class for query distribution."""
    
    @abstractmethod
    def sample(self) -> int:
        """Sample a vertex id from the distribution."""
        raise NotImplementedError
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the distribution state if needed."""
        raise NotImplementedError


class UniformDistribution(QueryDistribution):
    """Uniform random distribution over all vertices."""
    
    def __init__(self, vertex_num: int):
        self.vertex_num = vertex_num
    
    def sample(self) -> int:
        return random.randint(0, self.vertex_num - 1)
    
    def reset(self) -> None:
        pass


class ZipfDistribution(QueryDistribution):
    """Zipf distribution (power-law) for skewed access patterns."""
    
    def __init__(self, vertex_num: int, alpha: float = 1.0):
        self.vertex_num = vertex_num
        self.alpha = alpha
        # Precompute weights
        self.weights = [1.0 / ((i + 1) ** alpha) for i in range(vertex_num)]
        self.total_weight = sum(self.weights)
        self.cumulative = []
        cum = 0.0
        for w in self.weights:
            cum += w
            self.cumulative.append(cum / self.total_weight)
    
    def sample(self) -> int:
        r = random.random()
        for i, c in enumerate(self.cumulative):
            if r <= c:
                return i
        return self.vertex_num - 1
    
    def reset(self) -> None:
        pass


class SequentialDistribution(QueryDistribution):
    """Sequential access pattern (0, 1, 2, ..., n-1, 0, 1, ...)."""
    
    def __init__(self, vertex_num: int):
        self.vertex_num = vertex_num
        self.current = 0
    
    def sample(self) -> int:
        result = self.current
        self.current = (self.current + 1) % self.vertex_num
        return result
    
    def reset(self) -> None:
        self.current = 0


# ============================================================================
# D-Regular Graph Generator
# ============================================================================

def generate_d_regular_graph(vertex_num: int, degree: int) -> Dict[int, Set[int]]:
    """
    Generate a d-regular graph where each vertex has exactly 'degree' neighbors.
    
    Uses circulant graph construction: each vertex i connects to vertices
    (i ± 1), (i ± 2), ..., (i ± degree/2) mod vertex_num.
    For odd degree, also connects to vertex (i + vertex_num/2) if vertex_num is even.
    
    Requirements:
    - vertex_num * degree must be even (each edge has two endpoints)
    - degree < vertex_num (can't have more neighbors than other vertices)
    
    :param vertex_num: Number of vertices (0 to vertex_num-1)
    :param degree: Degree of each vertex (max_degree)
    :return: Adjacency dict {vertex_id: set of neighbor_ids}
    """
    # Validate inputs
    if vertex_num * degree % 2 != 0:
        raise ValueError(f"vertex_num * degree must be even: {vertex_num} * {degree} = {vertex_num * degree}")
    if degree >= vertex_num:
        raise ValueError(f"degree must be < vertex_num: {degree} >= {vertex_num}")
    
    # Initialize adjacency dict
    adjacency: Dict[int, Set[int]] = {v: set() for v in range(vertex_num)}
    
    # Use circulant graph construction
    # Connect each vertex i to i+k and i-k for k = 1, 2, ..., floor(degree/2)
    half_degree = degree // 2
    
    for v in range(vertex_num):
        for k in range(1, half_degree + 1):
            # Connect to (v + k) mod n and (v - k) mod n
            neighbor_plus = (v + k) % vertex_num
            neighbor_minus = (v - k) % vertex_num
            
            if neighbor_plus != v:
                adjacency[v].add(neighbor_plus)
                adjacency[neighbor_plus].add(v)
            if neighbor_minus != v and neighbor_minus != neighbor_plus:
                adjacency[v].add(neighbor_minus)
                adjacency[neighbor_minus].add(v)
    
    # If degree is odd, add diameter edges (only works if vertex_num is even)
    if degree % 2 == 1:
        if vertex_num % 2 != 0:
            raise ValueError(f"For odd degree, vertex_num must be even: degree={degree}, vertex_num={vertex_num}")
        for v in range(vertex_num // 2):
            opposite = v + vertex_num // 2
            adjacency[v].add(opposite)
            adjacency[opposite].add(v)
    
    # Verify all vertices have correct degree
    for v in range(vertex_num):
        if len(adjacency[v]) != degree:
            raise RuntimeError(f"Vertex {v} has degree {len(adjacency[v])}, expected {degree}")
    
    return adjacency


# ============================================================================
# Simplified Graph ORAM (No PosMap ORAM)
# ============================================================================

class SimplifiedGraphORAM:
    """
    Simplified Graph ORAM with local position map.
    
    Features:
    - Graph ORAM stores vertex data (adjacency lists with neighbor positions)
    - Meta ORAM stores duplications for position updates
    - Local position map (not ORAM-protected)
    - Direct initialization (not insertion-based)
    """
    
    def __init__(
        self,
        vertex_num: int,
        max_degree: int,
        meta_bucket_size: int = 4,
        graph_bucket_size: int = 4,
        stash_scale: int = 7,
        data_size: int = 64,
    ):
        """
        Initialize the simplified Graph ORAM.
        
        :param vertex_num: Number of vertices
        :param max_degree: Maximum degree (each vertex has exactly this many neighbors)
        :param meta_bucket_size: Bucket size for meta ORAM (user-specified, not computed bound)
        :param graph_bucket_size: Bucket size for graph ORAM
        :param stash_scale: Stash scale for ORAM
        :param data_size: Size of dummy data for padding
        """
        self.vertex_num = vertex_num
        self.max_degree = max_degree
        self.meta_bucket_size = meta_bucket_size
        self.graph_bucket_size = graph_bucket_size
        self.stash_scale = stash_scale
        self.data_size = data_size
        
        # Compute tree level
        self._level = int(math.ceil(math.log(vertex_num, 2))) + 1
        self._leaf_range = pow(2, self._level - 1)
        self._stash_size = stash_scale * (self._level - 1) if self._level > 1 else stash_scale
        
        # Local position map: vertex_id -> leaf position
        self._pos_map: Dict[int, int] = {}
        
        # Graph ORAM stash and storage
        self._graph_stash: List[Data] = []
        self._graph_tree: Optional[BinaryTree] = None
        
        # Meta ORAM stash and storage (for duplications)
        self._meta_stash: List[Data] = []
        self._meta_tree: Optional[BinaryTree] = None
        
        # Statistics tracking
        self._max_meta_stash_after_write: int = 0
        self._meta_stash_history: List[Tuple[int, int]] = []  # (access_count, stash_size)
        self._access_count: int = 0
    
    def initialize(self, adjacency: Dict[int, Set[int]]) -> None:
        """
        Initialize the ORAM with a pre-built graph.
        
        :param adjacency: Adjacency dict {vertex_id: set of neighbor_ids}
        """
        # Initialize position map with random leaves
        for v in range(self.vertex_num):
            self._pos_map[v] = secrets.randbelow(self._leaf_range)
        
        # Build data map for graph ORAM
        # Value format: {neighbor_id: neighbor_leaf_position}
        graph_data_map: Dict[int, Dict[int, int]] = {}
        for v in range(self.vertex_num):
            neighbors = adjacency.get(v, set())
            neighbor_positions = {n: self._pos_map[n] for n in neighbors}
            graph_data_map[v] = neighbor_positions
        
        # Create graph ORAM tree
        self._graph_tree = BinaryTree(
            num_data=self.vertex_num,
            bucket_size=self.graph_bucket_size,
            data_size=self.data_size,
        )
        
        # Fill graph data to tree based on position map
        for v, leaf in self._pos_map.items():
            data = Data(key=v, leaf=leaf, value=graph_data_map[v])
            self._graph_tree.fill_data_to_storage_leaf(data)
        
        # Create meta ORAM tree (empty initially, no duplications needed after init)
        self._meta_tree = BinaryTree(
            num_data=self.vertex_num,
            bucket_size=self.meta_bucket_size,
            data_size=self.data_size,
        )
        
        # Clear stashes (should be empty after init)
        self._graph_stash = []
        self._meta_stash = []
    
    def _get_new_leaf(self) -> int:
        """Generate a random new leaf position."""
        return secrets.randbelow(self._leaf_range)
    
    def _read_path(self, label: str, leaf: int) -> List[Data]:
        """Read a path from ORAM and return all real data. Clears buckets after read."""
        tree = self._graph_tree if label == "graph" else self._meta_tree
        path_indices = BinaryTree.get_path_indices(tree.start_leaf + leaf)
        result = []
        
        for idx in path_indices:
            bucket = tree.storage[idx]
            if bucket is None:
                continue
            for data in bucket:
                if data is not None and data.key is not None:
                    result.append(data)
            # Clear the bucket after reading (Path ORAM semantics)
            tree.storage[idx] = []
        
        return result
    
    def _write_path(self, label: str, leaf: int, stash: List[Data], bucket_size: int) -> List[Data]:
        """
        Evict stash to path and write back.
        
        :param label: "graph" or "meta"
        :param leaf: Leaf position of the path
        :param stash: Current stash
        :param bucket_size: Bucket size for this ORAM
        :return: Remaining stash after eviction
        """
        tree = self._graph_tree if label == "graph" else self._meta_tree
        path_indices = BinaryTree.get_path_indices(tree.start_leaf + leaf)
        
        # Create path dict
        path = {idx: [] for idx in path_indices}
        
        # Try to evict each item in stash
        remaining_stash = []
        for data in stash:
            inserted = BinaryTree.fill_data_to_path(
                data=data,
                path=path,
                leaves=[leaf],
                level=self._level,
                bucket_size=bucket_size,
            )
            if not inserted:
                remaining_stash.append(data)
        
        # Pad buckets with dummy data and write back
        for idx in path_indices:
            bucket = path[idx]
            while len(bucket) < bucket_size:
                bucket.append(Data(key=None, leaf=None, value=None))
            tree.storage[idx] = bucket
        
        return remaining_stash
    
    def _apply_duplications(self, vertex_key: int, vertex_data: Data, 
                            meta_leaves_set: Set[int] = None) -> Data:
        """
        Apply duplications from meta stash to update vertex's neighbor positions.
        
        Only applies dups where dup.leaf is in the paths we read (similar to Grove).
        
        :param vertex_key: The vertex being accessed
        :param vertex_data: The vertex data
        :param meta_leaves_set: Set of leaves whose paths were read (for filtering)
        :return: Updated vertex data
        """
        adjacency = vertex_data.value.copy()
        
        # Find and apply all relevant duplications
        new_meta_stash = []
        for dup in self._meta_stash:
            # Only apply if: 1) key matches, 2) dup.leaf is in read paths
            if dup.key == vertex_key:
                if meta_leaves_set is None or dup.leaf in meta_leaves_set:
                    # Dup format: value = (source_vertex, new_position)
                    source_vertex, new_pos = dup.value
                    if source_vertex in adjacency:
                        adjacency[source_vertex] = new_pos
                    # Remove applied dup (don't add to new stash)
                else:
                    # Dup not on read path, keep it
                    new_meta_stash.append(dup)
            else:
                new_meta_stash.append(dup)
        
        self._meta_stash = new_meta_stash
        vertex_data.value = adjacency
        return vertex_data
    
    def _create_duplications(self, vertex_key: int, new_leaf: int, adjacency: Dict[int, int]) -> List[Data]:
        """
        Create duplications to notify neighbors of position change.
        
        :param vertex_key: The vertex that changed position
        :param new_leaf: The new position
        :param adjacency: The vertex's adjacency dict
        :return: List of new duplications
        """
        dups = []
        for neighbor_key, neighbor_leaf in adjacency.items():
            # Create dup: notify neighbor that vertex_key moved to new_leaf
            dup = Data(
                key=neighbor_key,
                leaf=neighbor_leaf,
                value=(vertex_key, new_leaf)
            )
            dups.append(dup)
        return dups
    
    def _dedup_meta_stash(self) -> None:
        """
        De-duplicate meta stash: keep only the latest dup for each (target, source) pair.
        """
        # Key: (target_vertex, source_vertex) -> latest dup
        latest_dups: Dict[Tuple[int, int], Data] = {}
        
        for dup in self._meta_stash:
            if dup.key is None:
                continue
            source_vertex, new_pos = dup.value
            key = (dup.key, source_vertex)
            # Later dups overwrite earlier ones (assuming stash order is oldest first)
            latest_dups[key] = dup
        
        self._meta_stash = list(latest_dups.values())
    
    def _get_rl_leaves(self, count: int) -> List[int]:
        """Get random leaves for eviction (similar to Grove's get_rl_leaf)."""
        return [secrets.randbelow(self._leaf_range) for _ in range(count)]
    
    def _read_paths(self, label: str, leaves: List[int]) -> List[Data]:
        """Read multiple paths from ORAM and return all real data. Clears buckets after read."""
        tree = self._graph_tree if label == "graph" else self._meta_tree
        result = []
        visited_indices = set()
        
        for leaf in leaves:
            path_indices = BinaryTree.get_path_indices(tree.start_leaf + leaf)
            for idx in path_indices:
                if idx in visited_indices:
                    continue
                visited_indices.add(idx)
                bucket = tree.storage[idx]
                if bucket is None:
                    continue
                for data in bucket:
                    if data is not None and data.key is not None:
                        result.append(data)
                # Clear the bucket after reading (Path ORAM semantics)
                tree.storage[idx] = []
        
        return result
    
    def _write_paths(self, label: str, leaves: List[int], stash: List[Data], bucket_size: int) -> List[Data]:
        """
        Evict stash to multiple paths and write back.
        
        :param label: "graph" or "meta"
        :param leaves: List of leaf positions
        :param stash: Current stash
        :param bucket_size: Bucket size for this ORAM
        :return: Remaining stash after eviction
        """
        tree = self._graph_tree if label == "graph" else self._meta_tree
        
        # Get all bucket indices touched by these paths
        path = {}
        for leaf in leaves:
            path_indices = BinaryTree.get_path_indices(tree.start_leaf + leaf)
            for idx in path_indices:
                if idx not in path:
                    path[idx] = []
        
        # Try to evict each item in stash
        remaining_stash = []
        for data in stash:
            inserted = BinaryTree.fill_data_to_path(
                data=data,
                path=path,
                leaves=leaves,
                level=self._level,
                bucket_size=bucket_size,
            )
            if not inserted:
                remaining_stash.append(data)
        
        # Pad buckets with dummy data and write back
        for idx in path:
            bucket = path[idx]
            while len(bucket) < bucket_size:
                bucket.append(Data(key=None, leaf=None, value=None))
            tree.storage[idx] = bucket
        
        return remaining_stash

    def vertex_access(self, vertex_id: int) -> Dict[int, int]:
        """
        Access a vertex: download it, apply dups, create new dups, write back.
        
        Similar to Grove's neighbor query logic:
        1. Read graph ORAM path for the vertex
        2. Read meta ORAM paths (RL paths + vertex path) for dup application and eviction
        3. Apply dups to update neighbor positions
        4. Create new dups (D dups for D neighbors)
        5. Write back with eviction
        
        :param vertex_id: The vertex to access
        :return: The vertex's adjacency dict
        """
        # Step 1: Get current position from local pos map
        current_leaf = self._pos_map[vertex_id]
        
        # Step 2: Generate new position
        new_leaf = self._get_new_leaf()
        self._pos_map[vertex_id] = new_leaf
        
        # Step 3: Read graph path (just 1 path for the vertex)
        graph_path_data = self._read_path("graph", current_leaf)
        self._graph_stash.extend(graph_path_data)
        
        # Step 4: Read meta paths - D RL paths + 1 vertex path = D+1 total
        # D RL paths for eviction, 1 vertex path for applying dups to this vertex
        meta_rl_leaves = self._get_rl_leaves(self.max_degree)
        meta_all_leaves = meta_rl_leaves + [current_leaf]
        meta_path_data = self._read_paths("meta", meta_all_leaves)
        self._meta_stash.extend(meta_path_data)
        
        # Step 5: Find the vertex in stash
        vertex_data = None
        for data in self._graph_stash:
            if data.key == vertex_id:
                vertex_data = data
                break
        
        if vertex_data is None:
            raise KeyError(f"Vertex {vertex_id} not found in stash!")
        
        # Step 6: Apply duplications to update neighbor positions
        # Only apply dups whose dup.leaf is in the paths we read
        meta_leaves_set = set(meta_all_leaves)
        vertex_data = self._apply_duplications(vertex_id, vertex_data, meta_leaves_set)
        
        # Step 7: Update vertex's leaf and get adjacency
        vertex_data.leaf = new_leaf
        adjacency = vertex_data.value
        
        # Step 8: Update pos_map for neighbors based on current adjacency
        # (In simplified version, adjacency stores neighbor positions directly)
        
        # Step 9: Create new duplications for neighbors
        new_dups = self._create_duplications(vertex_id, new_leaf, adjacency)
        
        # Step 10: Add new dups to meta stash (at front for recency)
        self._meta_stash = new_dups + self._meta_stash
        
        # Step 11: De-duplicate meta stash
        self._dedup_meta_stash()
        
        # Step 12: Write back graph path
        self._graph_stash = self._write_path("graph", current_leaf, self._graph_stash, self.graph_bucket_size)
        
        # Step 13: Write back meta paths (same D+1 paths we read)
        self._meta_stash = self._write_paths("meta", meta_all_leaves, self._meta_stash, self.meta_bucket_size)
        
        # Step 14: Update statistics (after write)
        self._access_count += 1
        current_meta_stash_size = len(self._meta_stash)
        self._max_meta_stash_after_write = max(self._max_meta_stash_after_write, current_meta_stash_size)
        self._meta_stash_history.append((self._access_count, current_meta_stash_size))
        
        return adjacency
    
    def get_max_meta_stash_at_checkpoints(self, checkpoints: List[int]) -> Dict[int, int]:
        """
        Get the maximum meta stash size seen up to each checkpoint.
        
        :param checkpoints: Sorted list of access counts [a, b, c, ...]
        :return: Dict mapping checkpoint -> max stash size up to that point
        """
        result = {}
        max_so_far = 0
        checkpoint_idx = 0
        
        for access_num, stash_size in self._meta_stash_history:
            max_so_far = max(max_so_far, stash_size)
            
            while checkpoint_idx < len(checkpoints) and access_num >= checkpoints[checkpoint_idx]:
                result[checkpoints[checkpoint_idx]] = max_so_far
                checkpoint_idx += 1
        
        # Fill remaining checkpoints with current max
        while checkpoint_idx < len(checkpoints):
            result[checkpoints[checkpoint_idx]] = max_so_far
            checkpoint_idx += 1
        
        return result
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            "access_count": self._access_count,
            "graph_stash_size": len(self._graph_stash),
            "meta_stash_size": len(self._meta_stash),
            "max_meta_stash_after_write": self._max_meta_stash_after_write,
        }


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_benchmark(
    vertex_num: int,
    max_degree: int,
    meta_bucket_size: int,
    total_accesses: int,
    checkpoints: List[int],
    query_distribution: QueryDistribution = None,
    graph_bucket_size: int = 4,
    stash_scale: int = 7,
    seed: int = None,
) -> Dict[int, int]:
    """
    Run benchmark and return max meta stash size at each checkpoint.
    
    :param vertex_num: Number of vertices
    :param max_degree: Degree of each vertex
    :param meta_bucket_size: Bucket size for meta ORAM
    :param total_accesses: Total number of vertex accesses to perform
    :param checkpoints: Sorted list of access counts to report max stash size
    :param query_distribution: Distribution for selecting vertices (default: uniform)
    :param graph_bucket_size: Bucket size for graph ORAM
    :param stash_scale: Stash scale
    :param seed: Random seed for reproducibility
    :return: Dict mapping checkpoint -> max meta stash size up to that point
    """
    if seed is not None:
        random.seed(seed)
    
    # Default to uniform distribution
    if query_distribution is None:
        query_distribution = UniformDistribution(vertex_num)
    
    print(f"Generating {max_degree}-regular graph with {vertex_num} vertices...")
    adjacency = generate_d_regular_graph(vertex_num, max_degree)
    
    print(f"Initializing Graph ORAM (meta_bucket_size={meta_bucket_size})...")
    oram = SimplifiedGraphORAM(
        vertex_num=vertex_num,
        max_degree=max_degree,
        meta_bucket_size=meta_bucket_size,
        graph_bucket_size=graph_bucket_size,
        stash_scale=stash_scale,
    )
    oram.initialize(adjacency)
    
    print(f"Running {total_accesses} vertex accesses...")
    
    # 自定义进度输出
    report_step = max(1, total_accesses // 100)
    for i in range(total_accesses):
        vertex_id = query_distribution.sample()
        oram.vertex_access(vertex_id)
        if (i + 1) % report_step == 0 or i == total_accesses - 1:
            percent = int((i + 1) / total_accesses * 100)
            stats = oram.get_current_stats()
            print(f"[META_BUCKET_SIZE={meta_bucket_size}] Progress: {percent}% | max_stash_size={stats['max_meta_stash_after_write']}", flush=True)
    
    # Get results at checkpoints
    results = oram.get_max_meta_stash_at_checkpoints(checkpoints)
    
    print("\nResults:")
    for cp in checkpoints:
        print(f"  After {cp} accesses: max_meta_stash = {results.get(cp, 'N/A')}")
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

import argparse

if __name__ == "__main__":
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="Simplified Grove Graph ORAM Benchmark")
    parser.add_argument("--META_BUCKET_SIZE", type=int, default=4, help="Meta bucket size")
    parser.add_argument("--VERTEX_NUM", type=int, default=pow(2,20), help="Number of vertices")
    parser.add_argument("--MAX_DEGREE", type=int, default=10, help="Degree of each vertex")
    parser.add_argument("--GRAPH_BUCKET_SIZE", type=int, default=4, help="Graph bucket size")
    parser.add_argument("--TOTAL_ACCESSES", type=int, default=pow(2,21), help="Total accesses")
    parser.add_argument("--SEED", type=int, default=42, help="Random seed")
    # 兼容原有无效参数（不影响运行）
    parser.add_argument("--operation", type=str, default="lookup")
    parser.add_argument("--num_vertices", type=int, default=128)
    parser.add_argument("--num_edges", type=int, default=256)
    parser.add_argument("--max_deg", type=int, default=6)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--latency_ms", type=int, default=0)
    args = parser.parse_args()

    # ========== Configuration ==========
    # Graph parameters
    VERTEX_NUM = args.VERTEX_NUM             # Number of vertices
    MAX_DEGREE = args.MAX_DEGREE             # Each vertex has exactly this many neighbors
    
    # ORAM parameters
    META_BUCKET_SIZE = args.META_BUCKET_SIZE # User-specified meta bucket size
    GRAPH_BUCKET_SIZE = args.GRAPH_BUCKET_SIZE # Graph ORAM bucket size
    STASH_SCALE = 100000000000000            # Stash scale
    
    # Benchmark parameters
    TOTAL_ACCESSES = args.TOTAL_ACCESSES     # Total number of vertex accesses
    CHECKPOINTS = [pow(2,i) for i in range(10, 22)]  # Report max stash at these points
    
    # Random seed (set to None for random behavior)
    SEED = args.SEED
    
    # Query distribution (replace with your own implementation)
    # Options: UniformDistribution, ZipfDistribution, SequentialDistribution
    distribution = UniformDistribution(VERTEX_NUM)
    # distribution = ZipfDistribution(VERTEX_NUM, alpha=1.0)
    # distribution = SequentialDistribution(VERTEX_NUM)
    
    # ========== Run Benchmark ==========
    print("=" * 60)
    print("Simplified Grove Graph ORAM Benchmark")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  vertex_num = {VERTEX_NUM}")
    print(f"  max_degree = {MAX_DEGREE}")
    print(f"  meta_bucket_size = {META_BUCKET_SIZE}")
    print(f"  total_accesses = {TOTAL_ACCESSES}")
    print(f"  checkpoints = {CHECKPOINTS}")
    print("=" * 60)
    
    results = run_benchmark(
        vertex_num=VERTEX_NUM,
        max_degree=MAX_DEGREE,
        meta_bucket_size=META_BUCKET_SIZE,
        total_accesses=TOTAL_ACCESSES,
        checkpoints=CHECKPOINTS,
        query_distribution=distribution,
        graph_bucket_size=GRAPH_BUCKET_SIZE,
        stash_scale=STASH_SCALE,
        seed=SEED,
    )
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)
