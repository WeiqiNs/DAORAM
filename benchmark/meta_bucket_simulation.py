"""
Simulation to analyze meta bucket utilization in Grove's delayed duplication scheme.

This simulates how duplications are created and consumed over time to understand:
1. Actual bucket utilization vs theoretical bound from find_bound()
2. Distribution of duplications across buckets
3. Peak utilization patterns

Usage:
    # Basic simulation (uniform random access)
    python benchmark/meta_bucket_simulation.py --num-data 512 --max-deg 10 --num-ops 5000

    # With more trials for statistical confidence
    python benchmark/meta_bucket_simulation.py --num-data 512 --max-deg 10 --num-ops 5000 --num-trials 10

    # Zipf distribution (skewed access pattern)
    python benchmark/meta_bucket_simulation.py --zipf --alpha 1.5

    # Hotspot simulation (10% vertices get 80% accesses)
    python benchmark/meta_bucket_simulation.py --hotspot

    # Burst access pattern
    python benchmark/meta_bucket_simulation.py --burst

    # Compare different parameter configurations
    python benchmark/meta_bucket_simulation.py --compare

    # Run all skewed distribution tests
    python benchmark/meta_bucket_simulation.py --all-skewed

    # See all options
    python benchmark/meta_bucket_simulation.py --help

Key metrics:
    - Theoretical bound: Bucket size for 2^-128 overflow probability (from find_bound())
    - Max utilization: Peak bucket size observed during simulation
    - Utilization ratio: max_observed / theoretical_bound (typically ~13-19%)
"""

import math
import random
import secrets
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from typing import Dict, List, Tuple, Set


@dataclass
class SimulationConfig:
    """Configuration for the simulation."""
    num_data: int = 1024          # Number of vertices in the graph
    max_deg: int = 4              # Maximum degree of each vertex
    num_operations: int = 1000    # Number of lookup operations to simulate
    num_trials: int = 10          # Number of independent trials
    seed: int = None              # Random seed for reproducibility
    bucket_capacity: int = 4      # Bucket capacity (like in real ORAM)
    eviction_mode: str = "greedy" # "greedy" (deepest possible) or "random"


@dataclass
class BucketStats:
    """Statistics for a single bucket."""
    max_size: int = 0
    total_writes: int = 0
    total_reads: int = 0
    current_size: int = 0


@dataclass
class SimulationResult:
    """Results from a single simulation trial."""
    max_bucket_utilization: int = 0
    avg_bucket_utilization: float = 0.0
    total_duplications_created: int = 0
    total_duplications_consumed: int = 0
    bucket_size_distribution: Dict[int, int] = field(default_factory=dict)
    utilization_over_time: List[int] = field(default_factory=list)


class MetaBucketSimulator:
    """Simulates the meta bucket behavior in Grove."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.level = int(math.ceil(math.log2(config.num_data))) + 1
        self.leaf_range = 2 ** (self.level - 1)

        # RL counter for path generation
        self.rl_counter = 0

        # Buckets: indexed by tree node index (0 to 2^level - 2)
        # Each bucket contains a list of (vertex_key, source_key, new_leaf) tuples
        self.num_buckets = 2 ** self.level - 1
        self.buckets: Dict[int, List[Tuple]] = defaultdict(list)

        # Bucket capacity (for realistic eviction simulation)
        self.bucket_capacity = config.bucket_capacity

        # Track statistics per bucket
        self.bucket_stats: Dict[int, BucketStats] = defaultdict(BucketStats)

        # Graph structure: vertex_key -> {neighbor_key: leaf}
        self.graph: Dict[int, Dict[int, int]] = {}

        # Position map: vertex_key -> leaf
        self.pos_map: Dict[int, int] = {}

        # Stash for overflow (like in real ORAM)
        self.stash: List[Tuple] = []

        # Initialize random graph
        self._init_random_graph()

    def _init_random_graph(self):
        """Initialize a random bidirectional graph with exactly max_deg neighbors per vertex.

        In real Grove, when A links to B, B also links to A (undirected graph).
        Each vertex has EXACTLY max_deg neighbors to ensure:
        - Duplications created per op = max_deg
        - RL paths read per op = max_deg
        - These are balanced for steady-state operation
        """
        # First pass: assign random leaves to all vertices
        for v in range(self.config.num_data):
            self.pos_map[v] = secrets.randbelow(self.leaf_range)
            self.graph[v] = {}

        # Second pass: create bidirectional edges
        for v in range(self.config.num_data):
            current_deg = len(self.graph[v])
            needed = max(0, self.config.max_deg - current_deg)

            if needed > 0:
                # Find candidates that also need more neighbors
                candidates = [u for u in range(self.config.num_data)
                             if u != v and u not in self.graph[v]
                             and len(self.graph[u]) < self.config.max_deg]

                new_neighbors = random.sample(candidates, min(needed, len(candidates)))

                for n in new_neighbors:
                    self.graph[v][n] = self.pos_map[n]
                    self.graph[n][v] = self.pos_map[v]

        # Third pass: trim excess edges to ensure exactly max_deg
        for v in range(self.config.num_data):
            while len(self.graph[v]) > self.config.max_deg:
                # Remove a random neighbor
                neighbor = random.choice(list(self.graph[v].keys()))
                del self.graph[v][neighbor]
                if v in self.graph[neighbor]:
                    del self.graph[neighbor][v]

    def get_rl_leaf(self, count: int) -> List[int]:
        """Get the next 'count' RL leaves."""
        num_bits = self.level - 1
        leaves = []
        for i in range(count):
            val = (self.rl_counter + i) % self.leaf_range
            # Reverse the bits
            reversed_val = int(format(val, f'0{num_bits}b')[::-1], 2)
            leaves.append(reversed_val)
        self.rl_counter = (self.rl_counter + count) % self.leaf_range
        return leaves

    def get_path_indices(self, leaf: int) -> List[int]:
        """Get all bucket indices on the path from root to leaf.

        Returns indices in ROOT-TO-LEAF order (root first, leaf last).
        This ensures higher-level buckets have higher priority in deduplication.
        """
        indices = []
        # Start at the leaf level and go up to root
        # Leaf index in full tree = leaf_range - 1 + leaf
        node_idx = self.leaf_range - 1 + leaf
        while node_idx >= 0:
            indices.append(node_idx)
            if node_idx == 0:
                break
            node_idx = (node_idx - 1) // 2
        # Reverse to get root-to-leaf order (higher level = higher priority)
        return indices[::-1]

    def add_duplication_to_stash(self, target_key: int, source_key: int, new_leaf: int, target_leaf: int):
        """Add a duplication to the stash (will be evicted later)."""
        self.stash.append((target_key, source_key, new_leaf, target_leaf))

    def evict_stash_to_paths(self, paths: List[int]):
        """
        Evict duplications from stash to the given paths.

        This mimics the real ORAM eviction: try to place each duplication
        as deep as possible on any of the given paths.
        """
        # Get all bucket indices on all paths
        path_buckets: Dict[int, Set[int]] = {}  # bucket_idx -> set of paths it's on
        for leaf in paths:
            for bucket_idx in self.get_path_indices(leaf):
                if bucket_idx not in path_buckets:
                    path_buckets[bucket_idx] = set()
                path_buckets[bucket_idx].add(leaf)

        # Sort buckets by level (deepest first = higher index)
        sorted_buckets = sorted(path_buckets.keys(), reverse=True)

        new_stash = []
        for dup in self.stash:
            target_key, source_key, new_leaf, target_leaf = dup
            placed = False

            if self.config.eviction_mode == "greedy":
                # Try to place as deep as possible
                target_path_indices = set(self.get_path_indices(target_leaf))

                for bucket_idx in sorted_buckets:
                    # Can only place if bucket is on the duplication's target path
                    if bucket_idx not in target_path_indices:
                        continue

                    if len(self.buckets[bucket_idx]) < self.bucket_capacity:
                        # Store full 4-tuple including target_leaf for re-eviction after read
                        self.buckets[bucket_idx].append((target_key, source_key, new_leaf, target_leaf))
                        self._update_bucket_stats(bucket_idx)
                        placed = True
                        break
            else:
                # Random placement (original behavior)
                valid_buckets = [b for b in sorted_buckets
                                if b in set(self.get_path_indices(target_leaf))]
                if valid_buckets:
                    bucket_idx = random.choice(valid_buckets)
                    self.buckets[bucket_idx].append((target_key, source_key, new_leaf, target_leaf))
                    self._update_bucket_stats(bucket_idx)
                    placed = True

            if not placed:
                # Couldn't place, keep in stash (overflow)
                new_stash.append(dup)

        self.stash = new_stash

    def _update_bucket_stats(self, bucket_idx: int):
        """Update statistics for a bucket."""
        self.bucket_stats[bucket_idx].total_writes += 1
        self.bucket_stats[bucket_idx].current_size = len(self.buckets[bucket_idx])
        self.bucket_stats[bucket_idx].max_size = max(
            self.bucket_stats[bucket_idx].max_size,
            self.bucket_stats[bucket_idx].current_size
        )

    def read_path(self, leaf: int) -> List[Tuple]:
        """Read all duplications on a path (simulating a path read)."""
        path_indices = self.get_path_indices(leaf)
        collected = []

        for idx in path_indices:
            collected.extend(self.buckets[idx])
            self.bucket_stats[idx].total_reads += 1
            # Clear the bucket (duplications are consumed on read)
            self.buckets[idx] = []
            self.bucket_stats[idx].current_size = 0

        return collected

    def de_duplicate_stash(self):
        """
        Remove duplicate duplications from stash.

        Dedup key: (target_key, source_key)
        - If vertex A sends multiple updates to neighbor B, only keep the latest
        - Priority: front of stash = highest priority (most recent)
        """
        seen = set()
        new_stash = []

        for dup in self.stash:
            target_key, source_key, new_leaf, target_leaf = dup
            dedup_key = (target_key, source_key)

            if dedup_key not in seen:
                seen.add(dedup_key)
                new_stash.append(dup)

        removed = len(self.stash) - len(new_stash)
        self.stash = new_stash
        return removed

    def simulate_lookup(self, vertex_key: int) -> Tuple[int, int]:
        """
        Simulate a lookup operation on a vertex.

        Flow (matching Grove):
        1. Read RL paths + vertex path → duplications go to stash
        2. Deduplicate stash
        3. Apply duplications to accessed vertex
        4. Create new duplications (add to front of stash for highest priority)
        5. Deduplicate again
        6. Evict stash to paths

        Returns: (net_duplications_created, duplications_consumed)
        """
        # Get the vertex's current leaf
        old_leaf = self.pos_map[vertex_key]

        # Read RL paths + vertex path
        rl_paths = self.get_rl_leaf(count=self.config.max_deg)
        all_paths = list(set(rl_paths + [old_leaf]))  # Deduplicate paths

        # Step 1: Read paths - duplications from buckets go to stash
        for leaf in all_paths:
            dups = self.read_path(leaf)
            # Add read duplications directly to stash (they already have target_leaf)
            self.stash.extend(dups)

        # Step 2: Deduplicate stash (keep latest = front)
        self.de_duplicate_stash()

        # Step 3: Apply duplications to accessed vertex and remove consumed ones
        consumed = 0
        remaining_stash = []
        for dup in self.stash:
            target_key, source_key, new_leaf, target_leaf = dup
            if target_key == vertex_key and source_key in self.graph[vertex_key]:
                # Apply: update neighbor's leaf in adjacency list
                self.graph[vertex_key][source_key] = new_leaf
                consumed += 1
            else:
                remaining_stash.append(dup)
        self.stash = remaining_stash

        # Assign new leaf to the vertex
        new_leaf = secrets.randbelow(self.leaf_range)
        self.pos_map[vertex_key] = new_leaf

        # Step 4: Create new duplications (prepend to stash for highest priority)
        new_dups = []
        neighbors = self.graph[vertex_key]
        for neighbor_key, neighbor_leaf in neighbors.items():
            new_dups.append((neighbor_key, vertex_key, new_leaf, neighbor_leaf))

        # Prepend new duplications (they have highest priority)
        self.stash = new_dups + self.stash
        created = len(new_dups)

        # Step 5: Deduplicate again (new dups override old ones for same target,source)
        removed = self.de_duplicate_stash()

        # Step 6: Evict stash to paths
        self.evict_stash_to_paths(all_paths)

        return created - removed, consumed

    def get_max_bucket_size(self) -> int:
        """Get the current maximum bucket size across all buckets."""
        max_size = 0
        if self.buckets:
            max_size = max(len(b) for b in self.buckets.values())
        # Include stash overflow
        return max(max_size, len(self.stash))

    def get_total_duplications(self) -> int:
        """Get total duplications currently stored."""
        return sum(len(b) for b in self.buckets.values()) + len(self.stash)

    def get_stash_size(self) -> int:
        """Get current stash size (overflow)."""
        return len(self.stash)

    def run_trial(self) -> SimulationResult:
        """Run a single simulation trial."""
        result = SimulationResult()

        # Reset state
        self.rl_counter = 0
        self.buckets.clear()
        self.bucket_stats.clear()
        self.stash.clear()
        self._init_random_graph()

        max_stash_size = 0

        for op in range(self.config.num_operations):
            # Randomly select a vertex to lookup
            vertex_key = random.randint(0, self.config.num_data - 1)

            created, consumed = self.simulate_lookup(vertex_key)
            result.total_duplications_created += created
            result.total_duplications_consumed += consumed

            # Track utilization over time
            current_max = self.get_max_bucket_size()
            result.utilization_over_time.append(current_max)
            result.max_bucket_utilization = max(result.max_bucket_utilization, current_max)

            # Track stash overflow
            max_stash_size = max(max_stash_size, len(self.stash))

        # Compute final statistics
        bucket_sizes = [len(b) for b in self.buckets.values() if b]
        if bucket_sizes:
            result.avg_bucket_utilization = sum(bucket_sizes) / len(bucket_sizes)

        # Distribution of bucket sizes
        for bucket in self.buckets.values():
            size = len(bucket)
            result.bucket_size_distribution[size] = result.bucket_size_distribution.get(size, 0) + 1

        # Store max stash size in metadata
        result.bucket_size_distribution["max_stash"] = max_stash_size

        return result


def compute_theoretical_bound(num_operations: int, max_deg: int, level: int, prec: int = 80) -> int:
    """Compute the theoretical bucket size bound using Grove's formula."""

    def binomial(n: int, i: int, p: Decimal) -> Decimal:
        return Decimal(math.comb(n, i)) * (p ** i) * ((Decimal(1) - p) ** (n - i))

    def equation(m: int, K: int, Y: int, L: int) -> Decimal:
        sigma = math.ceil(math.log2(Y)) if Y > 0 else 1
        prob = Decimal(0)

        with localcontext() as ctx:
            ctx.prec = prec

            for j in range(sigma, L):
                temp_prob = Decimal(0)
                n = 2 ** j
                p = Decimal(1) / (Decimal(2) ** (j + 1))

                for i in range(math.floor(Y / K)):
                    temp_prob += binomial(n=n, i=i, p=p)

                term = (Decimal(2) ** j) * Decimal(math.ceil(m * K / (2 ** j))) * (Decimal(1) - temp_prob)
                prob += term

        return prob

    Y = 1
    target_prob = Decimal(1) / Decimal(2 ** 128)
    while equation(m=num_operations, K=max_deg, Y=Y, L=level) > target_prob:
        Y += 1

    return Y


def run_simulation(config: SimulationConfig) -> Dict:
    """Run the full simulation with multiple trials."""

    if config.seed is not None:
        random.seed(config.seed)

    simulator = MetaBucketSimulator(config)

    # Compute theoretical bound
    theoretical_bound = compute_theoretical_bound(
        num_operations=config.num_operations,
        max_deg=config.max_deg,
        level=simulator.level
    )

    print(f"Configuration:")
    print(f"  num_data: {config.num_data}")
    print(f"  max_deg: {config.max_deg}")
    print(f"  num_operations: {config.num_operations}")
    print(f"  bucket_capacity: {config.bucket_capacity}")
    print(f"  eviction_mode: {config.eviction_mode}")
    print(f"  tree_level: {simulator.level}")
    print(f"  leaf_range: {simulator.leaf_range}")
    print(f"  theoretical_bucket_bound: {theoretical_bound}")
    print()
    # Compute actual average degree
    avg_degree = sum(len(v) for v in simulator.graph.values()) / len(simulator.graph)

    print(f"Per-operation balance:")
    print(f"  Target max_deg: {config.max_deg}")
    print(f"  Actual avg degree: {avg_degree:.1f}")
    print(f"  Duplications created per op: ~{avg_degree:.1f} (one per neighbor)")
    print(f"  RL paths read per op: {config.max_deg}")
    print(f"  Total paths read per op: {config.max_deg} + 1 = {config.max_deg + 1}")
    print(f"  Ops to cycle all {simulator.leaf_range} leaves: ~{simulator.leaf_range // config.max_deg}")
    print()

    all_results = []
    max_utilizations = []

    for trial in range(config.num_trials):
        result = simulator.run_trial()
        all_results.append(result)
        max_utilizations.append(result.max_bucket_utilization)

        max_stash = result.bucket_size_distribution.get("max_stash", 0)
        # Compute total dups in system at end
        total_in_buckets = sum(len(b) for b in simulator.buckets.values())
        total_in_stash = len(simulator.stash)
        total_in_system = total_in_buckets + total_in_stash

        print(f"Trial {trial + 1}/{config.num_trials}:")
        print(f"  Max bucket utilization: {result.max_bucket_utilization}")
        print(f"  Avg bucket utilization: {result.avg_bucket_utilization:.2f}")
        print(f"  Max stash overflow: {max_stash}")
        print(f"  Total created: {result.total_duplications_created}")
        print(f"  Total consumed: {result.total_duplications_consumed}")
        print(f"  Total in system at end: {total_in_system} (buckets={total_in_buckets}, stash={total_in_stash})")

    print()
    print("Summary across all trials:")
    print(f"  Theoretical bound: {theoretical_bound}")
    print(f"  Max observed utilization: {max(max_utilizations)}")
    print(f"  Avg max utilization: {sum(max_utilizations) / len(max_utilizations):.2f}")
    print(f"  Min max utilization: {min(max_utilizations)}")
    print(f"  Utilization ratio (max_observed / theoretical): {max(max_utilizations) / theoretical_bound:.2%}")

    return {
        "config": config,
        "theoretical_bound": theoretical_bound,
        "results": all_results,
        "max_utilizations": max_utilizations,
    }


def analyze_utilization_over_time(result: SimulationResult, window_size: int = 50):
    """Analyze how utilization changes over time."""
    utilization = result.utilization_over_time

    print(f"\nUtilization over time (window={window_size}):")
    for i in range(0, len(utilization), window_size):
        window = utilization[i:i + window_size]
        avg = sum(window) / len(window)
        max_val = max(window)
        print(f"  Ops {i:4d}-{i + len(window) - 1:4d}: avg={avg:.2f}, max={max_val}")


def analyze_by_level(simulator: MetaBucketSimulator) -> Dict[int, Dict[str, float]]:
    """Analyze bucket utilization by tree level."""
    level_stats = {}

    for level in range(simulator.level):
        # Get bucket indices at this level
        start_idx = 2 ** level - 1
        end_idx = 2 ** (level + 1) - 1
        num_buckets_at_level = 2 ** level

        level_sizes = []
        level_max_sizes = []
        for idx in range(start_idx, min(end_idx, simulator.num_buckets)):
            level_sizes.append(len(simulator.buckets.get(idx, [])))
            if idx in simulator.bucket_stats:
                level_max_sizes.append(simulator.bucket_stats[idx].max_size)

        if level_sizes:
            level_stats[level] = {
                "avg_current": sum(level_sizes) / len(level_sizes),
                "max_current": max(level_sizes),
                "total_current": sum(level_sizes),
                "max_ever": max(level_max_sizes) if level_max_sizes else 0,
                "num_buckets": num_buckets_at_level,
            }

    return level_stats


def run_hotspot_simulation(config: SimulationConfig) -> Dict:
    """
    Simulate a hotspot access pattern where certain vertices are accessed frequently.

    This tests the worst-case scenario where duplications concentrate in certain areas.
    """
    print(f"\n{'='*80}")
    print("Hotspot Simulation")
    print(f"{'='*80}")

    simulator = MetaBucketSimulator(config)
    theoretical_bound = compute_theoretical_bound(
        num_operations=config.num_operations,
        max_deg=config.max_deg,
        level=simulator.level
    )

    print(f"Configuration:")
    print(f"  num_data: {config.num_data}")
    print(f"  max_deg: {config.max_deg}")
    print(f"  num_operations: {config.num_operations}")
    print(f"  theoretical_bound: {theoretical_bound}")
    print()

    # Create hotspot vertices (10% of vertices get 80% of accesses)
    num_hotspots = max(1, config.num_data // 10)
    hotspot_vertices = random.sample(range(config.num_data), num_hotspots)

    # Reset simulator
    simulator.rl_counter = 0
    simulator.buckets.clear()
    simulator.bucket_stats.clear()
    simulator.stash.clear()
    simulator._init_random_graph()

    max_utilization = 0
    max_stash = 0

    for op in range(config.num_operations):
        # 80% chance to access a hotspot, 20% chance for random
        if random.random() < 0.8:
            vertex_key = random.choice(hotspot_vertices)
        else:
            vertex_key = random.randint(0, config.num_data - 1)

        simulator.simulate_lookup(vertex_key)

        current_max = simulator.get_max_bucket_size()
        max_utilization = max(max_utilization, current_max)
        max_stash = max(max_stash, len(simulator.stash))

    # Analyze by level
    level_stats = analyze_by_level(simulator)

    print(f"Results:")
    print(f"  Max bucket utilization: {max_utilization}")
    print(f"  Max stash overflow: {max_stash}")
    print(f"  Utilization ratio: {max_utilization / theoretical_bound:.2%}")
    print()

    print("Utilization by tree level:")
    print(f"{'Level':>6} {'Buckets':>8} {'Avg':>8} {'Max':>8} {'MaxEver':>8}")
    print("-" * 42)
    for level, stats in level_stats.items():
        print(f"{level:>6} {stats['num_buckets']:>8} {stats['avg_current']:>8.2f} "
              f"{stats['max_current']:>8} {stats['max_ever']:>8}")

    return {
        "max_utilization": max_utilization,
        "theoretical_bound": theoretical_bound,
        "level_stats": level_stats,
    }


def run_burst_simulation(config: SimulationConfig) -> Dict:
    """
    Simulate burst access pattern where many duplications are created quickly.

    This tests the scenario where many operations happen before RL paths can clean up.
    """
    print(f"\n{'='*80}")
    print("Burst Access Simulation")
    print(f"{'='*80}")

    simulator = MetaBucketSimulator(config)
    theoretical_bound = compute_theoretical_bound(
        num_operations=config.num_operations,
        max_deg=config.max_deg,
        level=simulator.level
    )

    print(f"Configuration:")
    print(f"  num_data: {config.num_data}")
    print(f"  max_deg: {config.max_deg}")
    print(f"  num_operations: {config.num_operations}")
    print(f"  theoretical_bound: {theoretical_bound}")
    print()

    # Reset simulator
    simulator.rl_counter = 0
    simulator.buckets.clear()
    simulator.bucket_stats.clear()
    simulator.stash.clear()
    simulator._init_random_graph()

    # Find a vertex with maximum neighbors
    max_neighbors_vertex = max(range(config.num_data),
                                key=lambda v: len(simulator.graph.get(v, {})))

    # Repeatedly access this vertex and its neighbors in bursts
    max_utilization = 0
    utilization_over_time = []

    burst_size = 20  # Operations per burst
    num_bursts = config.num_operations // burst_size

    for burst in range(num_bursts):
        # In each burst, access the center vertex and a random neighbor
        center = max_neighbors_vertex
        neighbors = list(simulator.graph.get(center, {}).keys())

        for _ in range(burst_size):
            if random.random() < 0.5 and neighbors:
                vertex = random.choice(neighbors)
            else:
                vertex = center

            simulator.simulate_lookup(vertex)

            current_max = simulator.get_max_bucket_size()
            max_utilization = max(max_utilization, current_max)
            utilization_over_time.append(current_max)

    print(f"Results:")
    print(f"  Max bucket utilization: {max_utilization}")
    print(f"  Max stash overflow: {len(simulator.stash)}")
    print(f"  Utilization ratio: {max_utilization / theoretical_bound:.2%}")

    # Show utilization trend
    if utilization_over_time:
        chunks = [utilization_over_time[i:i+100] for i in range(0, len(utilization_over_time), 100)]
        print(f"\nUtilization trend (per 100 ops):")
        for i, chunk in enumerate(chunks[:10]):  # Show first 10 chunks
            print(f"  Ops {i*100:>4}-{i*100+len(chunk)-1:>4}: max={max(chunk)}, avg={sum(chunk)/len(chunk):.2f}")

    return {
        "max_utilization": max_utilization,
        "theoretical_bound": theoretical_bound,
        "utilization_over_time": utilization_over_time,
    }


def zipf_sample(n: int, alpha: float = 1.0) -> int:
    """Sample from Zipf distribution over [0, n-1] with parameter alpha."""
    # Use numpy-style Zipf if available, otherwise use inverse transform
    # Zipf: P(rank k) ∝ 1/(k+1)^alpha for k in [0, n-1]

    # Compute normalization constant (harmonic number)
    h_n = sum(1.0 / (k + 1) ** alpha for k in range(n))

    # Inverse transform sampling
    u = random.random() * h_n
    cumsum = 0.0
    for k in range(n):
        cumsum += 1.0 / (k + 1) ** alpha
        if u <= cumsum:
            return k

    return n - 1  # Fallback


def run_zipf_simulation(config: SimulationConfig, alpha: float = 1.0) -> Dict:
    """
    Simulate with Zipf-distributed access pattern.

    Zipf distribution: P(rank k) ∝ 1/k^alpha
    - alpha=1.0: Standard Zipf (80-20 rule approximately)
    - alpha>1.0: More skewed (fewer items get more accesses)
    - alpha<1.0: Less skewed (more uniform)

    This models real-world access patterns where popular items are accessed much more frequently.
    """
    print(f"\n{'='*80}")
    print(f"Zipf Distribution Simulation (alpha={alpha})")
    print(f"{'='*80}")

    simulator = MetaBucketSimulator(config)
    theoretical_bound = compute_theoretical_bound(
        num_operations=config.num_operations,
        max_deg=config.max_deg,
        level=simulator.level
    )

    print(f"Configuration:")
    print(f"  num_data: {config.num_data}")
    print(f"  max_deg: {config.max_deg}")
    print(f"  num_operations: {config.num_operations}")
    print(f"  zipf_alpha: {alpha}")
    print(f"  theoretical_bound: {theoretical_bound}")
    print()

    # Reset simulator
    simulator.rl_counter = 0
    simulator.buckets.clear()
    simulator.bucket_stats.clear()
    simulator.stash.clear()
    simulator._init_random_graph()

    # Sort vertices by their "popularity" (we'll use vertex index as rank)
    # Lower index = higher popularity under Zipf

    max_utilization = 0
    max_stash = 0
    utilization_over_time = []
    access_counts = defaultdict(int)

    for op in range(config.num_operations):
        # Sample vertex using Zipf distribution
        rank = zipf_sample(config.num_data, alpha)
        vertex_key = rank  # Vertex 0 is most popular, vertex n-1 is least

        access_counts[vertex_key] += 1
        simulator.simulate_lookup(vertex_key)

        current_max = simulator.get_max_bucket_size()
        max_utilization = max(max_utilization, current_max)
        max_stash = max(max_stash, len(simulator.stash))
        utilization_over_time.append(current_max)

    level_stats = analyze_by_level(simulator)

    print(f"Results:")
    print(f"  Max bucket utilization: {max_utilization}")
    print(f"  Max stash overflow: {max_stash}")
    print(f"  Utilization ratio: {max_utilization / theoretical_bound:.2%}")
    print()

    # Show access distribution
    sorted_access = sorted(access_counts.items(), key=lambda x: x[1], reverse=True)
    top_10_accesses = sum(count for _, count in sorted_access[:10])
    total_accesses = sum(access_counts.values())
    print(f"Access distribution:")
    print(f"  Top 10 vertices: {top_10_accesses}/{total_accesses} = {100*top_10_accesses/total_accesses:.1f}% of accesses")
    print(f"  Top 10 most accessed vertices: {sorted_access[:10]}")
    print()

    print("Utilization by tree level:")
    print(f"{'Level':>6} {'Buckets':>8} {'Avg':>8} {'Max':>8} {'MaxEver':>8}")
    print("-" * 42)
    for level, stats in level_stats.items():
        print(f"{level:>6} {stats['num_buckets']:>8} {stats['avg_current']:>8.2f} "
              f"{stats['max_current']:>8} {stats['max_ever']:>8}")

    return {
        "max_utilization": max_utilization,
        "theoretical_bound": theoretical_bound,
        "level_stats": level_stats,
        "access_counts": dict(access_counts),
        "utilization_over_time": utilization_over_time,
    }


def run_zipf_with_locality_simulation(config: SimulationConfig, alpha: float = 1.0) -> Dict:
    """
    Zipf access pattern combined with spatial locality in the ORAM tree.

    Popular vertices AND their neighbors are assigned to nearby leaves,
    creating concentrated duplication patterns.
    """
    print(f"\n{'='*80}")
    print(f"Zipf + Spatial Locality Simulation (alpha={alpha})")
    print(f"{'='*80}")

    simulator = MetaBucketSimulator(config)
    theoretical_bound = compute_theoretical_bound(
        num_operations=config.num_operations,
        max_deg=config.max_deg,
        level=simulator.level
    )

    print(f"Configuration:")
    print(f"  num_data: {config.num_data}")
    print(f"  max_deg: {config.max_deg}")
    print(f"  num_operations: {config.num_operations}")
    print(f"  zipf_alpha: {alpha}")
    print(f"  theoretical_bound: {theoretical_bound}")
    print()

    # Reset simulator
    simulator.rl_counter = 0
    simulator.buckets.clear()
    simulator.bucket_stats.clear()
    simulator.stash.clear()

    # Create graph with spatial locality:
    # Popular vertices (low rank) and their neighbors get nearby leaves

    # Assign leaves with locality - popular vertices share a small leaf range
    hot_zone_size = max(4, simulator.leaf_range // 8)

    for v in range(config.num_data):
        # Popular vertices (lower index) get leaves in hot zone
        popularity_factor = 1.0 - (v / config.num_data)  # 1.0 for v=0, 0.0 for v=n-1

        if random.random() < popularity_factor:
            # Assign to hot zone
            simulator.pos_map[v] = random.randint(0, hot_zone_size - 1)
        else:
            # Assign randomly
            simulator.pos_map[v] = secrets.randbelow(simulator.leaf_range)

        # Create neighbors - popular vertices connect to other popular vertices
        num_neighbors = random.randint(1, config.max_deg)

        # Bias neighbor selection toward similar popularity
        neighbor_range = max(10, int(config.num_data * 0.3))  # 30% of vertices around this one
        low = max(0, v - neighbor_range // 2)
        high = min(config.num_data, v + neighbor_range // 2)
        potential = [x for x in range(low, high) if x != v]

        if len(potential) < num_neighbors:
            potential = [x for x in range(config.num_data) if x != v]

        neighbors = random.sample(potential, min(num_neighbors, len(potential)))
        simulator.graph[v] = {n: simulator.pos_map[n] for n in neighbors}

    print(f"  Hot zone: leaves 0 to {hot_zone_size - 1}")
    hot_zone_vertices = sum(1 for v in range(config.num_data)
                           if simulator.pos_map[v] < hot_zone_size)
    print(f"  Vertices in hot zone: {hot_zone_vertices}/{config.num_data}")
    print()

    # Run simulation with Zipf access
    max_utilization = 0
    max_stash = 0
    utilization_over_time = []
    access_counts = defaultdict(int)

    for op in range(config.num_operations):
        rank = zipf_sample(config.num_data, alpha)
        vertex_key = rank

        access_counts[vertex_key] += 1
        simulator.simulate_lookup(vertex_key)

        current_max = simulator.get_max_bucket_size()
        max_utilization = max(max_utilization, current_max)
        max_stash = max(max_stash, len(simulator.stash))
        utilization_over_time.append(current_max)

    level_stats = analyze_by_level(simulator)

    print(f"Results:")
    print(f"  Max bucket utilization: {max_utilization}")
    print(f"  Max stash overflow: {max_stash}")
    print(f"  Utilization ratio: {max_utilization / theoretical_bound:.2%}")
    print()

    # Show access distribution
    sorted_access = sorted(access_counts.items(), key=lambda x: x[1], reverse=True)
    top_10_accesses = sum(count for _, count in sorted_access[:10])
    total_accesses = sum(access_counts.values())
    print(f"Access distribution:")
    print(f"  Top 10 vertices: {top_10_accesses}/{total_accesses} = {100*top_10_accesses/total_accesses:.1f}% of accesses")
    print()

    print("Utilization by tree level:")
    print(f"{'Level':>6} {'Buckets':>8} {'Avg':>8} {'Max':>8} {'MaxEver':>8}")
    print("-" * 42)
    for level, stats in level_stats.items():
        print(f"{level:>6} {stats['num_buckets']:>8} {stats['avg_current']:>8.2f} "
              f"{stats['max_current']:>8} {stats['max_ever']:>8}")

    # Show bucket distribution in hot zone
    print("\nBucket utilization in hot zone (first few leaves):")
    for leaf in range(min(4, hot_zone_size)):
        path = simulator.get_path_indices(leaf)
        total_in_path = sum(len(simulator.buckets.get(idx, [])) for idx in path)
        max_in_path = max((simulator.bucket_stats.get(idx, BucketStats()).max_size for idx in path), default=0)
        print(f"  Leaf {leaf}: current_total={total_in_path}, max_ever={max_in_path}")

    return {
        "max_utilization": max_utilization,
        "theoretical_bound": theoretical_bound,
        "level_stats": level_stats,
    }


def run_skewed_simulation(config: SimulationConfig) -> Dict:
    """
    Simulate a skewed distribution where duplications concentrate on specific paths.

    Strategy: Create a "star" topology where one central vertex connects to many others,
    and all those neighbors have leaves in a narrow range. This concentrates duplications.
    """
    print(f"\n{'='*80}")
    print("Skewed Distribution Simulation")
    print(f"{'='*80}")

    simulator = MetaBucketSimulator(config)
    theoretical_bound = compute_theoretical_bound(
        num_operations=config.num_operations,
        max_deg=config.max_deg,
        level=simulator.level
    )

    print(f"Configuration:")
    print(f"  num_data: {config.num_data}")
    print(f"  max_deg: {config.max_deg}")
    print(f"  num_operations: {config.num_operations}")
    print(f"  theoretical_bound: {theoretical_bound}")
    print()

    # Reset simulator
    simulator.rl_counter = 0
    simulator.buckets.clear()
    simulator.bucket_stats.clear()
    simulator.stash.clear()

    # Create a skewed graph topology:
    # - Select a "hot zone" of leaves (small range)
    # - Assign many vertices to this hot zone
    # - Create star patterns where central vertices connect to hot zone vertices

    hot_zone_size = max(4, simulator.leaf_range // 16)  # 1/16 of leaf range
    hot_zone_start = 0  # Leaves 0 to hot_zone_size-1

    print(f"  Hot zone: leaves {hot_zone_start} to {hot_zone_start + hot_zone_size - 1}")
    print(f"  (covers {hot_zone_size}/{simulator.leaf_range} = {100*hot_zone_size/simulator.leaf_range:.1f}% of leaves)")
    print()

    # Assign 80% of vertices to the hot zone
    num_hot_vertices = int(0.8 * config.num_data)
    hot_vertices = list(range(num_hot_vertices))
    cold_vertices = list(range(num_hot_vertices, config.num_data))

    # Initialize graph with skewed leaf assignment
    for v in range(config.num_data):
        if v in hot_vertices:
            # Hot vertices get leaves in the hot zone
            simulator.pos_map[v] = random.randint(hot_zone_start, hot_zone_start + hot_zone_size - 1)
        else:
            # Cold vertices get random leaves
            simulator.pos_map[v] = secrets.randbelow(simulator.leaf_range)

        # Create neighbors - hot vertices connect to other hot vertices
        num_neighbors = random.randint(1, config.max_deg)
        if v in hot_vertices:
            # Hot vertices mostly connect to other hot vertices
            potential_neighbors = [x for x in hot_vertices if x != v]
        else:
            potential_neighbors = [x for x in range(config.num_data) if x != v]

        neighbors = random.sample(potential_neighbors, min(num_neighbors, len(potential_neighbors)))
        simulator.graph[v] = {n: simulator.pos_map[n] for n in neighbors}

    # Run simulation - access hot vertices frequently
    max_utilization = 0
    max_stash = 0
    utilization_over_time = []

    for op in range(config.num_operations):
        # 90% chance to access a hot vertex
        if random.random() < 0.9:
            vertex_key = random.choice(hot_vertices)
        else:
            vertex_key = random.choice(cold_vertices) if cold_vertices else random.choice(hot_vertices)

        simulator.simulate_lookup(vertex_key)

        current_max = simulator.get_max_bucket_size()
        max_utilization = max(max_utilization, current_max)
        max_stash = max(max_stash, len(simulator.stash))
        utilization_over_time.append(current_max)

    # Analyze by level
    level_stats = analyze_by_level(simulator)

    print(f"Results:")
    print(f"  Max bucket utilization: {max_utilization}")
    print(f"  Max stash overflow: {max_stash}")
    print(f"  Utilization ratio: {max_utilization / theoretical_bound:.2%}")
    print()

    print("Utilization by tree level:")
    print(f"{'Level':>6} {'Buckets':>8} {'Avg':>8} {'Max':>8} {'MaxEver':>8}")
    print("-" * 42)
    for level, stats in level_stats.items():
        print(f"{level:>6} {stats['num_buckets']:>8} {stats['avg_current']:>8.2f} "
              f"{stats['max_current']:>8} {stats['max_ever']:>8}")

    # Show which buckets have highest utilization
    print("\nTop 10 buckets by max utilization:")
    sorted_buckets = sorted(simulator.bucket_stats.items(),
                           key=lambda x: x[1].max_size, reverse=True)[:10]
    for bucket_idx, stats in sorted_buckets:
        level = int(math.log2(bucket_idx + 1)) if bucket_idx > 0 else 0
        print(f"  Bucket {bucket_idx:>4} (level {level}): max={stats.max_size}, "
              f"writes={stats.total_writes}, reads={stats.total_reads}")

    return {
        "max_utilization": max_utilization,
        "theoretical_bound": theoretical_bound,
        "level_stats": level_stats,
        "utilization_over_time": utilization_over_time,
    }


def run_adversarial_simulation(config: SimulationConfig) -> Dict:
    """
    Adversarial simulation that tries to maximize bucket utilization.

    Strategy:
    1. Disable RL path reading (simulate scenario where RL doesn't help)
    2. Create duplications that all target the same small set of leaves
    3. Access pattern designed to create duplications faster than they're consumed
    """
    print(f"\n{'='*80}")
    print("Adversarial Simulation (RL disabled)")
    print(f"{'='*80}")

    simulator = MetaBucketSimulator(config)
    theoretical_bound = compute_theoretical_bound(
        num_operations=config.num_operations,
        max_deg=config.max_deg,
        level=simulator.level
    )

    print(f"Configuration:")
    print(f"  num_data: {config.num_data}")
    print(f"  max_deg: {config.max_deg}")
    print(f"  num_operations: {config.num_operations}")
    print(f"  theoretical_bound: {theoretical_bound}")
    print()

    # Reset simulator
    simulator.rl_counter = 0
    simulator.buckets.clear()
    simulator.bucket_stats.clear()
    simulator.stash.clear()

    # Create adversarial graph:
    # All vertices have neighbors that map to leaf 0
    target_leaf = 0

    for v in range(config.num_data):
        # Each vertex gets a random leaf
        simulator.pos_map[v] = secrets.randbelow(simulator.leaf_range)

        # But all neighbors are assigned to the target leaf!
        num_neighbors = config.max_deg  # Max neighbors
        neighbors = random.sample([x for x in range(config.num_data) if x != v],
                                 min(num_neighbors, config.num_data - 1))
        # Force all neighbors to have the same target leaf
        simulator.graph[v] = {n: target_leaf for n in neighbors}

    print(f"  All duplications target leaf {target_leaf}")
    print()

    # Custom lookup that doesn't use RL paths (adversarial)
    def adversarial_lookup(vertex_key: int) -> Tuple[int, int]:
        old_leaf = simulator.pos_map[vertex_key]

        # Only read the vertex's own path (no RL paths!)
        all_paths = [old_leaf]

        consumed = 0
        for leaf in all_paths:
            dups = simulator.read_path(leaf)
            consumed += len(dups)

        # Assign new leaf
        new_leaf = secrets.randbelow(simulator.leaf_range)
        simulator.pos_map[vertex_key] = new_leaf

        # Create duplications - all go to target_leaf
        created = 0
        for neighbor_key, neighbor_leaf in simulator.graph[vertex_key].items():
            simulator.add_duplication_to_stash(
                target_key=neighbor_key,
                source_key=vertex_key,
                new_leaf=new_leaf,
                target_leaf=neighbor_leaf  # This is target_leaf for all
            )
            created += 1

        # Evict to paths we read
        simulator.evict_stash_to_paths(all_paths)

        return created, consumed

    # Run simulation
    max_utilization = 0
    max_stash = 0
    utilization_over_time = []

    for op in range(config.num_operations):
        vertex_key = random.randint(0, config.num_data - 1)
        adversarial_lookup(vertex_key)

        current_max = simulator.get_max_bucket_size()
        max_utilization = max(max_utilization, current_max)
        max_stash = max(max_stash, len(simulator.stash))
        utilization_over_time.append(current_max)

    print(f"Results:")
    print(f"  Max bucket utilization: {max_utilization}")
    print(f"  Max stash overflow: {max_stash}")
    print(f"  Utilization ratio: {max_utilization / theoretical_bound:.2%}")

    # Analyze the target path
    target_path = simulator.get_path_indices(target_leaf)
    print(f"\nBuckets on target path (leaf {target_leaf}):")
    for bucket_idx in target_path:
        stats = simulator.bucket_stats.get(bucket_idx, BucketStats())
        current = len(simulator.buckets.get(bucket_idx, []))
        level = int(math.log2(bucket_idx + 1)) if bucket_idx > 0 else 0
        print(f"  Bucket {bucket_idx:>4} (level {level}): current={current}, max={stats.max_size}")

    return {
        "max_utilization": max_utilization,
        "theoretical_bound": theoretical_bound,
        "max_stash": max_stash,
        "utilization_over_time": utilization_over_time,
    }


def run_extreme_adversarial_simulation(config: SimulationConfig) -> Dict:
    """
    Extreme adversarial simulation: vertices are on opposite side of tree from target.

    This ensures read paths only intersect with target path at the root, forcing
    all duplications to compete for the root bucket.
    """
    print(f"\n{'='*80}")
    print("Extreme Adversarial Simulation (paths only share root)")
    print(f"{'='*80}")

    simulator = MetaBucketSimulator(config)
    theoretical_bound = compute_theoretical_bound(
        num_operations=config.num_operations,
        max_deg=config.max_deg,
        level=simulator.level
    )

    print(f"Configuration:")
    print(f"  num_data: {config.num_data}")
    print(f"  max_deg: {config.max_deg}")
    print(f"  num_operations: {config.num_operations}")
    print(f"  bucket_capacity: {config.bucket_capacity}")
    print(f"  theoretical_bound: {theoretical_bound}")
    print()

    # Reset simulator
    simulator.rl_counter = 0
    simulator.buckets.clear()
    simulator.bucket_stats.clear()
    simulator.stash.clear()

    # Target leaf is 0 (leftmost)
    # All accessed vertices are on leaves >= leaf_range/2 (right half)
    target_leaf = 0
    min_vertex_leaf = simulator.leaf_range // 2

    print(f"  Target leaf: {target_leaf} (leftmost)")
    print(f"  Vertex leaves: {min_vertex_leaf} to {simulator.leaf_range - 1} (right half)")
    print(f"  Paths only share root bucket!")
    print()

    for v in range(config.num_data):
        # Vertices on right half of tree
        simulator.pos_map[v] = random.randint(min_vertex_leaf, simulator.leaf_range - 1)

        # Neighbors map to target leaf (left side)
        num_neighbors = config.max_deg
        neighbors = random.sample([x for x in range(config.num_data) if x != v],
                                 min(num_neighbors, config.num_data - 1))
        simulator.graph[v] = {n: target_leaf for n in neighbors}

    # Custom lookup
    def extreme_lookup(vertex_key: int) -> Tuple[int, int]:
        old_leaf = simulator.pos_map[vertex_key]
        all_paths = [old_leaf]

        consumed = 0
        for leaf in all_paths:
            dups = simulator.read_path(leaf)
            consumed += len(dups)

        # Keep vertex on right side
        new_leaf = random.randint(min_vertex_leaf, simulator.leaf_range - 1)
        simulator.pos_map[vertex_key] = new_leaf

        created = 0
        for neighbor_key, neighbor_leaf in simulator.graph[vertex_key].items():
            simulator.add_duplication_to_stash(
                target_key=neighbor_key,
                source_key=vertex_key,
                new_leaf=new_leaf,
                target_leaf=neighbor_leaf
            )
            created += 1

        simulator.evict_stash_to_paths(all_paths)
        return created, consumed

    max_utilization = 0
    max_stash = 0
    utilization_over_time = []
    stash_over_time = []

    for op in range(config.num_operations):
        vertex_key = random.randint(0, config.num_data - 1)
        extreme_lookup(vertex_key)

        current_max = simulator.get_max_bucket_size()
        max_utilization = max(max_utilization, current_max)
        max_stash = max(max_stash, len(simulator.stash))
        utilization_over_time.append(current_max)
        stash_over_time.append(len(simulator.stash))

    print(f"Results:")
    print(f"  Max bucket utilization: {max_utilization}")
    print(f"  Max stash overflow: {max_stash}")
    print(f"  Final stash size: {len(simulator.stash)}")
    print(f"  Utilization ratio: {max_utilization / theoretical_bound:.2%}")

    # Root bucket analysis
    root_stats = simulator.bucket_stats.get(0, BucketStats())
    print(f"\nRoot bucket (only shared bucket):")
    print(f"  Current size: {len(simulator.buckets.get(0, []))}")
    print(f"  Max size seen: {root_stats.max_size}")
    print(f"  Total writes: {root_stats.total_writes}")
    print(f"  Total reads: {root_stats.total_reads}")

    # Stash growth over time
    if stash_over_time:
        print(f"\nStash growth (sampled every 50 ops):")
        for i in range(0, len(stash_over_time), 50):
            print(f"  Op {i:4d}: stash={stash_over_time[i]}")

    return {
        "max_utilization": max_utilization,
        "theoretical_bound": theoretical_bound,
        "max_stash": max_stash,
        "stash_over_time": stash_over_time,
    }


def run_synchronized_simulation(config: SimulationConfig) -> Dict:
    """
    Simulation where access pattern is synchronized with RL to avoid cleanup.

    Strategy: Access vertices in a pattern that creates duplications on paths
    that won't be read by RL for a long time.
    """
    print(f"\n{'='*80}")
    print("Synchronized Avoidance Simulation")
    print(f"{'='*80}")

    simulator = MetaBucketSimulator(config)
    theoretical_bound = compute_theoretical_bound(
        num_operations=config.num_operations,
        max_deg=config.max_deg,
        level=simulator.level
    )

    print(f"Configuration:")
    print(f"  num_data: {config.num_data}")
    print(f"  max_deg: {config.max_deg}")
    print(f"  num_operations: {config.num_operations}")
    print(f"  theoretical_bound: {theoretical_bound}")
    print()

    # Reset simulator
    simulator.rl_counter = 0
    simulator.buckets.clear()
    simulator.bucket_stats.clear()
    simulator.stash.clear()
    simulator._init_random_graph()

    # Pre-compute which leaves RL will visit
    # Then assign neighbors to leaves that are far from being visited

    max_utilization = 0
    max_stash = 0
    utilization_over_time = []

    for op in range(config.num_operations):
        # Predict next RL leaves
        next_rl = simulator.get_rl_leaf(count=config.max_deg)

        # Find a vertex whose neighbors are NOT on these RL paths
        # (to maximize time before duplications are consumed)
        best_vertex = None
        best_score = -1

        for v in random.sample(range(config.num_data), min(50, config.num_data)):
            neighbor_leaves = set(simulator.graph[v].values())
            # Score = how many neighbors are NOT on RL paths
            score = len(neighbor_leaves - set(next_rl))
            if score > best_score:
                best_score = score
                best_vertex = v

        vertex_key = best_vertex if best_vertex is not None else random.randint(0, config.num_data - 1)

        # Restore RL counter (we peeked but didn't actually use those paths)
        simulator.rl_counter = (simulator.rl_counter - config.max_deg) % simulator.leaf_range

        simulator.simulate_lookup(vertex_key)

        current_max = simulator.get_max_bucket_size()
        max_utilization = max(max_utilization, current_max)
        max_stash = max(max_stash, len(simulator.stash))
        utilization_over_time.append(current_max)

    level_stats = analyze_by_level(simulator)

    print(f"Results:")
    print(f"  Max bucket utilization: {max_utilization}")
    print(f"  Max stash overflow: {max_stash}")
    print(f"  Utilization ratio: {max_utilization / theoretical_bound:.2%}")
    print()

    print("Utilization by tree level:")
    print(f"{'Level':>6} {'Buckets':>8} {'Avg':>8} {'Max':>8} {'MaxEver':>8}")
    print("-" * 42)
    for level, stats in level_stats.items():
        print(f"{level:>6} {stats['num_buckets']:>8} {stats['avg_current']:>8.2f} "
              f"{stats['max_current']:>8} {stats['max_ever']:>8}")

    return {
        "max_utilization": max_utilization,
        "theoretical_bound": theoretical_bound,
        "level_stats": level_stats,
    }


def compare_parameters():
    """Compare different parameter configurations."""

    # Compare with theoretical bucket capacity (what find_bound would give)
    configs = [
        SimulationConfig(num_data=256, max_deg=4, num_operations=500, num_trials=5, bucket_capacity=120),
        SimulationConfig(num_data=512, max_deg=4, num_operations=500, num_trials=5, bucket_capacity=120),
        SimulationConfig(num_data=1024, max_deg=4, num_operations=500, num_trials=5, bucket_capacity=124),
        SimulationConfig(num_data=1024, max_deg=8, num_operations=500, num_trials=5, bucket_capacity=248),
        SimulationConfig(num_data=1024, max_deg=16, num_operations=500, num_trials=5, bucket_capacity=496),
    ]

    print("=" * 80)
    print("Parameter Comparison")
    print("=" * 80)

    results = []
    for config in configs:
        print(f"\n--- num_data={config.num_data}, max_deg={config.max_deg} ---")
        result = run_simulation(config)
        results.append(result)

    # Summary table
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'num_data':>10} {'max_deg':>8} {'theoretical':>12} {'max_observed':>12} {'ratio':>8}")
    print("-" * 50)

    for result in results:
        config = result["config"]
        theoretical = result["theoretical_bound"]
        max_obs = max(result["max_utilizations"])
        ratio = max_obs / theoretical if theoretical > 0 else 0
        print(f"{config.num_data:>10} {config.max_deg:>8} {theoretical:>12} {max_obs:>12} {ratio:>8.2%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simulate meta bucket utilization in Grove")
    parser.add_argument("--num-data", type=int, default=1024, help="Number of vertices")
    parser.add_argument("--max-deg", type=int, default=4, help="Maximum degree per vertex")
    parser.add_argument("--num-ops", type=int, default=1000, help="Number of operations")
    parser.add_argument("--num-trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--bucket-cap", type=int, default=None, help="Bucket capacity (default: theoretical bound)")
    parser.add_argument("--eviction", type=str, default="greedy", choices=["greedy", "random"],
                        help="Eviction mode: greedy (deepest) or random")
    parser.add_argument("--compare", action="store_true", help="Compare different parameters")
    parser.add_argument("--analyze-time", action="store_true", help="Analyze utilization over time")
    parser.add_argument("--hotspot", action="store_true", help="Run hotspot access simulation")
    parser.add_argument("--burst", action="store_true", help="Run burst access simulation")
    parser.add_argument("--zipf", action="store_true", help="Run Zipf distribution simulation")
    parser.add_argument("--zipf-locality", action="store_true", help="Run Zipf + spatial locality simulation")
    parser.add_argument("--alpha", type=float, default=1.0, help="Zipf alpha parameter (default: 1.0)")
    parser.add_argument("--skewed", action="store_true", help="Run skewed distribution simulation")
    parser.add_argument("--adversarial", action="store_true", help="Run adversarial simulation (RL disabled)")
    parser.add_argument("--extreme", action="store_true", help="Run extreme adversarial simulation (paths only share root)")
    parser.add_argument("--all-skewed", action="store_true", help="Run all skewed distribution tests")

    args = parser.parse_args()

    if args.compare:
        compare_parameters()
    elif args.all_skewed:
        # Run all skewed distribution tests
        level = int(math.ceil(math.log2(args.num_data))) + 1
        bucket_cap = args.bucket_cap
        if bucket_cap is None:
            bucket_cap = compute_theoretical_bound(
                num_operations=args.num_ops,
                max_deg=args.max_deg,
                level=level
            )

        config = SimulationConfig(
            num_data=args.num_data,
            max_deg=args.max_deg,
            num_operations=args.num_ops,
            num_trials=args.num_trials,
            seed=args.seed,
            bucket_capacity=bucket_cap,
            eviction_mode=args.eviction,
        )

        if args.seed is not None:
            random.seed(args.seed)

        results = []

        print("\n" + "=" * 80)
        print("RUNNING ALL SKEWED DISTRIBUTION TESTS")
        print("=" * 80)

        # Zipf with different alpha values
        for alpha in [0.5, 1.0, 1.5, 2.0]:
            result = run_zipf_simulation(config, alpha=alpha)
            results.append(("Zipf", alpha, result))

        # Zipf with locality
        result = run_zipf_with_locality_simulation(config, alpha=1.0)
        results.append(("Zipf+Locality", 1.0, result))

        # Hotspot
        run_hotspot_simulation(config)

        # Adversarial
        run_adversarial_simulation(config)

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY OF ALL SKEWED TESTS")
        print("=" * 80)
        print(f"{'Test':<20} {'Alpha':<8} {'Max Util':<10} {'Theoretical':<12} {'Ratio':<8}")
        print("-" * 60)
        for name, alpha, result in results:
            ratio = result['max_utilization'] / result['theoretical_bound']
            print(f"{name:<20} {alpha:<8} {result['max_utilization']:<10} "
                  f"{result['theoretical_bound']:<12} {ratio:<8.2%}")

    elif args.zipf or args.zipf_locality or args.skewed or args.adversarial or args.extreme or args.hotspot or args.burst:
        # Compute theoretical bound for bucket capacity if not provided
        level = int(math.ceil(math.log2(args.num_data))) + 1
        bucket_cap = args.bucket_cap
        if bucket_cap is None:
            bucket_cap = compute_theoretical_bound(
                num_operations=args.num_ops,
                max_deg=args.max_deg,
                level=level
            )

        config = SimulationConfig(
            num_data=args.num_data,
            max_deg=args.max_deg,
            num_operations=args.num_ops,
            num_trials=args.num_trials,
            seed=args.seed,
            bucket_capacity=bucket_cap,
            eviction_mode=args.eviction,
        )

        if args.seed is not None:
            random.seed(args.seed)

        if args.zipf:
            run_zipf_simulation(config, alpha=args.alpha)
        if args.zipf_locality:
            run_zipf_with_locality_simulation(config, alpha=args.alpha)
        if args.skewed:
            run_skewed_simulation(config)
        if args.adversarial:
            run_adversarial_simulation(config)
        if args.extreme:
            run_extreme_adversarial_simulation(config)
        if args.hotspot:
            run_hotspot_simulation(config)
        if args.burst:
            run_burst_simulation(config)
    else:
        # Compute theoretical bound for bucket capacity if not provided
        level = int(math.ceil(math.log2(args.num_data))) + 1
        bucket_cap = args.bucket_cap
        if bucket_cap is None:
            bucket_cap = compute_theoretical_bound(
                num_operations=args.num_ops,
                max_deg=args.max_deg,
                level=level
            )

        config = SimulationConfig(
            num_data=args.num_data,
            max_deg=args.max_deg,
            num_operations=args.num_ops,
            num_trials=args.num_trials,
            seed=args.seed,
            bucket_capacity=bucket_cap,
            eviction_mode=args.eviction,
        )

        result = run_simulation(config)

        if args.analyze_time and result["results"]:
            analyze_utilization_over_time(result["results"][0])
