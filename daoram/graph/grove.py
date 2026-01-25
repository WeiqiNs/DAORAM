import math
import random
import secrets
from decimal import Decimal, localcontext
from typing import List, Any, Dict

from daoram.dependency import InteractServer, Data, Encryptor
from daoram.omap.avl_omap_cache import AVLOmapCached
from daoram.oram.mul_path_oram import MulPathOram


class Grove:
    def __init__(self,
                 max_deg: int,
                 num_opr: int,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 client: InteractServer,
                 graph_depth: int = 1,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 filename: str = None,
                 encryptor: Encryptor = None):
        """
        Initializes the GraphOS.

        :param max_deg: The maximum number of neighbors a vertex can have.
        :param num_opr: The expected number of (lookup) operations the protocol will handle.
        :param num_data: The number of data points the oram should store.
        :param key_size: The number of bytes the random dummy key should have.
        :param data_size: The number of bytes the random dummy data should have.
        :param graph_depth: The number of intermediate vertex between nodes.
        :param client: The instance we use to interact with server.
        :param filename: The filename to save the oram data to.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param encryptor: The encryptor to use for encryption.
        """
        # Store the client.
        self._client = client

        # Store useful values.
        self._max_deg: int = max_deg
        self._num_opr: int = num_opr
        self._graph_depth: int = graph_depth

        # Compute the level of the binary tree needed.
        self._level: int = int(math.ceil(math.log(num_data, 2))) + 1

        # Compute the range of possible leafs [0, leaf_range).
        self._leaf_range: int = pow(2, self._level - 1)

        # Set counters for retrieving RL paths for meta ORAMs.
        self._graph_counter = 0  # For _graph_meta
        self._pos_counter = 0    # For _pos_meta

        # Compute the bucket size.
        meta_bucket_size = self.find_bound()

        # Initialize the OMAP with internal meta ORAM.
        # The internal meta (_pos_omap._meta) handles Graph ORAM -> PosMap updates:
        # - When vertex's graph_leaf changes, PosMap needs to update the stored value
        # - These updates are applied during batch_search when accessing PosMap
        self._pos_omap = AVLOmapCached(
            client=client,
            num_data=num_data,
            key_size=key_size,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            encryptor=encryptor,
            filename=f"{filename}_avl" if filename else None,
            enable_meta=True  # Enable internal meta for Graph ORAM -> PosMap updates
        )

        # Initialize the multi-path ORAMs.
        self._graph_oram = MulPathOram(
            name="graph",
            client=client,
            num_data=num_data,
            data_size=key_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            encryptor=encryptor,
            filename=f"{filename}_graph_mp" if filename else None
        )

        self._graph_meta = MulPathOram(
            name="g_meta",
            client=client,
            num_data=num_data,
            data_size=key_size,
            bucket_size=meta_bucket_size,
            stash_scale=stash_scale,
            encryptor=encryptor,
            filename=f"{filename}_graph_meta" if filename else None
        )

        self._pos_meta = MulPathOram(
            name="p_meta",
            client=client,
            num_data=num_data,
            data_size=key_size,
            bucket_size=meta_bucket_size,
            stash_scale=stash_scale,
            encryptor=encryptor,
            filename=f"{filename}_pos_meta" if filename else None
        )

    @staticmethod
    def binomial(n: int, i: int, p: Decimal) -> Decimal:
        """Compute (n choose i) * p^i * (1 - p)^(n - i) using Decimal."""
        return Decimal(math.comb(n, i)) * (p ** i) * ((Decimal(1) - p) ** (n - i))

    @staticmethod
    def equation(m: int, K: int, Y: int, L: int, prec: int) -> Decimal:
        """Compute the probability of overflow with the provided values."""
        # Sigma is the smallest value such that 2^sigma > Y.
        sigma = math.ceil(math.log2(Y))

        # Set prob to be zero to start with.
        prob = Decimal(0)

        # Set precision and continue the computation.
        with localcontext() as ctx:
            ctx.prec = prec

            for j in range(sigma, L):
                # Set temp_prob to be zero.
                temp_prob = Decimal(0)

                # Compute n and p (as Decimal).
                n = 2 ** j
                p = Decimal(1) / (Decimal(2) ** (j + 1))

                # Accumulate the binomial.
                for i in range(math.floor(Y / K)):
                    temp_prob += Grove.binomial(n=n, i=i, p=p)

                # Compute the actual term.
                term = (Decimal(2) ** j) * Decimal(math.ceil(m * K / (2 ** j))) * (Decimal(1) - temp_prob)

                # Accumulate the probability.
                prob += term

        # Return the probability.
        return prob

    def find_bound(self, prec: int = 80):
        """Use the provided information to find bound of number of delayed duplication."""
        # Find the bound start with 1.
        Y = 1
        while (self.equation(m=self._num_opr, L=self._level, K=self._max_deg, Y=Y, prec=prec) >
               Decimal(1) / Decimal(pow(2, 128))):
            Y += 1

        return Y

    def get_rl_leaf(self, count: int, for_pos_meta: bool = False) -> List[int]:
        """
        Get the next 'count' number of RL leaves.
        
        :param count: Number of RL leaves to generate.
        :param for_pos_meta: If True, use pos_counter; otherwise use graph_counter.
        :return: List of RL leaf indices.
        """
        # Compute the next "count" number of leaves.
        # Use (level - 1) bits since leaf range is [0, 2^(level-1)).
        num_bits = self._level - 1
        counter = self._pos_counter if for_pos_meta else self._graph_counter
        # Apply modulo to prevent counter overflow beyond leaf_range
        leaves = [int(format((counter + i) % self._leaf_range, f"0{num_bits}b")[::-1], 2) for i in range(count)]
        # Update the counter (also apply modulo to keep counter bounded).
        if for_pos_meta:
            self._pos_counter = (self._pos_counter + count) % self._leaf_range
        else:
            self._graph_counter = (self._graph_counter + count) % self._leaf_range
        # Return the leaves.
        return leaves

    def pos_meta_de_duplication(self):
        """Remove the duplications in position meta oram tree.
        
        For each key, keeps only the FIRST dup (highest priority).
        Priority: stash (front) > higher bucket > lower bucket.
        """
        seen_keys = set()
        temp_stash = []

        for data in self._pos_meta.stash:
            if data.key not in seen_keys:
                seen_keys.add(data.key)
                temp_stash.append(data)

        self._pos_meta.stash = temp_stash

    def graph_meta_de_duplication(self):
        """Remove the duplications in graph data meta oram tree.
        
        _graph_meta stores two types of duplications:
        1. Neighbor updates: value = (source_key, new_graph_leaf) - tuple
           Dedup by (key, source_key) - same neighbor can have multiple updates
        2. PosMap→GraphORAM updates: value = new_pos_leaf - int
           Dedup by key only - keep first (highest priority)
        
        Priority: stash (front) > higher bucket > lower bucket.
        """
        existing_neighbor_updates = set()  # (key, source_key)
        existing_pos_updates = set()       # key only
        temp_stash = []

        for data in self._graph_meta.stash:
            if isinstance(data.value, tuple):
                # Type 1: Neighbor update - use (key, source_key) as dedup key
                dedup_key = (data.key, data.value[0])
                if dedup_key not in existing_neighbor_updates:
                    existing_neighbor_updates.add(dedup_key)
                    temp_stash.append(data)
            else:
                # Type 2: PosMap→GraphORAM update - use key only
                if data.key not in existing_pos_updates:
                    existing_pos_updates.add(data.key)
                    temp_stash.append(data)

        self._graph_meta.stash = temp_stash

    def lookup(self, keys: List[Any]) -> Dict[Any, Any]:
        """
        Perform lookup operation on graph for multiple keys.

        :param keys: List of vertex keys to look up.
        :return: Dict mapping vertex key to (vertex_data, adjacency_dict).
        """
        # Batch OMAP search to get graph_leaf for all keys.
        # Also get all visited nodes' info for updating Graph ORAM.
        key_graph_leaf_dict, visited_nodes_map, total_posmap_paths = self._pos_omap.batch_search(
            keys=keys, return_visited_nodes=True
        )
        # key_graph_leaf_dict: {key: graph_leaf}
        # visited_nodes_map: {node_key: (new_pos_leaf, graph_leaf)} for all visited AVL nodes
        # total_posmap_paths: total ORAM paths read during batch_search

        # Filter out None values.
        missing_keys = [k for k, v in key_graph_leaf_dict.items() if v is None]
        if missing_keys:
            print(f"WARNING: Keys not found in PosMap: {missing_keys}")
        key_graph_leaf_dict = {k: v for k, v in key_graph_leaf_dict.items() if v is not None}
        
        # Debug: print graph_leaf for each key
        # print(f"DEBUG lookup: key_graph_leaf_dict = {key_graph_leaf_dict}")

        # Calculate pos_meta paths needed:
        # = total PosMap paths read - len(keys) (for non-downloaded vertices' pos_leaf updates)
        pos_meta_extra_paths = max(0, total_posmap_paths - len(keys))
        
        # Read vertices from graph ORAM, passing visited nodes info.
        return self.lookup_without_omap(key_graph_leaf_dict, visited_nodes_map, pos_meta_extra_paths)

    def lookup_without_omap(self, key_graph_leaf_dict: Dict[Any, int],
                               visited_nodes_map: Dict[Any, tuple] = None,
                               pos_meta_extra_paths: int = 0) -> Dict[Any, Any]:
        """
        Read multiple vertices from graph ORAM given their paths.

        :param key_graph_leaf_dict: Dict mapping vertex key to its graph_leaf.
        :param visited_nodes_map: Dict mapping node_key to (new_pos_leaf, graph_leaf) for nodes
                                  visited during PosMap access.
        :param pos_meta_extra_paths: Number of extra paths to read from _pos_meta.
                                     = total_posmap_paths - len(keys), for non-downloaded vertices.
        :return: Dict mapping vertex key to (vertex_data, adjacency_dict).
        
        Data structure in Graph ORAM:
            value = (vertex_data, adjacency_dict, pos_leaf)
            - Each vertex stores its OWN pos_leaf
            - adjacency_dict: {neighbor_key: neighbor_graph_leaf}
        
        Duplication types:
        1. _graph_meta (indexed by graph_leaf):
           - Neighbor updates: value = (source_key, new_graph_leaf)
           - PosMap→GraphORAM updates: value = new_pos_leaf (for non-downloaded vertices)
        2. _pos_meta (indexed by pos_leaf):
           - GraphORAM→PosMap updates: value = new_graph_leaf (for downloaded vertices)
        """
        leaves = list(key_graph_leaf_dict.values())
        if not leaves:
            return {}
        
        if visited_nodes_map is None:
            visited_nodes_map = {}

        # Get RL paths for meta ORAMs indexed by graph_leaf.
        # _graph_meta: for neighbor updates (max_deg neighbors per vertex)
        # _pos_meta: for pos_leaf updates of NON-DOWNLOADED vertices (pos_meta_extra_paths)
        # Both are indexed by graph_leaf, so combine them.
        graph_meta_rl_path = self.get_rl_leaf(
            count=self._max_deg * len(leaves) + pos_meta_extra_paths, 
            for_pos_meta=False
        ) + leaves
        
        # pos_meta_rl_path: for _pos_omap._meta (Graph ORAM -> PosMap updates)
        # = len(keys), because this many vertices are downloaded and their graph_leaf changes
        pos_meta_rl_path = self.get_rl_leaf(count=len(leaves), for_pos_meta=True)

        # Queue reads for graph ORAM and meta ORAMs (both use graph_meta_rl_path).
        self._graph_oram.queue_read(leaves=leaves)
        self._graph_meta.queue_read(leaves=graph_meta_rl_path)
        self._pos_meta.queue_read(leaves=pos_meta_rl_path)  # Same paths as _graph_meta
        
        result = self._client.execute()

        # Process read results.
        self._graph_oram.process_read_result(result)

        # Locate keys of interest and assign new paths.
        data_of_interest = {}
        
        # graph_meta_duplications: for _graph_meta (indexed by graph_leaf)
        # 1. Neighbor's graph_leaf updates
        # 2. Non-downloaded vertex's pos_leaf updates (PosMap -> Graph ORAM)
        graph_meta_duplications = []
        
        # pos_meta_duplications: for _pos_meta (indexed by pos_leaf)
        # Downloaded vertex's graph_leaf updates (Graph ORAM -> PosMap)
        pos_meta_duplications = []
        
        # Build index of downloaded vertices.
        downloaded_vertices = {data.key: data for data in self._graph_oram.stash 
                               if data.key in key_graph_leaf_dict}
        
        # Check which keys are missing from the stash (for debugging)
        # missing_from_stash = set(key_graph_leaf_dict.keys()) - set(downloaded_vertices.keys())
        
        # Only process TARGET vertices (not all vertices in stash)
        # Non-target vertices keep their original graph_leaf
        for data in self._graph_oram.stash:
            if data.key not in key_graph_leaf_dict:
                continue
                
            # Extract vertex_data, adjacency_dict, and pos_leaf from value.
            if len(data.value) == 3:
                vertex_data, adjacency_dict, old_pos_leaf = data.value
            else:
                vertex_data, adjacency_dict = data.value
                old_pos_leaf = None
            
            data_of_interest[data.key] = (vertex_data, adjacency_dict)
            
            # Assign new graph_leaf only for TARGET vertex
            new_graph_leaf = secrets.randbelow(self._leaf_range)
            data.leaf = new_graph_leaf
            
            # Update pos_leaf if this vertex was visited during PosMap access.
            # visited_nodes_map format: {key: (new_pos_leaf, graph_leaf)}
            visited_info = visited_nodes_map.get(data.key) if visited_nodes_map else None
            if visited_info is not None:
                new_pos_leaf, _ = visited_info  # Extract new_pos_leaf
            else:
                new_pos_leaf = old_pos_leaf
            data.value = (vertex_data, adjacency_dict, new_pos_leaf)
            
            # 1. Neighbor updates -> _graph_meta
            for neighbor_key, neighbor_graph_leaf in adjacency_dict.items():
                if isinstance(neighbor_graph_leaf, tuple):
                    neighbor_graph_leaf = neighbor_graph_leaf[0]
                graph_meta_duplications.append(
                    Data(key=neighbor_key, leaf=neighbor_graph_leaf, 
                         value=(data.key, new_graph_leaf))
                )
            
            # 2. Graph ORAM -> PosMap update -> _pos_omap._meta
            # Tell PosMap (AVL node) the new graph_leaf for this vertex
            if new_pos_leaf is not None:
                pos_meta_duplications.append(
                    Data(key=data.key, leaf=new_pos_leaf, value=new_graph_leaf)
                )
        
        # For visited AVL nodes whose vertex was NOT downloaded from Graph ORAM,
        # create duplication to update their pos_leaf -> _graph_meta
        # visited_nodes_map format: {vertex_key: (new_pos_leaf, graph_leaf)}
        for vertex_key, (new_pos_leaf, vertex_graph_leaf) in visited_nodes_map.items():
            if vertex_key not in downloaded_vertices:
                # The AVL node stores graph_leaf directly - no need to search in adjacency lists!
                if vertex_graph_leaf is not None:
                    # PosMap -> Graph ORAM: update vertex's pos_leaf -> _graph_meta
                    graph_meta_duplications.append(
                        Data(key=vertex_key, leaf=vertex_graph_leaf, value=new_pos_leaf)
                    )

        # Process graph_meta read results.
        self._graph_meta.process_read_result(result)
        self._graph_meta.stash = graph_meta_duplications + self._graph_meta.stash
        
        # Process pos_meta read results.
        self._pos_meta.process_read_result(result)
        # Note: pos_meta_duplications go to _pos_omap._meta (internal meta), not _pos_meta
        
        # Add Graph ORAM -> PosMap duplications to _pos_omap's internal meta ORAM.
        # These will be applied during the next batch_search when accessing PosMap.
        self._pos_omap.add_meta_duplications(pos_meta_duplications)

        # Perform de-duplication.
        self.graph_meta_de_duplication()
        self._pos_omap.meta_de_duplication()

        # Apply graph_meta duplications to downloaded vertices.
        # _graph_meta stores two types of duplications:
        # 1. Neighbor updates: value = (source_key, new_graph_leaf) - tuple
        # 2. PosMap->GraphORAM updates: value = new_pos_leaf - int
        retrieved_vertices = {data.key: idx for idx, data in enumerate(self._graph_oram.stash)}
        temp_stash = []
        for data in self._graph_meta.stash:
            if data.key in retrieved_vertices:
                idx = retrieved_vertices[data.key]
                vertex_value = self._graph_oram.stash[idx].value
                
                if isinstance(data.value, tuple):
                    # Type 1: Neighbor update (adjacency list update)
                    # Only UPDATE existing neighbors, don't ADD new ones
                    adjacency_dict = vertex_value[1]
                    source_key, new_graph_leaf = data.value[0], data.value[1]
                    
                    if new_graph_leaf < 0:
                        # Delete request
                        if source_key in adjacency_dict:
                            del adjacency_dict[source_key]
                    else:
                        # Update request - only update if neighbor already exists
                        if source_key in adjacency_dict:
                            adjacency_dict[source_key] = new_graph_leaf
                else:
                    # Type 2: PosMap->GraphORAM update (pos_leaf update)
                    # data.value is new_pos_leaf (int)
                    if len(vertex_value) >= 3:
                        vertex_data, adjacency_dict = vertex_value[0], vertex_value[1]
                        self._graph_oram.stash[idx].value = (vertex_data, adjacency_dict, data.value)
            else:
                temp_stash.append(data)
        self._graph_meta.stash = temp_stash
        
        # _pos_meta stores Graph ORAM -> PosMap duplications.
        # These are NOT applied here - they will be applied when PosMap is accessed.
        # Just keep them in _pos_meta.stash for now (no application needed in this round).

        # Queue writes for graph ORAM and meta ORAMs.
        # Use same paths as read to maintain obliviousness (read/write symmetry)
        self._graph_oram.queue_write()
        self._graph_meta.queue_write()  # Uses _tmp_leaves from queue_read
        self._pos_meta.queue_write()    # Uses _tmp_leaves from queue_read
        self._client.execute()

        return data_of_interest

    def insert(self, vertex: tuple) -> None:
        """
        Insert a new vertex to the graph with direct neighbor updates.

        :param vertex: the vertex to add, in format of (key, value, neighbors).
                       neighbors is a dict {neighbor_key: neighbor_leaf} or list of neighbor keys.
        
        For obliviousness:
        1. Always perform max_degree lookups (K real + max_degree-K dummy)
        2. Access max_degree+1 paths in Graph ORAM (max_degree neighbors + 1 new vertex path)
        3. Create duplications (RL path selection):
           - _graph_meta: max_degree * (max_degree-1) for neighbors' neighbors + AVL node updates
           - _pos_meta: max_degree for Graph ORAM -> PosMap updates
        
        Data structure in Graph ORAM:
            Data(key=vertex_key, leaf=graph_leaf, value=(vertex_data, adjacency_dict, pos_leaf))
        """
        # Get neighbor keys from vertex[2] (could be dict or list).
        neighbor_keys = list(vertex[2].keys()) if isinstance(vertex[2], dict) else list(vertex[2])
        K = len(neighbor_keys)  # Actual number of neighbors
        
        # Step 1: Sample a new path for the new vertex in Graph ORAM.
        new_graph_leaf = secrets.randbelow(self._leaf_range)

        # Step 2: Insert the new vertex key-path pair to PosMap ORAM.
        new_pos_leaf = self._pos_omap.insert(key=vertex[0], value=new_graph_leaf, return_pos_leaf=True)

        # Step 3: Use batch_search to get graph_leaf for all neighbors.
        # Also get visited_nodes_map for AVL node updates.
        if K > 0:
            key_graph_leaf_dict, visited_nodes_map, total_posmap_paths = self._pos_omap.batch_search(
                keys=neighbor_keys, return_visited_nodes=True
            )
            pos_meta_extra_paths = max(0, total_posmap_paths - K)
        else:
            key_graph_leaf_dict = {}
            visited_nodes_map = {}
            pos_meta_extra_paths = 0

        # Collect real neighbor leaves.
        real_neighbor_leaves = [v for k, v in key_graph_leaf_dict.items() if v is not None]
        
        # Step 4: Pad with dummy paths to max_degree for obliviousness.
        dummy_count = max(0, self._max_deg - K)
        dummy_leaves = [secrets.randbelow(self._leaf_range) for _ in range(dummy_count)]
        
        # Step 5: Read Graph ORAM (max_degree + 1 paths).
        # One extra path for downloading the new vertex's insertion path.
        new_path_download = secrets.randbelow(self._leaf_range)
        all_graph_leaves = [new_path_download] + real_neighbor_leaves + dummy_leaves
        
        # RL paths for meta ORAMs:
        # _graph_meta: 
        #   - Neighbor updates: max_degree neighbors, each has up to (max_degree-1) other neighbors
        #   - AVL node updates for non-downloaded vertices
        graph_meta_rl_count = self._max_deg * (self._max_deg - 1) + pos_meta_extra_paths
        graph_meta_rl_paths = self.get_rl_leaf(count=graph_meta_rl_count, for_pos_meta=False) + real_neighbor_leaves
        
        # _pos_meta: max_degree for Graph ORAM -> PosMap updates
        pos_meta_rl_paths = self.get_rl_leaf(count=self._max_deg, for_pos_meta=True)

        # Queue reads for all ORAMs.
        self._graph_oram.queue_read(leaves=all_graph_leaves)
        self._graph_meta.queue_read(leaves=graph_meta_rl_paths)
        self._pos_meta.queue_read(leaves=pos_meta_rl_paths)
        
        result = self._client.execute()

        # Process read results.
        self._graph_oram.process_read_result(result)
        self._graph_meta.process_read_result(result)
        self._pos_meta.process_read_result(result)

        # Build index of downloaded vertices.
        downloaded_vertices = {data.key: data for data in self._graph_oram.stash 
                               if data.key in key_graph_leaf_dict}
        
        # graph_meta_duplications: for neighbor updates and AVL node updates
        graph_meta_duplications = []
        
        # pos_meta_duplications: for Graph ORAM -> PosMap updates (go to _pos_omap._meta)
        pos_meta_duplications = []
        
        # Build adjacency dict for the new vertex: {neighbor_key: neighbor_new_graph_leaf}
        new_vertex_adjacency = {}

        # Step 6: Update each neighbor's adjacency list and create duplications.
        for neighbor_key in neighbor_keys:
            if neighbor_key not in downloaded_vertices:
                continue
                
            neighbor_data = downloaded_vertices[neighbor_key]
            
            # Extract neighbor's data.
            if len(neighbor_data.value) == 3:
                neighbor_vertex_data, neighbor_adjacency_dict, neighbor_old_pos_leaf = neighbor_data.value
            else:
                neighbor_vertex_data, neighbor_adjacency_dict = neighbor_data.value
                neighbor_old_pos_leaf = None
            
            # Add the new vertex to neighbor's adjacency list.
            neighbor_adjacency_dict[vertex[0]] = new_graph_leaf
            
            # Assign new random graph_leaf for the neighbor.
            neighbor_new_graph_leaf = secrets.randbelow(self._leaf_range)
            neighbor_data.leaf = neighbor_new_graph_leaf
            
            # Update new vertex's adjacency with neighbor's new graph_leaf.
            new_vertex_adjacency[neighbor_key] = neighbor_new_graph_leaf
            
            # Get neighbor's new pos_leaf from visited_nodes_map.
            visited_info = visited_nodes_map.get(neighbor_key)
            if visited_info is not None:
                neighbor_new_pos_leaf, _ = visited_info
            else:
                neighbor_new_pos_leaf = neighbor_old_pos_leaf
            neighbor_data.value = (neighbor_vertex_data, neighbor_adjacency_dict, neighbor_new_pos_leaf)
            
            # Type 1: Notify neighbor's neighbors about neighbor's new graph_leaf -> _graph_meta
            for nn_key, nn_graph_leaf in neighbor_adjacency_dict.items():
                if nn_key == vertex[0]:  # Skip the new vertex itself
                    continue
                if isinstance(nn_graph_leaf, tuple):
                    nn_graph_leaf = nn_graph_leaf[0]
                graph_meta_duplications.append(
                    Data(key=neighbor_key, leaf=nn_graph_leaf, value=(neighbor_key, neighbor_new_graph_leaf))
                )
            
            # Type 2: Notify PosMap about neighbor's new graph_leaf -> _pos_omap._meta
            if neighbor_new_pos_leaf is not None:
                pos_meta_duplications.append(
                    Data(key=neighbor_key, leaf=neighbor_new_pos_leaf, value=neighbor_new_graph_leaf)
                )
        
        # For visited AVL nodes whose vertex was NOT downloaded,
        # create duplication to update their pos_leaf -> _graph_meta
        for vertex_key, (new_pos_leaf_avl, vertex_graph_leaf) in visited_nodes_map.items():
            if vertex_key not in downloaded_vertices and vertex_key != vertex[0]:
                if vertex_graph_leaf is not None:
                    graph_meta_duplications.append(
                        Data(key=vertex_key, leaf=vertex_graph_leaf, value=new_pos_leaf_avl)
                    )
        
        # Step 7: Add the new vertex to stash.
        self._graph_oram.stash.append(
            Data(key=vertex[0], leaf=new_graph_leaf, value=(vertex[1], new_vertex_adjacency, new_pos_leaf))
        )
        
        # Add duplications to meta ORAMs (prepend for highest priority).
        self._graph_meta.stash = graph_meta_duplications + self._graph_meta.stash
        self._pos_omap.add_meta_duplications(pos_meta_duplications)

        # Perform de-duplication.
        self.graph_meta_de_duplication()
        self._pos_omap.meta_de_duplication()

        # Apply graph_meta duplications to downloaded vertices (same as lookup).
        retrieved_vertices = {data.key: idx for idx, data in enumerate(self._graph_oram.stash)}
        temp_stash = []
        for data in self._graph_meta.stash:
            if data.key in retrieved_vertices:
                idx = retrieved_vertices[data.key]
                vertex_value = self._graph_oram.stash[idx].value
                
                if isinstance(data.value, tuple):
                    # Type 1: Neighbor update (adjacency list update)
                    adjacency_dict = vertex_value[1]
                    source_key, new_graph_leaf_dup = data.value[0], data.value[1]
                    
                    if new_graph_leaf_dup < 0:
                        if source_key in adjacency_dict:
                            del adjacency_dict[source_key]
                    else:
                        if source_key in adjacency_dict:
                            adjacency_dict[source_key] = new_graph_leaf_dup
                else:
                    # Type 2: PosMap->GraphORAM update (pos_leaf update)
                    if len(vertex_value) >= 3:
                        self._graph_oram.stash[idx].value = (vertex_value[0], vertex_value[1], data.value)
            else:
                temp_stash.append(data)
        self._graph_meta.stash = temp_stash

        # Step 8: Queue writes for all ORAMs (use same paths as read for obliviousness).
        self._graph_oram.queue_write()
        self._graph_meta.queue_write()
        self._pos_meta.queue_write()
        self._client.execute()

    def delete(self, key: Any) -> None:
        """
        Delete a vertex from the graph using duplication-based neighbor notification.
        
        Instead of downloading all neighbors, we send duplications to notify them
        that this vertex has been deleted (using new_graph_leaf = -1 as deletion marker).

        :param key: The key of the vertex to delete.
        """
        # Step 1: Delete from PosMap ORAM (search with value=None performs lazy deletion)
        # This also returns the vertex's graph_leaf
        vertex_graph_leaf = self._pos_omap.delete(key=key)
        
        # Step 2: Get RL paths for graph_meta (to notify neighbors via duplication)
        # We need max_degree paths for obliviousness
        graph_meta_rl_paths = self.get_rl_leaf(count=self._max_deg, for_pos_meta=False)
        
        # Step 3: Queue reads for Graph ORAM and Graph Meta
        self._graph_oram.queue_read(leaves=[vertex_graph_leaf])
        self._graph_meta.queue_read(leaves=graph_meta_rl_paths + [vertex_graph_leaf])
        
        # Execute reads
        result = self._client.execute()
        self._graph_oram.process_read_result(result)
        self._graph_meta.process_read_result(result)
        
        # Step 4: Find the vertex in stash and get its neighbors
        vertex_data = None
        vertex_idx = None
        for i, data in enumerate(self._graph_oram.stash):
            if data.key == key:
                vertex_data = data
                vertex_idx = i
                break
        
        if vertex_data is None:
            # Vertex not found, write back and return
            self._graph_oram.queue_write()
            self._graph_meta.queue_write(leaves=graph_meta_rl_paths + [vertex_graph_leaf])
            self._client.execute()
            return
        
        # Get neighbor info: {neighbor_key: graph_leaf}
        neighbor_adjacency = vertex_data.value[1]
        neighbor_keys = list(neighbor_adjacency.keys())
        
        # Step 5: Apply duplications from graph_meta to this vertex (de-duplication)
        temp_stash = []
        for data in self._graph_meta.stash:
            if data.key == key:
                # This dup targets the deleted vertex, discard it
                pass
            else:
                temp_stash.append(data)
        self._graph_meta.stash = temp_stash
        
        # Step 6: Create deletion duplications for all neighbors
        # Use new_graph_leaf = -1 to indicate deletion
        DELETION_MARKER = -1
        deletion_dups = []
        for i, neighbor_key in enumerate(neighbor_keys):
            neighbor_graph_leaf = neighbor_adjacency[neighbor_key]
            # Duplication format: Data(key=neighbor_key, leaf=neighbor_graph_leaf, value=(deleted_key, DELETION_MARKER))
            dup = Data(key=neighbor_key, leaf=neighbor_graph_leaf, value=(key, DELETION_MARKER))
            deletion_dups.append(dup)
        
        # Pad with dummy dups for obliviousness (total = max_degree)
        dummy_leaf = secrets.randbelow(self._leaf_range)
        for _ in range(self._max_deg - len(deletion_dups)):
            dummy_dup = Data(key=None, leaf=dummy_leaf, value=(None, DELETION_MARKER))
            deletion_dups.append(dummy_dup)
        
        # Add deletion dups to graph_meta stash (prepend for highest priority)
        self._graph_meta.stash = deletion_dups + self._graph_meta.stash
        
        # Step 7: Remove the deleted vertex from graph_oram stash
        del self._graph_oram.stash[vertex_idx]
        
        # Step 8: Write back
        self._graph_oram.queue_write()
        self._graph_meta.queue_write(leaves=graph_meta_rl_paths + [vertex_graph_leaf])
        self._client.execute()

    def neighbor(self, keys: List[Any]) -> dict:
        """
        Perform a neighbor lookup query with proper duplication handling.
        
        For a single center vertex:
        1. Lookup the center vertex to get its neighbor list
        2. Download all neighbors in one round
        3. Update center's new path in all neighbors
        4. Send duplications to neighbors' neighbors about path changes
        5. Send duplications to PosMap for all path updates

        :param keys: List of vertex keys to find neighbors for.
        :return: Dict mapping neighbor_key to (vertex_data, adjacency_dict).
        """
        if not keys:
            return {}
        
        # For simplicity, handle one center vertex at a time
        # TODO: extend to multiple centers
        center_key = keys[0]
        
        # Step 1: Lookup the center vertex
        center_result = self.lookup([center_key])
        if center_key not in center_result:
            return {}
        
        center_data, center_adjacency = center_result[center_key]
        
        # Get neighbor keys and their graph_leaves
        neighbor_keys = list(center_adjacency.keys())
        K = len(neighbor_keys)
        
        if K == 0:
            return {}
        
        # Step 2: Get center's new graph_leaf from PosMap (it changed after lookup)
        # Also get center's pos_leaf for pos_meta duplication
        center_search = self._pos_omap.batch_search([center_key], return_pos_leaf=True)
        center_info = center_search.get(center_key)
        center_new_graph_leaf = center_info[0] if center_info else None
        center_pos_leaf = center_info[1] if center_info else None
        
        # Get graph_leaves and pos_leaves for all neighbors
        # return_pos_leaf=True returns {key: (graph_leaf, pos_leaf)}
        neighbor_search = self._pos_omap.batch_search(neighbor_keys, return_pos_leaf=True)
        neighbor_graph_leaves = []
        neighbor_pos_leaves = {}  # {neighbor_key: pos_leaf}
        valid_neighbor_keys = []  # neighbors that still exist
        for nk in neighbor_keys:
            info = neighbor_search.get(nk)
            if info is not None and info[0] is not None:
                neighbor_graph_leaves.append(info[0])  # graph_leaf
                neighbor_pos_leaves[nk] = info[1]  # pos_leaf
                valid_neighbor_keys.append(nk)
        
        # Update K to reflect only valid neighbors
        K = len(valid_neighbor_keys)
        
        # Pad with dummy leaves for obliviousness
        dummy_count = max(0, self._max_deg - K)
        dummy_leaves = [secrets.randbelow(self._leaf_range) for _ in range(dummy_count)]
        all_graph_leaves = neighbor_graph_leaves + dummy_leaves
        
        # RL paths for meta ORAMs
        # graph_meta: max_degree neighbors, each has up to max_degree-1 other neighbors
        graph_meta_rl_count = self._max_deg * (self._max_deg - 1)
        graph_meta_rl_paths = self.get_rl_leaf(count=graph_meta_rl_count, for_pos_meta=False) + neighbor_graph_leaves
        
        # pos_meta: max_degree neighbors + 1 center need to update their pos_leaf
        pos_meta_rl_paths = self.get_rl_leaf(count=self._max_deg + 1, for_pos_meta=True)
        
        # Step 3: Queue reads
        self._graph_oram.queue_read(leaves=all_graph_leaves)
        self._graph_meta.queue_read(leaves=graph_meta_rl_paths)
        self._pos_meta.queue_read(leaves=pos_meta_rl_paths)
        
        result = self._client.execute()
        
        self._graph_oram.process_read_result(result)
        self._graph_meta.process_read_result(result)
        self._pos_meta.process_read_result(result)
        
        # De-duplication for graph_meta
        self.graph_meta_de_duplication()
        
        # Step 4: Apply duplications and process neighbors
        downloaded_neighbors = {}
        for data in self._graph_oram.stash:
            if data.key in neighbor_keys:
                downloaded_neighbors[data.key] = data
        
        # Apply graph_meta duplications to downloaded neighbors
        temp_stash = []
        for dup in self._graph_meta.stash:
            if dup.key in downloaded_neighbors:
                neighbor_data = downloaded_neighbors[dup.key]
                vertex_value = neighbor_data.value
                if isinstance(dup.value, tuple) and len(dup.value) == 2:
                    # Type 1: Neighbor update
                    adjacency_dict = vertex_value[1]
                    source_key, new_graph_leaf = dup.value
                    if new_graph_leaf < 0:
                        if source_key in adjacency_dict:
                            del adjacency_dict[source_key]
                    else:
                        if source_key in adjacency_dict:
                            adjacency_dict[source_key] = new_graph_leaf
                else:
                    # Type 2: PosMap update
                    if len(vertex_value) >= 3:
                        neighbor_data.value = (vertex_value[0], vertex_value[1], dup.value)
            else:
                temp_stash.append(dup)
        self._graph_meta.stash = temp_stash
        
        # Prepare result and duplications
        graph_meta_duplications = []
        pos_meta_duplications = []
        neighbor_result = {}
        
        for neighbor_key, neighbor_data in downloaded_neighbors.items():
            if len(neighbor_data.value) == 3:
                neighbor_vertex_data, neighbor_adjacency, neighbor_pos_leaf = neighbor_data.value
            else:
                neighbor_vertex_data, neighbor_adjacency = neighbor_data.value
                neighbor_pos_leaf = None
            
            # Step 5: Update center's new path in this neighbor's adjacency
            if center_key in neighbor_adjacency:
                neighbor_adjacency[center_key] = center_new_graph_leaf
            
            # Assign new graph_leaf for this neighbor
            neighbor_new_graph_leaf = secrets.randbelow(self._leaf_range)
            neighbor_data.leaf = neighbor_new_graph_leaf
            
            # Step 6: Create duplications for this neighbor's other neighbors
            for other_key, other_graph_leaf in neighbor_adjacency.items():
                if other_key != center_key:  # Center already updated locally
                    dup = Data(key=other_key, leaf=other_graph_leaf, 
                              value=(neighbor_key, neighbor_new_graph_leaf))
                    graph_meta_duplications.append(dup)
            
            # Also notify center about this neighbor's new path
            dup_to_center = Data(key=center_key, leaf=center_new_graph_leaf,
                                value=(neighbor_key, neighbor_new_graph_leaf))
            graph_meta_duplications.append(dup_to_center)
            
            # Step 7: Create pos_meta duplication (Graph ORAM -> PosMap)
            # Use pos_leaf from batch_search, not from vertex data
            actual_pos_leaf = neighbor_pos_leaves.get(neighbor_key)
            if actual_pos_leaf is not None:
                pos_dup = Data(key=neighbor_key, leaf=actual_pos_leaf, value=neighbor_new_graph_leaf)
                pos_meta_duplications.append(pos_dup)
            
            # Update neighbor's value with pos_leaf
            neighbor_data.value = (neighbor_vertex_data, neighbor_adjacency, actual_pos_leaf)
            
            neighbor_result[neighbor_key] = (neighbor_vertex_data, neighbor_adjacency)
        
        # Pad duplications with dummies
        dummy_leaf = secrets.randbelow(self._leaf_range)
        while len(graph_meta_duplications) < self._max_deg * (self._max_deg - 1):
            graph_meta_duplications.append(Data(key=None, leaf=dummy_leaf, value=(None, 0)))
        
        while len(pos_meta_duplications) < self._max_deg + 1:
            pos_meta_duplications.append(Data(key=None, leaf=dummy_leaf, value=0))
        
        # Add duplications to stash
        self._graph_meta.stash = graph_meta_duplications + self._graph_meta.stash
        self._pos_omap.add_meta_duplications(pos_meta_duplications)
        
        # pos_meta de-duplication
        self.pos_meta_de_duplication()
        self._pos_omap.meta_de_duplication()
        
        # Step 8: Write back
        self._graph_oram.queue_write()
        self._graph_meta.queue_write(leaves=graph_meta_rl_paths)
        self._pos_meta.queue_write()
        self._client.execute()
        
        return neighbor_result

    def t_hop(self, key: Any, num_hop: int) -> dict:
        """
        Perform the t-hop query, finding all neighbors within t hops.
        
        Returns all vertices reachable within num_hop hops from the starting vertex.
        The starting vertex is included in the result.

        :param key: Starting vertex key.
        :param num_hop: Number of hops to traverse.
        :return: Dict mapping vertex key to (vertex_data, adjacency_dict) for all vertices found.
        """
        if num_hop <= 0:
            # Just return the starting vertex
            return self.lookup([key])
        
        # First, lookup the starting vertex
        result = self.lookup([key])
        if key not in result:
            return {}
        
        # BFS: keys we've seen and keys to explore next
        visited = {key}
        frontier = [key]
        
        for hop in range(num_hop):
            next_frontier = []
            
            for center_key in frontier:
                # Get neighbors of this vertex
                neighbors = self.neighbor([center_key])
                
                for neighbor_key, neighbor_value in neighbors.items():
                    if neighbor_key not in visited:
                        visited.add(neighbor_key)
                        next_frontier.append(neighbor_key)
                        result[neighbor_key] = neighbor_value
            
            frontier = next_frontier
            if not frontier:
                break  # No more vertices to explore
        
        return result

    def t_traversal(self, key: Any, num_hop: int) -> dict:
        """
        Perform t-hop traversal, randomly selecting one neighbor per hop.
        
        Starting from the given vertex, randomly walk through the graph for num_hop steps.
        Each step randomly selects one neighbor to visit next.

        :param key: Starting vertex key.
        :param num_hop: Number of hops to traverse.
        :return: Dict mapping vertex key to (vertex_data, adjacency_dict) for visited vertices.
        """
        if num_hop <= 0:
            return self.lookup([key])
        
        # Start with the initial vertex
        result = self.lookup([key])
        if key not in result:
            return {}
        
        current_key = key
        
        for hop in range(num_hop):
            # Get neighbors of current vertex
            neighbors = self.neighbor([current_key])
            
            if not neighbors:
                # No neighbors, stop traversal
                break
            
            # Randomly select one neighbor
            next_key = random.choice(list(neighbors.keys()))
            result[next_key] = neighbors[next_key]
            current_key = next_key
        
        return result
