"""
GraphOS: Oblivious graph data structure using ORAM.

This is a simpler implementation compared to Grove - it doesn't use delayed duplication.
Instead, it directly downloads and updates neighbor information.

Data model:
- Each vertex is stored as: key="V{vertex_id}", value=(vertex_data, neighbor_count)
- Each edge is stored as: key="E{vertex_id}{edge_num}", value=neighbor_vertex_id
- OMAP stores: vertex/edge key -> ORAM path
"""

import math
import random
import secrets
from typing import Any, Dict, List, Optional

from daoram.dependency import InteractServer, Data, Encryptor
from daoram.omap.avl_omap_cache import AVLOmapCached
from daoram.oram.mul_path_oram import MulPathOram


class GraphOS:
    """
    Oblivious graph data structure using ORAM.

    Stores vertices and edges separately:
    - Vertices: "V{id}" -> (data, neighbor_count)
    - Edges: "E{id}{num}" -> neighbor_id
    """

    def __init__(self,
                 max_deg: int,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 client: InteractServer,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 filename: str = None,
                 encryptor: Encryptor = None):
        """
        Initialize GraphOS.

        :param max_deg: Maximum number of neighbors a vertex can have.
        :param num_data: Number of data points the ORAM should store.
        :param key_size: Number of bytes for random dummy keys.
        :param data_size: Number of bytes for random dummy data.
        :param client: Instance to interact with server.
        :param bucket_size: Number of data items per bucket.
        :param stash_scale: Scaling factor for stash size.
        :param filename: Filename to persist ORAM data.
        :param encryptor: Encryptor instance for encryption.
        """
        # Store configuration.
        self._client = client
        self._max_deg: int = max_deg

        # Compute tree parameters.
        self._level: int = int(math.ceil(math.log(num_data, 2))) + 1
        self._leaf_range: int = pow(2, self._level - 1)

        # Initialize the OMAP for key -> path mapping.
        self._omap = AVLOmapCached(
            client=client,
            num_data=num_data,
            key_size=key_size,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            encryptor=encryptor,
            filename=f"{filename}_avl" if filename else None,
        )

        # Initialize the multi-path ORAM for actual data storage.
        self._oram = MulPathOram(
            client=client,
            num_data=num_data,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            encryptor=encryptor,
            name="graphos_oram",
            filename=f"{filename}_mp" if filename else None,
        )

    def init_server_storage(self, graph: Dict[Any, Any] = None) -> None:
        """
        Initialize the GraphOS server storage.

        :param graph: Optional dict mapping vertex_id -> (vertex_data, {neighbor_id: edge_data}).
                      If None, initializes empty storage.
        """
        if graph is None:
            self._omap.init_server_storage()
            self._oram.init_server_storage()
            return

        # Prepare OMAP entries and ORAM data.
        omap_data = []
        oram_data = []

        for vertex_id, (vertex_data, neighbors) in graph.items():
            # Sample path for this vertex.
            vertex_path = secrets.randbelow(self._leaf_range)
            vertex_key = f"V{vertex_id}"

            # OMAP: vertex_key -> path
            omap_data.append((vertex_key, vertex_path))

            # ORAM: vertex data with neighbor count
            # Value format: (vertex_data, neighbor_count)
            neighbor_count = len(neighbors) if neighbors else 0
            oram_data.append(Data(
                key=vertex_key,
                value=(vertex_data, neighbor_count),
                leaf=vertex_path
            ))

            # Add edges.
            for edge_num, neighbor_id in enumerate(neighbors.keys(), start=1):
                edge_path = secrets.randbelow(self._leaf_range)
                edge_key = f"E{vertex_id}_{edge_num}"

                # OMAP: edge_key -> path
                omap_data.append((edge_key, edge_path))

                # ORAM: edge points to neighbor
                oram_data.append(Data(
                    key=edge_key,
                    value=neighbor_id,
                    leaf=edge_path
                ))

        # Initialize storage.
        self._omap.init_server_storage(data=omap_data)
        self._oram.init_server_storage(data=oram_data)

    def lookup(self, keys: List[Any], update_count: int = None) -> Dict[Any, Any]:
        """
        Perform lookup on input vertices.

        :param keys: List of vertex keys to look up.
        :param update_count: If not None, add this to vertex neighbor count.
        :return: Dict mapping vertex_key -> (vertex_data, neighbor_count).
        """
        if not keys:
            return {}

        # Sample new leaves for each key.
        new_leaves = {k: secrets.randbelow(self._leaf_range) for k in keys}

        # Search OMAP for current paths, updating to new paths.
        # Filter out keys that don't exist (OMAP returns None or raises exception).
        key_to_path = {}
        for k in keys:
            try:
                path = self._omap.search(key=k, value=new_leaves[k])
                if path is not None:
                    key_to_path[k] = path
            except (KeyError, ValueError):
                # Key doesn't exist in OMAP
                pass

        if not key_to_path:
            return {}

        leaves = list(key_to_path.values())

        # Retrieve paths from ORAM.
        self._oram.queue_read(leaves=leaves)
        result = self._client.execute()
        self._oram.process_read_result(result)

        # Find requested data in stash.
        lookup_result = {}
        for data in self._oram.stash:
            if data.key in key_to_path:
                lookup_result[data.key] = data.value
                # Update leaf to new random path.
                data.leaf = new_leaves[data.key]
                # Update count if requested.
                if update_count is not None and isinstance(data.value, tuple):
                    vertex_data, count = data.value
                    data.value = (vertex_data, count + update_count)

        # Write back.
        self._oram.queue_write()
        self._client.execute()

        return lookup_result

    def neighbor(self, keys: List[Any]) -> Dict[Any, Any]:
        """
        Perform neighbor query for each vertex in the input list.

        :param keys: List of vertex IDs to find neighbors for.
        :return: Dict mapping neighbor_vertex_key -> (vertex_data, neighbor_count).
        """
        if not keys:
            return {}

        # First get vertex info (including neighbor count).
        vertex_keys = [f"V{k}" for k in keys]
        vertex_info = self.lookup(keys=vertex_keys)

        # Collect edge keys to fetch.
        edge_keys = []
        for vkey, value in vertex_info.items():
            if isinstance(value, tuple) and len(value) >= 2:
                vertex_id = vkey[1:]  # Remove "V" prefix
                neighbor_count = value[1]
                for i in range(1, neighbor_count + 1):
                    edge_keys.append(f"E{vertex_id}_{i}")

        if not edge_keys:
            return {}

        # Get edge values (neighbor IDs).
        edge_info = self.lookup(keys=edge_keys)

        # Collect neighbor vertex keys.
        neighbor_vertex_keys = []
        for edge_key, neighbor_id in edge_info.items():
            neighbor_vertex_keys.append(f"V{neighbor_id}")

        if not neighbor_vertex_keys:
            return {}

        # Get neighbor vertex data.
        return self.lookup(keys=neighbor_vertex_keys)

    def insert(self, vertex: tuple) -> None:
        """
        Insert a new vertex to the graph.

        :param vertex: Tuple of (vertex_id, vertex_data, neighbors).
                       neighbors is a dict {neighbor_id: edge_data} or list of neighbor IDs.
        """
        vertex_id, vertex_data, neighbors = vertex

        # Convert list to dict if needed.
        if isinstance(neighbors, list):
            neighbors = {n: None for n in neighbors}

        # Sample path for new vertex.
        vertex_path = secrets.randbelow(self._leaf_range)
        vertex_key = f"V{vertex_id}"

        # Insert vertex key -> path into OMAP.
        self._omap.insert(key=vertex_key, value=vertex_path)

        # Read a dummy path and add vertex to stash.
        dummy_path = secrets.randbelow(self._leaf_range)
        self._oram.queue_read(leaves=[dummy_path])
        result = self._client.execute()
        self._oram.process_read_result(result)

        # Add vertex to stash with neighbor count.
        self._oram.stash.append(Data(
            key=vertex_key,
            value=(vertex_data, len(neighbors)),
            leaf=vertex_path
        ))

        # Also add edges from new vertex to each neighbor.
        for edge_num, neighbor_id in enumerate(neighbors.keys(), start=1):
            edge_key = f"E{vertex_id}_{edge_num}"
            edge_path = secrets.randbelow(self._leaf_range)

            # Insert edge to OMAP.
            self._omap.insert(key=edge_key, value=edge_path)

            # Add edge to stash.
            self._oram.stash.append(Data(
                key=edge_key,
                value=neighbor_id,
                leaf=edge_path
            ))

        # Write back.
        self._oram.queue_write()
        self._client.execute()

        if not neighbors:
            return

        # For each neighbor, update their count and create edge pointing back to new vertex.
        for neighbor_id in neighbors.keys():
            neighbor_key = f"V{neighbor_id}"

            # Get current neighbor info and increment count.
            # We need to get the OLD count to know the edge number.
            new_leaf = secrets.randbelow(self._leaf_range)
            old_path = self._omap.search(key=neighbor_key, value=new_leaf)

            if old_path is None:
                continue

            # Read neighbor from ORAM.
            self._oram.queue_read(leaves=[old_path])
            result = self._client.execute()
            self._oram.process_read_result(result)

            # Find neighbor in stash and update count.
            old_count = 0
            for data in self._oram.stash:
                if data.key == neighbor_key:
                    neighbor_data, old_count = data.value
                    # Increment count.
                    data.value = (neighbor_data, old_count + 1)
                    # Update leaf.
                    data.leaf = new_leaf
                    break

            # Create edge: E{neighbor_id}_{new_edge_num} -> vertex_id
            new_edge_num = old_count + 1
            edge_key = f"E{neighbor_id}_{new_edge_num}"
            edge_path = secrets.randbelow(self._leaf_range)

            # Insert edge to OMAP.
            self._omap.insert(key=edge_key, value=edge_path)

            # Add edge to stash.
            self._oram.stash.append(Data(
                key=edge_key,
                value=vertex_id,
                leaf=edge_path
            ))

            # Write back.
            self._oram.queue_write()
            self._client.execute()

    def delete(self, key: Any) -> None:
        """
        Delete a vertex from the graph.

        :param key: Vertex ID to delete.
        """
        vertex_key = f"V{key}"

        # Delete from OMAP and get path.
        delete_path = self._omap.delete(key=vertex_key)

        # Download the path.
        self._oram.queue_read(leaves=[delete_path])
        result = self._client.execute()
        self._oram.process_read_result(result)

        # Find and remove vertex from stash.
        vertex_value = None
        temp_stash = []
        for data in self._oram.stash:
            if data.key == vertex_key:
                vertex_value = data.value
            else:
                temp_stash.append(data)
        self._oram.stash = temp_stash

        # Write back.
        self._oram.queue_write()
        self._client.execute()

        if vertex_value is None:
            return

        # Get neighbor count.
        neighbor_count = vertex_value[1] if isinstance(vertex_value, tuple) else 0

        # Delete edges.
        edge_keys = [f"E{key}_{i}" for i in range(1, neighbor_count + 1)]
        neighbor_ids = []

        for edge_key in edge_keys:
            # Delete edge from OMAP.
            edge_path = self._omap.delete(key=edge_key)

            # Download edge path.
            self._oram.queue_read(leaves=[edge_path])
            result = self._client.execute()
            self._oram.process_read_result(result)

            # Find and remove edge, save neighbor ID.
            temp_stash = []
            for data in self._oram.stash:
                if data.key == edge_key:
                    neighbor_ids.append(data.value)
                else:
                    temp_stash.append(data)
            self._oram.stash = temp_stash

            # Write back.
            self._oram.queue_write()
            self._client.execute()

        # Update neighbor counts (decrement by 1).
        if neighbor_ids:
            neighbor_keys = [f"V{n}" for n in neighbor_ids]
            self.lookup(keys=neighbor_keys, update_count=-1)

    def t_hop(self, key: Any, num_hop: int) -> Dict[Any, Any]:
        """
        Perform t-hop query: find all vertices within num_hop hops.

        :param key: Starting vertex ID.
        :param num_hop: Number of hops to traverse.
        :return: Dict mapping vertex_key -> (vertex_data, neighbor_count).
        """
        result = {}
        keys_to_search = [key]

        for _ in range(num_hop):
            if not keys_to_search:
                break

            # Get neighbors of current frontier.
            temp_result = self.neighbor(keys=keys_to_search)

            # Reset frontier.
            keys_to_search = []

            # Add new vertices to result and frontier.
            for vkey, value in temp_result.items():
                if vkey not in result:
                    keys_to_search.append(vkey[1:])  # Remove "V" prefix
                    result[vkey] = value

        return result

    def t_traversal(self, key: Any, num_hop: int) -> Dict[Any, Any]:
        """
        Perform t-hop traversal: randomly walk through the graph.

        :param key: Starting vertex ID.
        :param num_hop: Number of hops to traverse.
        :return: Dict mapping vertex_key -> (vertex_data, neighbor_count).
        """
        result = {}
        current_key = key

        for _ in range(num_hop):
            # Get neighbors of current vertex.
            temp_result = self.neighbor(keys=[current_key])

            if not temp_result:
                break

            # Select random neighbor.
            random_key = random.choice(list(temp_result.keys()))
            result[random_key] = temp_result[random_key]

            # Move to random neighbor.
            current_key = random_key[1:]  # Remove "V" prefix

        return result
