import math
import random
import secrets
from decimal import Decimal, localcontext
from typing import List, Any, Optional, Dict

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

        # Set a counter for retrieving paths.
        self._counter = 0

        # Compute the bucket size.
        meta_bucket_size = self.find_bound()

        # Initialize the OMAP.
        self._pos_omap = AVLOmapCached(
            client=client,
            num_data=num_data,
            key_size=key_size,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            encryptor=encryptor,
            filename=f"{filename}_avl" if filename else None
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

    def get_rl_leaf(self, count: int) -> List[int]:
        """Get the next 'count' number of RL leaves."""
        # Compute the next "count" number of leaves.
        leaves = [int(format(self._counter + i, f"0{self._level}b")[::-1], 2) for i in range(count)]
        # Update the counter.
        self._counter += count
        # Return the leaves.
        return leaves

    def pos_meta_de_duplication(self):
        """Remove the duplications in position meta oram tree."""
        # Record the delayed duplications that have been read.
        existing_dd = []
        # Set the temporary stash.
        temp_stash = []

        # For data in the pos_map stash, record the new ones.
        for data in self._pos_meta.stash:
            # If not previously read, add to both existing and stash.
            if (data.key, data.value[0]) not in existing_dd:
                existing_dd.append((data.key, data.value[0]))
                temp_stash.append(data)
            # If read, just ignore it.

        # Update the stash with duplications removed.
        self._pos_meta.stash = temp_stash

    def graph_meta_de_duplication(self):
        """Remove the duplications in graph data meta oram tree."""
        # Record the delayed duplications that have been read.
        existing_dd = []
        # Set the temporary stash.
        temp_stash = []

        # For data in the pos_map stash, record the new ones.
        for data in self._graph_meta.stash:
            # If not previously read, add to both existing and stash.
            if (data.key, data.value[0]) not in existing_dd:
                existing_dd.append((data.key, data.value[0]))
                temp_stash.append(data)
            # If read, just ignore it.

        # Update the stash with duplications removed.
        self._graph_meta.stash = temp_stash

    def lookup(self, keys: List[Any]) -> Dict[Any, Any]:
        """
        Perform lookup operation on graph for multiple keys.

        :param keys: List of vertex keys to look up.
        :return: Dict mapping vertex key to its value.
        """
        # Batch OMAP search to get paths for all keys.
        key_path_dict = self._pos_omap.batch_search(keys=keys)

        # Read vertices from graph ORAM.
        return self.lookup_without_omap(key_path_dict)

    def lookup_without_omap(self, key_path_dict: Dict[Any, int]) -> Dict[Any, Any]:
        """
        Read multiple vertices from graph ORAM given their paths.

        :param key_path_dict: Dict mapping vertex key to its path (leaf).
        :return: Dict mapping vertex key to its value.
        """
        leaves = list(key_path_dict.values())

        # Get the next degree number of rl paths.
        rl_path = self.get_rl_leaf(count=self._max_deg * len(leaves)) + leaves

        # Queue reads for both ORAMs.
        self._graph_oram.queue_read(leaves=leaves)
        self._graph_meta.queue_read(leaves=rl_path)
        result = self._client.execute()

        # Process read results.
        self._graph_oram.process_read_result(result)

        # Locate keys of interest and assign new paths.
        data_of_interest = {}
        duplications = []
        for data in self._graph_oram.stash:
            if data.key in key_path_dict:
                data_of_interest[data.key] = data.value
                data.leaf = secrets.randbelow(self._leaf_range)
                # Generate delayed duplications.
                duplications += [
                    Data(key=neighbor, leaf=leaf, value=(data.key, data.leaf))
                    for neighbor, leaf in data.value[1].items()
                ]

        # Process the meta block paths.
        self._graph_meta.process_read_result(result)
        # Add the new duplications to the front.
        self._graph_meta.stash = duplications + self._graph_meta.stash

        # Perform de duplication within graph meta.
        self.graph_meta_de_duplication()

        # Perform further de duplication by inspecting whether graph oram has recipients.
        retrieved_vertices = {data.key: index for index, data in enumerate(self._graph_oram.stash)}

        # Check if recipients exist.
        temp_stash = []
        for data in self._graph_meta.stash:
            if data.key in retrieved_vertices:
                # If the updated path is less than 1, the neighbor should be deleted.
                if data.value[1] < 0:
                    del self._graph_oram.stash[retrieved_vertices[data.key]].value[1][data.value[0]]
                # Otherwise update the value or add the new vertex.
                else:
                    self._graph_oram.stash[retrieved_vertices[data.key]].value[1][data.value[0]] = data.value[1]
            # Else add data back to temp stash.
            else:
                temp_stash.append(data)

        # Update graph meta stash.
        self._graph_meta.stash = temp_stash

        # Queue writes for both ORAMs.
        self._graph_oram.queue_write()
        self._graph_meta.queue_write()
        self._client.execute()

        return data_of_interest

    def insert(self, vertex: tuple) -> None:
        """
        Insert a new vertex to the graph with direct neighbor updates.

        :param vertex: the vertex to add, in format of (key, value, neighbors).
                       neighbors is a dict {neighbor_key: neighbor_leaf} or list of neighbor keys.
        """
        # Sample a new path for the new vertex.
        new_path = secrets.randbelow(self._leaf_range)

        # Insert the new vertex key-path pair to the position OMAP.
        self._pos_omap.insert(key=vertex[0], value=new_path)

        # Get neighbor keys from vertex[2] (could be dict or list).
        neighbor_keys = list(vertex[2].keys()) if isinstance(vertex[2], dict) else list(vertex[2])

        # Use batch_search to get paths for all neighbors.
        neighbor_path_map = self._pos_omap.batch_search(keys=neighbor_keys)

        # Build adjacency dict for the new vertex: {neighbor_key: neighbor_leaf}.
        new_vertex_adjacency = {k: neighbor_path_map[k] for k in neighbor_keys}

        # Collect all leaves to retrieve: new vertex path + all neighbor paths.
        all_leaves = [new_path] + list(neighbor_path_map.values())

        # Queue read and execute.
        self._graph_oram.queue_read(leaves=all_leaves)
        result = self._client.execute()
        self._graph_oram.process_read_result(result)

        # Add the new vertex to stash.
        self._graph_oram.stash.append(
            Data(key=vertex[0], leaf=new_path, value=(vertex[1], new_vertex_adjacency))
        )

        # Build index of vertices in stash for quick lookup.
        stash_index = {data.key: i for i, data in enumerate(self._graph_oram.stash)}

        # Update each neighbor's adjacency list to include the new vertex.
        for neighbor_key in neighbor_keys:
            if neighbor_key in stash_index:
                idx = stash_index[neighbor_key]
                neighbor_data = self._graph_oram.stash[idx]
                # Add the new vertex to neighbor's adjacency list.
                neighbor_data.value[1][vertex[0]] = new_path
                # Assign new random leaf for the neighbor.
                neighbor_data.leaf = secrets.randbelow(self._leaf_range)

        # Queue write and execute.
        self._graph_oram.queue_write()
        self._client.execute()

    def delete(self, key: Any) -> None:
        """
        Delete a vertex from the graph with direct neighbor updates.

        :param key: The key of the vertex to delete.
        """
        # Look up the vertex path from OMAP (and perform lazy deletion by setting value to None).
        vertex_leaf = self._pos_omap.search(key=key, value=None)

        # First retrieve the vertex to get its neighbor list.
        self._graph_oram.queue_read(leaves=[vertex_leaf])
        result = self._client.execute()
        self._graph_oram.process_read_result(result)

        # Find the vertex in stash and get its neighbors.
        vertex_data = None
        vertex_idx = None
        for i, data in enumerate(self._graph_oram.stash):
            if data.key == key:
                vertex_data = data
                vertex_idx = i
                break

        if vertex_data is None:
            # Vertex not found, just write back and return.
            self._graph_oram.queue_write()
            self._client.execute()
            return

        # Get neighbor keys from the vertex's adjacency list.
        neighbor_adjacency = vertex_data.value[1]  # {neighbor_key: neighbor_leaf}
        neighbor_keys = list(neighbor_adjacency.keys())

        if not neighbor_keys:
            # No neighbors, just remove the vertex and write back.
            del self._graph_oram.stash[vertex_idx]
            self._graph_oram.queue_write()
            self._client.execute()
            return

        # Use batch_search to get current paths for all neighbors.
        neighbor_path_map = self._pos_omap.batch_search(keys=neighbor_keys)

        # Retrieve all neighbor paths from graph ORAM.
        neighbor_leaves = list(neighbor_path_map.values())
        self._graph_oram.queue_read(leaves=neighbor_leaves)
        result = self._client.execute()
        self._graph_oram.process_read_result(result)

        # Build index of vertices in stash for quick lookup.
        stash_index = {data.key: i for i, data in enumerate(self._graph_oram.stash)}

        # Remove the deleted vertex from each neighbor's adjacency list.
        for neighbor_key in neighbor_keys:
            if neighbor_key in stash_index:
                idx = stash_index[neighbor_key]
                neighbor_data = self._graph_oram.stash[idx]
                # Remove the deleted vertex from neighbor's adjacency list.
                if key in neighbor_data.value[1]:
                    del neighbor_data.value[1][key]
                # Assign new random leaf for the neighbor.
                neighbor_data.leaf = secrets.randbelow(self._leaf_range)

        # Remove the deleted vertex from stash.
        # Need to re-find index since stash may have changed.
        for i, data in enumerate(self._graph_oram.stash):
            if data.key == key:
                del self._graph_oram.stash[i]
                break

        # Evict to all paths (vertex path + neighbor paths) and write back.
        all_leaves = [vertex_leaf] + neighbor_leaves
        self._graph_oram.queue_write(leaves=all_leaves)
        self._client.execute()

    def neighbor(self, keys: List[Any]) -> dict:
        """
        Perform a neighbor lookup query.

        :param keys: List of vertex keys to find neighbors for.
        :return: Dict mapping vertex key to its value.
        """
        # First lookup using OMAP.
        vertices = self.lookup(keys=keys)

        # Get the neighbors to search (they already have paths in adjacency list).
        search_dict = {}
        for vertex_value in vertices.values():
            search_dict.update(vertex_value[1])

        # For desired number of layers, keep searching.
        for _ in range(self._graph_depth):
            vertices = self.lookup_without_omap(key_path_dict=search_dict)

            # Update the search dictionary with new neighbors.
            search_dict = {}
            for vertex_value in vertices.values():
                search_dict.update(vertex_value[1])

        return vertices

    def t_hop(self, key: Any, num_hop: int) -> dict:
        """
        Perform the t-hop query, finding all neighbors within t hops.

        :param key: Starting vertex key.
        :param num_hop: Number of hops to traverse.
        :return: Dict mapping vertex key to its value for all vertices found.
        """
        result = {}
        keys_to_search = [key]

        for _ in range(num_hop):
            temp_result = self.neighbor(keys=keys_to_search)

            keys_to_search = []
            for k, value in temp_result.items():
                if k not in result:
                    keys_to_search.append(k)
                    result[k] = value

        return result

    def t_traversal(self, key: Any, num_hop: int) -> dict:
        """
        Perform t-hop traversal, randomly selecting one neighbor per hop.

        :param key: Starting vertex key.
        :param num_hop: Number of hops to traverse.
        :return: Dict mapping vertex key to its value for visited vertices.
        """
        result = {}
        keys_to_search = [key]

        for _ in range(num_hop):
            temp_result = self.neighbor(keys=keys_to_search)

            key_of_interest = random.choice(list(temp_result.keys()))
            result[key_of_interest] = temp_result[key_of_interest]

            keys_to_search = [key_of_interest]

        return result
