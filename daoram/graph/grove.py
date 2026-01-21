import math
import random
import secrets
from decimal import Decimal, localcontext
from typing import List, Any, Optional, Dict

from daoram.dependency import InteractServer, Data
from daoram.omap.avl_ods_omap_opt import AVLOdsOmapOptimized
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
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True):
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
        :param aes_key: The key to use for the AES instance.
        :param num_key_bytes: The number of bytes the aes key should have.
        :param use_encryption: A boolean indicating whether to use encryption.
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
        self._pos_omap = AVLOdsOmapOptimized(
            client=client,
            aes_key=aes_key,
            num_data=num_data,
            key_size=key_size,
            data_size=data_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            num_key_bytes=num_key_bytes,
            use_encryption=use_encryption,
            filename=f"{filename}_avl" if filename else None
        )

        # Initialize the multi-path ORAMs.
        self._graph_oram = MulPathOram(
            name="graph",
            client=client,
            aes_key=aes_key,
            num_data=num_data,
            data_size=key_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            num_key_bytes=num_key_bytes,
            use_encryption=use_encryption,
            filename=f"{filename}_graph_mp" if filename else None
        )

        self._graph_meta = MulPathOram(
            name="g_meta",
            client=client,
            aes_key=aes_key,
            num_data=num_data,
            data_size=key_size,
            bucket_size=meta_bucket_size,
            stash_scale=stash_scale,
            num_key_bytes=num_key_bytes,
            use_encryption=use_encryption,
            filename=f"{filename}_graph_meta" if filename else None
        )

        self._pos_meta = MulPathOram(
            name="p_meta",
            client=client,
            aes_key=aes_key,
            num_data=num_data,
            data_size=key_size,
            bucket_size=meta_bucket_size,
            stash_scale=stash_scale,
            num_key_bytes=num_key_bytes,
            use_encryption=use_encryption,
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

    def lookup(self, key: List[Any]) -> Optional[dict]:
        """Perform lookup operation on graph."""
        # First lookup the pos omap.
        leaf = self._pos_omap.search_with_meta(key=key, meta=self._pos_meta)

        # Get the next degree number of rl paths (for each key to lookup).
        rl_path = self.get_rl_leaf(count=self._max_deg * len(key)) + [leaf]

        # Use the leaf to retrieve graph oram.
        paths = self._client.read_mul_query(label=["graph", "g_meta"], leaf=[leaf, rl_path])
        self._graph_oram.process_path_to_stash(path=paths[0])

        # Locate the key of interest and give it a new path.
        data_of_interest = {}
        duplications = []
        for data in self._graph_oram.stash:
            # todo: key is a list here, so check it
            if data.key in key:
                data_of_interest[data.key] = data.value
                data.leaf = secrets.randbelow(self._leaf_range)
                # Generate delayed duplications.
                duplications += [
                    Data(key=neighbor, leaf=leaf, value=(data.key, data.leaf))
                    for neighbor, leaf in data.value[1].items()
                ]

        # Process the meta block paths.
        self._graph_meta.process_path_to_stash(path=paths[1])
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
                # Otherwise update the value.
                else:
                    self._graph_oram.stash[retrieved_vertices[data.key]].value[1][data.value[0]] = data.value[1]
            # Else add data back to temp stash.
            else:
                temp_stash.append(data)

        # Update graph meta stash.
        self._graph_meta.stash = temp_stash

        # Write things back.
        graph_oram_path = self._graph_oram.prepare_evict_path(leaves=[leaf])
        graph_meta_path = self._graph_meta.prepare_evict_path(leaves=rl_path)

        # Write both orams together.
        self._client.write_mul_query(
            label=["graph", "g_meta"], leaf=[leaf, rl_path], data=[graph_oram_path, graph_meta_path]
        )

        return data_of_interest

    def lookup_without_omap(self, key_path_dict: Dict[Any, int]) -> Optional[dict]:
        """Perform lookup operation on graph."""
        # First get the leaves to grab.
        leaves = list(key_path_dict.values())

        # Get the next degree number of rl paths.
        rl_path = self.get_rl_leaf(count=self._max_deg * len(leaves)) + leaves

        # Use the leaf to retrieve graph oram.
        paths = self._client.read_mul_query(label=["graph", "g_meta"], leaf=[leaves, rl_path])
        self._graph_oram.process_path_to_stash(path=paths[0])

        # Locate the key of interest and give it a new path.
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
        self._graph_meta.process_path_to_stash(path=paths[1])
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
                # Otherwise update the value or add the new vertex (same operation).
                else:
                    self._graph_oram.stash[retrieved_vertices[data.key]].value[1][data.value[0]] = data.value[1]
            # Else add data back to temp stash.
            else:
                temp_stash.append(data)

        # Update graph meta stash.
        self._graph_meta.stash = temp_stash

        # Write things back.
        graph_oram_path = self._graph_oram.prepare_evict_path(leaves=leaves)
        graph_meta_path = self._graph_meta.prepare_evict_path(leaves=rl_path)

        # Write both orams together.
        self._client.write_mul_query(
            label=["graph", "g_meta"], leaf=[leaves, rl_path], data=[graph_oram_path, graph_meta_path]
        )

        return data_of_interest

    def insert(self, vertex: tuple) -> None:
        """
        Insert a new vertex to the graph.

        :param vertex: the vertex to add, in format of (key, value, neighbors).
        """
        # Sample a new path.
        path = secrets.randbelow(self._leaf_range)

        # First insert vertex and path pair to the OMAP.
        self._pos_omap.insert(key=vertex[0], value=path)

        # Search all the neighbors.
        neighbor_paths = self._pos_omap.search_with_meta(key=vertex[2], meta=self._pos_meta)

        # Get the next degree number of rl paths.
        rl_path = self.get_rl_leaf(count=self._max_deg) + [path]

        # Use the leaf to retrieve graph oram.
        retrieved_paths = self._client.read_mul_query(label=["graph", "g_meta"], leaf=[path, rl_path])

        # Process the graph ORAM.
        self._graph_oram.process_path_to_stash(path=retrieved_paths[0])
        # Add the new value.
        self._graph_oram.stash += Data(key=vertex[0], leaf=path, value=(vertex[1], vertex[2]))

        # Add delayed duplications.
        duplications = []
        for i, neighbor in enumerate(vertex[2]):
            # Add the delayed duplications with updated information.
            duplications += [Data(key=neighbor, leaf=neighbor_paths[i], value=(vertex[0], path))]

        # Process the meta block paths.
        self._graph_meta.process_path_to_stash(path=retrieved_paths[1])
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
                # Otherwise update the value or add the new vertex (same operation).
                else:
                    self._graph_oram.stash[retrieved_vertices[data.key]].value[1][data.value[0]] = data.value[1]
            # Else add data back to temp stash.
            else:
                temp_stash.append(data)

        # Update graph meta stash.
        self._graph_meta.stash = temp_stash

        # Write things back.
        graph_oram_path = self._graph_oram.prepare_evict_path(leaves=[path])
        graph_meta_path = self._graph_meta.prepare_evict_path(leaves=rl_path)

        # Write both orams together.
        self._client.write_mul_query(
            label=["graph", "g_meta"], leaf=[path, rl_path], data=[graph_oram_path, graph_meta_path]
        )

    def delete(self, key: Any) -> None:
        """Perform lookup operation on graph."""
        # First lookup the pos omap (and perform a lazy deletion).
        leaf = self._pos_omap.search_with_meta(key=key, value=None, meta=self._pos_meta)

        # Get the next degree number of rl paths (for each key to lookup).
        rl_path = self.get_rl_leaf(count=self._max_deg * len(key)) + [leaf]

        # Use the leaf to retrieve graph oram.
        paths = self._client.read_mul_query(label=["graph", "g_meta"], leaf=[leaf, rl_path])
        self._graph_oram.process_path_to_stash(path=paths[0])

        # Locate the key of interest and give it a new path.
        temp_stash = []
        duplications = []

        for data in self._graph_oram.stash:
            if data.key == key:
                # Generate delayed duplications.
                duplications += [
                    Data(key=neighbor, leaf=leaf, value=(data.key, -1))
                    for neighbor, leaf in data.value[1].items()
                ]
            else:
                temp_stash.append(data)

        # Update the stash of graph oram.
        self._graph_oram.stash = temp_stash

        # Process the meta block paths.
        self._graph_meta.process_path_to_stash(path=paths[1])
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
                # Otherwise update the value.
                else:
                    self._graph_oram.stash[retrieved_vertices[data.key]].value[1][data.value[0]] = data.value[1]
            # Else add data back to temp stash.
            else:
                temp_stash.append(data)

        # Update graph meta stash.
        self._graph_meta.stash = temp_stash

        # Write things back.
        graph_oram_path = self._graph_oram.prepare_evict_path(leaves=[leaf])
        graph_meta_path = self._graph_meta.prepare_evict_path(leaves=rl_path)

        # Write both orams together.
        self._client.write_mul_query(
            label=["graph", "g_meta"], leaf=[leaf, rl_path], data=[graph_oram_path, graph_meta_path]
        )

    def neighbor(self, key: List[Any]) -> dict:
        """Perform a neighbor lookup query."""
        # First lookup the pos omap.
        vertices = self.lookup(key=key)

        # Get the neighbors to search.
        search_dict = {}
        for vertex_value in vertices.values():
            search_dict += vertex_value[1]

        # For desired number of layers, keep searching.
        for i in range(self._graph_depth):
            vertices = self.lookup_without_omap(key_path_dict=search_dict)

            # Update the search dictionary accordingly.
            # Get the neighbors to search.
            search_dict = {}
            for vertex_value in vertices.values():
                search_dict += vertex_value[1]

        return vertices

    def t_hop(self, key: Any, num_hop: int) -> dict:
        """Perform the t-hop query, for each hop we find all neighbor of current retrieved nodes."""
        # Set an empty dictionary.
        result = {}

        # Set the initial key as a dictionary.
        keys_to_search = [key]

        # For each hop, we will grab all neighbors and keep looking for neighbors with all neighbors.
        for _ in range(num_hop):
            # Get all neighbors and current search keys.
            temp_result = self.neighbor(key=keys_to_search)

            # Reset keys_to_search to empty.
            keys_to_search = []

            # For the ones that aren't in result already, keep searching.
            for key, value in temp_result.items():
                if key not in result:
                    # Add it to search key.
                    keys_to_search.append(key)
                    # Add it to result.
                    result[key] = value

        # Return the result dictionary.
        return result

    def t_traversal(self, key: Any, num_hop: int) -> dict:
        """Perform the t-hop query, for each hop we find all neighbor of current retrieved nodes."""
        # Set an empty dictionary.
        result = {}

        # Set the initial key as a dictionary.
        keys_to_search = [key]

        # For each hop, we will grab all neighbors and keep looking for neighbors with all neighbors.
        for _ in range(num_hop):
            # Get all neighbors and current search keys.
            temp_result = self.neighbor(key=keys_to_search)

            # Select a random key to keep searching.
            key_of_interest = random.choice(list(temp_result.keys()))
            result[key_of_interest] = temp_result[key_of_interest]

            # Update the key to search.
            keys_to_search = [key_of_interest]

        # Return the result dictionary.
        return result
