import math
import random
import secrets
from typing import Any, List

from daoram.dependency import InteractServer, Data
from daoram.dependency.graph import Graph
from daoram.omap.avl_ods_omap_opt import AVLOdsOmapOptimized
from daoram.oram.mul_path_oram import MulPathOram


class GraphOS:
    def __init__(self,
                 max_deg: int,
                 num_data: int,
                 key_size: int,
                 data_size: int,
                 client: InteractServer,
                 bucket_size: int = 4,
                 stash_scale: int = 7,
                 filename: str = None,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True):
        """
        Initializes the GraphOS.

        :param max_deg: The maximum number of neighbors a vertex can have.
        :param num_data: The number of data points the oram should store.
        :param key_size: The number of bytes the random dummy key should have.
        :param data_size: The number of bytes the random dummy data should have.
        :param client: The instance we use to interact with server.
        :param filename: The filename to save the oram data to.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param aes_key: The key to use for the AES instance.
        :param num_key_bytes: The number of bytes the aes key should have.
        :param use_encryption: A boolean indicating whether to use encryption.
        """
        # Store the maximum degree of the graph.
        self.max_deg: int = max_deg

        # Compute the level of the binary tree needed.
        self._level: int = int(math.ceil(math.log(num_data, 2))) + 1

        # Compute the range of possible leafs [0, leaf_range).
        self._leaf_range: int = pow(2, self._level - 1)

        # Initialize the OMAP.
        self._omap = AVLOdsOmapOptimized(
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

        # Initialize the multi-path ORAM.
        self._oram = MulPathOram(
            client=client,
            aes_key=aes_key,
            num_data=num_data,
            data_size=key_size,
            bucket_size=bucket_size,
            stash_scale=stash_scale,
            num_key_bytes=num_key_bytes,
            use_encryption=use_encryption,
            filename=f"{filename}_mp" if filename else None
        )

    def init_server_storage(self, graph: Graph):
        """Initialize the graphos server storage."""
        # Sample the path number for each key.
        path_map = {vertex: secrets.randbelow(self._leaf_range) for vertex in graph}
        # Store them in the OMAP.
        self._omap.init_server_storage(data=[(key, value) for key, value in path_map.items()])
        # Prepare the correct data item.
        data = [Data(key=key, value=value, leaf=path_map[key]) for key, value in graph.items()]
        # Initialize the multiple path ORAM.
        self._oram.init_server_storage(data=data)

    def lookup(self, key: List[Any], update: int = None) -> dict:
        """
        Perform lookup on input vertices.

        :param key: A list of vertices of interest.
        :param update: If update is not None, it is added to the vertex value (for updating counts).
        :return: The values corresponding to the vertices of interest.
        """
        # Sample the new leaves for each key.
        new_leaves = {key: secrets.randbelow(self._leaf_range) for _ in key}

        # Find the leaves, result are the paths of keys of interest.
        leaves = self._omap.search(key=key, value=new_leaves.values())

        # Retrieve the paths of interest.
        self._oram.retrieve_path(leaves=leaves)

        # Get a dictionary for returned result.
        result = {}

        # Obtain the stash.
        stash = self._oram.stash

        # Retrieve the desired keys.
        for data in stash:
            if data.key in key:
                # Save the result.
                result[data.key] = data.value
                # Update the data leaf.
                data.leaf = new_leaves[data.key]
                # If update is not None, update the last position (counter) by 1.
                if update is not None:
                    data.value[:-1] += update

        # Update the stash.
        self._oram.stash = stash

        # Write back.
        self._oram.evict_path(leaves=leaves)

        return result

    def neighbor(self, key: List[Any]) -> dict:
        """
        Perform a neighbor query, for each key in the input list, download all its neighbors.

        :param key: A list of vertices of interest.
        :return: The neighbors of the vertice of interest and their values.
        """
        # For each key first find its value and get their number of neighbors.
        num_neighbors = self.lookup(key=[f"V{each_key}" for each_key in key])

        # Get the number of edges for each vertice and prepare keys.
        search_keys = []

        # Loop through the retrieved keys.
        for key in num_neighbors:
            # Number of neighbors is the last position.
            num_edge = num_neighbors[key][-1]
            for i in range(1, num_edge + 1):
                search_keys.append(f"E{key}{i}")

        # Get the value corresponding to the edges.
        neighbors = self.lookup(key=search_keys)

        # Loop through the retrieved neighbors.
        new_search_key = []
        for key in neighbors:
            new_search_key.append(f"V{neighbors[key]}")

        # Finally search for the neighbors.
        return self.lookup(key=new_search_key)

    def insert(self, vertex: tuple) -> None:
        """
        Insert a new vertex to the graph.

        :param vertex: the vertex to add, in format of (key, value, neighbors).
        """
        # First sample the path to put the value in ORAM.
        path_for_value = secrets.randbelow(self._leaf_range)

        # First insert the new vertex and then update the corresponding neighbors.
        self._omap.insert(vertex[0], path_for_value)

        # Grab a path and add the value.
        self._oram.retrieve_path(leaves=[secrets.randbelow(self._leaf_range)])

        # Add the value to stash.
        self._oram.stash.append(Data(key=f"V{vertex[0]}", value=vertex[1], leaf=path_for_value))

        # Retrieve the neighbor vertex and add their value by 1.
        neighbors = self.lookup(key=[f"V{neighbor}" for neighbor in vertex[2]], update=1)

        # Finally insert the edge values for all neighbors.
        insert_keys = [f"E{key}{value[-1] + 1}" for key, value in neighbors.items()]
        # Sample paths for these keys.
        leaves = [secrets.randbelow(self._leaf_range) for _ in insert_keys]
        # Insert keys and their ORAM paths to OMAP.
        self._omap.insert(key=insert_keys, value=leaves)

        # Download paths.
        download_paths = [secrets.randbelow(self._leaf_range) for _ in insert_keys]
        self._oram.retrieve_path(leaves=download_paths)
        # Add to stash.
        self._oram.stash += [
            Data(key=each_key, value=vertex[0], leaf=leaves[index]) for index, each_key in enumerate(insert_keys)
        ]
        # Perform eviction.
        self._oram.evict_path(leaves=download_paths)

    def delete(self, key: Any) -> None:
        # First remove the key (lazily) from the omap and get vertex value.
        delete_path = self._omap.search(key=key, value=-1)

        # Download the desired path.
        self._oram.retrieve_path(leaves=[delete_path])

        # Find the value in stash.
        value, temp_stash = None, []
        for data in self._oram.stash:
            if data.key == key:
                # Store the value.
                value = data.value
            else:
                # Add to temp stash if not the correct one.
                temp_stash.append(data)

        # Update the stash.
        self._oram.stash = temp_stash
        self._oram.evict_path(leaves=[delete_path])

        # Delete the edges as well.
        edge_keys = [f"E{key}{i}" for i in range(1, value[-1] + 1)]
        edge_paths = self._omap.search(key=edge_keys, value=[-1 for _ in edge_keys])

        # Download the desired path.
        self._oram.retrieve_path(edge_paths)

        # Find the desired values in stash.
        vertices, temp_stash = [], []
        for data in self._oram.stash:
            if data.key in edge_keys:
                # Store the vertex value.
                vertices.append(f"V{data.value}")
            else:
                # Add to temp stash.
                temp_stash.append(data)

        # Update the stash.
        self._oram.stash = temp_stash
        self._oram.evict_path(leaves=[edge_paths])

        # Finally update the count.
        # For fairness, we assume a simple edit here; the edge to delete can be postponed during neighbor traversal.
        vertices_path = self._omap.search(key=vertices)
        # Download the vertices.
        self._oram.retrieve_path(leaves=vertices_path)
        # Update the vertice values.
        for data in self._oram.stash:
            if data.key in edge_keys:
                # Store the vertex value.
                data.value[-1] -= 1

            # Add to temp stash.
            temp_stash.append(data)
        # Write them back.
        self._oram.evict_path(leaves=vertices_path)

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
