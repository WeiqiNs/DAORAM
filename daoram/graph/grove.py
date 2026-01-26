import math
import random
import secrets
from decimal import Decimal, localcontext
from typing import List, Any, Dict

from daoram.dependency import InteractServer, Data, Encryptor
from daoram.omap.avl_omap_cache import AVLOmapCached
from daoram.oram.mul_path_oram import MulPathOram


class MetaMulPathOram(MulPathOram):
    def process_read_result(self, result: Any) -> None:
        if self._name not in result.results: return
        path_data = result.results[self._name]
        path = self.decrypt_path_data(path=path_data)
        for bucket in path.values():
            for data in bucket:
                if data.key is None: continue
                self._stash.append(data)


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
        """Initializes the GraphOS."""
        self._client = client
        self._max_deg: int = max_deg
        self._num_opr: int = num_opr
        self._graph_depth: int = graph_depth
        self._level: int = int(math.ceil(math.log(num_data, 2))) + 1
        self._leaf_range: int = pow(2, self._level - 1)
        self._graph_counter = 0
        self._pos_counter = 0
        meta_bucket_size = self.find_bound()

        self._pos_omap = AVLOmapCached(
            client=client, num_data=num_data, key_size=key_size, data_size=data_size,
            bucket_size=bucket_size, stash_scale=stash_scale, encryptor=encryptor,
            filename=f"{filename}_avl" if filename else None, enable_meta=True
        )

        self._graph_oram = MulPathOram(
            name="graph", client=client, num_data=num_data, data_size=key_size,
            bucket_size=bucket_size, stash_scale=stash_scale, encryptor=encryptor,
            filename=f"{filename}_graph_mp" if filename else None
        )

        self._graph_meta = MetaMulPathOram(
            name="g_meta", client=client, num_data=num_data, data_size=key_size,
            bucket_size=meta_bucket_size, stash_scale=stash_scale, encryptor=encryptor,
            filename=f"{filename}_graph_meta" if filename else None
        )
        self._pos_meta = self._pos_omap._meta

    @staticmethod
    def binomial(n: int, i: int, p: Decimal) -> Decimal:
        return Decimal(math.comb(n, i)) * (p ** i) * ((Decimal(1) - p) ** (n - i))

    @staticmethod
    def equation(m: int, K: int, Y: int, L: int, prec: int) -> Decimal:
        sigma = math.ceil(math.log2(Y))
        prob = Decimal(0)
        with localcontext() as ctx:
            ctx.prec = prec
            for j in range(sigma, L):
                temp_prob = Decimal(0)
                n = 2 ** j
                p = Decimal(1) / (Decimal(2) ** (j + 1))
                for i in range(math.floor(Y / K)):
                    temp_prob += Grove.binomial(n=n, i=i, p=p)
                term = (Decimal(2) ** j) * Decimal(math.ceil(m * K / (2 ** j))) * (Decimal(1) - temp_prob)
                prob += term
        return prob

    def find_bound(self, prec: int = 80):
        Y = 1
        while (self.equation(m=self._num_opr, L=self._level, K=self._max_deg, Y=Y, prec=prec) >
               Decimal(1) / Decimal(pow(2, 128))):
            Y += 1
        return Y

    def get_rl_leaf(self, count: int, for_pos_meta: bool = False) -> List[int]:
        num_bits = self._level - 1
        counter = self._pos_counter if for_pos_meta else self._graph_counter
        leaves = [int(format((counter + i) % self._leaf_range, f"0{num_bits}b")[::-1], 2) for i in range(count)]
        if for_pos_meta: self._pos_counter = (self._pos_counter + count) % self._leaf_range
        else: self._graph_counter = (self._graph_counter + count) % self._leaf_range
        return leaves

    def pos_meta_de_duplication(self):
        seen = set(); temp = []
        for d in self._pos_meta.stash:
            if d.key is not None and d.key not in seen:
                seen.add(d.key); temp.append(d)
        self._pos_meta.stash = temp

    def graph_meta_de_duplication(self):
        """Remove duplicate duplications in graph meta ORAM stash."""
        existing_neighbor_updates = {}  # key -> {source_key: data}
        existing_pos_updates = {}       # key -> data
        temp_stash = []
        for data in self._graph_meta.stash:
            if data.key is None: continue
            if isinstance(data.value, tuple) and len(data.value) == 2:
                # Type 1: Neighbor update
                if data.key not in existing_neighbor_updates:
                    existing_neighbor_updates[data.key] = {}
                source_key = data.value[0]
                if source_key not in existing_neighbor_updates[data.key]:
                    existing_neighbor_updates[data.key][source_key] = data
                    temp_stash.append(data)
            else:
                # Type 2: PosMap update
                if data.key not in existing_pos_updates:
                    existing_pos_updates[data.key] = data
                    temp_stash.append(data)
        self._graph_meta.stash = temp_stash

    def lookup(self, keys: List[Any], return_visited_nodes: bool = False) -> Dict[Any, Any]:
        res, v_map, total_paths = self._pos_omap.batch_search(keys=keys, return_visited_nodes=True)
        key_gl_dict = {k: v for k, v in res.items() if v is not None}
        pos_meta_extra = max(0, total_paths - len(keys))
        result = self.lookup_without_omap(key_gl_dict, v_map, pos_meta_extra)
        ext_res = {k: (v[0], v[1]) for k, v in result.items()}
        return (ext_res, v_map) if return_visited_nodes else ext_res

    def lookup_without_omap(self, key_gl_dict: Dict[Any, int], v_map: Dict[Any, tuple] = None, extra_paths: int = 0) -> Dict[Any, Any]:
        leaves = list(key_gl_dict.values())
        if not leaves: return {}
        if v_map is None: v_map = {}
        rl_path = self.get_rl_leaf(count=self._max_deg * len(leaves) + extra_paths, for_pos_meta=False) + leaves
        if self._pos_omap._enable_meta: self._pos_omap._meta.queue_read(leaves=leaves)
        
        self._graph_oram.queue_read(leaves=leaves); self._graph_meta.queue_read(leaves=rl_path)
        result = self._client.execute()
        self._graph_oram.process_read_result(result); self._graph_meta.process_read_result(result)
        if self._pos_omap._enable_meta:
            self._pos_omap._meta.process_read_result(result)
            self._pos_omap.meta_de_duplication()
        self.graph_meta_de_duplication()

        target_keys = set(key_gl_dict.keys()); retrieved = {d.key: i for i, d in enumerate(self._graph_oram.stash)}
        temp_meta = []
        for dup in self._graph_meta.stash:
            if dup.key in target_keys and dup.key in retrieved:
                idx = retrieved[dup.key]; v_val = self._graph_oram.stash[idx].value
                if isinstance(dup.value, tuple) and len(dup.value) == 2:
                    adj = v_val[1]; s_k, n_gl = dup.value
                    if n_gl < 0:
                        if s_k in adj: del adj[s_k]
                    else: 
                        if s_k in adj: adj[s_k] = n_gl
                else: self._graph_oram.stash[idx].value = (v_val[0], v_val[1], dup.value)
            else: temp_meta.append(dup)
        self._graph_meta.stash = temp_meta

        for v_key, (n_pl, _) in v_map.items():
            if v_key in retrieved:
                idx = retrieved[v_key]; v_val = self._graph_oram.stash[idx].value
                self._graph_oram.stash[idx].value = (v_val[0], v_val[1], n_pl)

        res = {}; g_dups = []; p_dups = []; new_paths = {k: secrets.randbelow(self._leaf_range) for k in target_keys}
        for data in self._graph_oram.stash:
            if data.key not in target_keys: continue
            new_gl = new_paths[data.key]; data.leaf = new_gl
            v_d, adj, old_pl = data.value if len(data.value) == 3 else (data.value[0], data.value[1], None)
            new_pl = v_map.get(data.key, (old_pl, None))[0]
            data.value = (v_d, adj, new_pl); res[data.key] = (v_d, adj, new_gl)
            for nk, ngl in adj.items():
                if isinstance(ngl, tuple): ngl = ngl[0]
                t_gl = new_paths.get(nk, ngl)
                if nk in new_paths: adj[nk] = t_gl
                g_dups.append(Data(key=nk, leaf=t_gl, value=(data.key, new_gl)))
            if new_pl is not None: p_dups.append(Data(key=data.key, leaf=new_pl, value=new_gl))

        for v_key, (n_pl, v_gl) in v_map.items():
            if v_key not in retrieved and v_gl is not None:
                g_dups.append(Data(key=v_key, leaf=v_gl, value=n_pl))

        self._graph_meta.stash = g_dups + self._graph_meta.stash
        self._pos_omap.add_meta_duplications(p_dups)
        self.graph_meta_de_duplication(); self._pos_omap.meta_de_duplication()
        self._graph_oram.queue_write(leaves=leaves); self._graph_meta.queue_write(leaves=rl_path)
        if self._pos_omap._enable_meta: self._pos_omap._meta.queue_write(leaves=leaves)
        self._client.execute(); return res

    def insert(self, vertex: tuple) -> None:
        """Insert a new vertex."""
        neighbor_keys = list(vertex[2].keys()) if isinstance(vertex[2], dict) else list(vertex[2])
        new_gl = secrets.randbelow(self._leaf_range)
        new_pl, i_visited = self._pos_omap.insert(key=vertex[0], value=new_gl, return_pos_leaf=True, return_visited_nodes=True)
        key_gl_dict, s_visited, total_pl_paths = ({}, {}, 0)
        if neighbor_keys: key_gl_dict, s_visited, total_pl_paths = self._pos_omap.batch_search(keys=neighbor_keys, return_visited_nodes=True)
        visited = {**i_visited, **s_visited}; neighbor_leaves = [v for v in key_gl_dict.values() if v is not None]
        all_gls = [secrets.randbelow(self._leaf_range)] + neighbor_leaves + [secrets.randbelow(self._leaf_range) for _ in range(max(0, self._max_deg - len(neighbor_keys)))]
        rl_paths = self.get_rl_leaf(count=self._max_deg*(self._max_deg-1)+len(visited), for_pos_meta=False) + all_gls
        
        self._graph_oram.queue_read(leaves=all_gls); self._graph_meta.queue_read(leaves=rl_paths)
        if self._pos_omap._enable_meta: self._pos_omap._meta.queue_read(leaves=all_gls)
        result = self._client.execute()
        self._graph_oram.process_read_result(result); self._graph_meta.process_read_result(result)
        if self._pos_omap._enable_meta: self._pos_omap._meta.process_read_result(result); self._pos_omap.meta_de_duplication()
        self.graph_meta_de_duplication()

        target_keys = set(key_gl_dict.keys()); retrieved = {d.key: i for i, d in enumerate(self._graph_oram.stash)}
        temp_meta = []
        for dup in self._graph_meta.stash:
            if dup.key in target_keys and dup.key in retrieved:
                idx = retrieved[dup.key]; v_val = self._graph_oram.stash[idx].value
                if isinstance(dup.value, tuple) and len(dup.value) == 2:
                    adj = v_val[1]; s_k, n_gl = dup.value
                    if n_gl < 0:
                        if s_k in adj: del adj[s_k]
                    else: 
                        if s_k in adj: adj[s_k] = n_gl
                else: self._graph_oram.stash[idx].value = (v_val[0], v_val[1], dup.value)
            else: temp_meta.append(dup)
        self._graph_meta.stash = temp_meta

        g_dups = []
        for v_key, (n_pl, _) in visited.items():
            if v_key in retrieved:
                idx = retrieved[v_key]; v_val = self._graph_oram.stash[idx].value
                self._graph_oram.stash[idx].value = (v_val[0], v_val[1], n_pl)
    
        # Also notify updates for visited PosMap nodes (not in retrieved)
        for v_key, (n_pl, v_gl) in visited.items():
            if v_key not in retrieved and v_key != vertex[0] and v_gl is not None:
                g_dups.append(Data(key=v_key, leaf=v_gl, value=n_pl))

        p_dups = []; new_v_adj = {}; batch_paths = {vertex[0]: new_gl}
        for nk in neighbor_keys:
            if nk in retrieved: batch_paths[nk] = secrets.randbelow(self._leaf_range)

        for nk in neighbor_keys:
            if nk not in retrieved: continue
            n_data = self._graph_oram.stash[retrieved[nk]]; n_ngl = batch_paths[nk]
            v_d, n_adj, n_pl = n_data.value if len(n_data.value) == 3 else (n_data.value[0], n_data.value[1], None)
            n_new_pl = visited.get(nk, (n_pl, None))[0]
            n_adj[vertex[0]] = new_gl; n_data.leaf = n_ngl; new_v_adj[nk] = n_ngl
            for nn_key, nn_gl in n_adj.items():
                if nn_key == vertex[0]: continue
                if isinstance(nn_gl, tuple): nn_gl = nn_gl[0]
                t_gl = batch_paths.get(nn_key, nn_gl); 
                if nn_key in batch_paths: n_adj[nn_key] = t_gl
                g_dups.append(Data(key=nn_key, leaf=t_gl, value=(nk, n_ngl)))
            g_dups.append(Data(key=vertex[0], leaf=new_gl, value=(nk, n_ngl)))
            if n_new_pl is not None: p_dups.append(Data(key=nk, leaf=n_new_pl, value=n_ngl))
            n_data.value = (v_d, n_adj, n_new_pl)

        for v_key, (n_pl, v_gl) in visited.items():
            if v_key not in retrieved and v_key != vertex[0] and v_gl is not None:
                g_dups.append(Data(key=v_key, leaf=v_gl, value=n_pl))

        self._graph_oram.stash.append(Data(key=vertex[0], leaf=new_gl, value=(vertex[1], new_v_adj, new_pl)))
        self._graph_meta.stash = g_dups + self._graph_meta.stash
        self._pos_omap.add_meta_duplications(p_dups)
        self.graph_meta_de_duplication(); self._pos_omap.meta_de_duplication()
        self._graph_oram.queue_write(leaves=all_gls); self._graph_meta.queue_write(leaves=rl_paths)
        if self._pos_omap._enable_meta: self._pos_omap._meta.queue_write(leaves=all_gls)
        self._client.execute()

    def delete(self, key: Any) -> None:
        """Delete a vertex."""
        v_gl, visited = self._pos_omap.delete(key=key, return_visited_nodes=True)
        if v_gl is None: return
        rl_paths = self.get_rl_leaf(count=self._max_deg + len(visited), for_pos_meta=False)
        
        self._graph_oram.queue_read(leaves=[v_gl]); self._graph_meta.queue_read(leaves=rl_paths + [v_gl])
        if self._pos_omap._enable_meta: self._pos_omap._meta.queue_read(leaves=[v_gl])
        result = self._client.execute()
        self._graph_oram.process_read_result(result); self._graph_meta.process_read_result(result)
        if self._pos_omap._enable_meta: self._pos_omap._meta.process_read_result(result); self._pos_omap.meta_de_duplication()
        self.graph_meta_de_duplication()
        
        retrieved = {d.key: i for i, d in enumerate(self._graph_oram.stash)}
        if key not in retrieved:
            self._graph_oram.queue_write(leaves=[v_gl]); self._graph_meta.queue_write(leaves=rl_paths + [v_gl])
            if self._pos_omap._enable_meta: self._pos_omap._meta.queue_write(leaves=[v_gl])
            self._client.execute(); return
        
        idx = retrieved[key]; temp_meta = []
        for dup in self._graph_meta.stash:
            if dup.key == key:
                v_val = self._graph_oram.stash[idx].value
                if isinstance(dup.value, tuple) and len(dup.value) == 2:
                    adj = v_val[1]; s_k, n_gl = dup.value
                    if n_gl < 0:
                        if s_k in adj: del adj[s_k]
                    else: 
                        if s_k in adj: adj[s_k] = n_gl
                else: self._graph_oram.stash[idx].value = (v_val[0], v_val[1], dup.value)
                continue
            temp_meta.append(dup)
        self._graph_meta.stash = temp_meta
        
        adj = self._graph_oram.stash[idx].value[1]; neighbor_keys = list(adj.keys())
        # Optimization: Use stored leaves in adjacency list instead of PosMap batch search
        
        del_dups = []
        for nk, ngl in adj.items():
            if isinstance(ngl, tuple): ngl = ngl[0]
            if ngl is not None: 
                del_dups.append(Data(key=nk, leaf=ngl, value=(key, -1)))
        
        while len(del_dups) < self._max_deg: del_dups.append(Data(key=None, leaf=secrets.randbelow(self._leaf_range), value=(None, -1)))
        
        # Also notify updates for visited PosMap nodes (same as lookup/insert)
        # for v_key, (n_pl, v_gl) in visited.items():
        #     if v_key != key and v_gl is not None:
        #         del_dups.append(Data(key=v_key, leaf=v_gl, value=n_pl))

        self._graph_meta.stash = del_dups + self._graph_meta.stash
        del self._graph_oram.stash[idx]
        
        self._graph_oram.queue_write(leaves=[v_gl]); self._graph_meta.queue_write(leaves=rl_paths + [v_gl])
        if self._pos_omap._enable_meta: self._pos_omap._meta.queue_write(leaves=[v_gl])
        self._client.execute()

    def neighbor(self, keys: List[Any]) -> dict:
        """Perform a neighbor lookup query."""
        if not keys: return {}
        c_key = keys[0]
        key_gl_dict, c_visited, total_pl_paths = self._pos_omap.batch_search(keys=[c_key], return_visited_nodes=True)
        if c_key not in key_gl_dict or (key_graph_leaf := key_gl_dict[c_key]) is None: return {}
        c_lookup = self.lookup_without_omap({c_key: key_graph_leaf}, c_visited, max(0, total_pl_paths - 1))
        if c_key not in c_lookup: return {}
        c_vdata, c_adj, c_new_gl = c_lookup[c_key]; nk_keys = list(c_adj.keys())
        # Optimization: Use stored leaves in adjacency list instead of PosMap batch search
        nk_gls = []
        for nk, gl in c_adj.items():
            if isinstance(gl, tuple): gl = gl[0]
            if gl is not None: nk_gls.append(gl)
        
        # Padding for obliviousness
        all_gls = nk_gls + [secrets.randbelow(self._leaf_range) for _ in range(max(0, self._max_deg - len(nk_keys)))]
        all_gls_with_center = all_gls + [c_new_gl]
        
        # Since we skipped batch_search, we only have visited nodes from center lookup
        all_visited = c_visited
        rl_count = self._max_deg * self._max_deg
        rl_paths = self.get_rl_leaf(count=rl_count + len(all_visited), for_pos_meta=False) + all_gls_with_center
        
        self._graph_oram.queue_read(leaves=all_gls_with_center); self._graph_meta.queue_read(leaves=rl_paths)
        if self._pos_omap._enable_meta: self._pos_omap._meta.queue_read(leaves=all_gls_with_center)
        result = self._client.execute()
        self._graph_oram.process_read_result(result); self._graph_meta.process_read_result(result)
        if self._pos_omap._enable_meta: self._pos_omap._meta.process_read_result(result); self._pos_omap.meta_de_duplication()
        self.graph_meta_de_duplication()
        
        retrieved = {d.key: i for i, d in enumerate(self._graph_oram.stash)}; all_t = set(nk_keys) | {c_key}
        temp_meta = []
        for dup in self._graph_meta.stash:
            if dup.key in all_t and dup.key in retrieved:
                idx = retrieved[dup.key]; target = self._graph_oram.stash[idx]; v_v = target.value
                if isinstance(dup.value, tuple) and len(dup.value) == 2:
                    adj = v_v[1]; s_k, n_gl = dup.value
                    if n_gl < 0:
                        if s_k in adj: del adj[s_k]
                    else: 
                        if s_k in adj: adj[s_k] = n_gl
                else: target.value = (v_v[0], v_v[1], dup.value)
            else: temp_meta.append(dup)
        self._graph_meta.stash = temp_meta
        
        for v_key, (n_pl, _) in all_visited.items():
            if v_key in retrieved:
                idx = retrieved[v_key]; v_val = self._graph_oram.stash[idx].value
                self._graph_oram.stash[idx].value = (v_val[0], v_val[1], n_pl)

        graph_dups = []; pos_dups = []; res = {}; batch_p = {c_key: c_new_gl}
        for nk in nk_keys:
            if nk in retrieved:
                # OPTIMIZATION: Do not move neighbors to avoid needing to update PosMap
                # This avoids the need for batch_search to find neighbor's PosMap location.
                # Valid for 'Vertex-based' checking where we trust the edge pointer.
                target_data = self._graph_oram.stash[retrieved[nk]]
                batch_p[nk] = target_data.leaf

        if c_key in retrieved:
            target = self._graph_oram.stash[retrieved[c_key]]; v_v = target.value; adj = v_v[1]
            for k, p in batch_p.items():
                if k != c_key and k in adj: adj[k] = p
            target.leaf = c_new_gl; target.value = (v_v[0], adj, v_v[2] if len(v_v) > 2 else None)

        for nk in nk_keys:
            if nk not in retrieved: continue
            n_data = self._graph_oram.stash[retrieved[nk]]; n_ngl = batch_p[nk]
            v_d, n_adj, n_pl = n_data.value if len(n_data.value) == 3 else (n_data.value[0], n_data.value[1], None)
            if c_key in n_adj: n_adj[c_key] = c_new_gl
            n_data.leaf = n_ngl
            for ok, ogl in n_adj.items():
                if ok == c_key: continue
                if isinstance(ogl, tuple): ogl = ogl[0]
                if ogl is None or ogl < 0: continue
                # Update pointers IF the neighbor moved (which they don't now)
                t_gl = batch_p.get(ok, ogl)
                if ok in batch_p: n_adj[ok] = t_gl
                # Only queue duplication if target actually moved?
                # Since we don't move ok (neighbor's neighbor), t_gl == ogl.
                # So we don't need to queue dup.
                if t_gl != ogl:
                    graph_dups.append(Data(key=ok, leaf=t_gl, value=(nk, n_ngl)))
                
            # Queue duplicate for reverse edge to center (Center DID move)
            graph_dups.append(Data(key=c_key, leaf=c_new_gl, value=(nk, n_ngl)))
            
            # PosMap update: Only if neighbor moved.
            # Since n_ngl == n_data.leaf (old leaf), we DON'T update PosMap.
            # if n_pl is not None: pos_dups.append(Data(key=nk, leaf=n_pl, value=n_ngl))
            
            n_data.value = (v_d, n_adj, n_pl); res[nk] = (v_d, n_adj)
        
        d_leaf = secrets.randbelow(self._leaf_range)
        while len(graph_dups) < rl_count: graph_dups.append(Data(key=None, leaf=d_leaf, value=(None, 0)))
        while len(pos_dups) < self._max_deg + 1: pos_dups.append(Data(key=None, leaf=d_leaf, value=0))
        self._graph_meta.stash = graph_dups + self._graph_meta.stash
        self._pos_omap.add_meta_duplications(pos_dups)
        self.graph_meta_de_duplication(); self.pos_meta_de_duplication(); self._pos_omap.meta_de_duplication()
        
        self._graph_oram.queue_write(leaves=all_gls_with_center); self._graph_meta.queue_write(leaves=rl_paths)
        if self._pos_omap._enable_meta: self._pos_omap._meta.queue_write(leaves=all_gls_with_center)
        self._client.execute(); return res

    def t_hop(self, key: Any, num_hop: int) -> dict:
        """Perform the t-hop query."""
        if num_hop <= 0: return self.lookup([key])
        result = self.lookup([key])
        if key not in result: return {}
        visited = {key}; frontier = [key]
        for hop in range(num_hop):
            next_f = []
            for c_key in frontier:
                neighbors = self.neighbor([c_key])
                for nk, nv in neighbors.items():
                    if nk not in visited:
                        visited.add(nk); next_f.append(nk); result[nk] = nv
            frontier = next_f
            if not frontier: break
        return result

    def t_traversal(self, key: Any, num_hop: int) -> dict:
        """Perform t-hop traversal."""
        if num_hop <= 0: return self.lookup([key])
        result = self.lookup([key])
        if key not in result: return {}
        current_key = key
        for hop in range(num_hop):
            neighbors = self.neighbor([current_key])
            if not neighbors: break
            next_key = random.choice(list(neighbors.keys()))
            result[next_key] = neighbors[next_key]
            current_key = next_key
        return result
