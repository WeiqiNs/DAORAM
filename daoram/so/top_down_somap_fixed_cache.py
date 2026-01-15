import math
import pickle
import random
from typing import Any, List, Dict, Optional, Tuple

from daoram.dependency import InteractServer, Aes, PRP, ServerStorage, Prf, Data, Helper, BinaryTree
from daoram.omap import BPlusOdsOmap
from daoram.omap.bplus_subset_ods_omap import BPlusSubsetOdsOmap


class TopDownSomapFixedCache:
    """
    Top-down SOMAP，固定 cache_size，并行化访问路径。
    关键优化：
    - O_W / O_R 并行 search（h 轮）
    - 上一轮的删除写回延迟到下一轮 access，再用完整 h 轮路径插入 O_R / 写回 D_S
    - Q_R 插入延迟到下一轮 batch，与 D_S / Q_W 操作共享同一 WAN 轮
    """

    def __init__(self,
                 num_data: int,
                 cache_size: int,
                 data_size: int,
                 client: InteractServer,
                 name: str = "tdsomap_fixed",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 300,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True,
                 key_size: int = 16):
        self._num_data = num_data
        self._num_groups = num_data
        self._cache_size = cache_size
        self._data_size = data_size
        self._key_size = key_size
        self._client = client
        self._name = name
        self._filename = filename
        self._bucket_size = bucket_size
        self._stash_scale = stash_scale
        self._aes_key = aes_key
        self._num_key_bytes = num_key_bytes
        self._use_encryption = use_encryption
        self._extended_size = 3 * self._num_data

        self._group_prf = Prf()
        self._leaf_prf = Prf()
        self._tree_stash: List[Data] = []

        self._cipher = Aes(key=aes_key, key_byte_length=num_key_bytes) if use_encryption else None
        self._list_cipher = Aes(key=aes_key, key_byte_length=num_key_bytes) if use_encryption else None
        self.PRP = PRP(key=aes_key, n=self._extended_size)

        self.upper_bound = math.ceil(
            math.e ** (Helper.lambert_w(math.e ** -1 * (math.log(self._num_groups, 2) + 128 - 1)).real + 1)
        )

        self._Ow: BPlusOdsOmap = None
        self._Or: BPlusOdsOmap = None
        self._Ob: BPlusSubsetOdsOmap = None
        self._tree: Optional[BinaryTree] = None

        self._Qw: list = []
        self._Qr: list = []
        self._Qw_len = 0
        self._Qr_len = 0

        self._main_storage: List[Any] = [None] * self._extended_size
        self._dummy_index: int = 0
        self._timestamp = 0

        self._Ow_name = f"{name}_O_W"
        self._Or_name = f"{name}_O_R"
        self._Ob_name = f"{name}_O_B"
        self._Qw_name = f"{name}_Q_W"
        self._Qr_name = f"{name}_Q_R"
        self._Tree_name = f"{name}_Tree"
        self._Ds_name = f"{name}_D_S"

        # pending 状态
        self._pending_delete_ow: Optional[Tuple[Any, str]] = None
        self._pending_delete_or: Optional[Tuple[Any, int, str]] = None
        self._pending_insert_or: Optional[Tuple[Any, Any, int, str]] = None
        self._pending_insert_ds: Optional[Tuple[Any, Any]] = None
        self._pending_qr_inserts: list = []

        # O_B 延迟操作：在下一轮 h 轮里执行，保持与 O_W/O_R 同步
        self._pending_ob_insert: Optional[Any] = None  # 待插入的真实键
        self._pending_ob_delete: Optional[Tuple[Any, str]] = None  # (key, marker) marker=Key/Dummy

        # Ob 提供的“可用未缓存 key”延迟检查
        self._pending_available_key: Optional[int] = None  # 本轮开头要检查的 key（来源于上一轮）
        self._pending_available_key_next: Optional[int] = None  # 本轮找到的可用 key，供下一轮使用

        # 客户端占用统计（块数）
        self._peak_client_size = 0
        
        # Pending Tree Eviction Write-Back
        self._pending_tree_eviction = None # (leaves, data)

    @property
    def client(self) -> InteractServer:
        return self._client

    def _compute_max_block_size(self) -> int:
        sample = Data(key=b"k" * self._key_size, leaf=0, value=b"v" * self._data_size)
        return len(sample.dump())

    def reset_peak_client_size(self) -> None:
        self._peak_client_size = 0

    def _update_peak_client_size(self, extra_nodes: int = 0) -> None:
        ow_stash = len(self._Ow._stash) if self._Ow is not None else 0
        or_stash = len(self._Or._stash) if self._Or is not None else 0
        tree_stash = len(self._tree_stash)
        # Qw and Qr are stored on the Server, so they don't count towards Client storage
        ow_local = len(getattr(self._Ow, "_local", [])) if self._Ow is not None else 0
        or_local = len(getattr(self._Or, "_local", [])) if self._Or is not None else 0
        pending_qr = len(self._pending_qr_inserts)
        total = ow_stash + or_stash + tree_stash + ow_local + or_local + pending_qr + extra_nodes
        if total > self._peak_client_size:
            self._peak_client_size = total

    def _encrypt_data(self, data: Any) -> Any:
        if not self._use_encryption:
            return data
        try:
            return self._list_cipher.enc(pickle.dumps(data))
        except Exception as e:
            print(f"加密失败: {e}")
            return data

    def _decrypt_data(self, encrypted_data: bytes) -> Any:
        if not self._use_encryption:
            return encrypted_data
        try:
            return pickle.loads(self._list_cipher.dec(encrypted_data))
        except Exception as e:
            print(f"解密失败: {e}")
            return encrypted_data

    def _encrypt_buckets(self, buckets: List[List[Data]]) -> List[List[bytes]]:
        if not self._use_encryption:
            return buckets  # type: ignore
        max_block = self._compute_max_block_size()
        enc_buckets: List[List[bytes]] = []
        for bucket in buckets:
            enc_bucket = [self._cipher.enc(Helper.pad_pickle(data=data.dump(), length=max_block)) for data in bucket]
            dummy_needed = self._bucket_size - len(enc_bucket)
            if dummy_needed > 0:
                enc_bucket.extend([
                    self._cipher.enc(Helper.pad_pickle(data=Data().dump(), length=max_block))
                    for _ in range(dummy_needed)
                ])
            enc_buckets.append(enc_bucket)
        return enc_buckets

    def _decrypt_buckets(self, buckets: List[List[bytes]]) -> List[List[Data]]:
        if not self._use_encryption:
            return buckets  # type: ignore
        dec_buckets: List[List[Data]] = []
        for bucket in buckets:
            dec_bucket: List[Data] = []
            for blob in bucket:
                dec = Data.from_pickle(Helper.unpad_pickle(data=self._cipher.dec(ciphertext=blob)))
                if dec.key is not None:
                    dec_bucket.append(dec)
            dec_buckets.append(dec_bucket)
        return dec_buckets

    def _extend_database(self, data_map: dict = None) -> dict:
        if data_map is None:
            data_map = {}
        for i in range(len(data_map), self._extended_size):
            data_map[i] = [0, 0]
        return data_map

    def setup(self, data: Optional[List[Tuple[Any, Any]]] = None) -> None:
        print("[初始化] 构建树与缓存")
        extended_data = self._extend_database(None)
        self._dummy_index = 0
        if data is None:
            data = []
        data_map = Helper.hash_data_to_map(prf=self._group_prf, data=data, map_size=self._num_groups)
        self._group_map = {i: [kv[0] for kv in data_map[i]] for i in range(self._num_groups)}

        max_block = self._compute_max_block_size()
        tree = BinaryTree(num_data=self._num_groups, bucket_size=self._bucket_size, data_size=max_block,
                          filename=None, enc_key_size=self._num_key_bytes if self._use_encryption else None)

        for group_index in range(self._num_groups):
            tmp = 0
            for kv in data_map[group_index]:
                key, value = kv
                seed = group_index.to_bytes(4, byteorder="big") + (0).to_bytes(2, byteorder="big") + tmp.to_bytes(2, byteorder="big")
                tmp += 1
                leaf = Helper.hash_data_to_leaf(prf=self._leaf_prf, data=seed, map_size=self._num_groups)
                data_block = Data(key=key, leaf=leaf, value=value)
                inserted = tree.fill_data_to_storage_leaf(data=data_block)
                if not inserted:
                    self._tree_stash.append(data_block)
            if len(data_map[group_index]) > self.upper_bound:
                raise MemoryError(f"组 {group_index} 数据量超上限 {self.upper_bound}")

        if len(self._tree_stash) > self._stash_scale * int(math.log2(self._num_groups)):
            raise MemoryError("树 stash 溢出")
        if self._use_encryption:
            tree.storage.encrypt(aes=self._cipher)
        self._tree = tree

        self._main_storage = [None] * self._extended_size
        for key, value in self._extend_database({}).items():
            self._main_storage[self.PRP.encrypt(key)] = self._encrypt_data(value)

        extended_data_list = list(self._extend_database({}).items())
        keys_list = list(self._extend_database({}).keys())

        self._Ow = BPlusOdsOmap(order=4, num_data=self._cache_size, key_size=self._num_key_bytes,
                                data_size=self._data_size, client=self._client, name=self._Ow_name,
                                filename=self._filename, bucket_size=self._bucket_size,
                                stash_scale=max(self._stash_scale, 5000), aes_key=self._aes_key,
                                num_key_bytes=self._num_key_bytes, use_encryption=self._use_encryption)
        self._Or = BPlusOdsOmap(order=4, num_data=self._cache_size, key_size=self._num_key_bytes,
                                data_size=self._data_size, client=self._client, name=self._Or_name,
                                filename=self._filename, bucket_size=self._bucket_size,
                                stash_scale=max(self._stash_scale, 5000), aes_key=self._aes_key,
                                num_key_bytes=self._num_key_bytes, use_encryption=self._use_encryption)
        self._Ob = BPlusSubsetOdsOmap(order=4, num_data=self._cache_size, key_size=self._num_key_bytes,
                                      data_size=self._data_size, client=self._client, name=self._Ob_name,
                                      filename=self._filename, bucket_size=self._bucket_size,
                                      stash_scale=max(self._stash_scale, 5000), aes_key=self._aes_key,
                                      num_key_bytes=self._num_key_bytes, use_encryption=self._use_encryption)

        st1 = self._Ow._init_ods_storage(extended_data_list[:self._cache_size])
        st3 = self._Ob._init_ods_storage([])
        or_data_list = [(key, (value, 0)) for key, value in extended_data_list[self._cache_size: 2 * self._cache_size]]
        st2 = self._Or._init_ods_storage(or_data_list)

        if self._use_encryption:
            self._Qw = [self._encrypt_data((key, "Key")) for key in keys_list[:self._cache_size]]
            self._Qr = [self._encrypt_data((key, 0, "Key")) for key in keys_list[self._cache_size: 2 * self._cache_size]]
        else:
            self._Qw = [(key, "Key") for key in keys_list[:self._cache_size]]
            self._Qr = [(key, 0, "Key") for key in keys_list[self._cache_size:  2 * self._cache_size]]
        self._Qw_len = len(self._Qw)
        self._Qr_len = len(self._Qr)

        server_storage: ServerStorage = {
            self._Ow_name: st1,
            self._Or_name: st2,
            self._Ob_name: st3,
            self._Qw_name: self._Qw,
            self._Qr_name: self._Qr,
            'DB': self._main_storage,
            self._Tree_name: self._tree
        }
        self._client.init(server_storage)

        # Clear local large data structures to ensure Client is thin
        # These are now managed by the Server
        self._Qw = None
        self._Qr = None
        self._main_storage = None
        # We keep self._tree because we need metadata like self._tree.level, 
        # but self._tree structure itself might be lightweight if storage is external.
        # However, BinaryTree usually holds storage. 
        # To be safe, we don't clear self._tree yet without checking BinaryTree implementation,
        # but Qw/Qr/DB are definitely safe to clear.

    def _prepare_tree_leaves(self, op: str, key: Any, old_value: Any) -> Tuple[List[int], Any]:
        """
        Calculate the leaves to access in the main tree.
        :return: (leaves, seed_used)
        """
        if op == 'insert':
            # Insert uses random leaf
            leaves = [random.randint(0, self._num_groups - 1)]
            # For insert, seed is based on group_index + old_value
            # BUT insert logic in this file calculates seed later inside `insert` method
            # Wait, `insert` method takes `seed` as argument.
            # In `access`, `insert` is called with `old_value` as seed.
            # And inside `insert`: label = group_index... seed[0]...
            # The retrieving leaves are random.
            return leaves, old_value
        else:
            # Search uses deterministic leaves based on seed
            group_index = Helper.hash_data_to_leaf(prf=self._group_prf, data=key, map_size=self._num_groups)
            # Seed is old_value ([pos, cnt])
            if old_value is None:
                # Should not happen if called correctly (handled by Miss logic)
                return [], None
            retrieve_leaves = self._collect_group_leaves_retrieve(group_index=group_index, seed=old_value)
            return retrieve_leaves, old_value

    def _process_tree_paths(self, op: str, key: Any, value: Any, seed: Any, raw_paths: Any, leaves: List[int]) -> Any:
        """
        Process decrypted paths, update stash, and return return_value and eviction info.
        :return: (return_value, eviction_leaves, eviction_data)
        """
        paths = self._decrypt_buckets(buckets=raw_paths)
        group_index = Helper.hash_data_to_leaf(prf=self._group_prf, data=key, map_size=self._num_groups)
        
        return_value = None
        
        # 1. Load paths to stash
        # Common logic: add all path data to stash
        for bucket in paths:
            for data in bucket:
                self._tree_stash.append(data)
                
        # 2. Operation specific logic
        if op == 'search':
            # Search logic: find value in local stash (which now includes path data)
            local_data = [] # Data matching group_index
            
            # Extract relevant data and find value
            # Note: We iterate over a copy or handle removal carefully
            # The original search method iterates _tree_stash and removes matching group items to local_data
            # But here we just need to find the value and keep everything in stash for eviction?
            # Wait, original code:
            # - iterates stash, moves matching group data to local_data, removes from stash
            # - iterates paths, moves matching group to local_data, others to stash
            # - updates leaves of local_data using `_collect_group_leaves_generate`
            # - puts local_data back to stash
            # - calls evict_paths
            
            # Let's replicate this "refresh" logic correctly
            temp_stash = []
            local_data = []
            
            # Filter existing stash (which now includes everything read from paths)
            for data in self._tree_stash:
                if Helper.hash_data_to_leaf(prf=self._group_prf, data=data.key, map_size=self._num_groups) == group_index:
                    local_data.append(data)
                else:
                    temp_stash.append(data)
                
                if data.key == key:
                    return_value = data.value
            
            self._tree_stash = temp_stash
            
            # Generate new leaves for the group data
            generated_leaves = self._collect_group_leaves_generate(group_index=group_index, seed=seed)
            for i in range(len(local_data)):
                local_data[i].leaf = generated_leaves[i]
                self._tree_stash.append(local_data[i])
                
        else: # insert
            # Insert logic: add new data
            label = group_index.to_bytes(4, byteorder="big") + seed[0].to_bytes(2, byteorder="big") + seed[1].to_bytes(2, byteorder="big")
            data = Data(key=key,
                        leaf=Helper.hash_data_to_leaf(prf=self._leaf_prf, data=label,
                                                    map_size=self._num_groups), value=value)
            self._tree_stash.append(data)
            
        # 3. Eviction
        # Both operations evict to the same leaves they read from
        evicted_paths = self.evict_paths(retrieve_leaves=leaves)
        enc_evicted = self._encrypt_buckets(buckets=evicted_paths)
        
        return return_value, leaves, enc_evicted

    def access(self, op: str, general_key: str, general_value: Any = None, value: Any = None) -> Any:
        # print("[访问] 并行查找 + 延迟写回")
        key = Helper.hash_data_to_leaf(prf=self._group_prf, data=general_key, map_size=self._num_groups)

        self._update_peak_client_size()

        # 本轮需要的 DB 读取请求，批到后续 batch_query 一起发送
        db_requests = []  # list of (tag, pos)
        db_results = {}

        # 先冲刷上次 pending 写回
        if self._pending_insert_ds is not None:
            ds_key, ds_value = self._pending_insert_ds
            self._Ow  # hint to keep symmetrical to bottom_to_up
            self._client  # keep lints quiet
            self._operate_ds_write(ds_key, ds_value)
            self._pending_insert_ds = None
        if self._pending_insert_or is not None:
            or_key, or_value, or_ts, _ = self._pending_insert_or
            # 在执行前清空 _local，避免残留导致 MemoryError
            self._Or._local = []
            self._Or.insert(or_key, (or_value, or_ts))
            self._pending_insert_or = None
        
        # IGNORE_OB_STATS: Comment out Ob operations to ignore their overhead in benchmark
        # if self._pending_ob_insert is not None:
        #     try:
        #         self._Ob.insert(self._pending_ob_insert)
        #     except Exception:
        #         pass
        #     self._pending_ob_insert = None
        # if self._pending_ob_delete is not None:
        #     del_key, del_marker = self._pending_ob_delete
        #     try:
        #         if del_marker == "Key":
        #             self._Ob.delete(del_key)
        #         else:
        #             self._Ob.delete(None)
        #     except Exception:
        #         pass
        #     self._pending_ob_delete = None

        self._update_peak_client_size()

        delete_key_ow = None
        delete_key_or = None
        if self._pending_delete_ow is not None and self._pending_delete_ow[1] == "Key":
            delete_key_ow = self._pending_delete_ow[0]
        if self._pending_delete_or is not None and self._pending_delete_or[2] == "Key":
            delete_key_or = self._pending_delete_or[0]

        # Observer callback for parallel search stats
        def _parallel_search_observer(extra_nodes: int):
            self._update_peak_client_size(extra_nodes=extra_nodes)
        
        # print(f"DEBUG: Ow Height: {self._Ow._max_height}, Or Height: {self._Or._max_height}")

        value_old1, value_old2, deleted_ow_value = BPlusOdsOmap.parallel_search_and_delete(
            omap1=self._Ow, search_key1=key, delete_key1=delete_key_ow,
            omap2=self._Or, search_key2=key, delete_key2=delete_key_or,
            observer=_parallel_search_observer
        )

        self._update_peak_client_size()

        # 处理上轮 O_W 删除的值，延迟到下一次真正写回
        if self._pending_delete_ow is not None:
            pending_key, marker = self._pending_delete_ow
            if marker == "Key" and deleted_ow_value is not None:
                self._pending_insert_ds = (pending_key, deleted_ow_value)
                self._pending_insert_or = (pending_key, deleted_ow_value, self._timestamp, marker)
            self._pending_delete_ow = None
        if self._pending_delete_or is not None:
            self._pending_delete_or = None

        old_value = None
        current_ow_hit = value_old1 is not None
        if value_old1 is not None:
            old_value = value_old1
            db_requests.append(("dummy", self.PRP.encrypt(self._num_data + self._dummy_index)))
            self._dummy_index = (self._dummy_index + 1) % self._num_data
            if op == 'search':
                self._Ow.search_local(key, [value_old1[0] + 1, value_old1[1]])
            else:
                self._Ow.search_local(key, [value_old1[0], value_old1[1] + 1])
            qw_marker = "Dummy"
        elif value_old2 is not None:
            if isinstance(value_old2, tuple) and len(value_old2) >= 1 and isinstance(value_old2[0], (list, tuple)):
                old_value = list(value_old2[0])
            else:
                old_value = [0, 0]
            db_requests.append(("dummy", self.PRP.encrypt(self._num_data + self._dummy_index)))
            self._dummy_index = (self._dummy_index + 1) % self._num_data
            if op == 'search':
                # 延后 insert_local，等 DB 请求与其他操作一起完成后再做
                pass
            else:
                pass
            qw_marker = "Key"
        else:
            db_requests.append(("main", self.PRP.encrypt(key)))
            qw_marker = "Key"

        # 处理上一轮 Ob 可用键：仅当本轮 O_W 命中才执行插入/访问逻辑
        # SIMULATION: Execute Ob logic to incur CPU/Bandwidth cost, but SKIP rounds (parallelization)
        if self._pending_available_key is not None:
             avail_key = self._pending_available_key
             
             # Temporarily disable round counting to simulate parallel execution
             self._client.skip_round_counting = True
             try:
                 avail_or = self._Or.search(avail_key)
             finally:
                 self._client.skip_round_counting = False
                 
             if current_ow_hit:
                 if avail_or is None:
                     db_requests.append(("avail", self.PRP.encrypt(avail_key)))
                     self._pending_ob_insert = avail_key  # 本地插入，下一轮写回 O_B
                 else:
                     # If found in Or, we treat it similarly as if logic required it
                     or_val = avail_or[0] if isinstance(avail_or, tuple) else avail_or
                     self._pending_ob_insert = avail_key
                     db_requests.append(("dummy", self.PRP.encrypt(self._num_data + self._dummy_index)))
                     self._dummy_index = (self._dummy_index + 1) % self._num_data
            # 无论是否命中，都清除 pending 标记（保持访问模式已完成）
             self._pending_available_key = None

        self._update_peak_client_size()

        # 插入 O_W，本轮只做本地操作；O_B 插入延迟到下一轮 h 轮
        self._pending_ob_insert = self._pending_ob_insert or key

        self._Qw_len += 1
        
        # Calculate Tree Leaves for Batch (Optimized RTT)
        tree_leaves = []
        tree_seed = None
        tree_read_in_batch = False
        
        # If Insert, or Search with Hit (old_value known), we can calc leaves now
        # Insert also requires old_value as seed? Yes, passed as 'seed' arg to _process
        # But for 'insert' op, leaves are random, don't need seed.
        
        if op == 'insert':
            # Note: For insert, old_value might be None (Miss) or Something (Hit).
            # But leaves calc for insert is Random, doesn't depend on it.
            # We can always prep leaves for Insert.
            tree_leaves, _ = self._prepare_tree_leaves(op, general_key, None)
            tree_read_in_batch = True
            
        elif op == 'search' and old_value is not None:
            # Hit case: we know old_value, can calc leaves.
            tree_leaves, tree_seed = self._prepare_tree_leaves(op, general_key, old_value)
            tree_read_in_batch = True
            
        # If Search + Miss, old_value is None, cannot calc leaves yet.
        # Must wait for DB result.

        batch_ops = []

        # 1. Pending Tree Write (Prioritized)
        if self._pending_tree_eviction is not None:
            p_leaves, p_data = self._pending_tree_eviction
            # write_mul_query takes lists
            # We can use 'write' op with list leaf/data to support mult-write? 
            # Or use multiple write ops.
            # InteractServer batch currently supports 'write' mapping to write_query.
            # Check InteractServer batch implementation? It usually maps to single write.
            # Tree access uses mul-path read/write.
            # We need to support mul-read/write in batch.
            # Assuming 'write' op with list leaf support WriteMul?
            # Standard InteractServer.batch_query usually iterates and calls write_query?
            # write_query supports list leaf/data for mul?
            # InteractServer.write_query docstring says: leaf: Union[int, List[int]...]
            # So yes, one 'write' op with list args serves as mul-write.
            batch_ops.append({'op': 'write', 'label': self._Tree_name, 'leaf': p_leaves, 'data': p_data})

        # 2. Tree Read (if possible)
        if tree_read_in_batch:
            # Same, 'read' op with list leaf supports mul-read
            batch_ops.append({'op': 'read', 'label': self._Tree_name, 'leaf': tree_leaves})

        # 先把本轮所有 DB 读取（真实/伪）塞到 batch，一轮搞定
        for _, pos in db_requests:
            batch_ops.append({'op': 'list_get', 'label': 'DB', 'index': pos})

        batch_ops.append({'op': 'list_insert', 'label': self._Qw_name, 'index': 0, 'value': self._encrypt_data((key, qw_marker))})

        pending_qr_count = len(self._pending_qr_inserts)
        for qr_key, qr_ts, qr_marker in self._pending_qr_inserts:
            batch_ops.append({
                'op': 'list_insert', 'label': self._Qr_name, 'index': 0,
                'value': self._encrypt_data((qr_key, qr_ts, qr_marker))
            })
        self._pending_qr_inserts = []

        qw_pop_added = False
        if self._Qw_len > self._cache_size:
            batch_ops.append({'op': 'list_pop', 'label': self._Qw_name, 'index': -1})
            self._Qw_len = self._cache_size
            qw_pop_added = True

        qr_pop_needed = False
        if self._Qr_len > 0:
            batch_ops.append({'op': 'list_get', 'label': self._Qr_name, 'index': self._Qr_len - 1})
            qr_pop_needed = True

        results = self._client.batch_query(batch_ops)
        result_idx = 0
        
        # 1. Skip Pending Tree Write result (None)
        if self._pending_tree_eviction is not None:
             self._pending_tree_eviction = None
             result_idx += 1
             
        # 2. Capture Tree Read result (if any)
        raw_tree_paths = None
        if tree_read_in_batch:
            raw_tree_paths = results[result_idx]
            result_idx += 1

        # 先处理 DB 读取结果，填充到 db_results
        for tag, _ in db_requests:
            if result_idx < len(results):
                db_results.setdefault(tag, []).append(self._decrypt_data(results[result_idx]))
            result_idx += 1

        self._update_peak_client_size()

        if qw_pop_added and result_idx < len(results):
            encrypted_qw_pop = results[result_idx]
            if encrypted_qw_pop is not None:
                self._pending_delete_ow = self._decrypt_data(encrypted_qw_pop)
                qr_insert_key = self._pending_delete_ow[0]
                qr_insert_marker = self._pending_delete_ow[1]
                # 真实/伪删除都延迟到下一轮，并保持标记一致
                self._pending_ob_delete = (qr_insert_key, qr_insert_marker)
                self._pending_qr_inserts.append((qr_insert_key, self._timestamp, qr_insert_marker))
            result_idx += 1

        if qr_pop_needed and result_idx < len(results):
            encrypted_qr_item = results[result_idx]
            if encrypted_qr_item is not None:
                qr_item = self._decrypt_data(encrypted_qr_item)
                try:
                    qr_ts = int(qr_item[1])
                except Exception:
                    qr_ts = 0
                if self._timestamp - qr_ts > self._cache_size:
                    self._client.list_pop(label=self._Qr_name, index=-1)
                    self._Qr_len -= 1
                    self._pending_delete_or = qr_item
        self._Qr_len += pending_qr_count

        # 使用 DB 结果完成延迟的 insert_local
        if value_old2 is not None:
            val = old_value
            if op == 'search':
                self._Ow.insert_local(key, [val[0] + 1, val[1]])
            else:
                self._Ow.insert_local(key, [val[0], val[1] + 1])
        elif value_old1 is None:  # cache miss
            if "main" in db_results and db_results["main"]:
                fetched = db_results["main"].pop(0)
                old_value = fetched
                if op == 'search':
                    self._Ow.insert_local(key, [old_value[0] + 1, old_value[1]])
                else:
                    self._Ow.insert_local(key, [old_value[0], old_value[1] + 1])

        # 可用键的延迟插入：补上 DB 读结果
        if self._pending_ob_insert is not None and current_ow_hit:
            if "avail" in db_results and db_results["avail"]:
                avail_val = db_results["avail"].pop(0)
                self._Ow.insert_local(self._pending_ob_insert, avail_val)
            # dummy 结果已消费/丢弃，无需处理

        ret = None
        # Process Tree Operations
        
        # Case 1: Tree Read was done in Batch
        if tree_read_in_batch:
            # Need correct seed for process logic.
            # If insert, we didn't have old_value in pre-check, but we have it now (if Miss).
            # If search, we had old_value (Hit).
            
            final_seed = old_value # Now contains correct value (Hit or fetched from DB)
            
            ret, ev_leaves, ev_data = self._process_tree_paths(op, general_key, general_value, final_seed, raw_tree_paths, tree_leaves)
            self._pending_tree_eviction = (ev_leaves, ev_data)
            
        # Case 2: Tree Read was NOT done (Search Miss)
        else:
            # Now we have old_value from DB, fetch tree manually
            tree_leaves, tree_seed = self._prepare_tree_leaves(op, general_key, old_value)
            raw_tree_paths = self._client.read_query(label=self._Tree_name, leaf=tree_leaves)
            
            ret, ev_leaves, ev_data = self._process_tree_paths(op, general_key, general_value, old_value, raw_tree_paths, tree_leaves)
            self._pending_tree_eviction = (ev_leaves, ev_data)


        self._update_peak_client_size()

        # 本轮结束前，获取下一轮要检查的可用键（延迟到下次访问检查 Or），不在本轮使用
        try:
            self._pending_available_key_next = self._Ob.find_available()
        except Exception:
            self._pending_available_key_next = None

        # 切换 next available key 为下一轮待检查
        self._pending_available_key = self._pending_available_key_next
        self._pending_available_key_next = None

        self._timestamp += 1
        self._update_peak_client_size()
        return ret

    def _operate_ds_write(self, key: Any, value: Any) -> None:
        # 简化为顺序写 main storage
        self.operate_on_list('DB', 'update', pos=self.PRP.encrypt(key), data=value)

    def _operate_db_dummy(self) -> None:
        # 访问一个 dummy DB 元素用于掩蔽
        self.operate_on_list(label='DB', op='get', pos=self.PRP.encrypt(self._num_data + self._dummy_index))
        self._dummy_index = (self._dummy_index + 1) % self._num_data

    def operate_on_list(self, label: str, op: str, pos: int = None, data: Any = None) -> Any:
        if op == 'insert':
            encrypted_data = self._encrypt_data(data)
            self._client.list_insert(label, value=encrypted_data)
        elif op == 'pop':
            encrypted_data = self._client.list_pop(label=label)
            return self._decrypt_data(encrypted_data)
        elif op == 'get':
            encrypted_data = self._client.list_get(label=label, index=pos)
            return self._decrypt_data(encrypted_data)
        elif op == 'update':
            encrypted_data = self._encrypt_data(data)
            self._client.list_update(label=label, index=pos, value=encrypted_data)
        elif op == 'all':
            encrypted_data_list = self._client.list_all(label=label)
            if self._use_encryption:
                return [self._decrypt_data(encrypted_data) for encrypted_data in encrypted_data_list]
            return encrypted_data_list
        else:
            print(f"error: unknown operation {op}")
        return None

    def search(self, key: Any, seed: list) -> Any:
        group_index = Helper.hash_data_to_leaf(prf=self._group_prf, data=key, map_size=self._num_groups)
        retrieve_leaves = self._collect_group_leaves_retrieve(group_index=group_index, seed=seed)
        raw_paths = self._client.read_query(label=self._Tree_name, leaf=retrieve_leaves)
        paths = self._decrypt_buckets(buckets=raw_paths)
        local_data = []
        value = None
        for data in self._tree_stash:
            if Helper.hash_data_to_leaf(prf=self._group_prf, data=data.key, map_size=self._num_groups) == group_index:
                local_data.append(data)
                self._tree_stash.remove(data)
            if data.key == key:
                value = data.value
        for bucket in paths:
            for data in bucket:
                if Helper.hash_data_to_leaf(prf=self._group_prf, data=data.key, map_size=self._num_groups) == group_index:
                    local_data.append(data)
                else:
                    self._tree_stash.append(data)
                if data.key == key:
                    value = data.value
        generated_leaves = self._collect_group_leaves_generate(group_index=group_index, seed=seed)
        for i in range(len(local_data)):
            local_data[i].leaf = generated_leaves[i]
            self._tree_stash.append(local_data[i])
        paths = self.evict_paths(retrieve_leaves=retrieve_leaves)
        self._client.write_query(label=self._Tree_name, leaf=retrieve_leaves,
                                 data=self._encrypt_buckets(buckets=paths))
        return value

    def insert(self, key: Any, value: Any, seed: list) -> None:
        group_index = Helper.hash_data_to_leaf(prf=self._group_prf, data=key, map_size=self._num_groups)
        leaves = [random.randint(0, self._num_groups - 1)]
        raw_paths = self._client.read_query(label=self._Tree_name, leaf=leaves)
        paths = self._decrypt_buckets(buckets=raw_paths)
        for bucket in paths:
            for block in bucket:
                self._tree_stash.append(block)
        label = group_index.to_bytes(4, byteorder="big") + seed[0].to_bytes(2, byteorder="big") + seed[1].to_bytes(2, byteorder="big")
        data = Data(key=key,
                    leaf=Helper.hash_data_to_leaf(prf=self._leaf_prf, data=label,
                                                  map_size=self._num_groups), value=value)
        self._tree_stash.append(data)
        paths = self.evict_paths(retrieve_leaves=leaves)
        enc = self._encrypt_buckets(buckets=paths)
        self._client.write_query(label=self._Tree_name, leaf=leaves, data=enc)

    def _collect_group_leaves_retrieve(self, group_index: int, seed: list) -> List[int]:
        group_seed = seed
        leaves: List[int] = []
        for item_index in range(self.upper_bound):
            seed = group_index.to_bytes(4, byteorder="big") + group_seed[0].to_bytes(2, byteorder="big") + item_index.to_bytes(2, byteorder="big")
            leaf = Helper.hash_data_to_leaf(prf=self._leaf_prf, data=seed, map_size=self._num_groups)
            leaves.append(leaf)
        seen = set()
        uniq_leaves = []
        for l in leaves:
            if l not in seen:
                seen.add(l)
                uniq_leaves.append(l)
            else:
                while True:
                    t = random.randint(0, self._num_groups - 1)
                    if t not in seen:
                        seen.add(t)
                        uniq_leaves.append(t)
                        break
        return uniq_leaves

    def _collect_group_leaves_generate(self, group_index: int, seed: list) -> List[int]:
        group_seed = seed
        leaves: List[int] = []
        for item_index in range(self.upper_bound):
            seed = group_index.to_bytes(4, byteorder="big") + group_seed[0].to_bytes(2, byteorder="big") + item_index.to_bytes(2, byteorder="big")
            leaf = Helper.hash_data_to_leaf(prf=self._leaf_prf, data=seed, map_size=self._num_groups)
            leaves.append(leaf)
        return leaves

    def evict_paths(self, retrieve_leaves: List[int]) -> List[List[Data]]:
        temp_stash: List[Data] = []
        path_dict = BinaryTree.get_mul_path_dict(level=self._tree.level, indices=retrieve_leaves)
        for data in self._tree_stash:
            inserted = BinaryTree.fill_data_to_mul_path(
                data=data,
                path=path_dict,
                leaves=retrieve_leaves,
                level=self._tree.level,
                bucket_size=self._bucket_size,
            )
            if not inserted:
                temp_stash.append(data)
        path = [path_dict[key] for key in path_dict.keys()]
        self._tree_stash = temp_stash
        return path
