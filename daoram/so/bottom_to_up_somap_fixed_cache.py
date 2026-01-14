import os
from daoram.dependency.interact_server import InteractServer, ServerStorage
from daoram.dependency.crypto import Aes
from daoram.omap.bplus_ods_omap import BPlusOdsOmap
from daoram.oram.static_oram import StaticOram
from typing import Any, Dict, Tuple, Optional
import pickle

class BottomUpSomapFixedCache:
    """
    Bottom-to-Up SOMAP with Fixed Cache Size (Optimized Version)
    
    This is an optimized version where cache_size is fixed after initialization.
    Key optimizations:
    1. In each query, batch D_S write + Q_W insert + Q_W pop + Q_R pop in one round
    2. Cache the popped elements locally
    3. In the next query's parallel_search (h rounds), perform O_W/O_R deletions
       in parallel with the search, utilizing the same h rounds
    
    This reduces WAN interaction rounds compared to the dynamic cache version.
    """
    
    def __init__(self,
                 num_data: int,
                 cache_size: int,
                 data_size: int,
                 client: InteractServer,
                 name: str = "busomap_fixed",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 300,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True):
        """
        Initialize Bottom-to-Up SOMAP with Fixed Cache
        
        :param num_data: Database size (N)
        :param cache_size: Cache size (fixed, window parameter c)
        :param data_size: Data block size
        :param client: Server interaction instance
        :param name: Protocol name
        :param filename: Storage filename
        :param bucket_size: Bucket size for ORAM
        :param stash_scale: Stash scaling factor
        :param aes_key: AES encryption key
        :param num_key_bytes: Number of bytes for keys
        :param use_encryption: Whether to use encryption
        """
        self._num_data = num_data
        self._cache_size = cache_size  # Fixed, will not change
        self._data_size = data_size
        self._client = client
        self._name = name
        self._filename = filename
        self._bucket_size = bucket_size
        self._stash_scale = stash_scale
        self._aes_key = aes_key
        self._num_key_bytes = num_key_bytes
        self._use_encryption = use_encryption
        
        # Initialize encryption for list data
        self._list_cipher = Aes(key=aes_key, key_byte_length=num_key_bytes) if use_encryption else None
        
        # OMAP caches
        self._Ow: BPlusOdsOmap = None  
        self._Or: BPlusOdsOmap = None  
        
        # Queues
        self._Qw: list = []  
        self._Qr: list = [] 
        
        # Static ORAM tree (server state D_S)
        self._Ds: StaticOram = None
        
        # Timestamp management
        self._timestamp = 0
        
        self._Qw_len = 0
        self._Qr_len = 0

        # Name identifiers
        self._Ow_name = f"{name}_O_W"
        self._Or_name = f"{name}_O_R"
        self._Qw_name = f"{name}_Q_W"
        self._Qr_name = f"{name}_Q_R"
        self._Ds_name = f"{name}_D_S"
        
        # Pending deletions from previous query (to be executed in next query's parallel_search)
        # Format: (key, marker) for O_W, (key, timestamp, marker) for O_R
        self._pending_delete_ow: Optional[Tuple[Any, str]] = None
        self._pending_delete_or: Optional[Tuple[Any, int, str]] = None

        # Pending inserts produced by O_W deletions, to be flushed in the next access
        # O_R insert needs full h-round path, so we defer it instead of using insert_local
        self._pending_insert_or: Optional[Tuple[Any, Any, int, str]] = None  # (key, value, ts, marker)
        self._pending_insert_ds: Optional[Tuple[Any, Any]] = None  # (key, value)

        # Pending Q_R inserts (enqueue in next batch round with D_S/Q_W ops)
        self._pending_qr_inserts: list = []
        
        # Pending D_S eviction write-back (delayed to next access to merge with read)
        self._pending_ds_eviction: Optional[Tuple[int, Any]] = None  # (leaf, encrypted_path)

        # 客户端占用统计（块数）
        self._peak_client_size = 0

    @property
    def client(self) -> InteractServer:
        """Return the client object."""
        return self._client

    def _encrypt_data(self, data: Any) -> Any:
        """Encrypt data if encryption is enabled"""
        if not self._use_encryption:
            return data
        
        try:
            serialized_data = pickle.dumps(data)
            encrypted_data = self._list_cipher.enc(serialized_data)
            return encrypted_data
        except Exception as e:
            print(f"Error encrypting data: {e}")
            return data
    
    def _decrypt_data(self, encrypted_data: Any) -> Any:
        """Decrypt data if encryption is enabled"""
        if not self._use_encryption:
            return encrypted_data
        
        try:
            decrypted_data = self._list_cipher.dec(encrypted_data)
            data = pickle.loads(decrypted_data)
            return data
        except Exception as e:
            print(f"Error decrypting data: {e}")
            return encrypted_data

    def reset_peak_client_size(self) -> None:
        self._peak_client_size = 0

    def _update_peak_client_size(self, extra_nodes: int = 0) -> None:
        ow_stash = len(self._Ow._stash) if self._Ow is not None else 0
        or_stash = len(self._Or._stash) if self._Or is not None else 0
        ds_stash = len(getattr(self._Ds, "_stash", [])) if self._Ds is not None else 0
        q_sizes = self._Qw_len + self._Qr_len
        ow_local = len(getattr(self._Ow, "_local", [])) if self._Ow is not None else 0
        or_local = len(getattr(self._Or, "_local", [])) if self._Or is not None else 0
        pending_qr = len(self._pending_qr_inserts)
        total = ow_stash + or_stash + ds_stash + q_sizes + ow_local + or_local + pending_qr + extra_nodes
        if total > self._peak_client_size:
            self._peak_client_size = total
    
    def setup(self, data_map: Dict[int, Any] = None) -> None:
        """
        Initialization phase: Build client and server states
        
        :param data_map: Initial database
        """
        if data_map is None:
            data_map = {}
            
        # Initialize static ORAM tree D_S
        self._Ds = StaticOram(
            num_data=self._num_data,
            data_size=self._data_size,
            client=self._client,
            name=self._Ds_name,
            filename=self._filename,
            bucket_size=self._bucket_size,
            stash_scale=self._stash_scale,
            aes_key=self._aes_key,
            num_key_bytes=self._num_key_bytes,
            use_encryption=self._use_encryption
        )
        
        # Initialize OMAP caches
        # O_W and O_R each store at most cache_size + 1 entries
        self._Ow = BPlusOdsOmap(
            order=4,
            num_data=self._cache_size + 1,
            key_size=self._num_key_bytes,
            data_size=self._data_size,
            client=self._client,
            name=self._Ow_name,
            filename=self._filename,
            bucket_size=self._bucket_size,
            stash_scale=self._stash_scale,
            aes_key=self._aes_key,
            num_key_bytes=self._num_key_bytes,
            use_encryption=self._use_encryption
        )
        
        self._Or = BPlusOdsOmap(
            order=4,
            num_data=self._cache_size + 1,
            key_size=self._num_key_bytes,
            data_size=self._data_size,
            client=self._client,
            name=self._Or_name,
            filename=self._filename,
            bucket_size=self._bucket_size,
            stash_scale=self._stash_scale,
            aes_key=self._aes_key,
            num_key_bytes=self._num_key_bytes,
            use_encryption=self._use_encryption
        )

        # print(f"[DEBUG] O_W tree height: {self._Ow._max_height}, O_R tree height: {self._Or._max_height}")
        
        # Initialize OMAP storage
        st_ow = self._Ow._init_ods_storage([])
        st_or = self._Or._init_ods_storage([])

        # Upload server state
        server_storage: ServerStorage = {
            self._Ow_name: st_ow,
            self._Or_name: st_or,
            self._Qw_name: self._Qw,
            self._Qr_name: self._Qr,
            self._Ds_name: self._Ds._init_storage_on_pos_map(data_map=data_map)
        }
        
        self._client.init(server_storage)

    def access(self, key: Any, op: str, value: Any = None) -> Any:
        # print("[访问] 并行查找 + 延迟写回")
        # key = Helper.hash_data_to_leaf(prf=self._group_prf, data=general_key, map_size=self._num_groups)
        
        # DEBUG PRINTS
        # print(f"DEBUG: Ow MaxHeight: {self._Ow._max_height}, Level: {self._Ow._level}, LeafRange: {self._Ow._leaf_range}")
        # print(f"DEBUG: Or MaxHeight: {self._Or._max_height}, Level: {self._Or._level}, LeafRange: {self._Or._leaf_range}")

        # The access needs to use hash?
        # Note: The benchmark script calls access(key, op).
        # In TopDown: access(op, general_key, value).
        # In BottomUp: access(key, op, value).
        # The key passed is int?
        # Benchmark says: keys = zipf_keys... (ints).
        
        old_value = None

        self._update_peak_client_size()

        # 先处理上一次删除留下的待写回（O_R 插入 + D_S 写回），使用完整 h 轮保证路径正确
        if self._pending_insert_ds is not None:
            ds_key, ds_value = self._pending_insert_ds
            self._Ds.operate_on_key(op="w", key=ds_key, value=ds_value)
            self._pending_insert_ds = None
        if self._pending_insert_or is not None:
            or_key, or_value, or_ts, _ = self._pending_insert_or
            self._Or.insert(or_key, (or_value, or_ts))
            self._pending_insert_or = None

        self._update_peak_client_size()

        # Get pending delete keys for parallel execution
        delete_key_ow = None
        delete_key_or = None
        
        if self._pending_delete_ow is not None:
            pending_key, marker = self._pending_delete_ow
            if marker == "Key":
                delete_key_ow = pending_key
            # else: marker == "Dummy", do dummy deletion (key=None)
        
        if self._pending_delete_or is not None:
            pending_key, pending_ts, marker = self._pending_delete_or
            if marker == "Key":
                delete_key_or = pending_key
            # else: marker == "Dummy", do dummy deletion (key=None)

        # Observer callback for parallel search stats
        def _parallel_search_observer(extra_nodes: int):
            self._update_peak_client_size(extra_nodes=extra_nodes)

        # Parallel search with pending deletions - all 4 operations in h rounds
        value_ow, value_or, deleted_ow_value = BPlusOdsOmap.parallel_search_and_delete(
            omap1=self._Ow, search_key1=key, delete_key1=delete_key_ow,
            omap2=self._Or, search_key2=key, delete_key2=delete_key_or,
            observer=_parallel_search_observer
        )

        self._update_peak_client_size()
        
        # Handle the deleted O_W value: move to O_R and update D_S
        if self._pending_delete_ow is not None:
            pending_key, marker = self._pending_delete_ow
            if marker == "Key" and deleted_ow_value is not None:
                # 这里仅记录，真正的写回与 O_R 插入推迟到下一次 access 的 h 轮
                self._pending_insert_ds = (pending_key, deleted_ow_value)
                self._pending_insert_or = (pending_key, deleted_ow_value, self._timestamp, marker)
            self._pending_delete_ow = None
        
        # Handle O_R pending deletion cleanup
        if self._pending_delete_or is not None:
            self._pending_delete_or = None

        # Pre-calculated markers and keys for D_S batch access
        ds_read_key = None
        ds_read_needed = False
        
        # Process the current access
        # Case a: Key is in write cache O_W
        if value_ow is not None:
            old_value = value_ow
            ds_read_key = None # Dummy
            qw_marker = "Dummy"
               
        # Case b: Key is in read cache O_R
        elif value_or is not None:
            old_value, _ = value_or
            ds_read_key = None # Dummy
            qw_marker = "Key"
              
        # Case c: Key is not in cache
        else:
            ds_read_key = key
            ds_read_needed = True
            qw_marker = "Key"

        self._Qw_len += 1

        self._update_peak_client_size()
        
        # Calculate D_S leaf (random for dummy, hash for real)
        ds_leaf = self._Ds.get_path_leaf(ds_read_key)

        # Build batch operations list
        batch_ops = []
        
        # 1. Pending D_S Write from previous access (Must be first to handle path collisions)
        if self._pending_ds_eviction is not None:
            p_leaf, p_data = self._pending_ds_eviction
            batch_ops.append({'op': 'write', 'label': self._Ds_name, 'leaf': p_leaf, 'data': p_data})
            
        # 2. Current D_S Read
        batch_ops.append({'op': 'read', 'label': self._Ds_name, 'leaf': ds_leaf})

        # 3. Q_W insert
        batch_ops.append({'op': 'list_insert', 'label': self._Qw_name, 'index': 0, 'value': self._encrypt_data((key, qw_marker))})

        # Flush pending Q_R inserts in this batch round以共享 WAN 轮次
        pending_qr_count = len(self._pending_qr_inserts)
        for qr_key, qr_ts, qr_marker in self._pending_qr_inserts:
            batch_ops.append({
                'op': 'list_insert', 'label': self._Qr_name, 'index': 0,
                'value': self._encrypt_data((qr_key, qr_ts, qr_marker))
            })
        self._pending_qr_inserts = []
        
        # If Q_W exceeds cache_size, pop the oldest element
        qw_pop_added = False
        if self._Qw_len > self._cache_size:
            batch_ops.append({'op': 'list_pop', 'label': self._Qw_name, 'index': -1})
            self._Qw_len = self._cache_size
            qw_pop_added = True
        
        # Always pop one element from Q_R if available (no expiry check)
        qr_pop_added = False
        if self._Qr_len > 0:
            batch_ops.append({'op': 'list_pop', 'label': self._Qr_name, 'index': -1})
            qr_pop_added = True
        
        # Execute batch query
        results = self._client.batch_query(batch_ops)
        
        # Process results
        result_idx = 0
        
        # 1. Check Pending Write result (None)
        if self._pending_ds_eviction is not None:
            result_idx += 1
            self._pending_ds_eviction = None
            
        # 2. Process D_S Read
        if result_idx < len(results):
            encrypted_path = results[result_idx]
            result_idx += 1
            
            # Decrypt and process path
            path = self._Ds._decrypt_buckets(encrypted_path)
            # Use 'r' op to retrieve value and put into stash
            # If dummy read, ds_read_key is None, still valid
            ds_val = self._Ds.process_retrieved_path('r', ds_read_key, path, None)
            
            # If this was real access (Case C), update O_W
            if ds_read_needed:
                old_value = ds_val
                if op == 'read':
                    self._Ow.insert_local(key, old_value)  
                else:
                    self._Ow.insert_local(key, value)
            else:
                # Case a/b post-batch logic
                if value_ow is not None:
                    if op == 'read':
                        self._Ow.search_local(key)
                    else:
                        self._Ow.search_local(key, value)
                elif value_or is not None:
                    if op == 'read':
                        self._Ow.insert_local(key, old_value)  
                    else:
                        self._Ow.insert_local(key, value)

            # Prepare Eviction for next round
            new_evicted_path = self._Ds.prepare_eviction(ds_leaf)
            self._pending_ds_eviction = (ds_leaf, new_evicted_path)

        # 3. Skip Q_W insert result
        result_idx += 1

        # 4. Skip Pending Q_R inserts results
        result_idx += pending_qr_count

        self._update_peak_client_size()
        
        # Handle Q_W pop result

        if qw_pop_added:
            encrypted_qw_pop = results[result_idx]
            if encrypted_qw_pop is not None:
                self._pending_delete_ow = self._decrypt_data(encrypted_qw_pop)
                # 排队到下次 batch 的 Q_R 插入
                qr_insert_key = self._pending_delete_ow[0]
                qr_insert_marker = self._pending_delete_ow[1]
                self._pending_qr_inserts.append((qr_insert_key, self._timestamp, qr_insert_marker))
            result_idx += 1
        
        # Handle Q_R pop result (always pop one when available)
        if qr_pop_added and result_idx < len(results):
            encrypted_qr_item = results[result_idx]
            if encrypted_qr_item is not None:
                qr_item = self._decrypt_data(encrypted_qr_item)
                self._pending_delete_or = qr_item
            self._Qr_len -= 1
            result_idx += 1

        # 新插入的 Q_R 元素现在才计入长度（已发送本轮 batch）
        self._Qr_len += pending_qr_count

        # Update timestamp
        self._timestamp += 1

        self._update_peak_client_size()
        
        return old_value
    
    def operate_on_list(self, label: str, op: str, pos: int = None, data: Any = None) -> Any:
        """Perform an operation on a list stored on the server"""
        if op == 'insert':
            encrypted_data = self._encrypt_data(data)
            self._client.list_insert(label=label, value=encrypted_data)
        elif op == 'pop':            
            encrypted_data = self._client.list_pop(label=label)
            return self._decrypt_data(encrypted_data)
        elif op == 'get':
            encrypted_data = self._client.list_get(label=label, index=pos)
            return self._decrypt_data(encrypted_data)
        elif op == 'all':
            encrypted_data_list = self._client.list_all(label=label)
            if self._use_encryption:
                return [self._decrypt_data(item) for item in encrypted_data_list]
            return encrypted_data_list
        else:
            print(f"error: unknown operation '{op}'")
        return None
