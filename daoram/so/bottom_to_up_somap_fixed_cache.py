import logging
import math
import os
import warnings
from collections import deque
from daoram.dependency.interact_server import InteractServer, ServerStorage
from daoram.dependency.crypto import Aes
from daoram.dependency.helper import Data, Helper
from daoram.omap.bplus_ods_omap import BPlusOdsOmap
from daoram.oram.static_oram import StaticOram
from typing import Any, Dict, Tuple, Optional
import pickle

class BottomUpSomapFixedCache:
    """Bottom-to-Up SOMAP (Algorithm 3) with parallel access optimization.

    Architecture:
        - D_S: Static ORAM storing the full database
        - O_W / O_R: B+ tree OMAPs acting as write/read caches
        - Q_W / Q_R: Server-side encrypted queues tracking cache contents

    Key optimizations over the dynamic-cache version:
        1. Batch D_S write + Q_W insert + Q_W/Q_R pop in one server round
        2. Cache popped elements locally (pending deletes)
        3. Execute O_W/O_R deletions in parallel with the next query's
           search, reusing the same h rounds of OMAP traversal

    Dynamic features:
        - Timestamp-based cache eviction via set_target_cache_size()
        - Effective OMAP height tracks max(|Q_W|, |Q_R|) for round efficiency
        - O_W/O_R initialized with num_data capacity to avoid rebuild on resize

    TODO: Support dynamic ORAM tree resizing so that O_W/O_R storage
          scales with actual cache size rather than always using num_data.
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
                 use_encryption: bool = True,
                 order: int = 4,
                 key_length: int = 4,
                 value_length: int = 256):
        self._num_data = num_data
        self._cache_size = cache_size  # Fixed, will not change
        self._data_size = data_size
        self._key_length = key_length
        self._value_length = value_length
        self._client = client
        self._name = name
        self._filename = filename
        self._bucket_size = bucket_size
        self._stash_scale = stash_scale
        self._aes_key = aes_key
        self._num_key_bytes = num_key_bytes
        self._order = order
        self._use_encryption = use_encryption
        
        # Initialize encryption for list data
        self._list_cipher = Aes(key=aes_key, key_byte_length=num_key_bytes) if use_encryption else None
        self._extended_size = 3 * self._num_data
        self._list_pad_length = self._compute_list_pad_length() if use_encryption else 0
        
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
        self._pending_ds_eviction_secondary: Optional[Tuple[int, Any]] = None  # Secondary pending write

        # Dynamic cache adjustment (timestamp-based eviction)
        self._target_cache_size = cache_size
        self._adjust_cap = 1
        self._qw_push_times: deque = deque()
        self._qr_push_times: deque = deque()
        self._pending_delete_ow_queue: deque = deque()
        self._pending_delete_or_queue: deque = deque()

        # 客户端占用统计（块数）
        self._peak_client_size = 0

    @property
    def client(self) -> InteractServer:
        """Return the client object."""
        return self._client

    def _compute_list_pad_length(self) -> int:
        """Compute fixed padding length so all list-encrypted ciphertexts have identical size.
        Considers both original keys (key_length bytes) and hashed indices (from num_data)."""
        max_idx = self._extended_size
        worst_keys = [max_idx, bytes(self._key_length)]
        s_qw = max(len(pickle.dumps((k, "Dummy"))) for k in worst_keys)
        s_qr = max(len(pickle.dumps((k, max_idx, "Dummy"))) for k in worst_keys)
        s_val = len(pickle.dumps(bytes(self._value_length)))
        return max(s_qw, s_qr, s_val)

    def _encrypt_data(self, data: Any) -> Any:
        if not self._use_encryption:
            return data
        serialized = pickle.dumps(data)
        padded = Helper.pad_pickle(data=serialized, length=self._list_pad_length)
        return self._list_cipher.enc(padded)

    def _decrypt_data(self, encrypted_data: Any) -> Any:
        if not self._use_encryption:
            return encrypted_data
        decrypted = self._list_cipher.dec(encrypted_data)
        return pickle.loads(Helper.unpad_pickle(data=decrypted))

    def reset_peak_client_size(self) -> None:
        self._peak_client_size = 0

    def set_target_cache_size(self, target_c: int, adjust_cap: int = 1) -> None:
        """Dynamically adjust the security window size using timestamp-based eviction.

        Items in Q_W/Q_R are evicted when current_timestamp - push_timestamp >= target_c.
        A cap limits burst evictions when shrinking: at most (1 + adjust_cap) pops per queue
        per operation.

        :param target_c: New target cache size. If > initial cache_size, OMAP trees may
                         need to be initialized with sufficient capacity beforehand.
        :param adjust_cap: Max extra pops per operation beyond the normal 1. Default 1.
        """
        if target_c > self._num_data:
            warnings.warn(
                f"target_c={target_c} > OMAP capacity={self._num_data}. "
                f"O_W/O_R trees may overflow.",
                stacklevel=2
            )
        self._target_cache_size = target_c
        self._adjust_cap = adjust_cap

    def _update_peak_client_size(self, extra_nodes: int = 0) -> None:
        """Update peak client storage size (in blocks/entries).
        
        Client storage includes:
        - O_W stash and local buffers
        - O_R stash and local buffers
        - D_S (StaticOram) stash
        - Q_W and Q_R queue lengths (client-side tracking)
        - Pending Q_R inserts
        - Any extra nodes passed as parameter
        """
        # O_W components
        ow_stash = len(self._Ow._stash) if self._Ow is not None else 0
        ow_local = len(getattr(self._Ow, "_local", [])) if self._Ow is not None else 0
        
        # O_R components
        or_stash = len(self._Or._stash) if self._Or is not None else 0
        or_local = len(getattr(self._Or, "_local", [])) if self._Or is not None else 0
        
        # D_S (StaticOram) stash
        ds_stash = len(getattr(self._Ds, "_stash", [])) if self._Ds is not None else 0
        
        # Queue lengths (client tracks lengths, actual data is on server)
        q_sizes = self._Qw_len + self._Qr_len
        
        # Pending operations
        pending_qr = len(self._pending_qr_inserts)
        pending_others = (1 if self._pending_delete_ow else 0) + \
                         (1 if self._pending_delete_or else 0) + \
                         (1 if self._pending_insert_or else 0) + \
                         (1 if self._pending_insert_ds else 0) + \
                         (1 if self._pending_ds_eviction else 0) + \
                         (1 if self._pending_ds_eviction_secondary else 0)
        
        total = (ow_stash + ow_local + or_stash + or_local + 
                 ds_stash + q_sizes + pending_qr + pending_others + extra_nodes)
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
            order=self._order,
            num_data=self._num_data,
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
            order=self._order,
            num_data=self._num_data,
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

    def setup_caches_only(self) -> Dict[str, Any]:
        """
        Initialize only the OMAP caches (O_W, O_R, Q_W, Q_R) and return their storage.
        Assumes D_S is already valid on server.
        """
        # Initialize O_W (B+ ODS OMAP)
        self._Ow = BPlusOdsOmap(
            order=self._order,
            num_data=self._num_data,
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
        
        # Initialize O_R (B+ ODS OMAP)
        self._Or = BPlusOdsOmap(
            order=self._order,
            num_data=self._num_data,
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
        
        # Initialize OMAP storage
        st_ow = self._Ow._init_ods_storage([])
        st_or = self._Or._init_ods_storage([])
        
        # Initialize queues (empty)
        self._Qw = []
        self._Qr = []
        self._Qw_len = 0
        self._Qr_len = 0
        self._qw_push_times = deque()
        self._qr_push_times = deque()
        self._pending_delete_ow_queue = deque()
        self._pending_delete_or_queue = deque()
        
        # Return components to be merged into server storage
        return {
            self._Ow_name: st_ow,
            self._Or_name: st_or,
            self._Qw_name: self._Qw,
            self._Qr_name: self._Qr
        }

    def restore_client_state(self, force_reset_caches: bool = False) -> None:
        """
        Restore the client-side objects (O_W, O_R, D_S) mirroring a completed setup(),
        without uploading any data to the server, EXCEPT if force_reset_caches is True.
        
        Use this when the server has already loaded a pre-built storage.

        :param force_reset_caches: If True, it will RE-INITIALIZE the cache components
               (O_W, O_R, Q_W, Q_R) on the server side using the current cache_size/order
               parameters, while keeping D_S intact. This is useful if the loaded storage
               file has a different cache_size than the one we want to use now.
        """
        # Initialize D_S (Static ORAM)
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
        
        # Initialize O_W (B+ ODS OMAP)
        self._Ow = BPlusOdsOmap(
            order=self._order,
            num_data=self._num_data,
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

        # Initialize O_R (B+ ODS OMAP)
        self._Or = BPlusOdsOmap(
            order=self._order,
            num_data=self._num_data,
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
        
        # Initialize queues (empty)
        self._Qw = []
        self._Qr = []
        self._Qw_len = 0
        self._Qr_len = 0
        self._qw_push_times = deque()
        self._qr_push_times = deque()
        self._pending_delete_ow_queue = deque()
        self._pending_delete_or_queue = deque()
        
        # If not forcing reset, we must restore the caches' client state (Root pointers)
        # from the server metadata, otherwise the client is disconnected from the loaded B+ trees.
        if not force_reset_caches:
             self._Ow.restore_client_state()
             self._Or.restore_client_state()
        
        if force_reset_caches:
            # We need to tell the server to replace the current O_W, O_R, Q_W, Q_R
            # with EMPTY, FRESHLY initialized versions matching our current config.
            # D_S is kept as is.
            
            st_ow = self._Ow._init_ods_storage([])
            st_or = self._Or._init_ods_storage([])
            
            # We send an helper 'update' or 'init' query.
            # Using 'init' merges/updates the storage dict on server side.
            # So if we send new objects for existing keys, they will be overwritten.
            partial_storage = {
                self._Ow_name: st_ow,
                self._Or_name: st_or,
                self._Qw_name: self._Qw,
                self._Qr_name: self._Qr,
                # Do NOT include self._Ds_name here, so it remains touched on server
            }
            self._client.init(partial_storage)

        # Ensure stashes are empty (or consistent with initial state)
        # BPlusOdsOmap and StaticOram init with empty stash by default.
        # Queues are already empty lists [] in __init__.
        
        # Re-assign client reference just in case
        self._Ow._client = self._client
        self._Or._client = self._client
        self._Ds._client = self._client

    def access(self, key: Any, op: str, value: Any = None) -> Any:
        old_value = None

        self._update_peak_client_size()

        # 记录 pending D_S insert 信息，稍后在 batch 中处理
        pending_ds_insert_key = None
        pending_ds_insert_value = None
        if self._pending_insert_ds is not None:
            pending_ds_insert_key, pending_ds_insert_value = self._pending_insert_ds
            self._pending_insert_ds = None
        
        # O_R insert merged into parallel batch (h rounds)
        insert_key_or = None
        insert_val_or = None
        if self._pending_insert_or is not None:
            or_key, or_value, or_ts, _ = self._pending_insert_or
            insert_key_or = or_key
            insert_val_or = (or_value, or_ts)
            self._pending_insert_or = None

        self._update_peak_client_size()

        # Dequeue from overflow queues if primary pending slot is empty
        if self._pending_delete_ow is None and self._pending_delete_ow_queue:
            self._pending_delete_ow = self._pending_delete_ow_queue.popleft()
        if self._pending_delete_or is None and self._pending_delete_or_queue:
            self._pending_delete_or = self._pending_delete_or_queue.popleft()

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

        # Set effective tree height based on current queue sizes.
        # This avoids full-height OMAP traversal when cache is smaller than max capacity.
        current_max = max(self._Qw_len, self._Qr_len, 1)
        eff_h = math.ceil(math.log(current_max, math.ceil(self._order / 2))) + 1
        self._Ow.effective_height = eff_h
        self._Or.effective_height = eff_h

        # Parallel search with pending deletions & insertions - all 5 operations in h rounds
        value_ow, value_or, deleted_ow_value = BPlusOdsOmap.parallel_search_and_delete(
            omap1=self._Ow, search_key1=key, delete_key1=delete_key_ow,
            omap2=self._Or, search_key2=key, delete_key2=delete_key_or,
            insert_key2=insert_key_or, insert_value2=insert_val_or,
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
        
        # Calculate pending D_S insert leaf (if any)
        pending_ds_insert_leaf = None
        if pending_ds_insert_key is not None:
            pending_ds_insert_leaf = self._Ds.get_path_leaf(pending_ds_insert_key)

        # Build batch operations list
        batch_ops = []
        
        # 1. Pending D_S Eviction Write from previous access (Must be first)
        if self._pending_ds_eviction is not None:
            p_leaf, p_data = self._pending_ds_eviction
            batch_ops.append({'op': 'write', 'label': self._Ds_name, 'leaf': p_leaf, 'data': p_data})
        
        # 1b. Secondary Pending D_S Eviction Write (from pending insert)
        if self._pending_ds_eviction_secondary is not None:
            p_leaf2, p_data2 = self._pending_ds_eviction_secondary
            batch_ops.append({'op': 'write', 'label': self._Ds_name, 'leaf': p_leaf2, 'data': p_data2})
        
        # 2. Pending D_S Insert Read (if any) - read path to update with new value
        pending_ds_insert_read_added = False
        if pending_ds_insert_key is not None:
            batch_ops.append({'op': 'read', 'label': self._Ds_name, 'leaf': pending_ds_insert_leaf})
            pending_ds_insert_read_added = True
            
        # 3. Current D_S Read
        batch_ops.append({'op': 'read', 'label': self._Ds_name, 'leaf': ds_leaf})

        # 4. Q_W insert
        batch_ops.append({'op': 'list_insert', 'label': self._Qw_name, 'index': 0, 'value': self._encrypt_data((key, qw_marker))})
        self._qw_push_times.appendleft(self._timestamp)

        # Flush pending Q_R inserts in this batch round以共享 WAN 轮次
        pending_qr_count = len(self._pending_qr_inserts)
        for qr_key, qr_ts, qr_marker in self._pending_qr_inserts:
            batch_ops.append({
                'op': 'list_insert', 'label': self._Qr_name, 'index': 0,
                'value': self._encrypt_data((qr_key, qr_ts, qr_marker))
            })
            self._qr_push_times.appendleft(self._timestamp)
        self._pending_qr_inserts = []
        
        # Timestamp-based Q_W eviction with cap
        qw_pops_requested = 0
        max_qw_pops = 1 + self._adjust_cap
        for i in range(min(max_qw_pops, self._Qw_len, len(self._qw_push_times))):
            if self._timestamp - self._qw_push_times[-(1 + i)] >= self._target_cache_size:
                qw_pops_requested += 1
            else:
                break
        for _ in range(qw_pops_requested):
            batch_ops.append({'op': 'list_pop', 'label': self._Qw_name, 'index': -1})

        # Timestamp-based Q_R eviction with cap
        qr_pops_requested = 0
        max_qr_pops = 1 + self._adjust_cap
        for i in range(min(max_qr_pops, self._Qr_len, len(self._qr_push_times))):
            if self._timestamp - self._qr_push_times[-(1 + i)] >= self._target_cache_size:
                qr_pops_requested += 1
            else:
                break
        for _ in range(qr_pops_requested):
            batch_ops.append({'op': 'list_pop', 'label': self._Qr_name, 'index': -1})
        
        # Execute batch query
        results = self._client.batch_query(batch_ops)
        
        # Process results
        result_idx = 0
        
        # 1. Check Pending Eviction Write result (None)
        if self._pending_ds_eviction is not None:
            result_idx += 1
            self._pending_ds_eviction = None
        
        # 1b. Check Secondary Pending Eviction Write result (None)
        if self._pending_ds_eviction_secondary is not None:
            result_idx += 1
            self._pending_ds_eviction_secondary = None
        
        # 2. Process Pending D_S Insert Read (if any)
        if pending_ds_insert_read_added and result_idx < len(results):
            pending_insert_path = results[result_idx]
            result_idx += 1
            # Process the path and update with pending value
            is_decrypted = False
            if pending_insert_path and isinstance(pending_insert_path, list) and len(pending_insert_path) > 0:
                if isinstance(pending_insert_path[0], list) and len(pending_insert_path[0]) > 0:
                    if isinstance(pending_insert_path[0][0], Data):
                        is_decrypted = True
            if is_decrypted:
                original_enc = self._Ds._use_encryption
                self._Ds._use_encryption = False
                try:
                    self._Ds.process_retrieved_path('w', pending_ds_insert_key, pending_insert_path, pending_ds_insert_value)
                finally:
                    self._Ds._use_encryption = original_enc
            else:
                self._Ds.process_retrieved_path('w', pending_ds_insert_key, pending_insert_path, pending_ds_insert_value)
            # Prepare eviction for this path - will be written in next batch
            pending_insert_evicted = self._Ds.prepare_eviction(pending_ds_insert_leaf)
            # Store as secondary pending eviction for next batch
            self._pending_ds_eviction_secondary = (pending_ds_insert_leaf, pending_insert_evicted)
            
        # 3. Process Current D_S Read
        if result_idx < len(results):
            encrypted_path = results[result_idx]
            result_idx += 1
            
            # Correct logic:
            # 1. Check if path_raw is already Data objects.
            # 2. If so, call process_retrieved_path with encryption disabled (to skip 2nd dec).
            # 3. If not (bytes), call process_retrieved_path normally (it will decrypt).
            
            path_for_oram = encrypted_path # results[result_idx]
            
            is_decrypted = False
            if path_for_oram and isinstance(path_for_oram, list) and len(path_for_oram) > 0:
                 if isinstance(path_for_oram[0], list) and len(path_for_oram[0]) > 0:
                      if isinstance(path_for_oram[0][0], Data):
                           is_decrypted = True
                           
            if is_decrypted:
                 original_enc = self._Ds._use_encryption
                 self._Ds._use_encryption = False
                 try:
                      ds_val = self._Ds.process_retrieved_path('r', ds_read_key, path_for_oram, None)
                 finally:
                      self._Ds._use_encryption = original_enc
            else:
                 # Standard encrypted path, let ORAM decrypt it
                 ds_val = self._Ds.process_retrieved_path('r', ds_read_key, path_for_oram, None)
            

            
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
        
        # Handle Q_W pop results (may be multiple with timestamp-based eviction)
        for i in range(qw_pops_requested):
            encrypted_qw_pop = results[result_idx]
            if encrypted_qw_pop is not None:
                qw_item = self._decrypt_data(encrypted_qw_pop)
                if i == 0:
                    self._pending_delete_ow = qw_item
                else:
                    self._pending_delete_ow_queue.append(qw_item)
                qr_insert_key = qw_item[0]
                qr_insert_marker = qw_item[1]
                self._pending_qr_inserts.append((qr_insert_key, self._timestamp, qr_insert_marker))
            self._qw_push_times.pop()
            self._Qw_len -= 1
            result_idx += 1

        # Handle Q_R pop results (may be multiple with timestamp-based eviction)
        for i in range(qr_pops_requested):
            if result_idx < len(results):
                encrypted_qr_item = results[result_idx]
                if encrypted_qr_item is not None:
                    qr_item = self._decrypt_data(encrypted_qr_item)
                    if i == 0:
                        self._pending_delete_or = qr_item
                    else:
                        self._pending_delete_or_queue.append(qr_item)
                self._qr_push_times.pop()
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
            raise ValueError(f"Unknown operation '{op}'")
        return None
