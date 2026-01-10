import pickle
from typing import Any, Dict

from daoram.dependency.crypto import Aes
from daoram.dependency.interact_server import InteractServer, ServerStorage
from daoram.omap.avl_omap import AVLOmap
from daoram.omap.avl_omap_cache import AVLOmapOptimized
from daoram.oram.static_oram import StaticOram


class BottomUpSomap:
    """
    Bottom-to-Up SOMAP (Snapshot-Oblivious Map with Dynamic Security)
    
    Implementation of Algorithm 3: A snapshot-oblivious key-value store with dynamic security level adjustment.
    This protocol maintains two OMAP caches (O_W for writes and O_R for reads) and two queues (Q_W and Q_R)
    to provide oblivious access with adjustable security parameters.
    """

    def __init__(self,
                 num_data: int,
                 cache_size: int,
                 data_size: int,
                 client: InteractServer,
                 name: str = "busomap",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 300,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True):
        """
        Initialize Bottom-to-Up SOMAP
        
        :param num_data: Database size (N)
        :param cache_size: Cache size (window parameter c)
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
        self._cache_size = cache_size
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
        self._Ow: AVLOmap = None
        self._Or: AVLOmap = None

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

    @property
    def client(self) -> InteractServer:
        """Return the client object."""
        return self._client

    # todo: @weiqi check if all ciphertexts have the same length
    # since the value component of dummy pair is "dummy"
    def _encrypt_data(self, data: Any) -> Any:
        """Encrypt data if encryption is enabled"""
        if not self._use_encryption:
            return data

        try:
            # Serialize the data
            serialized_data = pickle.dumps(data)
            # Encrypt the serialized data
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
            # Decrypt the data
            decrypted_data = self._list_cipher.dec(encrypted_data)
            # Deserialize the data
            data = pickle.loads(decrypted_data)
            return data
        except Exception as e:
            print(f"Error decrypting data: {e}")
            return encrypted_data

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

        # Initialize OMAP caches and queues
        self._Ow = AVLOmapOptimized(
            num_data=self._cache_size,
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

        self._Or = AVLOmapOptimized(
            num_data=self._cache_size,
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

    def access(self, key: Any, op: str, value: Any = None) -> Any:
        """
        Access phase: Process a single operation
        
        :param key: Key to access
        :param op: Operation type ('lookup', 'update', 'insert', 'delete')
        :param value: New value (for update/insert operations)
        :return: Previous value of the key
        """
        old_value = None

        # Retrieve key-value pair
        value_ow = self._Ow.search(key)
        value_or = self._Or.search(key)

        # Case a: Key is in write cache O_W
        if value_ow is not None:
            old_value = value_ow
            # Dummy access: randomly access a path
            self._Ds.operate_on_key(op="r", key=None)
            # update (𝑘,𝑣) in O_W
            if op == 'read':
                self._Ow.search(key)
            else:
                self._Ow.search(key, value)

            self.operate_on_list(label=self._Qw_name, op="insert", data=(key, "Dummy"))

        # Case b: Key is in read cache O_R
        elif value_or is not None:
            old_value, _ = value_or
            # Dummy access: randomly access a path
            self._Ds.operate_on_key(op="r", key=None)
            #  insert (𝑘,𝑣) to O_W
            if op == 'read':
                self._Ow.insert(key, old_value)
            else:
                self._Ow.insert(key, value)

            self.operate_on_list(label=self._Qw_name, op="insert", data=(key, "Key"))

        # Case c: Key is not in cache
        else:
            # Retrieve from static ORAM tree using path number
            old_value = self._Ds.operate_on_key(op="r", key=key)
            #  insert (𝑘,𝑣) to O_W
            if op == 'read':
                self._Ow.insert(key, old_value)
            else:
                self._Ow.insert(key, value)

            self.operate_on_list(label=self._Qw_name, op="insert", data=(key, "Key"))

        self._Qw_len += 1

        # Adjust security level
        self._adjust_security_level()

        # Update timestamp and operation count
        self._timestamp += 1

        return old_value

    def _adjust_security_level(self) -> None:
        """
        Adjust security level: Dynamically manage cache sizes
        
        This implements the security level adjustment mechanism of Algorithm 3:
        - If |O_W| > c, pop h oldest key-value pairs from Q_W
        - For each popped pair with marker "Key", move from O_W to O_R and update D_S
        - Maintain O_R size by removing oldest entries when necessary
        """
        # Adjust O_W and O_R cache
        if self._Qw_len > self._cache_size:

            # Pop the oldest h key-value pairs
            popped_items = []
            for _ in range(self._Qw_len - self._cache_size):
                popped_items.append(self.operate_on_list(label=self._Qw_name, op="pop"))
            self._Qw_len = self._cache_size

            # Process each popped key-value pair
            for key, marker in popped_items:
                # Actual key
                if marker == "Key":

                    # delete form Ow and Move to O_R cache
                    value = self._Ow.delete(key)

                    # Update static ORAM tree
                    self._Ds.operate_on_key(op="w", key=key, value=value)

                    tmp = self._Or.search(key)
                    if tmp is None:
                        self._Or.insert(key, (value, self._timestamp))

                    # If some of them already exist in Or
                    else:
                        self._Or.search(key, (value, self._timestamp))

                    self.operate_on_list(label=self._Qr_name, op="insert", data=(key, self._timestamp, "Dummy"))
                    self._Qr_len += 1

                else:
                    self._Ow.delete(None)
                    self._Ds.operate_on_key(op="r", key=None)
                    self._Or.insert(None)
                    self.operate_on_list(label=self._Qr_name, op="insert", data=(key, self._timestamp, "Dummy"))

            # Adjust O_R cache
            match_Qr = None
            while True:

                if self._Qr_len == 0:
                    break

                # expired
                item = self.operate_on_list(label=self._Qr_name, op="r", pos=0)
                if self._timestamp - item[1] > self._cache_size:
                    match_Qr = self.operate_on_list(label=self._Qr_name, op="pop")
                    self._Qr_len -= 1

                    if match_Qr[2] == "Dummy":
                        self._Or.search(None)
                        self._Or.delete(None)
                    else:
                        match_Or = self._Or.search(match_Qr[0])
                        if match_Qr[1] == match_Or[1]:
                            self._Or.delete(match_Or[0])
                        else:
                            self._Or.delete(None)
                else:
                    break

    def adjust_cache_size(self, new_cache_size: int) -> None:
        """
        Dynamically adjust cache size (window parameter c)
        
        :param new_cache_size: New cache size
        """
        if new_cache_size > 0:
            self._cache_size = new_cache_size
            # Immediately adjust security level to adapt to new size
            self._adjust_security_level()

    def operate_on_list(self, label: str, op: str, pos: int = None, data: Any = None) -> Any:
        """Perform an operation on a list stored on the server"""
        if op == 'insert':
            # Encrypt data before inserting if encryption is enabled
            encrypted_data = self._encrypt_data(data)
            self._client.list_insert(label=label, value=encrypted_data)
        elif op == 'pop':
            # Get the encrypted data from the list
            encrypted_data = self._client.list_pop(label=label)
            # Decrypt the data if encryption is enabled
            return self._decrypt_data(encrypted_data)
        elif op == 'r':
            encrypted_data = self._client.list_get(label=label, index=pos)
            return self._decrypt_data(encrypted_data)
        elif op == 'all':
            # Get all encrypted data from the list
            encrypted_data_list = self._client.list_all(label=label)
            # Decrypt each item if encryption is enabled
            if self._use_encryption:
                return [self._decrypt_data(item) for item in encrypted_data_list]
            return encrypted_data_list
        else:
            print(f"error: unknown operation '{op}'")
        return None
