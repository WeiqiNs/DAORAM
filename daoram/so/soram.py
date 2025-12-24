"""
SORAM is a type of ORAM that provides access pattern protection under a weaker threat model
where the adversary can only observe access patterns of consecutive c operations.
It is more efficient than traditional ORAM while still providing security guarantees.
"""
import os
from queue import Queue
from typing import Any, List
from daoram.dependency import InteractServer, ServerStorage
from daoram.omap import AVLOdsOmap, AVLOdsOmapOptimized
# Define ServerStorage type

class Soram():
    def __init__(self,
                 num_data: int,
                 cache_size: int,
                 data_size: int,
                 client: InteractServer,
                 name: str = "sor",
                 filename: str = None,
                 bucket_size: int = 10,
                 stash_scale: int = 300,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True):
        """
        Initialize the SORAM with the following parameters.

        :param num_data: The number of data points the SORAM should store.
        :param cache_size: The size of the caches (O_W and O_R), denoted as 'c' in the algorithm.
        :param data_size: The number of bytes the random dummy data should have.
        :param client: The instance we use to interact with server.
        :param name: The name of the protocol, this should be unique if multiple schemes are used together.
        :param filename: The filename to save the SORAM data to.
        :param bucket_size: The number of data each bucket should have.
        :param stash_scale: The scaling scale of the stash.
        :param aes_key: The key to use for the AES instance.
        :param num_key_bytes: The number of bytes the aes key should have.
        :param use_encryption: A boolean indicating whether to use encryption.
        """
        # Initialize the parent TreeBaseOram class.

        # SORAM specific parameters
        self._extended_size = num_data + 2 * cache_size
        self._cache_size = cache_size
        # self._prp = Prp(key=os.urandom(16))
        self._data_size = data_size
        self._client = client
        self._name = name
        self._filename = filename
        self._bucket_size = bucket_size
        self._stash_scale = stash_scale
        self._aes_key = aes_key
        self._use_encryption = use_encryption
        self._num_key_bytes = num_key_bytes
        self._num_data = num_data


        # Use ServerStorage type directly for O_W, O_R, Q_W, Q_R
        self._Ow: AVLOdsOmap = None  # OMAP O_W
        self._Or: AVLOdsOmap = None  # OMAP O_R
        self._Qw: Queue = None      # Queue Q_W
        self._Qr: Queue = None     # Queue Q_R

        self._Ow_name = "O_W"
        self._Or_name = "O_R"
        self._Qw_name = "Q_W"
        self._Qr_name = "Q_R"

        # Main storage
        self._main_storage: List[Any] = [None] * self._extended_size

        # Virtual data index (client state)
        self._dummy_index: int = 0

    def operate_on_list(self, label:str, op: str, pos: int = None, data: Any = None)-> Any:
        """Perform an operation on a key in the SORAM"""
        if op == 'insert':
            self._client.list_insert(label=label, value=data)
        elif op == 'pop':            
            return self._client.list_pop(label=label)
        elif op == 'get':
            return self._client.list_get(label=label, index=pos)
        elif op == 'update':
            self._client.list_update(label=label, index=pos, value=data)
        else:
            print(f"error: unkonw operation'{op}'")
        return None

    def _extend_database(self, data_map: dict = None) -> dict:
        """
        Extend the database with virtual data entries.

        :param data_map: Original data map.
        :return: Extended data map.
        """
        if data_map is None:
            data_map = {}

        # Add virtual data entries with raw values
        for i in range(len(data_map), self._extended_size):
            data_map[i] = f"raw value of {i}"

        return data_map

    def setup(self, data_map: dict = None) -> None:
        """
        Setup phase of SORAM algorithm.

        :param data_map: Original data map.
        """
        # 1. Extend database
        extended_data = self._extend_database(data_map)

        # 2.  initializes a variable 𝑑 = 0 as the index for dummy data
        self._dummy_index = 0
            

        # 3. creates two OMAPs denoted by (O𝑊,O𝑅) used to store𝑐 KV pairs, and two queues (𝑄𝑊,𝑄𝑅) of length c
        self._main_storage = [None] * self._extended_size
        self._Ow  = AVLOdsOmapOptimized(num_data=self._cache_size, key_size=self._num_key_bytes, data_size=self._data_size, client=self._client, name=self._Ow_name, 
                                         filename=self._filename, bucket_size=self._bucket_size, stash_scale = self._stash_scale, aes_key=self._aes_key,
                                         num_key_bytes=self._num_key_bytes, use_encryption=False)
        self._Or  = AVLOdsOmapOptimized(num_data = self._cache_size, key_size=self._num_key_bytes, data_size=self._data_size, client=self._client, name=self._Or_name, 
                                         filename=self._filename, bucket_size=self._bucket_size, stash_scale = self._stash_scale, aes_key=self._aes_key,
                                         num_key_bytes=self._num_key_bytes, use_encryption=False)
        self._Qw:list = []
        self._Qr:List = []

        # 4. PRP function 𝐸𝑠𝑘 on Z𝑛+2𝑐−1 to permute (𝑖,𝑣𝑖) to 𝐸𝑠𝑘(𝑖)
        # self._prp = Prp(key=os.urandom(16))  # Randomly generate PRP key
        for i, (key, value) in enumerate(extended_data.items()):
            # encrypted_key = self._prp.digest_mod_n(str(key).encode(), self._extended_size)
            self._main_storage[key] = value

        # 5.  The client initializes the queues (𝑄𝑊,𝑄𝑅) with the initial dataset
        for i, (key, value) in enumerate(extended_data.items()):
            if i < self._cache_size:
                self._Qr.insert(0,key)
            elif i < 2 * self._cache_size :
                self._Qw.insert(0,key)
            else:
                continue

        # 6.  The client initializes the OMAPs (O𝑊,O𝑅) with the initial dataset
        st1 = self._Ow._init_ods_storage(list(extended_data.items())[self._cache_size: 2*self._cache_size])
        st2 = self._Or._init_ods_storage(list(extended_data.items())[:self._cache_size])

        # 7.  The client initializes the server storage with the OMAPs and queues
        Serverstorage:ServerStorage = {
            self._Ow_name: st1,
            self._Or_name: st2,
            self._Qw_name: self._Qw,
            self._Qr_name: self._Qr,
            'DB': self._main_storage
        }

        self._client.init(Serverstorage)

    def access(self, key: int, op: str, value: Any = None) -> Any:
        """
        Access phase of SORAM algorithm.

        :param key: The key to access.
        :param op: The operation ('read' or 'write').
        :param value: The value to write (for write operations).
        :return: the old value of the key
        """
        # The client retrieves (𝑘,𝑣𝑘) by checking if 𝑘 exists in O𝑊 and O𝑅:
        value_old1 = self._Ow.search(key)
        value_old2 = self._Or.search(key)
        value_old = None
        # Case a: key in cache Ow
        if value_old1 is not None:  
            value_old = value_old1
            # self.operate_on_list(label='DB', op='get', pos=self._prp.digest_mod_n(str(self._num_data+self._dummy_index).encode(), self._extended_size))
            self.operate_on_list(label='DB', op='get', pos=self._num_data+self._dummy_index)
            #If 𝑘 ∈ O𝑊, update (𝑘,𝑣𝑘) in O𝑊 and push 𝑛 +𝑑 into 𝑄w
            if op == 'read':
                self._Ow.search(key)
            else:
                self._Ow.search(key, value)
            self.operate_on_list(label= self._Qw_name, op='insert', data=self._num_data + self._dummy_index)
            # execute𝑑 = 𝑑 + 1 mod 2c           
            self._dummy_index += 1 
            self._dummy_index = self._dummy_index % (2 * self._cache_size) 
            
        # Case b: key in cache Or 
        elif value_old2 is not None:  
            value_old = value_old2
            # visit (𝑛+𝑑,𝑣_𝑛+𝑑) from D
            # self.operate_on_list('DB', 'get', pos=self._prp.digest_mod_n(str(self._num_data+self._dummy_index).encode(), self._extended_size))
            self.operate_on_list(label='DB', op='get',pos = self._num_data+self._dummy_index)

            #Otherwise, insert (𝑘,𝑣) to Ow and push 𝑘 into 𝑄w.
            if op == 'read':
                self._Ow.insert(key, value_old)
            else:
                self._Ow.insert(key, value)
            self.operate_on_list(self._Qw_name, 'insert', data=key)
            # execute𝑑 = 𝑑 + 1 mod 2c
            self._dummy_index += 1 
            self._dummy_index = self._dummy_index % (2 * self._cache_size) 

        # Case c: key not in cache
        else:  
            # visit (key,vale) from D
            # value_old = self.operate_on_list('DB', 'get', pos = self._prp.digest_mod_n(str(key).encode(), self._extended_size))
            value_old = self.operate_on_list('DB', 'get', pos = key)
            #Otherwise, insert (𝑘,𝑣) to Ow and push 𝑘 into 𝑄w.
            if op == 'read':
                self._Ow.insert(key, value_old)
            else:
                self._Ow.insert(key, value)
            self.operate_on_list(self._Qw_name, 'insert', data = key)

        # pop from Qw, push what have been poped into Qr
        key = self.operate_on_list(self._Qw_name, 'pop')
        value = self._Ow.delete(key) 

        self.operate_on_list(self._Qr_name, 'insert', data = key)

        # delete what have been poped from Ow and insert into Or and DB
        self._Or.insert(key, value)
        # self.operate_on_list('DB', 'update', pos=self._prp.digest_mod_n(str(key).encode(), self._extended_size), data=value)
        self.operate_on_list('DB', 'update', pos=key, data=value)

        # pop from Qr,delete what have been poped from Or
        key = self.operate_on_list(self._Qr_name, 'pop')
        self._Or.delete(key)

        return value_old
