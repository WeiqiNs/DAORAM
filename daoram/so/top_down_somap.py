import pickle

from daoram.dependency import InteractServer, Aes, PRP, ServerStorage
from typing import Any, List, Dict
from daoram.omap import AVLOdsOmap, AVLOdsOmapOptimized

class TopDownSomap:
    """
    Top-down SOMAP (Efficient snapshot-oblivious map with dynamic security)

    Implementation of Algorithm 2: A snapshot-oblivious map with dynamic security level adjustment.
    This protocol maintains two OMAP caches (O_w for writes and O_R for reads) and two queues (Q_W and Q_R)
    to provide oblivious access with adjustable security parameters.

    Also, O_B is used to provide the target pair which has not been cached
    """

    def __init__(self, num_data: int,
                 cache_size: int,
                 data_size: int,
                 client: InteractServer,
                 name: str = "tdsomap",
                 filename: str = None,
                 bucket_size: int = 4,
                 stash_scale: int = 300,
                 aes_key: bytes = None,
                 num_key_bytes: int = 16,
                 use_encryption: bool = True):
        """
        Initialize Top-down SOMAP.

        :param num_data: Database size (N).
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
        self._extended_size = 2*self._num_data

        # Initialize cipher for list encryption if encryption is enabled
        self._list_cipher = Aes(key=aes_key, key_byte_length=num_key_bytes) if use_encryption else None
        self.PRP = PRP(key=aes_key, n=self._extended_size)

        # Use ServerStorage type directly for O_W, O_R, Q_W, Q_R
        self._Ow: AVLOdsOmap = None  # OMAP O_W
        self._Or: AVLOdsOmap = None  # OMAP O_R
        self._Ob: AVLOdsOmap = None

        # Queues
        self._Qw: list = []  # Queue Q_W
        self._Qr: list = []  # Queue Q_R

        # Main storage
        self._main_storage: List[Any] = [None] * self._extended_size

        # Virtual data index (client state)
        self._dummy_index: int = 0

        # Timestamp management
        self._timestamp = 0

        # Name identifiers
        self._Ow_name = f"{name}_O_W"
        self._Or_name = f"{name}_O_R"
        self._Ob_name = f"{name}_O_B"
        self._Qw_name = f"{name}_Q_W"
        self._Qr_name = f"{name}_Q_R"
        self._Ds_name = f"{name}_D_S"


    @property
    def client(self) -> InteractServer:
        """Return the client object"""
        return self._client


    # todo: @weqi check if all ciphertexts have the same length
    # since the value component of dummy pair is "dummy"
    def _encrypt_data(self, data: Any) -> Any:
        """
        Encrypt data using AES if encryption is enabled.

        :param data: The data to encrypt.
        :return: Encrypted data as bytes.
        """
        if not self._use_encryption:
            return data

        try:
            # Serialize the data
            serialized_data = pickle.dumps(data)
            # Encrypt the serialized data
            encrypted_data = self._list_cipher.enc(serialized_data)
            return encrypted_data
        except Exception as e:
            print(f"Error encryption data: {e}")
            return data

    def _decrypt_data(self, encrypted_data: bytes) -> Any:
        """
        Decrypt data using AES if encryption is enabled.

        :param encrypted_data: The encrypted data to decrypt.
        :return: Decrypted data.
        """
        if not self._use_encryption:
            return encrypted_data

        try:
            # Decrypt the data
            decrypted_data = self._list_cipher.dec(encrypted_data)
            # Deserialize the data
            data = pickle.loads(decrypted_data)
            return data
        except Exception as e:
            print(f"Error decryption data: {e}")
            return encrypted_data

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

    def setup(self, data_map: Dict[int, Any] = None) -> None:
        """
        Setup phase of SOMAP algorithm.

        :param data_map: Original data map.
        """
        # Extend database
        extended_data = self._extend_database(data_map)

        # initializes a variable 𝑑 = 0 as the index for dummy data
        self._dummy_index = 0

        # creates two OMAPs denoted by (O𝑊,O𝑅) used to store𝑐 KV pairs, and two queues (𝑄𝑊,𝑄𝑅) of length c
        self._main_storage = [None] * self._extended_size
        self._Ow = AVLOdsOmapOptimized(num_data=self._cache_size, key_size=self._num_key_bytes,
                                       data_size=self._data_size, client=self._client, name=self._Ow_name,
                                       filename=self._filename, bucket_size=self._bucket_size,
                                       stash_scale=self._stash_scale, aes_key=self._aes_key,
                                       num_key_bytes=self._num_key_bytes, use_encryption=self._use_encryption)
        self._Or = AVLOdsOmapOptimized(num_data=self._cache_size, key_size=self._num_key_bytes,
                                       data_size=self._data_size, client=self._client, name=self._Or_name,
                                       filename=self._filename, bucket_size=self._bucket_size,
                                       stash_scale=self._stash_scale, aes_key=self._aes_key,
                                       num_key_bytes=self._num_key_bytes, use_encryption=self._use_encryption)

        # PRP function 𝐸𝑠𝑘 on Z_{2n} to permute (𝑖,𝑣𝑖) to 𝐸𝑠𝑘(𝑖)
        # self._prp = Prp(key=os.urandom(16))  # Randomly generate PRP key
        for i, (key, value) in enumerate(extended_data.items()):
            # encrypted_key = self._prp.digest_mod_n(str(key).encode(), self._extended_size)
            # Encrypt the main storage data if encryption is enabled
            self._main_storage[self.PRP.encrypt(key)] = self._encrypt_data(value)

        # The client initializes the OMAPs (O𝑊,O𝑅) with the initial dataset
        extended_data_list = list(extended_data.items())
        keys_list = list(extended_data.keys())
        st1 = self._Ow._init_ods_storage(extended_data_list[:self._cache_size])
        st2 = self._Or._init_ods_storage(extended_data_list[self._cache_size: 2 * self._cache_size])

        # Encrypt queue data if encryption is enabled
        if self._use_encryption:
            self._Qw = [self._encrypt_data(key) for key in keys_list[:self._cache_size]]
            self._Qr = [self._encrypt_data(key) for key in keys_list[self._cache_size: 2 * self._cache_size]]
        else:
            self._Qw = keys_list[:self._cache_size]
            self._Qr = keys_list[self._cache_size:  2 * self._cache_size]

        print("Qw:", self._Qw)
        print("Qr:", self._Qr)

        # The client initializes the server storage with the OMAPs and queues
        Serverstorage: ServerStorage = {
            self._Ow_name: st1,
            self._Or_name: st2,
            self._Qw_name: self._Qw,
            self._Qr_name: self._Qr,
            'DB': self._main_storage
        }

        self._client.init(Serverstorage)

    def access(self, key: int, op: str, value: Any = None) -> Any:
        """
        Access phase of SOMAP algorithm.

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
            self.operate_on_list(label='DB', op='get', pos=self.PRP.encrypt(self._num_data + self._dummy_index))
            # If 𝑘 ∈ O𝑊, update (𝑘,𝑣𝑘) in O𝑊 and push 𝑛 +𝑑 into 𝑄w
            if op == 'read':
                self._Ow.search(key)
            else:
                self._Ow.search(key, value)
            self.operate_on_list(label=self._Qw_name, op='insert', data=(self._num_data + self._dummy_index, "Dummy"))
            # execute𝑑 = 𝑑 + 1 mod 2c
            self._dummy_index += 1
            self._dummy_index = self._dummy_index % self._num_data

            # Case b: key in cache Or
        elif value_old2 is not None:
            value_old = value_old2
            # visit (𝑛+𝑑,𝑣_𝑛+𝑑) from D
            # self.operate_on_list('DB', 'get', pos=self._prp.digest_mod_n(str(self._num_data+self._dummy_index).encode(), self._extended_size))
            self.operate_on_list(label='DB', op='get', pos=self.PRP.encrypt(self._num_data + self._dummy_index))

            # Otherwise, insert (𝑘,𝑣) to Ow and push 𝑘 into 𝑄w.
            if op == 'read':
                self._Ow.insert(key, value_old)
            else:
                self._Ow.insert(key, value)
            self.operate_on_list(self._Qw_name, 'insert', data=(key, "Key"))
            # execute𝑑 = 𝑑 + 1 mod 2c
            self._dummy_index += 1
            self._dummy_index = self._dummy_index % self._num_data

            # Case c: key not in cache
        else:
            # visit (key,vale) from D
            # value_old = self.operate_on_list('DB', 'get', pos = self._prp.digest_mod_n(str(key).encode(), self._extended_size))
            value_old = self.operate_on_list('DB', 'get', pos=self.PRP.encrypt(key))
            # Otherwise, insert (𝑘,𝑣) to Ow and push 𝑘 into 𝑄w.
            if op == 'read':
                self._Ow.insert(key, value_old)
            else:
                self._Ow.insert(key, value)
            self.operate_on_list(self._Qw_name, 'insert', data=(key, "Key"))

        # Adjust security level
        self._adjust_security_level()
        self._timestamp += 1

        return value_old


    def adjust_security_level(self) -> None:
        """
        Adjust the security level of SOMAP algorithm.

        This implements the security lvele adjustment mechanism of Algorithm 2:
        - If |Q_W| > c, pop h oldesr KV pairs from Q_W
        - For each popped pair with marker "Key", move from O_W to O_R and update D_S
        - Maintain Q_R size by removing oldest entries when necessary
        """

        # Adjust O_W and O_R cache
        if len(self._Qw) > self._cache_size:

            # Pop the oldest h KV pairs
            popped_items = []
            for _ in range(len(self._Qw)-self._cache_size):
                popped_items.append(self.operate_on_list(label=self._Qw_name, op="pop"))

            # Process each popped KV pair
            for key, marker in popped_items:
                # Actual key
                if marker == "Key":

                    # delete from Ow and Move to Or cache
                    value = self._Ow.delete(key)

                    # Update the main storage
                    self.operate_on_list('DB', 'update', pos=self.PRP.encrypt(key), data=value)

                    tmp = self._Or.search(key)
                    if tmp is None:
                        self._Or.insert(key, (value, self._timestamp))

                    # If some of them already exist in Or
                    else:
                        self._Or.insert(key, (value, self._timestamp))

                    self.operate_on_list(label=self._Qr_name, op='insert', data=(key, self._timestamp, "Dummy"))

                else:
                    value = self._Ow.delete(None)
                    # Update the main storage
                    # todo: @weiqi, check if the encryption works if value is none here
                    self.operate_on_list('DB', 'update', pos=self.PRP.encrypt(self._dummy_index), data=value)
                    self._Or.insert(None)
                    self.operate_on_list(label=self._Qr_name, op='insert', data=(key, self._timestamp, "Dummy"))

            # Adjust O_R cache
            while True:

                if len(self._Qr) == 0:
                    break

                # expired
                item = self.operate_on_list(label=self._Qr_name, op="get", pos=0)
                if self._timestamp - item[1] > self._cache_size:
                    match_Qr = self.operate_on_list(label=self._Qr_name, op="pop")

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

            :param new_cache_size: new cache size
            """
            if new_cache_size > 0:
                self._cahce_size = new_cache_size
                # Immediately adjust security level to adapt to new size
                self._adjust_security_level()


        def operate_on_list(self, label: str, op: str, pos: int = None, data: Any = None) -> Any:
            """Perform an operation on a list stored on the server"""
            if op == 'insert':
                # Encrypt data before inserting if encryption is enabled
                encrypted_data = self._encrypt_data(data)
                self._client.list_insert(label, data=encrypted_data)
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
