import math
import pickle
import random

from daoram.dependency import InteractServer, Aes, PRP, ServerStorage, Prf, Data, Helper, BinaryTree
from typing import Any, List, Dict, Optional, Tuple
from daoram.omap import BPlusOdsOmap
from daoram.omap.bplus_subset_ods_omap import BPlusSubsetOdsOmap

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
                 use_encryption: bool = True,
                 key_size: int = 16):
        """
        Initialize Top-down SOMAP.

        :param num_data: Database size (N). Must be a power of 2 and >= total data count.
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
        :param key_size: The number of bytes for keys (used for padding).
        """
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
        self._extended_size = 3*self._num_data

        # PRFs: one for group hashing, one for leaf mapping
        self._group_prf = Prf()
        self._leaf_prf = Prf()
        self._tree_stash: List[Data] = []

        # Initialize cipher for encryption
        self._cipher = Aes(key=aes_key, key_byte_length=num_key_bytes) if use_encryption else None
        # Initialize cipher for list encryption if encryption is enabled
        self._list_cipher = Aes(key=aes_key, key_byte_length=num_key_bytes) if use_encryption else None
        self.PRP = PRP(key=aes_key, n=self._extended_size)

        # First compute how many are in the buckets, according to https://eprint.iacr.org/2021/1280.
        self.upper_bound = math.ceil(
            math.e ** (Helper.lambert_w(math.e ** -1 * (math.log(self._num_groups, 2) + 128 - 1)).real + 1)
        )

        # Use ServerStorage type directly for O_W, O_R, Q_W, Q_R
        self._Ow: BPlusOdsOmap = None  # OMAP O_W
        self._Or: BPlusOdsOmap = None  # OMAP O_R
        self._Ob: BPlusSubsetOdsOmap = None  # OMAP O_B for uncached data

        # Underlying BinaryTree storage (created at init_server_storage)
        self._tree: Optional[BinaryTree] = None

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
        self._Tree_name = f"{name}_Tree"

    def _compute_max_block_size(self) -> int:
        """Compute a conservative block size for storage padding similar to other OMAPs."""
        # Create a sample Data and measure its dump length.
        sample = Data(key=b"k" * self._key_size, leaf=0, value=b"v" * self._data_size)
        return len(sample.dump())

    def _encrypt_buckets(self, buckets: List[List[Data]]) -> List[List[bytes]]:
        """Encrypt and pad buckets for writing to the server."""
        if not self._use_encryption:
            return buckets  # type: ignore

        max_block = self._compute_max_block_size()

        enc_buckets: List[List[bytes]] = []
        for bucket in buckets:
            # Ensure each element's value is pickled bytes when necessary; Data.dump handles it
            enc_bucket = [self._cipher.enc(plaintext=Helper.pad_pickle(data=data.dump(), length=max_block))
                          for data in bucket]
            # pad with dummy encrypted blocks to bucket_size
            dummy_needed = self._bucket_size - len(enc_bucket)
            if dummy_needed > 0:
                enc_bucket.extend([self._cipher.enc(plaintext=Helper.pad_pickle(data=Data().dump(), length=max_block))
                                   for _ in range(dummy_needed)])
            enc_buckets.append(enc_bucket)

        return enc_buckets

    def _decrypt_buckets(self, buckets: List[List[bytes]]) -> List[List[Data]]:
        """Decrypt buckets read from server into Data objects (dropping dummies)."""
        if not self._use_encryption:
            return buckets  # type: ignore

        dec_buckets: List[List[Data]] = []
        for bucket in buckets:
            dec_bucket: List[Data] = []
            for blob in bucket:
                # decrypt and unpad then unpickle via Helper and Data.from_pickle
                dec = Data.from_pickle(Helper.unpad_pickle(data=self._cipher.dec(ciphertext=blob)))
                if dec.key is not None:
                    dec_bucket.append(dec)
            dec_buckets.append(dec_bucket)

        return dec_buckets


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
            data_map[i] = [0, 0]

        return data_map


    def setup(self, data: Optional[List[Tuple[Any, Any]]] = None) -> None:
        """
        Setup phase of SOMAP algorithm.

        :param data_map: Original data map.
        """

        # Extend database
        extended_data = self._extend_database(None)

        # initializes a variable 𝑑 = 0 as the index for dummy data
        self._dummy_index = 0

        if data is None:
            data = []

        # Partition into groups
        data_map = Helper.hash_data_to_map(prf=self._group_prf, data=data, map_size=self._num_groups)
        # Save group map locally (store only keys)
        self._group_map = {i: [kv[0] for kv in data_map[i]] for i in range(self._num_groups)}

        # Initialize the BinaryTree storage
        max_block = self._compute_max_block_size()
        tree = BinaryTree(num_data=self._num_groups, bucket_size=self._bucket_size, data_size=max_block,
                          filename=None, enc_key_size=self._num_key_bytes if self._use_encryption else None)

        # Fill the tree with each KV mapped to a PRF-determined leaf. Use a small rehash loop upon collision.
        for group_index in range(self._num_groups):
            tmp = 0
            for kv in data_map[group_index]:
                key, value = kv
                seed = group_index.to_bytes(4, byteorder="big") + (0).to_bytes(2,byteorder="big") + tmp.to_bytes(
                    2, byteorder="big")
                tmp += 1
                leaf = Helper.hash_data_to_leaf(prf=self._leaf_prf, data=seed, map_size=self._num_groups)

                data_block = Data(key=key, leaf=leaf, value=value)
                inserted = tree.fill_data_to_storage_leaf(data=data_block)

                if not inserted:
                    self._tree_stash.append(data_block)
            if len(data_map[group_index]) > self.upper_bound:
                raise MemoryError(f"Group {group_index} has more items ({len(data_map[group_index])}) than upper bound "
                                  f"({self.upper_bound}); increase the bound.")

        if len(self._tree_stash) > self._stash_scale * int(math.log2(self._num_groups)):
            raise MemoryError(f"Stash size {len(self._stash)} exceeds allowed limit "
                              f"{self._stash_scale * int(math.log2(self._num_groups))}; increase the limit.")

        # Encrypt storage if needed
        if self._use_encryption:
            tree.storage.encrypt(aes=self._cipher)

        self._tree = tree

        # creates two OMAPs denoted by (O𝑊,O𝑅) used to store𝑐 KV pairs, and two queues (𝑄𝑊,𝑄𝑅) of length c
        self._main_storage = [None] * self._extended_size
        # Use non-optimized B+ tree OMAP for both O_W and O_R caches
        # Use a slightly larger bucket size for B+ caches to reduce insertion collisions
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
        # Use BPlusSubsetOdsOmap for O_B to return uncached data
        self._Ob = BPlusSubsetOdsOmap(order=4, num_data=self._cache_size, key_size=self._num_key_bytes,
                data_size=self._data_size, client=self._client, name=self._Ob_name,
                filename=self._filename, bucket_size=self._bucket_size,
                stash_scale=max(self._stash_scale, 5000), aes_key=self._aes_key,
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
        # Initialize O_B as an empty subset OMAP so all keys are initially available
        st3 = self._Ob._init_ods_storage([])
        # O_R stores (seed, timestamp) format where seed is [read_count, write_count]
        # Initialize with timestamp 0 for all entries
        or_data_list = [(key, (value, 0)) for key, value in extended_data_list[self._cache_size: 2 * self._cache_size]]
        st2 = self._Or._init_ods_storage(or_data_list)

        # Encrypt queue data if encryption is enabled
        # Q_W entries are stored as (key, marker) tuples where marker is "Key" or "Dummy"
        # Q_R entries are stored as (key, timestamp, marker) tuples
        if self._use_encryption:
            self._Qw = [self._encrypt_data((key, "Key")) for key in keys_list[:self._cache_size]]
            self._Qr = [self._encrypt_data((key, 0, "Key")) for key in keys_list[self._cache_size: 2 * self._cache_size]]
        else:
            self._Qw = [(key, "Key") for key in keys_list[:self._cache_size]]
            self._Qr = [(key, 0, "Key") for key in keys_list[self._cache_size:  2 * self._cache_size]]



        # The client initializes the server storage with the OMAPs and queues
        Serverstorage: ServerStorage = {
            self._Ow_name: st1,
            self._Or_name: st2,
            self._Ob_name: st3,
            self._Qw_name: self._Qw,
            self._Qr_name: self._Qr,
            'DB': self._main_storage,
            self._Tree_name: self._tree
        }

        self._client.init(Serverstorage)

    def access(self,  op: str, general_key: str, general_value: Any = None, value: Any = None) -> Any:
        """
        Access phase of SOMAP algorithm.

        :param key: The key to access.
        :param op: The operation ('read' or 'write').
        :param value: The value to write (for write operations).
        :return: the old value of the key
        """
        key = group_index = Helper.hash_data_to_leaf(prf=self._group_prf, data=general_key, map_size=self._num_groups)
        # The client retrieves (𝑘,𝑣𝑘) by checking if 𝑘 exists in O𝑊 and O𝑅:
        value_old1 = self._Ow.search(key)
        value_old2 = self._Or.search(key)
        uncached_key = self._Ob.find_available()

        # this is used to simulate the cost
        validate_key = self._Or.search(uncached_key)

        value_old = None
        # Case a: key in cache Ow
        if value_old1 is not None:
            value_old = value_old1
            # Piggyback: fetch an uncached DB element if available
            # Skip if subset OMAP is full (find_available returns None)
            if uncached_key is not None:
                self.operate_on_list(label='DB', op='get', pos=self.PRP.encrypt(uncached_key))
            # If 𝑘 ∈ O𝑊, update (𝑘,𝑣𝑘) in O𝑊 and push 𝑛 +𝑑 into 𝑄w
            # seed = [a, b]: a = read_count (search), b = write_count (insert)
            if op == 'search':
                # Key exists in O_W; update its seed counts in-place
                self._Ow.search(key, value=[value_old[0]+1, value_old[1]])
            else:
                # Key exists in O_W; update write count in-place
                self._Ow.search(key, value=[value_old[0], value_old[1]+1])
            if uncached_key is not None:
                self.operate_on_list(label=self._Qw_name, op='insert', data=(uncached_key, "Key"))


            # Case b: key in cache Or
        elif value_old2 is not None:
            # O_R stores (seed, timestamp); guard against unexpected shapes
            if isinstance(value_old2, tuple) and len(value_old2) >= 1 and isinstance(value_old2[0], (list, tuple)):
                value_old = list(value_old2[0])
            else:
                # Fallback seed when Or returns unexpected value shape
                value_old = [0, 0]
            # visit (𝑛+𝑑,𝑣_𝑛+𝑑) from D
            # self.operate_on_list('DB', 'get', pos=self._prp.digest_mod_n(str(self._num_data+self._dummy_index).encode(), self._extended_size))
            self.operate_on_list(label='DB', op='get', pos=self.PRP.encrypt(self._num_data + self._dummy_index))

            # Otherwise, insert (𝑘,𝑣) to Ow and push 𝑘 into 𝑄w.
            # seed = [a, b]: a = read_count (search), b = write_count (insert)
            if op == 'search':
                self._Ow.insert(key, [value_old[0]+1, value_old[1]])  # a+1 for search
            else:
                self._Ow.insert(key, [value_old[0], value_old[1]+1])  # b+1 for insert
            self.operate_on_list(self._Qw_name, 'insert', data=(key, "Key"))
            # execute𝑑 = 𝑑 + 1 mod n
            self._dummy_index = (self._dummy_index + 1) % self._num_data
            # Do not insert dummy into subset OMAP

            # Case c: key not in cache
        else:
            # visit (key,vale) from D
            # value_old = self.operate_on_list('DB', 'get', pos = self._prp.digest_mod_n(str(key).encode(), self._extended_size))
            value_old = self.operate_on_list('DB', 'get', pos=self.PRP.encrypt(key))
            # Otherwise, insert (𝑘,𝑣) to Ow and push 𝑘 into 𝑄w.
            # seed = [a, b]: a = read_count (search), b = write_count (insert)
            if op == 'search':
                self._Ow.insert(key, [value_old[0]+1, value_old[1]])  # a+1 for search
            else:
                self._Ow.insert(key, [value_old[0], value_old[1]+1])  # b+1 for insert
            self.operate_on_list(self._Qw_name, 'insert', data=(key, "Key"))

        if op == 'search':
            value = self.search(general_key, value_old)
        else:
            value = self.insert(general_key, general_value, value_old)

        # Adjust security level
        self.adjust_security_level()
        self._timestamp += 1

        return value


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
                    # For dummy, keep a structured placeholder in Or
                    self._Or.insert(None, (None, self._timestamp))
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
            self._cache_size = new_cache_size
            # Immediately adjust security level to adapt to new size
            self.adjust_security_level()


    def operate_on_list(self, label: str, op: str, pos: int = None, data: Any = None) -> Any:
        """Perform an operation on a list stored on the server"""
        if op == 'insert':
            # Encrypt data before inserting if encryption is enabled
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

    def _collect_group_leaves_retrieve(self, group_index: int, seed: list) -> List[int]:
        """Deterministically compute the leaf indices for all keys in a group (unique, in stable order)."""
        group_seed = seed

        # Compute the leaves for all keys in the group
        leaves: List[int] = []
        for item_index in range(self.upper_bound):
            seed = group_index.to_bytes(4, byteorder="big") + group_seed[0].to_bytes(2,byteorder="big") + item_index.to_bytes(2, byteorder="big")
            leaf = Helper.hash_data_to_leaf(prf=self._leaf_prf, data=seed, map_size=self._num_groups)
            leaves.append(leaf)

        # Remove duplicates while preserving order
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
        """Deterministically compute the leaf indices for all keys in a group (in stable order)."""
        group_seed = seed

        # Compute the leaves for all keys in the group
        leaves: List[int] = []
        for item_index in range(self.upper_bound):
            seed = group_index.to_bytes(4, byteorder="big") + group_seed[0].to_bytes(2, byteorder="big") + item_index.to_bytes(2, byteorder="big")
            leaf = Helper.hash_data_to_leaf(prf=self._leaf_prf, data=seed, map_size=self._num_groups)
            leaves.append(leaf)

        return leaves

    def search(self, key: Any, seed: list) -> Any:
        """Given a key, batch-download its group's paths and return the value for the key.

        This performs a single client.read_query with a list of leaves corresponding to the group's members.
        """
        group_index = Helper.hash_data_to_leaf(prf=self._group_prf, data=key, map_size=self._num_groups)

        retrieve_leaves = self._collect_group_leaves_retrieve(group_index=group_index, seed = seed)

        # Batch read
        raw_paths = self._client.read_query(label=self._Tree_name, leaf=retrieve_leaves)
        # Decrypt/path -> list of buckets of Data
        paths = self._decrypt_buckets(buckets=raw_paths)

        # Flatten and find the desired key
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

        # evict stash to write paths back
        paths = self.evict_paths(retrieve_leaves=retrieve_leaves)
        self._client.write_query(label=self._Tree_name, leaf=retrieve_leaves,
                                 data=self._encrypt_buckets(buckets=paths))

        return value

    def insert(self, key: Any, value: Any, seed: list) -> None:
        """Insert or update a key by batch reading its group, updating, and batch-writing modified paths back.

        This keeps the single-round batch read/write property: one read_query and one write_query.
        """
        group_index = Helper.hash_data_to_leaf(prf=self._group_prf, data=key, map_size=self._num_groups)
        leaves = [random.randint(0, self._num_groups - 1)]

        raw_paths = self._client.read_query(label=self._Tree_name, leaf=leaves)
        paths = self._decrypt_buckets(buckets=raw_paths)

        for bucket in paths:
            for block in bucket:
                self._tree_stash.append(block)

        label = group_index.to_bytes(4, byteorder="big") + seed[0].to_bytes(2, byteorder="big") + seed[1].to_bytes(
            2, byteorder="big")

        data = Data(key=key,
                    leaf=Helper.hash_data_to_leaf(prf=self._leaf_prf, data=label,
                                                  map_size=self._num_groups), value=value)
        self._tree_stash.append(data)

        # evict stash to write paths back
        paths = self.evict_paths(retrieve_leaves=leaves)

        # Encrypt and write back the modified paths (the server expects buckets in the same order)
        enc = self._encrypt_buckets(buckets=paths)
        self._client.write_query(label=self._Tree_name, leaf=leaves, data=enc)

    def evict_paths(self, retrieve_leaves: List[int]) -> List[List[Data]]:
        """
        Evict data blocks in the stash to multiple paths (corresponding to `retrieve_leaves`).

        This prepares the list of buckets (Data objects) to be written back to the server for the
        collection of leaves passed in. It mirrors the behavior of other multi-path eviction
        helpers in the repository (for example `_evict_stash_to_mul` in DA-ORAM).

        :param retrieve_leaves: list of leaf labels to evict to.
        :return: A list of buckets (list of lists of `Data`) ordered as the server expects for
                 multi-path writes (bottom-up ordering consistent with BinaryTree.get_mul_path_dict).
        """
        # Temporary stash for items that couldn't be placed back into the provided paths.
        temp_stash: List[Data] = []

        # Create a dict keyed by storage indices that covers all nodes on the multiple paths.
        path_dict = BinaryTree.get_mul_path_dict(level=self._tree.level, indices=retrieve_leaves)

        # Try to place every real data item from stash into one of the provided paths.
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

        # Convert dict to list following the dict key order (which matches get_mul_path_indices ordering).
        path = [path_dict[key] for key in path_dict.keys()]

        # Update the stash to only those elements that could not be placed.
        self._tree_stash = temp_stash

        # Return raw Data buckets; caller will encrypt if needed.
        return path
