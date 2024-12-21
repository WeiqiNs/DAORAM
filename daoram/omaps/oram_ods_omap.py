from typing import Any, List, Optional, Tuple, Union

from daoram.dependency.crypto import Prf
from daoram.dependency.helpers import Helper
from daoram.omaps.tree_ods_omap import TreeOdsOmap
from daoram.orams.tree_base_oram import TreeBaseOram


class OramTreeOdsOmap:
    def __init__(self, num_data: int, ods: TreeOdsOmap, oram: TreeBaseOram):
        """Initialize the proposed construction for optimal omap.

        :param num_data: the number of data points the oram should store.
        :param ods: some tree structured ods omap that inherits the TreeOdsOmap class.
        :param oram: some tree based oram that inherits the TreeBaseOmap class.
        """
        # Save the input as class attributes.
        self.__num_data: int = num_data
        self.__ods: TreeOdsOmap = ods
        self.__oram: TreeBaseOram = oram

        # The ods for tree needs to adjust its height only.
        self.__ods.update_mul_tree_height(num_tree=num_data)

        # A new PRF used to hash the input keys.
        self.__prf: Prf = Prf()

    def init_server_storage(self, data: Optional[List[Tuple[Union[str, int, bytes], Any]]] = None) -> None:
        """
        Initialize the server storage for the input list of key-value pairs.

        :param data: a list of key-value pairs.
        """
        # If the data list is not provided, we set it to an empty list.
        if data is None:
            data = []

        # Create a dictionary where keys are consecutive integers and values are list of values mapped to the integer.
        data_map = Helper.hash_data_to_map(prf=self.__prf, data=data, map_size=self.__num_data)

        # Convert the values to a list.
        data_list = [data_map[key] for key in range(self.__num_data)]

        # Initialize multiple trees for the ODS.
        roots = self.__ods.init_mul_tree_server_storage(data_list=data_list)

        # Initialize DAOram with the roots.
        self.__oram.init_server_storage(data_map={key: root for key, root in enumerate(roots)})

    def insert(self, key: Union[str, int, bytes], value: Any):
        """
        Given key-value pair, insert the pair to the omap.

        :param key: the search key of interest.
        :param value: the value to insert.
        """
        # First hash the key of interest and find where it is stored in the oram.
        oram_key = Helper.hash_data_to_leaf(prf=self.__prf, data=key, map_size=self.__num_data)

        # Get the root of the ods tree storing the input key of interest.
        root = self.__oram.operate_on_key_without_eviction(op="r", key=oram_key, value=None)

        # Set the ods root and perform insert.
        self.__ods.root = root
        self.__ods.insert(key=key, value=value)

        # Update the stored root in oram.
        self.__oram.eviction_with_update_stash(key=oram_key, value=self.__ods.root)

    def search(self, key: Union[str, int, bytes], value: Any = None) -> Any:
        """
        Given a search key, return its corresponding value.

        If input value is not None, the value corresponding to the search tree will be updated.
        :param key: the search key of interest.
        :param value: the updated value.
        :return: the (old) value corresponding to the search key.
        """
        # First hash the key of interest and find where it is stored in the oram.
        oram_key = Helper.hash_data_to_leaf(prf=self.__prf, data=key, map_size=self.__num_data)

        # Get the root of the ods tree storing the input key of interest.
        root = self.__oram.operate_on_key_without_eviction(op="r", key=oram_key, value=None)

        # Set the ods root and perform insert.
        self.__ods.root = root
        value = self.__ods.search(key=key, value=value)

        # Update the stored root in oram.
        self.__oram.eviction_with_update_stash(key=oram_key, value=self.__ods.root)

        # Return the retrieved value from tree omap.
        return value
