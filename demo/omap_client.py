"""This file demonstrates how to set up client for each of the omap we include in this library.

Each function shows how the client should initialize the server and how to perform operations.
"""

from daoram.dependency import InteractRemoteServer
from daoram.omaps import AVLOdsOmap, BPlusOdsOmap, OramTreeOdsOmap
from daoram.orams import DAOram


def avl_ods_omap_client():
    # Define number of data to store.
    num_data = pow(2, 10)

    # Create the omap instance.
    omap = AVLOdsOmap(num_data=num_data, key_size=10, data_size=10, client=InteractRemoteServer())

    # Initialize the client to make connection.
    omap.client.init_connection()

    # Set the storage to the server.
    omap.init_server_storage()

    # Issue some insert queries.
    for i in range(num_data):
        omap.insert(key=i, value=i)

    # Issue some search queries.
    for i in range(num_data):
        print(f"Read key {i} have value {omap.search(key=i)}")

    # Finally close the connection.
    omap.client.close_connection()


def bplus_ods_omap_client():
    # Define number of data to store.
    num_data = pow(2, 10)

    # Create the omap instance.
    omap = BPlusOdsOmap(order=10, num_data=num_data, key_size=10, data_size=10, client=InteractRemoteServer())

    # Initialize the client to make connection.
    omap.client.init_connection()

    # Set the storage to the server.
    omap.init_server_storage()

    # Issue some insert queries.
    for i in range(num_data):
        omap.insert(key=i, value=i)

    # Issue some search queries.
    for i in range(num_data):
        print(f"Read key {i} have value {omap.search(key=i)}")

    # Finally close the connection.
    omap.client.close_connection()


def daoram_avl_omap_client():
    # Define number of data to store.
    num_data = pow(2, 10)

    # Create a client object for shared usage.
    client = InteractRemoteServer()

    # Create the ods object.
    ods = AVLOdsOmap(num_data=num_data, key_size=10, data_size=10, client=client)

    # Create the oram object.
    oram = DAOram(num_data=num_data, data_size=10, client=client)

    # Create the omap object.
    omap = OramTreeOdsOmap(num_data=num_data, ods=ods, oram=oram)

    # Initialize the client to make connection.
    client.init_connection()

    # Set the storage to the server.
    omap.init_server_storage()

    # Issue some insert queries.
    for i in range(num_data):
        omap.insert(key=i, value=i)

    # Issue some search queries.
    for i in range(num_data):
        print(f"Read key {i} have value {omap.search(key=i)}")

    # Finally close the connection.
    client.close_connection()


def daoram_bplus_omap_client():
    # Define number of data to store.
    num_data = pow(2, 10)

    # Create a client object for shared usage.
    client = InteractRemoteServer()

    # Create the ods object.
    ods = BPlusOdsOmap(order=10, num_data=num_data, key_size=10, data_size=10, client=client)

    # Create the oram object.
    oram = DAOram(num_data=num_data, data_size=10, client=client)

    # Create the omap object.
    omap = OramTreeOdsOmap(num_data=num_data, ods=ods, oram=oram)

    # Initialize the client to make connection.
    client.init_connection()

    # Set the storage to the server.
    omap.init_server_storage()

    # Issue some insert queries.
    for i in range(num_data):
        omap.insert(key=i, value=i)

    # Issue some search queries.
    for i in range(num_data):
        print(f"Read key {i} have value {omap.search(key=i)}")

    # Finally close the connection.
    client.close_connection()


if __name__ == '__main__':
    # avl_ods_omap_client()
    # bplus_ods_omap_client()
    # daoram_avl_omap_client()
    daoram_bplus_omap_client()
