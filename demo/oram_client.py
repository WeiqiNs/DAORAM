"""This file demonstrates how to set up client for each of the oram we include in this library.

Each function shows how the client should initialize the server and how to perform operations.
"""

from daoram.dependency import InteractRemoteServer
from daoram.orams import DAOram, FreecursiveOram, PathOram, RecursivePathOram


def path_oram_client():
    # Define number of data to store.
    num_data = pow(2, 10)

    # Create the path oram object.
    oram = PathOram(num_data=num_data, data_size=10, client=InteractRemoteServer())

    # Initialize the client to make connection.
    oram.client.init_connection()

    # Initialize the server storage.
    oram.init_server_storage()

    # Perform some operations.
    for i in range(num_data):
        oram.operate_on_key(op="w", key=i, value=i)

    for i in range(num_data):
        print(f"Read key {i} have value {oram.operate_on_key(op='r', key=i, value=None)}")

    # Finally close the connection.
    oram.client.close_connection()


def recursive_oram_client():
    # Define number of data to store.
    num_data = pow(2, 10)

    # Create the path oram object.
    oram = RecursivePathOram(num_data=num_data, data_size=10, client=InteractRemoteServer())

    # Initialize the client to make connection.
    oram.client.init_connection()

    # Initialize the server storage.
    oram.init_server_storage()

    # Perform some operations.
    for i in range(num_data):
        oram.operate_on_key(op="w", key=i, value=i)

    for i in range(num_data):
        print(f"Read key {i} have value {oram.operate_on_key(op='r', key=i, value=None)}")

    # Finally close the connection.
    oram.client.close_connection()


def freecursive_oram_client():
    # Define number of data to store.
    num_data = pow(2, 10)

    # Create the path oram object.
    oram = FreecursiveOram(num_data=num_data, data_size=10, client=InteractRemoteServer())

    # Initialize the client to make connection.
    oram.client.init_connection()

    # Initialize the server storage.
    oram.init_server_storage()

    # Perform some operations.
    for i in range(num_data):
        oram.operate_on_key(op="w", key=i, value=i)

    for i in range(num_data):
        print(f"Read key {i} have value {oram.operate_on_key(op='r', key=i, value=None)}")

    # Finally close the connection.
    oram.client.close_connection()


def da_oram_client():
    # Define number of data to store.
    num_data = pow(2, 10)

    # Create the path oram object.
    oram = DAOram(num_data=num_data, data_size=10, client=InteractRemoteServer())

    # Initialize the client to make connection.
    oram.client.init_connection()

    # Initialize the server storage.
    oram.init_server_storage()

    # Perform some operations.
    for i in range(num_data):
        oram.operate_on_key(op="w", key=i, value=i)

    for i in range(num_data):
        print(f"Read key {i} have value {oram.operate_on_key(op='r', key=i, value=None)}")

    # Finally close the connection.
    oram.client.close_connection()


if __name__ == '__main__':
    # path_oram_client()
    # recursive_oram_client()
    # freecursive_oram_client()
    da_oram_client()
