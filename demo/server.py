"""This file simulates a universal server for either an ORAM or an OMAP instance."""

from daoram.dependency import RemoteServer


def simulate():
    # Create the server socket and run it.
    server = RemoteServer()
    server.run()


if __name__ == '__main__':
    simulate()
