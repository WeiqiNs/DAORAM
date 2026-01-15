"""This file simulates a universal server for either an ORAM or an OMAP instance."""

import os
import sys

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from daoram.dependency import RemoteServer


def simulate():
    # Create the server socket and run it.
    server = RemoteServer()
    server.run()


if __name__ == '__main__':
    simulate()
