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
    while True:
        try:
            print("Server starting/restarting...")
            server = RemoteServer(ip="0.0.0.0")
            server.run()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Server error: {e}")
            pass


if __name__ == '__main__':
    simulate()
