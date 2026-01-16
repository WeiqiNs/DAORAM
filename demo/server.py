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
            print("Server starting/restarting...", flush=True)
            server = RemoteServer(ip="0.0.0.0")
            server.run()
        except KeyboardInterrupt:
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Server error: {e}", flush=True)
            pass


if __name__ == '__main__':
    simulate()
