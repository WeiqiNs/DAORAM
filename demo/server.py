"""This file simulates a universal server for either an ORAM or an OMAP instance."""

import os
import sys

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
from daoram.dependency import RemoteServer


def simulate():
    parser = argparse.ArgumentParser(description="Run Remote Server")
    parser.add_argument("--port", type=int, default=10000, help="Port to listen on")
    parser.add_argument("--mock-crypto", action="store_true", help="Use mock encryption")
    args = parser.parse_args()

    if args.mock_crypto:
        from daoram.dependency import crypto
        print("[Mock Encryption Enabled]")
        crypto.MOCK_ENCRYPTION = True

    # Create the server socket and run it.
    while True:
        try:
            print(f"Server starting/restarting on port {args.port}...", flush=True)
            server = RemoteServer(ip="0.0.0.0", port=args.port)
            print("RemoteServer initialized", flush=True)
            server.run()
        except KeyboardInterrupt:
            print("Server Stopped by KeyboardInterrupt", flush=True)
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Server error: {e}", flush=True)
            pass


if __name__ == '__main__':
    simulate()
