"""ORAM/OMAP Server Demo.

Usage:
    python server.py [--ip IP] [--port PORT] [--storage-dir DIR]

Examples:
    python server.py
    python server.py --port 6666 --storage-dir /tmp/oblivlib-store
"""

import argparse

from oblivlib.dependency import StorageServer, ZmqListener, serve


def main():
    parser = argparse.ArgumentParser(description="ORAM/OMAP Server")
    parser.add_argument("--ip", default="*", help="IP to bind (default: *)")
    parser.add_argument("--port", type=int, default=5555, help="Port (default: 5555)")
    parser.add_argument("--storage-dir", default=None, help="Keep hosted trees in files here (default: memory)")
    args = parser.parse_args()

    print(f"Server listening on {args.ip}:{args.port}...")
    listener = ZmqListener(f"tcp://{args.ip}:{args.port}")
    server = StorageServer(storage_dir=args.storage_dir)

    try:
        serve(listener, server)
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.close()
        listener.close()


if __name__ == "__main__":
    main()
