import threading
import time
from contextlib import contextmanager

from oblivlib.dependency import ZMQSocket


@contextmanager
def _serving(port, handle):
    def run():
        server = ZMQSocket("127.0.0.1", port, is_server=True)
        handle(server)
        server.close()

    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(0.1)
    try:
        yield ZMQSocket("127.0.0.1", port, is_server=False)
    finally:
        thread.join()


class TestZMQSocket:
    def test_send_recv_dict(self):
        received = []

        def handle(server):
            received.append(server.recv())
            server.send({"response": "ok"})

        with _serving(9899, handle) as client:
            client.send({"test": 123, "data": [1, 2, 3]})
            response = client.recv()
            client.close()

        assert received == [{"test": 123, "data": [1, 2, 3]}]
        assert response == {"response": "ok"}

    def test_multiple_messages(self):
        received = []

        def handle(server):
            for _ in range(3):
                msg = server.recv()
                received.append(msg)
                server.send(f"ack-{msg}")

        with _serving(9896, handle) as client:
            responses = []
            for i in range(3):
                client.send(i)
                responses.append(client.recv())
            client.close()

        assert received == [0, 1, 2]
        assert responses == ["ack-0", "ack-1", "ack-2"]
