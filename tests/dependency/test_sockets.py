import threading
import time

from daoram.dependency.sockets import ZMQSocket


class TestZMQSocket:
    def test_send_recv_dict(self):
        """Test sending and receiving a dictionary."""
        ip = "127.0.0.1"
        port = 9899
        received = []

        def run_server():
            server = ZMQSocket(ip, port, is_server=True)
            msg = server.recv()
            received.append(msg)
            server.send({"response": "ok"})
            server.close()

        server_thread = threading.Thread(target=run_server)
        server_thread.start()

        time.sleep(0.1)

        client = ZMQSocket(ip, port, is_server=False)
        client.send({"test": 123, "data": [1, 2, 3]})
        response = client.recv()
        client.close()

        server_thread.join()

        assert received[0] == {"test": 123, "data": [1, 2, 3]}
        assert response == {"response": "ok"}

    def test_send_recv_list(self):
        """Test sending and receiving a list."""
        ip = "127.0.0.1"
        port = 9898
        received = []

        def run_server():
            server = ZMQSocket(ip, port, is_server=True)
            msg = server.recv()
            received.append(msg)
            server.send([4, 5, 6])
            server.close()

        server_thread = threading.Thread(target=run_server)
        server_thread.start()

        time.sleep(0.1)

        client = ZMQSocket(ip, port, is_server=False)
        client.send([1, 2, 3])
        response = client.recv()
        client.close()

        server_thread.join()

        assert received[0] == [1, 2, 3]
        assert response == [4, 5, 6]

    def test_send_recv_bytes(self):
        """Test sending and receiving bytes."""
        ip = "127.0.0.1"
        port = 9897
        received = []

        def run_server():
            server = ZMQSocket(ip, port, is_server=True)
            msg = server.recv()
            received.append(msg)
            server.send(b"response bytes")
            server.close()

        server_thread = threading.Thread(target=run_server)
        server_thread.start()

        time.sleep(0.1)

        client = ZMQSocket(ip, port, is_server=False)
        client.send(b"test bytes")
        response = client.recv()
        client.close()

        server_thread.join()

        assert received[0] == b"test bytes"
        assert response == b"response bytes"

    def test_multiple_messages(self):
        """Test sending multiple messages in sequence."""
        ip = "127.0.0.1"
        port = 9896
        received = []

        def run_server():
            server = ZMQSocket(ip, port, is_server=True)
            for _ in range(3):
                msg = server.recv()
                received.append(msg)
                server.send(f"ack-{msg}")
            server.close()

        server_thread = threading.Thread(target=run_server)
        server_thread.start()

        time.sleep(0.1)

        client = ZMQSocket(ip, port, is_server=False)
        responses = []
        for i in range(3):
            client.send(i)
            responses.append(client.recv())
        client.close()

        server_thread.join()

        assert received == [0, 1, 2]
        assert responses == ["ack-0", "ack-1", "ack-2"]
