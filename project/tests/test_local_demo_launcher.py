from __future__ import annotations

import socket
import unittest

from scripts.web.local_demo import check_port


class TestLocalDemoPorts(unittest.TestCase):
    def test_live_listener_is_not_replaced(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaisesRegex(RuntimeError, "端口"):
                check_port(listener.getsockname()[1])

    def test_just_closed_http_connection_does_not_prevent_restart(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            listener.listen()
            with socket.create_connection(("127.0.0.1", port)) as client:
                connection, _ = listener.accept()
                connection.close()
                self.assertEqual(client.recv(1), b"")
        check_port(port)


if __name__ == "__main__":
    unittest.main()
