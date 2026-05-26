"""Tests for exercises/03-raw-http.py (stdlib only)."""

from __future__ import annotations

import asyncio
import importlib.util
import socket
import threading
import unittest
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parent / "03-raw-http.py"
    spec = importlib.util.spec_from_file_location("raw_http_exercise", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


raw_http = _load_module()
HTTPMessage = raw_http.HTTPMessage
HTTPClient = raw_http.HTTPClient


class LocalRawHTTPServer:
    """Minimal HTTP/1.1 server for deterministic wire tests."""

    def __init__(self, responses: dict[str, bytes]):
        self.responses = responses
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(5)
        self.host, self.port = self._sock.getsockname()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            with socket.create_connection((self.host, self.port), timeout=0.5):
                pass
        except OSError:
            pass
        self._sock.close()
        self._thread.join(timeout=2)

    def _serve(self) -> None:
        self._sock.settimeout(0.5)
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                continue
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            try:
                request = conn.recv(65536)
                if not request:
                    return
                first_line = request.split(b"\r\n", 1)[0]
                path = first_line.split(b" ")[1].decode("ascii", errors="replace")
                payload = self.responses.get(path, self.responses.get("/", b""))
                conn.sendall(payload)
            except OSError:
                pass


def _run(coro):
    return asyncio.run(coro)


async def _get(host: str, port: int, path: str, *, connection: str = "close") -> HTTPMessage:
    client = HTTPClient()
    client.connect(host, port)
    try:
        client.send_status_line("GET", path)
        client.send_headers({"Host": f"{host}:{port}", "Connection": connection})
        return await client.receive_response()
    finally:
        client.close()


class TestHTTPMessage(unittest.TestCase):
    def test_from_resp_line_and_headers(self) -> None:
        raw = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/plain\r\n"
            b"Content-Length: 5\r\n"
            b"\r\n"
        )
        msg = HTTPMessage.from_resp_line_and_headers(raw)
        self.assertEqual(msg.protocol, "HTTP/1.1")
        self.assertEqual(msg.status_code, 200)
        self.assertEqual(msg.status_message, "OK")
        self.assertEqual(msg.headers["Content-Type"], "text/plain")
        self.assertEqual(msg.headers["Content-Length"], "5")
        self.assertIsNone(msg.body)

    def test_invalid_message(self) -> None:
        with self.assertRaises(ValueError):
            HTTPMessage.from_resp_line_and_headers(b"HTTP/1.1 200 OK")


class TestHTTPClientLocal(unittest.TestCase):
    def setUp(self) -> None:
        self.server = LocalRawHTTPServer(
            {
                "/length": (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Length: 11\r\n"
                    b"Connection: close\r\n"
                    b"\r\n"
                    b"hello world"
                ),
                "/chunked": (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Transfer-Encoding: chunked\r\n"
                    b"Connection: close\r\n"
                    b"\r\n"
                    b"5\r\n"
                    b"hello\r\n"
                    b"7\r\n"
                    b", world\r\n"
                    b"0\r\n"
                    b"\r\n"
                ),
                "/close-body": (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Connection: close\r\n"
                    b"\r\n"
                    b"no length header"
                ),
                "/204": (
                    b"HTTP/1.1 204 No Content\r\n"
                    b"Connection: close\r\n"
                    b"\r\n"
                ),
            }
        )
        self.server.start()

    def tearDown(self) -> None:
        self.server.stop()

    def test_content_length(self) -> None:
        msg = _run(_get(self.server.host, self.server.port, "/length"))
        self.assertEqual(msg.status_code, 200)
        self.assertEqual(msg.body, b"hello world")

    def test_chunked(self) -> None:
        msg = _run(_get(self.server.host, self.server.port, "/chunked"))
        self.assertEqual(msg.status_code, 200)
        self.assertEqual(msg.body, b"hello, world")

    def test_connection_close_body(self) -> None:
        msg = _run(_get(self.server.host, self.server.port, "/close-body"))
        self.assertEqual(msg.status_code, 200)
        self.assertEqual(msg.body, b"no length header")

    def test_204_no_content(self) -> None:
        msg = _run(_get(self.server.host, self.server.port, "/204"))
        self.assertEqual(msg.status_code, 204)
        self.assertIsNone(msg.body)


class TestHTTPClientInternet(unittest.TestCase):
    """Plain HTTP against stable public hosts (no TLS in the exercise client)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._network_ok = True
        try:
            socket.getaddrinfo("example.com", 80, type=socket.SOCK_STREAM)
        except OSError:
            cls._network_ok = False

    def setUp(self) -> None:
        if not self._network_ok:
            self.skipTest("no network / DNS")

    def test_example_com(self) -> None:
        try:
            msg = _run(_get("example.com", 80, "/"))
        except (OSError, ValueError, AssertionError) as exc:
            self.skipTest(f"example.com unreachable: {exc}")
        self.assertEqual(msg.status_code, 200)
        self.assertIsNotNone(msg.body)
        assert msg.body is not None
        self.assertIn(b"Example Domain", msg.body)

    def test_cloudflare_cp(self) -> None:
        try:
            msg = _run(_get("cp.cloudflare.com", 80, "/"))
        except (OSError, ValueError, AssertionError) as exc:
            self.skipTest(f"cp.cloudflare.com unreachable: {exc}")
        self.assertIn(msg.status_code, (200, 204))
        if msg.status_code == 200:
            self.assertIsNotNone(msg.body)
            assert msg.body is not None
            self.assertIn(b"success", msg.body)


if __name__ == "__main__":
    unittest.main()
