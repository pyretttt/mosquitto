import socket

class HTTPClient():
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self, host: str, port: int):
        self.socket.connect((host, port))

    def close(self):
        self.socket.close()


    def send_status_line(self, method: str, path: str):
        status_line = f"{method} {path} HTTP/1.1\r\n"
        self.socket.sendall(status_line.encode("utf-8"))


    def send_headers(self, headers: dict[str, str]):
        header_lines = []
        for header, value in headers.items():
            header_lines.append(f"{header}: {value}\r\n")
        self.socket.sendall(("\r\n".join(header_lines) + "\r\n").encode("utf-8"))


    def send_body(self, body: str):
        self.socket.sendall(body.encode("utf-8"))


    async def receive_response(self) -> str:
        status_line = ""
        finished_status_line = False
        finished_headers = False
        headers = {}

        body = ""

