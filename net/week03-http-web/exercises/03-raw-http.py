import socket

class HTTPMessage:

    def __init__(self, protocol: str, status_code: int, status_message: str, headers: dict[str, str], body: bytes | None):
        self.protocol = protocol
        self.status_code = status_code
        self.status_message = status_message
        self.headers = headers
        self.body = body

    @staticmethod
    def from_resp_line_and_headers(data: bytes) -> 'HTTPMessage':
        lines = data.split(b"\r\n")
        if len(lines) < 2: # status + blank line are required
            raise ValueError("Invalid HTTP message")
        protocol, status_code, status_message = lines[0].decode("utf-8").split(" ", maxsplit=2)

        headers = {}
        header_line_idx = 1
        while lines[header_line_idx] != b"":
            header_name, header_value = lines[header_line_idx].decode("utf-8").split(":", maxsplit=1)
            headers[header_name.strip()] = header_value.strip()
            header_line_idx += 1

        return HTTPMessage(protocol, int(status_code), status_message, headers, None)



class HTTPClient():
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.recv_buffer = bytearray()

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
            header_lines.append(f"{header}: {value}")
        self.socket.sendall(("\r\n".join(header_lines) + "\r\n\r\n").encode("utf-8"))


    def send_body(self, body: bytes):
        self.socket.sendall(body)


    async def receive_response(self) -> HTTPMessage | None:
        msg = self.try_get_msg_updating_buffer(None) # May be received in previous recv
        if msg:
            return msg
        while recv := self.socket.recv(1024):
            msg = self.try_get_msg_updating_buffer(recv)
            if msg:
                return msg

        # Try to get a message after receving b''
        msg = self.try_get_msg_updating_buffer(b'')
        if msg:
            return msg
        raise ValueError("Failed to obtain response")


    def try_get_msg_updating_buffer(self, data: bytes | None):
        self.recv_buffer.extend(data or [])
        header_end_idx = self.recv_buffer.find(b"\r\n\r\n")
        if header_end_idx != -1:
            resp_line_and_headers = self.recv_buffer[:header_end_idx + 4]
            http_msg = HTTPMessage.from_resp_line_and_headers(resp_line_and_headers)

            if content_length := http_msg.headers.get("Content-Length"):
                assert "Transfer-Encoding" not in http_msg.headers
                body_end_idx = header_end_idx + 4 + int(content_length)
                if len(self.recv_buffer) >= body_end_idx:
                    http_msg.body = bytes(self.recv_buffer[header_end_idx + 4:body_end_idx])
                    self.recv_buffer = self.recv_buffer[body_end_idx:]

                    return http_msg
            elif http_msg.headers.get("Transfer-Encoding") == "chunked":
                assert "Content-Length" not in http_msg.headers
                chunk_idx = header_end_idx+4
                chunks = bytearray()
                while chunk_idx < len(self.recv_buffer):
                    chunk_size_end = self.recv_buffer.find(b"\r\n", chunk_idx)
                    if chunk_size_end == -1: # Has not received chunk end yet, or no CRLF after chunk
                        break
                    chunk_size = int(self.recv_buffer[chunk_idx:chunk_size_end], 16)
                    if chunk_size == 0:
                        http_msg.body = bytes(chunks)
                        self.recv_buffer = self.recv_buffer[chunk_size_end+4:] # hex-size(0) + CRLF + CRLF
                        return http_msg
                    if chunk_size_end + 2 + chunk_size + 2 > len(self.recv_buffer): # Has not received chunk end yet, or no CRLF after chunk
                        break
                    chunks.extend(self.recv_buffer[chunk_size_end+2:chunk_size_end + 2 + chunk_size])
                    chunk_idx = chunk_size_end + 2 + chunk_size + 2 # hex-size + CRLF + chunk + CRLF
                return None
            elif (
                "Transfer-Encoding" not in http_msg.headers
                and "Content-Length" not in http_msg.headers
                and http_msg.headers.get("Connection") == "close"
                and data is not None and len(data) == 0
            ):
                http_msg.body = bytes(self.recv_buffer[header_end_idx + 4:])
                self.recv_buffer.clear()
                return http_msg
            elif http_msg.status_code in (204, ):
                return http_msg
            else:
                assert False, "Unexpected HTTP message"
                return http_msg

        return None

