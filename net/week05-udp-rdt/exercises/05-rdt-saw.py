import socket
from dataclasses import dataclass
import struct
import contextlib
from io import BytesIO

WINDOW_SIZE = 2

@dataclass
class Segment:
    @dataclass
    class Header:
        src_port: int
        dst_port: int
        length: int # of header + data
        checksum: int
        seq_num: int
        flags: int

        HEADER_LENGTH = 12

        @staticmethod
        def from_bytes(bytes: bytes) -> 'Header':
            return Segment.Header(*struct.unpack('!HHHHHH', bytes))


    header: Header
    data: bytes

    def calculate_checksum(self) -> int:
        # One's-complement sum over header fields (checksum excluded) + data.
        total = (
            self.header.src_port
            + self.header.dst_port
            + self.header.length
            + self.header.seq_num
            + self.header.flags
        )
        data = self.data
        if len(data) % 2:
            data = data + b'\x00'
        if data:
            total += sum(struct.unpack(f'!{len(data) // 2}H', data))
        while total >> 16:
            total = (total & 0xFFFF) + (total >> 16)
        return total

    @staticmethod
    def make(bytes: bytes, src_port: int, dst_port: int, seq_num: int, flags: int = 0) -> 'Segment':
        header = Segment.Header(
            src_port=src_port,
            dst_port=dst_port,
            length=len(bytes) + Segment.Header.HEADER_LENGTH, # 10 is the length of the header
            checksum=0,
            seq_num=seq_num,
            flags=flags
        )
        segment = Segment(header=header, data=bytes)
        segment.header.checksum = (~segment.calculate_checksum()) & 0xFFFF
        return segment

    def not_corrupted(self) -> bool:
        total = self.calculate_checksum() + self.header.checksum
        while total >> 16:
            total = (total & 0xFFFF) + (total >> 16)
        return total == 0xFFFF

    def to_bytes(self) -> bytes:
        return struct.pack('!HHHHHH', self.header.src_port, self.header.dst_port, self.header.length, self.header.checksum, self.header.seq_num, self.header.flags) + self.data

class Sender:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.2)
        self.seq_num = 0
        self.is_free = True

    @contextlib.contextmanager
    def lock(self) -> None:
        if not self.is_free:
            raise ValueError("Sender is not free")
        self.is_free = False
        try:
            yield
        finally:
            self.is_free = True

    def _send(self, dst_addr: str, dst_port: int, data: bytes):
        segment = Segment.make(data, self.port, dst_port, self.seq_num)
        self.sock.sendto(segment.to_bytes(), (dst_addr, dst_port))
        try:
            segment_bytes, _ = self.sock.recvfrom(2 ** 16)
            io = BytesIO(segment_bytes)
            header = Segment.Header.from_bytes(io.read(Segment.Header.HEADER_LENGTH))

            if header.length - Segment.Header.HEADER_LENGTH > 0:
                data = io.read(header.length - Segment.Header.HEADER_LENGTH)
            else:
                data = bytes()
            received_segment = Segment(header=header, data=data)
        except TimeoutError:
            print(f"Timeout waiting for ACK")
            raise TimeoutError("Timeout waiting for ACK")

        if header.flags & 0x01 and received_segment.not_corrupted() and header.seq_num == self.seq_num:
            self.seq_num = (self.seq_num + 1) % WINDOW_SIZE
        else:
            print(f"Invalid ACK: {header.flags}, {received_segment.not_corrupted()}, {header.seq_num == self.seq_num}")
            raise RuntimeError("Invalid ACK")


    def send(self, dst_addr: str, dst_port: int, data: bytes):
        retry_count = 15
        while retry_count > 0:
            try:
                with self.lock():
                    self._send(dst_addr, dst_port, data)
                return
            except TimeoutError:
                retry_count -= 1
                continue
            except RuntimeError:
                retry_count -= 1
                continue
        raise TimeoutError("Timeout sending data")



class Receiver:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.ack_num = 0

    def recv(self):
        while True:
            b, (dst_addr, dst_port) = self.sock.recvfrom(2 ** 16)
            io = BytesIO(b)
            header = Segment.Header.from_bytes(io.read(Segment.Header.HEADER_LENGTH))
            data = io.read(header.length - Segment.Header.HEADER_LENGTH)
            segment = Segment(header=header, data=data)

            if not segment.not_corrupted():
                self.sock.sendto(
                    Segment.make(bytes(), self.port, dst_port, (self.ack_num - 1) % WINDOW_SIZE, flags=0x01).to_bytes(),
                    (dst_addr, dst_port)
                )
            elif segment.not_corrupted() and header.seq_num != self.ack_num:
                self.sock.sendto(
                    Segment.make(bytes(), self.port, dst_port, header.seq_num, flags=0x01).to_bytes(),
                    (dst_addr, dst_port)
                )
            elif segment.not_corrupted() and header.seq_num == self.ack_num and header.flags == 0x00:
                self.ack_num = (header.seq_num + 1) % WINDOW_SIZE
                self.sock.sendto(
                    Segment.make(bytes(), self.port, dst_port, header.seq_num, flags=0x01).to_bytes(),
                    (dst_addr, dst_port)
                )
                print(f"Received data: {segment.data}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sender", action="store_true")
    parser.add_argument("--receiver", action="store_true")
    args = parser.parse_args()

    if args.sender:
        sender = Sender("10.0.0.1", 12345)
        for chunk in range(10):
            sender.send("10.0.0.2", 12346, b"Hello, world " + str(chunk).encode())
    elif args.receiver:
        receiver = Receiver("10.0.0.2", 12346)
        receiver.recv()