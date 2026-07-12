import socket
from dataclasses import dataclass
import struct
from io import BytesIO

SEQ_NUMBERS = 2 ** 3
WINDOW_SIZE = 2 ** 2
SPECIAL_VALUE = "RECEIVED"

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

# Usage:
# for chunk in chunks:
#    sender.send(host, port, chunk)
# sender.flush(host, port)

class Sender:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.base = 0
        self.next_seq_num = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(0.2)
        self.retry_buffer = [None] * WINDOW_SIZE

    def is_in_current_window(self, next_seq_num: int) -> bool:
        if (self.base + WINDOW_SIZE) % SEQ_NUMBERS < self.base:
            return next_seq_num in range(self.base, SEQ_NUMBERS) or next_seq_num in range(0, (self.base + WINDOW_SIZE) % SEQ_NUMBERS)
        else:
            return next_seq_num in range(self.base, (self.base + WINDOW_SIZE) % SEQ_NUMBERS)

    def send(self, dst_addr, dst_port, data: bytes):
        while not self.is_in_current_window(self.next_seq_num):
            self._recv_ack_or_retransmit(dst_addr, dst_port)

        seq = self.next_seq_num
        seg = Segment.make(data, self.port, dst_port, seq).to_bytes()
        self.retry_buffer[(seq - self.base) % SEQ_NUMBERS] = seg
        self.next_seq_num = (seq + 1) % SEQ_NUMBERS
        self.sock.sendto(seg, (dst_addr, dst_port))

    def _recv_ack_or_retransmit(self, dst_addr: str, dst_port: int):
        # All are none, so we dont wait for responses
        if not any(self.retry_buffer):
            return
        try:
            raw, _ = self.sock.recvfrom(2 ** 16)
        except TimeoutError:
            if isinstance(self.retry_buffer[0], bytes):
                self.sock.sendto(self.retry_buffer[0], (dst_addr, dst_port))
            return

        io = BytesIO(raw)
        header = Segment.Header.from_bytes(io.read(Segment.Header.HEADER_LENGTH))
        segment = Segment(header=header, data=io.read(header.length - Segment.Header.HEADER_LENGTH))
        if segment.not_corrupted() and header.flags == 0x01 and self.is_in_current_window(header.seq_num):
            idx = (header.seq_num - self.base) % SEQ_NUMBERS
            self.retry_buffer[idx] = SPECIAL_VALUE

            while self.retry_buffer[0] == SPECIAL_VALUE:
                self.retry_buffer = self.retry_buffer[1:]
                self.base = (self.base + 1) % SEQ_NUMBERS
            self.retry_buffer += [None] * WINDOW_SIZE
            self.retry_buffer = self.retry_buffer[:WINDOW_SIZE]

    def flush(self, dst_addr: str, dst_port: int, retry_count: int = 15):
        while retry_count and self.base != self.next_seq_num:
            self._recv_ack_or_retransmit(dst_addr, dst_port)
            retry_count -= 1


class Receiver:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.base = 0
        self.buffer = [None] * WINDOW_SIZE

    def is_seq_num_in_current_window(self, seq_num: int) -> bool:
        if (self.base + WINDOW_SIZE) % SEQ_NUMBERS < self.base:
            return seq_num in range(self.base, SEQ_NUMBERS) or seq_num in range(0, (self.base + WINDOW_SIZE) % SEQ_NUMBERS)
        else:
            return seq_num in range(self.base, (self.base + WINDOW_SIZE) % SEQ_NUMBERS)

    def is_seq_num_in_previous_window(self, seq_num: int) -> bool:
        return (seq_num - (self.base - WINDOW_SIZE)) % SEQ_NUMBERS < WINDOW_SIZE

    def recv(self):
        while True:
            b, (dst_addr, dst_port) = self.sock.recvfrom(2 ** 16)
            io = BytesIO(b)
            header = Segment.Header.from_bytes(io.read(Segment.Header.HEADER_LENGTH))
            data = io.read(header.length - Segment.Header.HEADER_LENGTH)
            segment = Segment(header=header, data=data)

            if not segment.not_corrupted():
                print(f"Corrupted packet: {header.seq_num}")
            elif segment.not_corrupted() and self.is_seq_num_in_previous_window(header.seq_num):
                # acknowledge packets for which ack was lost
                self.sock.sendto(
                    Segment.make(bytes(), self.port, dst_port, header.seq_num, flags=0x01).to_bytes(),
                    (dst_addr, dst_port)
                )
            elif segment.not_corrupted() and self.is_seq_num_in_current_window(header.seq_num) and header.flags == 0x00:
                self.buffer[(header.seq_num - self.base) % SEQ_NUMBERS] = segment.to_bytes()
                segments_to_flush = []
                while self.buffer[0] is not None:
                    segments_to_flush.append(self.buffer[0])
                    self.buffer = self.buffer[1:] + [None]
                    self.base = (self.base + 1) % SEQ_NUMBERS

                self.sock.sendto(
                    Segment.make(bytes(), self.port, dst_port, header.seq_num, flags=0x01).to_bytes(),
                    (dst_addr, dst_port)
                )
                print(f"Received data segment: {segment.header.seq_num}")

                for segment in segments_to_flush:
                    io = BytesIO(segment)
                    print(f"Flushed segment to upper layer: {Segment.Header.from_bytes(io.read(Segment.Header.HEADER_LENGTH)).seq_num}")
