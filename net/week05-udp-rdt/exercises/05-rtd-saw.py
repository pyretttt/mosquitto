import socket
from dataclasses import dataclass
import struct

@dataclass
class Segment:
    @dataclass
    class Header:
        src_port: int
        dst_port: int
        length: int # of header + data
        checksum: int
        seq_num: int

    header: Header
    data: bytes

    def calculate_checksum(self) -> bytes:
        total = 0
        total += struct.pack('!H', self.header.src_port)
        total += struct.pack('!H', self.header.dst_port)
        total += struct.pack('!H', self.header.length)
        total += struct.pack('!H', self.header.checksum)
        total += struct.pack('!H', self.header.seq_num)
        if len(self.data) % 2:
            data = self.data.copy()
            data += b'\x00'
        words = struct.unpack(f'!{len(data)//2}H', data)
        total += sum(words)
        while total >> 16:
            total = (total & 0xFFFF) + (total >> 16)
        return total

    @staticmethod
    def make(bytes: bytes, src_port: int, dst_port: int, seq_num: int) -> 'Segment':
        header = Segment.Header(
            src_port=src_port,
            dst_port=dst_port,
            length=len(bytes) + 10, # 10 is the length of the header
            checksum=0,
            seq_num=seq_num
        )
        checksum = Segment.calc_checksum(bytes, header)
        return Segment(header=header, data=bytes)

@dataclass
class PseudoSegment:
    @dataclass
    class PseudoHeader:
        src_ip: str
        dst_ip: str
        protocol: int
        length: int

    pseudo_header: PseudoHeader
    segment: Segment

    def calc_checksum(self) -> int:
        checksum = 0


class Sender:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_RAW)
        self.sock.bind((host, port))
        self.seq_num = 0
        self.timeout = 200

    def send(self, dst_addr: str, dst_port: int, data: bytes):
        header = PseudoSegment.Header(
            src_port=self.port,
            dst_port=dst_port,
            length=len(data),
            checksum=0
        )
        pseudo_segment = PseudoSegment(
            pseudo_header=PseudoSegment.PseudoHeader(
                src_ip=self.host,
                dst_ip=dst_addr,
                protocol=socket.IPPROTO_UDP,
                length=len(data) + 16
            ),
            header=header,
            data=data
        )
        checksum = pseudo_segment.calc_checksum()

        self.sock.sendto(segment.to_bytes(), (self.host, self.port))


class Receiver:
    pass