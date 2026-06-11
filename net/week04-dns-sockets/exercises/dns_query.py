import socket
import struct
from dataclasses import dataclass, astuple
from io import BytesIO
from typing import List


@dataclass
class DNSHeader:
    id: int
    flags: int
    num_questions: int = 0
    num_answers: int = 0
    num_authorities: int = 0
    num_additionals: int = 0

    def as_bytes(self):
        return struct.pack("!HHHHHH", *astuple(self))

    @classmethod
    def from_reader(cls, reader: BytesIO):
        items = struct.unpack("!HHHHHH", reader.read(12))
        return cls(*items)


@dataclass
class DNSQuestion:
    name: bytes
    type_: int = 1
    class_: int = 1

    def as_bytes(self):
        return self.name + struct.pack("!HH", self.type_, self.class_)

    @classmethod
    def from_str_name(cls, name: str, type_: int = 1, class_: int = 1):
        encoded_name = b""
        for part in name.encode("ascii").split(b"."):
            encoded_name += bytes([len(part)]) + part
        encoded_name += b"\x00"

        return cls(name=encoded_name, type_=type_, class_=class_)

    @classmethod
    def from_reader(cls, reader: BytesIO):
        parts = []
        while (length := reader.read(1)[0]) != 0:
            parts.append(reader.read(length))
        name = b'.'.join(parts)
        type_, class_= struct.unpack("!HH", reader.read(4))
        return cls(name=name, type_=type_, class_=class_)


@dataclass
class DNSQuery:
    header: DNSHeader
    question: DNSQuestion

    def to_bytes(self):
        return self.header.as_bytes() + self.question.as_bytes()


#### RESPONSE

@dataclass
class DNSRecord:
    name: bytes
    type_: int
    class_: int
    ttl: int
    data: bytes

    @classmethod
    def from_reader(cls, reader: BytesIO):
        name = decode_name(reader)
        data = reader.read(10)
        type_, class_, ttl, data_len = struct.unpack('!HHIH', data)
        data = reader.read(data_len)
        return cls(name=name, type_=type_, class_=class_, ttl=ttl, data=data)


def decode_name(reader: BytesIO):
    parts = []
    while (length := reader.read(1)[0]) != 0:
        if length & 192:
            parts.append(decode_compressed_name(length, reader))
            break
        else:
            parts.append(reader.read(length))
    return b'.'.join(parts)

def decode_compressed_name(length, reader):
    pointer_bytes = bytes([length & 63]) + reader.read(1)
    pointer = struct.unpack('!H', pointer_bytes)[0]
    current_pos = reader.tell()
    reader.seek(pointer)
    result = decode_name(reader)
    reader.seek(current_pos)
    return result


@dataclass
class DNSPacket:
    header: DNSHeader
    questions: List[DNSQuestion]
    answers: List[DNSRecord]
    authorities: List[DNSRecord]
    additionals: List[DNSRecord]



def parse_dns_packet(data):
    reader = BytesIO(data)
    header = DNSHeader.from_reader(reader)
    questions = [DNSQuestion.from_reader(reader) for _ in range(header.num_questions)]
    answers = [DNSRecord.from_reader(reader) for _ in range(header.num_answers)]
    authorities = [DNSRecord.from_reader(reader) for _ in range(header.num_authorities)]
    additionals = [DNSRecord.from_reader(reader) for _ in range(header.num_additionals)]
    return DNSPacket(header, questions, answers, authorities, additionals)


if __name__ == "__main__":
    query = DNSQuery(
        header=DNSHeader(id=12345, flags=1 << 8, num_questions=1),
        question=DNSQuestion.from_str_name("cursor.com")
    )
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    res = sock.sendto(query.to_bytes(), ("1.1.1.1", 53))
    data, addr = sock.recvfrom(1024)
    dns_packet =parse_dns_packet(data)
    print("dns_packet: ", dns_packet)
    print("IP: ", '.'.join([str(x) for x in dns_packet.answers[0].data]))