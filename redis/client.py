import socket as sock
import struct


def main():
    fd = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
    fd.connect((str(sock.INADDR_LOOPBACK), 8000))
    
    buffer = bytearray()
    message = bytes("hey there", encoding='utf-8')
    msg = struct.pack('!{}s'.format(len(message)), message)
    buffer += struct.pack("!I", len(msg))
    buffer += msg
    fd.send(buffer)
    
    data_response = fd.recv(4096)

def process_request(connection_sock):
    buffer = bytearray()
    connection_sock.recv_into(buffer, 4)
    
    print('message len: ', int.from_bytes(buffer[:4]))

if __name__ == '__main__':
    main()