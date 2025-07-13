import socket as sock
import struct

def main():
    fd = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
    fd.setsockopt(sock.SOL_SOCKET, sock.SO_REUSEADDR, 1)
    fd.bind(('0.0.0.0', 8000))
    fd.listen(sock.SOMAXCONN)

    while True:
        connection_sock, addr = fd.accept()
        process_request(connection_sock)
        
def process_request(connection_sock):
    buffer = bytearray(4096)
    connection_sock.recv_into(buffer, 4)
    
    message_len = struct.unpack('!I', buffer[:4])
    
    print('message len: ', message_len)

if __name__ == '__main__':
    main()