import socket as sock
import struct


def main():
    fd = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
    fd.connect((str(sock.INADDR_LOOPBACK), 8000))
    
    buffer = bytearray()
    message = "hey there"
    msg = bytearray(message, 'utf-8')
    buffer.append(struct.pack("!I", len(msg)))
    buffer.append(msg)
    fd.send(buffer)
    
    bytes = fd.recv(4096)
           
def process_request(connection_sock):
    buffer = bytearray()
    connection_sock.recv_into(buffer, 4)
    
    print('message len: ', int.from_bytes(buffer[:4]))

if __name__ == '__main__':
    main()