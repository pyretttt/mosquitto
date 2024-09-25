import socket as sock

def main():
    fd = sock.socket(sock.AF_INET, sock.SOCK_STREAM)
    fd.setsockopt(sock.SOL_SOCKET, sock.SO_REUSEADDR, 1)
    fd.bind(('0.0.0.0', 8000))
    fd.listen(sock.SOMAXCONN)

    while True:
        connection_sock, addr = fd.accept()
        process_request(connection_sock)
        
def process_request(connection_sock):
    buffer = bytearray()
    connection_sock.recv_into(buffer, 4)
    
    print('message len: ', int.from_bytes(buffer[:4]))

if __name__ == '__main__':
    main()