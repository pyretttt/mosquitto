import asyncio
import socket

class EchoClient:
    def __init__(self):
        self.host = 'localhost'
        self.port = '9000'

    async def send_message(self, message: str):
        loop = asyncio.get_event_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)
        await loop.sock_connect(sock, (self.host, self.port))
        print(f"Connected to Server ({sock.getpeername()})")

        await loop.sock_sendall(sock, message.encode('utf-8'))
        print(f"Sent Message to Server ({sock.getpeername()}): ", message)
        while True:
            response, _ = await loop.sock_recvfrom(sock, 1024)
            print(f"Received From Server ({sock.getpeername()}): ", response.decode('utf-8'))
            if not response:
                break
            print("Received From Server: ", response.decode('utf-8'))

        sock.close()

async def main():
    clients = [EchoClient() for _ in range(10)]
    try:
        async with asyncio.TaskGroup() as tg:
            for idx, client in enumerate(clients):
                tg.create_task(client.send_message(f"Hello, world! {idx}"))
    except ExceptionGroup as eg:
        for exception in eg.exceptions:
            print(exception)

if __name__ == "__main__":
    asyncio.run(main())
