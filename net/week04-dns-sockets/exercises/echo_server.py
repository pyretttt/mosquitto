import asyncio
import socket

class EchoServer:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 9000
        self.should_serve = True

    async def handle_client(
        self,
        client_sock: socket.socket,
        loop: asyncio.AbstractEventLoop
    ):
        while True:
            await asyncio.sleep(2) # Emulate processing time
            try:
                data = await loop.sock_recv(client_sock, 1024)
            except ConnectionResetError as e:
                client_sock.close()
                print(f"Error receiving data from client: {e}")
                break
            if not data:
                break
            print(f"Received From Client ({client_sock.getpeername()}): ", data.decode('utf-8'))
            await loop.sock_sendall(client_sock, data)
        client_sock.close()

    async def serve(self):
        loop = asyncio.get_event_loop()

        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind((self.host, self.port))
        server_sock.listen(5)
        server_sock.setblocking(False)

        tasks = []
        while self.should_serve:
            client_sock, _ = await loop.sock_accept(server_sock)
            print(f"Accepted Connection from Client ({client_sock.getpeername()})")
            # Can't await here for concurrency reasons
            tasks.append(asyncio.create_task(self.handle_client(client_sock, loop)))

        server_sock.close()

        results = await asyncio.gather(*tasks)

        return results

if __name__ == "__main__":
    server = EchoServer()
    asyncio.run(server.serve())