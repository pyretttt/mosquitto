#include <iostream>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>

static void process(int fd) {
    char buff[64] = {0};
    ssize_t n = read(fd, buff, sizeof(buff) - 1);
    if (n < 0) {
        std::cout << "Nothing to read" << std::endl;
        return;
    }

    std::cout << "Client send: " << buff << std::endl;

    write(fd, buff, n);
}

int main() {
    using namespace std;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int val{1};
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &val, sizeof(val));

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = ntohs(1234);
    addr.sin_addr.s_addr = ntohl(0);

    int rv = bind(fd, reinterpret_cast<const sockaddr *>(&addr), sizeof(addr));
    if (rv) {
        cerr << "Failed to bind socket" << endl;
    }
    
    rv = listen(fd, SOMAXCONN);
    if (rv) {
        cerr << "Failed to listen socket" << endl;
    }

    while (true) {
        struct sockaddr_in client_addr;
        socklen_t socklen = sizeof(client_addr);
        int connfd = accept(fd, reinterpret_cast<sockaddr *>(&client_addr), &socklen);

        if (connfd < 0) {
            continue;
        }

        process(fd);
        close(connfd);
    }
}
