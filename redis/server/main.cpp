#include <iostream>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <assert.h>

#include "consts.h"
#include "utils.h"

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

static int32_t one_request(int fd) {
    char rbuf[4 + k_max_msg + 1];
    errno = 0;
    int32_t err = read_full(fd, rbuf, 4);
    if (err) {
        if (errno == 0) {
            std::cerr << "EOF" << std::endl;
        } else {
            std::cerr << "read() error" << std::endl;
        }
        return err;
    }

    uint32_t len = 0;
    memcpy(&len, rbuf, 4);

    if (len > k_max_msg) {
        std::cerr << "too long message" << std::endl;
        return -1;
    }

    err = read_full(fd, rbuf + 4, len);
    if (err) {
        std::cerr << "read() error" << std::endl;
    }
    rbuf[4 + len] = 0;
    std::cout << "Client says: " << rbuf + 4 << std::endl;

    std::string reply = "Gotcha";
    len = static_cast<uint32_t>(reply.size());
    char wbuf[4 + len];
    memcpy(wbuf, &len, 4);
    memcpy(wbuf + 4, reply.data(), len);
    return write_all(fd, wbuf, 4 + len);
}

int main() {
    using namespace std;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    int val{1};
    int rv = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &val, sizeof(val));
    if (rv) {
        cerr << "Failed to configure socket" << endl;
    }

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = ntohs(8000);
    addr.sin_addr.s_addr = ntohl(0);

    socklen_t sock_len = sizeof(addr);
    rv = bind(fd, reinterpret_cast<const sockaddr *>(&addr), sock_len);
    if (rv) {
        cerr << "Failed to bind socket" << endl;
    }
    
    rv = listen(fd, SOMAXCONN);
    if (rv) {
        cerr << "Failed to listen socket" << endl;
    }

    while (true) {
        struct sockaddr_in client_addr{};
        socklen_t socklen = sizeof(client_addr);
        int connfd = accept(fd, reinterpret_cast<sockaddr *>(&client_addr), &socklen);

        if (connfd < 0) {
            continue;
        }

        while (true) {
            int32_t err = one_request(connfd);
            if (err) {
                break;
            }
        }
        close(connfd);
    }
}
