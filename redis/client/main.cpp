#include <iostream>
#include <string>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#include "consts.h"
#include "utils.h"

static int32_t query(int fd, const char *text) {
    uint32_t len = (uint32_t)strlen(text);
    if (len > k_max_msg) {
        return -1;
    }

    char wbuf[4 + k_max_msg];
    memcpy(wbuf, &len, 4);
    memcpy(wbuf + 4, text, len);

    if (int32_t err = write_all(fd, wbuf, 4 + len)) {
        return err;
    }

    char rbuf[4 + k_max_msg + 1];
    errno = 0;
    int32_t err = read_full(fd, rbuf, 4);
    if (err) {
        if (errno == 0) {
            std::cerr << "EOF" << std::endl;
        } {
            std::cerr << "read() error" << std::endl;
        }
        return err;
    }

    memcpy(&len, rbuf, 4);
    if (len > k_max_msg) {
        std::cerr << "too long msg" << std::endl;
        return -1;
    }

    err = read_full(fd, rbuf + 4, len);
    if (err) { 
        std::cerr << "read() error" << std::endl;
        return err;
    }

    rbuf[4 + len] = 0;
    std::cout << "Server says: " << rbuf + 4 << std::endl;
    return 0;
}

int main() {
    using namespace std;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        cerr << "Failed to create socket" << endl;
    }

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = ntohs(8000);
    addr.sin_addr.s_addr = ntohl(INADDR_LOOPBACK);
    int rv = connect(fd, reinterpret_cast<const sockaddr *>(&addr), sizeof(addr));
    if (rv) {
        cerr << "Failed to connect to socket" << endl;
    }

    string const msg = "Hey, there!";
    int err = query(fd, msg.data());
    if (err) {
        cerr << "Error message 1" << endl;
    }
    string const msg2 = "Hey, there2!";
    err = query(fd, msg2.data());
    if (err) {
        cerr << "Error message 2" << endl;
    }
    string const msg3 = "Hey, there3!";
    err = query(fd, msg3.data());
    if (err) {
        cerr << "Error message 3" << endl;
    }
    // write(fd, msg.data(), msg.size());

    // char buff[64] = {0};
    // int n = read(fd, buff, sizeof(buff) - 1);
    // if (n < 0) {
    //     cerr << "Failed to read socket" << endl;
    // }

    // cout << buff << endl;
    close(fd);
}
