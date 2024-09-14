#include <iostream>
#include <string>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>

int main() {
    using namespace std;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        cerr << "Failed to create socket" << endl;
    }

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = ntohs(1234);
    addr.sin_addr.s_addr = ntohl(INADDR_LOOPBACK);
    int rv = connect(fd, reinterpret_cast<const sockaddr *>(&addr), sizeof(addr));
    if (rv) {
        cerr << "Failed to bind socket" << endl;
    }

    string msg = "Hey, there!";
    write(fd, msg.data(), msg.size());

    char buff[64] = {0};
    int n = read(fd, buff, sizeof(buff) - 1);
    if (n < 0) {
        cerr << "Failed to read socket" << endl;
    }

    cout << buff << endl;
    close(fd);
}
