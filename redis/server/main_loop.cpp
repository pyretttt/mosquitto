#include <iostream>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <assert.h>
#include <vector>
#include <poll.h>

#include "consts.h"
#include "utils.h"
#include "event_loop.h"

static void conn_put(std::vector<Conn *> &fd2conn, Conn *conn) {
    if (fd2conn.size() < (size_t)conn->fd) {
        fd2conn.resize(conn->fd + 1);
    }
    fd2conn[conn->fd] = conn;
}

static int32_t accept_new_conn(std::vector<Conn *> &fd2conn, int fd) {
    struct sockaddr_in client_addr = {};
    socklen_t socklen = sizeof(client_addr);

    int connfd = accept(fd, reinterpret_cast<struct sockaddr *>(&client_addr), &socklen);
    if (connfd < 0) {
        std::cerr << "accept() err" << std::endl;
        return -1;
    }

    fd_set_nb(connfd);
    Conn *conn = new Conn;
    if (!conn) {
        close(connfd);
        return -1;
    }

    conn->fd = connfd;
    conn_put(fd2conn, conn);
    return 0;
}

static bool try_fill_buffer(Conn *conn) {
    assert(conn->rbuf_size < sizeof(conn->rbuf));
    ssize_t rv = 0;
    do {
        size_t cap = sizeof(conn->rbuf) - conn->rbuf_size;
        rv = read(conn->fd, &conn->rbuf[conn->rbuf_size], cap);
    } while (rv < 0 && errno == EINTR);

    if (rv < 0 && errno == EAGAIN) {
        return false;
    }

    if (rv < 0) {
        std::cerr << "read() err" << std::endl;
        conn->state = State::End;
        return false;
    }

    if (rv == 0) {
        if (conn->rbuf_size > 0) {
            std::cerr << "Unexpected EOF" << std::endl;
        } else {
            std::cout << "EOF" << std::endl;
        }
        conn->state = State::End;
        return false;
    }

    conn->rbuf_size += (size_t)rv;
    assert(conn->rbuf_size <= sizeof(conn->rbuf) - conn->rbuf_size);

}

static void state_req(Conn *conn) {
    while (try_fill_buffer(conn)) {}
}

static void connection_io(Conn *conn) {
    switch (conn->state) {
        case State::Req:
            state_req(conn);
            break;
        case State::Res:
            state_res(conn);
            break;
        case State::End:
            assert(0);
            break;
    }
}

int main() {
    using namespace std;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        cerr << "failed create socket" << endl;
    }

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

    vector<Conn *> fd2con;
    fd_set_nb(fd);

    vector<struct pollfd> poll_args;

    while (true) {
        poll_args.clear();
        struct pollfd pfd = {fd, POLL_IN, 0};

        poll_args.push_back(pfd);

        for (auto conn : fd2con) {
            if (!conn) {
                continue;
            }

            struct pollfd pdf = {};
            pfd.fd = conn->fd;
            pfd.events = (conn->state == State::Req) ? POLL_IN : POLL_OUT;
            pfd.events |= POLL_ERR;
            poll_args.push_back(pfd);
        }

        int rv = poll(poll_args.data(), (nfds_t) poll_args.size(), 1000);
        if (rv < 0) {
            cerr << "failed to poll events" << endl;
        }

        for (size_t i = 1; i < poll_args.size(); ++i) {
            if (poll_args[i].revents) {
                Conn *conn = fd2con[poll_args[i].fd];
                connection_io(conn);

                if (conn->state == State::End) {
                    fd2con[conn->fd] = nullptr;
                    close(conn->fd);
                    free(conn);
                }
            }
        }

        if (poll_args[0].revents) {
            accept_new_conn(fd2con, fd);
        }
    }

    return 0;
}