#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <sstream>
#include <functional>

#include <sys/socket.h>
#include <arpa/inet.h> // This contains inet_addr
#include <unistd.h> // This contains close
#include <netdb.h>
#include <fcntl.h>

using socket_t = int;
#define INVALID_SOCKET ((socket_t) -1)

using Headers = std::unordered_multimap<std::string, std::string>;

template<typename T>
T modify(T&& arg, std::function<void (T&)> mutation) {
    mutation(arg);
    return arg;
}

enum class Errors {
    socket_send_failure
};

struct Request {
    std::string path;
    Headers headers;
    std::string method;
};

struct Response {

};

namespace NodeEnum {
    struct Host { std::string host; };
    struct Ip { std::string ip; };
    using E = std::variant<Host, Ip>;
};

struct Socket {
    Socket() = default;
    Socket(socket_t sock, addrinfo addrinfo) : sock(sock), addrinfo_(addrinfo) {}

    socket_t sock = INVALID_SOCKET;
    addrinfo addrinfo_;

    bool is_valid() const {
        return sock != INVALID_SOCKET;
    }

    int try_connect() const {
        return connect(this->sock, addrinfo_.ai_addr, static_cast<socklen_t>(addrinfo_.ai_addrlen));
    }

    auto inline static create_raw_socket(
        NodeEnum::E const &host_or_ip, 
        int port
    ) {
        const char *node = nullptr;
        struct addrinfo hints;
        struct addrinfo *result;
        memset(&hints, 0, sizeof(struct addrinfo));
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = 0;

        if (auto host = std::get_if<NodeEnum::Host>(&host_or_ip)) {
            node = host->host.c_str();
        } else if (auto ip = std::get_if<NodeEnum::Ip>(&host_or_ip)) {
            node = ip->ip.c_str();
            hints.ai_family = AF_UNSPEC;
            hints.ai_flags = AI_NUMERICHOST;
        }

        auto service = std::to_string(port);
        if (getaddrinfo(node, service.c_str(), &hints, &result)) {
            for (auto rp = result; rp; rp = rp->ai_next) {
                auto sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);

                if (sock == INVALID_SOCKET) continue;
                if (fcntl(sock, F_SETFD, FD_CLOEXEC) == -1) {
                    close(sock);
                    continue;
                }
                // if (config.tcp_nodelay) {
                //     auto yes = 1;
                //     setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const void *>(&yes), sizeof(yes));
                // }

                if (rp->ai_family == AF_INET6) {
                    auto no = 0;
                    setsockopt(sock, IPPROTO_IPV6, IPV6_V6ONLY, reinterpret_cast<const void *>(&no), sizeof(no));
                }

                freeaddrinfo(result);
                return modify<Socket>({}, [sock, rp](Socket &arg) {
                    arg.sock = sock;
                    arg.addrinfo_ = *rp;
                });
            }
        }

        free(result);
        return modify<Socket>({}, [](Socket &arg) {
            arg.sock = INVALID_SOCKET;
            arg.addrinfo_ = {};
        });
    }

    void inline static set_nonblocking(socket_t sock, bool nonblocking) {
        auto flags = fcntl(sock, F_GETFL, 0);
        fcntl(sock, F_SETFL, nonblocking ? (flags | O_NONBLOCK) : (flags & (~O_NONBLOCK)));
    }

    inline size_t raw_socket_send_all(char const *data, size_t size, int flags) const {
        size_t all_sent = 0;
        while (all_sent < size) {
            auto sent = send(sock, reinterpret_cast<const void *>(data[all_sent]), size - all_sent, flags);
            if (sent < 0) {
                throw Errors::socket_send_failure;
            }
            all_sent += sent;
        }

        return all_sent;
    }

    void write_request(
        Request &req
    ) const {
        std::ostringstream strm;
        strm << req.method << " " << req.path << " HTTP/1.1\r\n";
        for (auto const &header : req.headers) {
            strm << header.first << " : " << header.second << "\r\n";
        }
        strm << "\r\n";
        std::string str{strm.str()};
        char const *http_header = str.c_str();
        raw_socket_send_all(http_header, strlen(http_header), 0);

    }
};