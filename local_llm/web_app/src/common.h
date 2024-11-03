#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <sstream>

#include <sys/socket.h>
#include <arpa/inet.h> // This contains inet_addr
#include <unistd.h> // This contains close
#include <netdb.h>
#include <fcntl.h>

template<typename T>
using Modify = std::function<T(T&&)>;

using socket_t = int;
#define INVALID_SOCKET ((socket_t) -1)

using Headers = std::unordered_multimap<std::string, std::string>;

struct Request {
    std::string path;
    Headers headers;
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

        std::visit([&node](auto&& arg)
        {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, NodeEnum::Host>) {
                node = host.c_str();
            }
            else if constexpr (std::is_same_v<T, NodeEnum::IP>) {
                node = ip.c_str();
                hints.ai_family = AF_UNSPEC;
                hints.ai_flags = AI_NUMERICHOST;
            }
        }, host_or_ip);

        auto service = std::to_string(port);
        if (getaddrinfo(node, service.c_str(), &hints, &result)) {
            for (auto rp = result; rp; rp = rp->ai_next) {
                auto sock = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);

                if (sock == INVALID_SOCKET) continue;
                if (fcntl(sock, F_SETFD, FD_CLOEXEC) == -1) {
                    close(sock);
                    continue;
                }
                if (config.tcp_nodelay) {
                    auto yes = 1;
                    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const void *>(&yes), sizeof(yes));
                }

                if (config.set_sock_option) {
                    config.set_sock_option(sock);
                }
                if (rp->ai_family == AF_INET6) {
                    auto no = 0;
                    setsockopt(sock, IPPROTO_IPV6, IPV6_V6ONLY, reinterpret_cast<const void *>(&no), sizeof(no));
                }

                freeaddrinfo(result);
                return std::make_pair(sock, *rp);
            }
        }

        free(result);
        return std::make_pair(INVALID_SOCKET, addrinfo{});
    }

    void inline static set_nonblocking(socket_t sock, bool nonblocking) {
        auto flags = fcntl(sock, F_GETFL, 0);
        fcntl(sock, F_SETFL, nonblocking ? (flags | O_NONBLOCK) : (flags & (~O_NONBLOCK)));
    }

    Socket inline static create_client_socket(
        std::string const &host, 
        std::string const &ip, 
        int port,
    ) {
        auto [sock, ai] = create_raw_socket(host, ip, port, config, socket_flags, std::move(setSockOptions));
        return Socket{sock, ai};
    }

    inline size_t raw_socket_send_all(char const *data, size_t size, int flags) const {
        int all_sent = 0;
        while (all_sent < size) {
            auto sent = send(sock, reinterpret_cast<const void *>(data[all_sent]), size - all_sent, flags);
            if (sent < 0) {
                throw Errors::socket_send_failure;
            }
            all_sent += sent;
        }
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
        char const *http_header = strm.str().c_str();
        raw_socket_send_all(http_header, strlen(http_header), 0);

    }
};