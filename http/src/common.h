#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <sstream>
#include <functional>
#include <regex>

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
    socket_send_failure,
    http_response_failed_to_parse
};

struct Request {
    std::string path;
    Headers headers;
    std::string method;
};

struct Response {
    std::string version;
    int status;
    std::string reason;
    Headers headers;
};

namespace NodeEnum {
    struct Host { std::string host; };
    struct Ip { std::string ip; };
    using E = std::variant<Host, Ip>;
};

bool ends_with_crlf(std::string const &str) {
    if (str.size() < 2) return false;
    auto end = str.data() + str.size();
    return end[-1] == '\n' && end[-2] == '\r';
} 

struct Socket {
    Socket() = default;
    Socket(socket_t sock, addrinfo addrinfo) : sock(sock), addrinfo_(addrinfo) {}

    socket_t sock = INVALID_SOCKET;
    addrinfo addrinfo_;

    bool is_valid() const {
        return sock != INVALID_SOCKET;
    }

    int try_connect() const {
        return connect(sock, addrinfo_.ai_addr, static_cast<socklen_t>(addrinfo_.ai_addrlen));
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

    std::string read_line() const {
        std::array<char, 2048> fix_size_buffer;
        std::string flex_buffer;
        size_t used{0};
        while (;;) {
            char byte;
            auto n = read(sock, &byte, 1);
            if (n == 1) {
                if (used < fix_size_buffer.max_size()) {
                    fix_size_buffer[used++] = byte;
                } else {
                    flex_buffer.append(byte);
                    used++;
                }
            } else {
                break;
            }
            if (byte == '\n') {
                break;
            }
        }
        
        if (flex_buffer.empty()) {
            flex_buffer.assign(fix_size_buffer, used);
        }
        return flex_buffer;
    }

    void read_response(
        Response &res
    ) const {
        auto const response_line = read_line();
        if (response_line.empty() || !ends_with_crlf(response_line)) {
            throw Errors::http_response_failed_to_parse;
        }

        const static std::regex re("(HTTP/1\\.[01]) (\\d{3})(?: (.*?))?\r\n");
        std::cmatch m;
        if (!std::regex_match(response_line, m, re)) {
            throw Errors::http_response_failed_to_parse;
        }

        res.version = std::string(m[1]);
        res.status = std::stoi(std::string(m[2]));
        res.reason = std::string(m[3]);

        for (;;) {
            auto const line = read_line();
            if (ends_with_crlf(line)) {
                if (line.size() == 2) {
                    break
                } else {
                    line.
                    res.headers.insert(make_pair())
                }
            }
        }
    }
};