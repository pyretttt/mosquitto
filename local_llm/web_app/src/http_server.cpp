#include <iostream>
#include <functional>
#include <optional>
#include <utility>
#include <unordered_map>
#include <string_view>
#include <sstream>

#include <sys/socket.h>
#include <arpa/inet.h> // This contains inet_addr
#include <unistd.h> // This contains close
#include <netdb.h>
#include <fcntl.h>

#include <cctype>

using socket_t = int;
#define INVALID_SOCKET ((socket_t) -1)
#define AI_NUMERICHOST 0x00000004
#define TCP_NODELAY 0x01

// Utilities
template<typename T>
using Modify = std::function<T(T&&)>;
using Headers = std::unordered_multimap<std::string, std::string>;

using SetSockOptions = std::function<void(socket_t)>;

// Types
enum class Errors {
    failed_to_connect,
    socket_send_failure
};

class Schema {
    constexpr static inline std::string_view http = "http";
    constexpr static inline std::string_view https = "https";
};



struct Request {
    Headers headers;
    std::string path;
    std::string method;
};

struct Response {};
using ResponseCallback = std::function<void(Response)>;

struct HttpClient {
    void virtual Get(Request request, ResponseCallback completion) const = 0;
    void virtual Post(Request request, ResponseCallback completion) const = 0;
    // void virtual Put(Request request) const = 0;
    // void virtual Patch(Request request) const = 0;
    virtual ~HttpClient() {};
};

struct HttpConfig {
    std::string client_cert_path;
    std::string client_key_path;
    time_t connection_timeout_sec;
    time_t read_timeout_sec;
    time_t read_timeout_usec;
    time_t write_timeout_sec;
    time_t write_timeout_usec;
    std::string basic_auth_username;
    std::string basic_auth_pass;
    std::string bearer_token_auth_token;
    bool keep_alive;
    bool follow_location;
    bool url_encode;
    int address_family = AF_UNSPEC;
    bool tcp_nodelay;
    SetSockOptions set_sock_option = nullptr;
    bool compress = false;
    bool decompress = true;
    std::string interface;
    std::string proxy_host;
    int proxy_port = -1;
    std::string proxy_basic_auth_username;
    std::string proxy_basic_auth_pass;
    std::string proxy_bearer_token_auth_token;

    Headers default_headers;
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

    std::pair<socket_t, addrinfo> inline static create_raw_socket(
        std::string const &host, 
        std::string const &ip, 
        int port,
        HttpConfig const &config,
        int socket_flags,
        SetSockOptions &&setSockOptions
    ) {
        const char *node = nullptr;
        struct addrinfo hints;
        struct addrinfo *result;
        memset(&hints, 0, sizeof(struct addrinfo));
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = 0;
        if (!ip.empty()) {
            node = ip.c_str();
            hints.ai_family = AF_UNSPEC;
            hints.ai_flags = AI_NUMERICHOST;
        } else {
            if (!host.empty()) { node = host.c_str(); }
            hints.ai_family = config.address_family;
            hints.ai_flags = socket_flags;
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
        HttpConfig const &config,
        int socket_flags,
        SetSockOptions &&setSockOptions
    ) {
        auto [sock, ai] = create_raw_socket(host, ip, port, config, socket_flags, std::move(setSockOptions));
        timeval tv;
        tv.tv_sec = static_cast<long>(config.read_timeout_sec);
        tv.tv_usec = static_cast<decltype(tv.tv_usec)>(config.read_timeout_usec);
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,
                    reinterpret_cast<const void *>(&tv), sizeof(tv));

        tv.tv_sec = static_cast<long>(config.write_timeout_sec);
        tv.tv_usec = static_cast<decltype(tv.tv_usec)>(config.write_timeout_usec);
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO,
                    reinterpret_cast<const void *>(&tv), sizeof(tv));

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

    void write_socket(
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

struct HttpClientImpl: public virtual HttpClient {
    HttpClientImpl(
        std::string schema, 
        std::string host,
        int port,
        HttpConfig const &config
    ) : 
        HttpClient(),
        config(config),
        schema(schema), 
        host(host),
        port(port)
         {}

    void Get(Request request, ResponseCallback completion) const override {
    }
    void Post(Request request, ResponseCallback completion) const override {
    }

    int create_and_connect_socket(int socket_flags = 0) {
        sock = Socket::create_client_socket(host, "", port, config, socket_flags, nullptr);
        auto ret = sock.try_connect();
        std::cout << "Socket connection status " << ret << std::endl;
        return ret;
    }

    ~HttpClientImpl() {
        std::cout << "deinited" << std::endl;
    }

    HttpConfig config;
    Socket sock;
private:
    void send(Request req, ResponseCallback &&completion) {
        if (sock.is_valid()) {
            for (const auto &header : config.default_headers) {
                if (req.headers.find(header.first) == req.headers.end()) {
                    req.headers.insert(header);
                }
            }
            req.path = schema + "://" + host + ":" + std::to_string(port) + req.path;



        } else {
            auto ret = create_and_connect_socket();
            if (ret) {
                throw Errors::failed_to_connect;
            }
            send(req, std::move(completion));
        }


    }




    std::string schema;
    std::string host;
    int port;
};

int main() {
    auto client = HttpClientImpl("123", "123", 123, HttpConfig{});
    return 0;
}