#pragma once

#include <sys/event.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <atomic>
#include <iostream>
#include <string>
#include <functional>
#include <unordered_map>
#include <thread>
#include <sstream>
#include <optional>

#include "HttpMessage.hpp"

std::string_view trim(std::string_view s)
{
    s.remove_prefix(std::min(s.find_first_not_of(" \t\r\v\n"), s.size()));
    s.remove_suffix(std::min(s.size() - s.find_last_not_of(" \t\r\v\n") - 1, s.size()));

    return s;
}

namespace http {
    constexpr size_t MaxBufferSize = 4096;
    constexpr size_t ThreadPoolSize = 4;
    constexpr size_t BacklogSize = 1000;

    using RequestHandler = std::function<HttpResponse (HttpRequest const &)>;
    using RequestHandlersMap = std::unordered_map<std::string, std::unordered_map<size_t, RequestHandler>>;

    struct MessageProcessor final {
        MessageProcessor() : dataBuff() {
            dataBuff.reserve(MaxBufferSize);
        }

        std::optional<HttpRequest> processRead(char *data, size_t n) {
            std::optional<HttpRequest> ret;

            dataBuff.append(data, n);
            size_t headersEnd = dataBuff.find("\r\n\r\n");
            if (headersEnd == std::string::npos) {
                return ret;
            } else if (!filledHeaders) {
                std::string_view sv = std::string_view(dataBuff).substr(0, headersEnd + 4);
                parseStartLineAndHeaders(sv);
                filledHeaders = true;
            }

            if (request.headers.find("Content-Length") != request.headers.end()) {
                size_t contentLength = static_cast<size_t>(std::stoi(request.headers["Content-Length"]));
                auto contentReceived = dataBuff.length() - (headersEnd + 4);
                if (contentReceived >= contentLength) {
                    request.body = dataBuff.substr(headersEnd + 4, contentLength);
                    reset(headersEnd + 4 + contentLength);
                } else {
                    reset(headersEnd + 4);
                    return ret;
                }
                ret = request;
            } else {
                ret = request;
            }
            return ret;
        }

        void reset(size_t bufferPos) {
            request = HttpRequest();
            dataBuff.erase(dataBuff.begin() + bufferPos);
        }

        void parseStartLineAndHeaders(std::string_view &startLineAndHeaders) {
            size_t start = 0;
            size_t end = 0;
            bool hasParsedStartLine = false;
            while ((end = startLineAndHeaders.find("\r\n", start)) != std::string_view::npos) {
                auto substr = startLineAndHeaders.substr(start, end - start);
                if (hasParsedStartLine) {
                    parseHeader(substr);
                } else {
                    parseStartLine(substr);
                    hasParsedStartLine = true;
                }
                start = end;
            }
        }

        void parseStartLine(std::string_view &startLine) {
            size_t start = 0;
            size_t end = 0;
            size_t i = 0;
            while ((end = startLine.find(" ", start)) != std::string_view::npos) {
                auto substr = startLine.substr(start, end - start);
                switch (i) {
                case 0:
                    request.method = HttpMethod::fromString(std::string(substr));
                case 1:
                    request.url = std::string(substr);
                case 2:
                    request.version = httpVersionFromString(std::string(substr));
                default:
                    throw std::runtime_error("Wrong starting line of http request");
                }
                i++;
                start = end;
            } 
        }

        void parseHeader(std::string_view &header) {
            auto semicolonPos = header.find(':');
            if (semicolonPos == std::string_view::npos) {
                throw std::runtime_error("Wrong header format");
            }
            std::string key = std::string(trim(header.substr(0, semicolonPos)));
            std::string value = std::string(trim(header.substr(semicolonPos + 1)));
            request.headers.insert({key, value});
        }

        std::string dataBuff;
        HttpRequest request;
        bool filledHeaders = false;
    };

    struct ClientData {
        ClientData() : fd(-1) {}
        int fd;
        MessageProcessor messageProcessor;
        // size_t length;
        // size_t cursor;
        // char buffer[MaxBufferSize];
    };

    // Os specific functions
    int createNonBlockingSocket() {
        int sock_fd = socket(AF_INET, SOCK_STREAM | O_NONBLOCK, 0);
        if (sock_fd < 0) {
            throw std::runtime_error("Failed to create socket");
        }
        return sock_fd;
    }

    struct SocketManager {
        SocketManager() = default;
        SocketManager(int connectionSocket) : kq(kqueue()), connectionSock(connectionSocket) {
            registerSock(connectionSocket);
        }

        void registerSock(int sock) {
            struct kevent evSet;
            EV_SET(&evSet, sock, EVFILT_READ, EV_ADD, 0, 0, nullptr);
            if (kevent(kq, &evSet, 1, nullptr, 0, nullptr) == -1) { // Adds event to kqueue
                throw std::runtime_error("Failed to register event");
            }
        }

        void unregisterSock(int sock) {
            struct kevent evSet;
            EV_SET(&evSet, sock, EVFILT_READ, EV_DELETE, 0, 0, nullptr);
            if (kevent(kq, &evSet, 1, nullptr, 0, nullptr) == -1) { // Adds event to kqueue
                throw std::runtime_error("Failed to unregister event");
            }
        }

        void pollEvents(
            std::unordered_map<int, ClientData> &clients,
            RequestHandlersMap const &requestHandlers
        ) {
            struct kevent evList[BacklogSize];
            int nevent = kevent(kq, nullptr, 0, evList, BacklogSize, nullptr);
            if (nevent < 0) {
                std::runtime_error("Failed to poll events");
            }
            struct sockaddr_storage addr;
            socklen_t socklen = sizeof(addr);
            for (int i = 0; i < nevent; i++) {
                if (evList[i].ident == connectionSock) {
                    int clientSocket = accept(
                        evList[i].ident, 
                        reinterpret_cast<struct sockaddr *>(&addr), 
                        &socklen
                    );
                    if (clientSocket < 0) {
                        std::runtime_error("Failed to accept client socket");
                    }
                    auto clientData = ClientData();
                    clientData.fd = clientSocket;
                    clients[clientSocket] = clientData;
                    registerSock(clientSocket);
                } else if (evList[i].flags & EV_EOF) {
                    int fd = evList[i].ident;
                    unregisterSock(fd);
                    clients.erase(fd);
                } else if (evList[i].filter == EVFILT_READ || evList[i].filter == EVFILT_WRITE) {
                    handleReadWrite(evList[i], clients, requestHandlers);
                }
            }
        }

        void handleReadWrite(
            struct kevent event,
            std::unordered_map<int, ClientData> &clients,
            RequestHandlersMap const &requestHandlers
        ) {
            if (event.filter == EVFILT_READ) {
                auto &clientData = clients.at(event.ident);
                char buffer[MaxBufferSize];
                ssize_t nbyte = recv(event.ident, buffer, MaxBufferSize, 0);

                if (nbyte > 0) {
                    try {
                        auto httpRequest = clientData.messageProcessor.processRead(buffer, nbyte);
                        if (httpRequest) {
                            try {
                                RequestHandler const &handler = requestHandlers
                                    .at(httpRequest->url)
                                    .at(httpRequest->method.id());
                                auto resp = handler(httpRequest.value());
                                // TODO: Write response
                            } catch (std::out_of_range &err) {
                                std::cerr << "Can not find request handler" << std::endl;
                            }
                        }
                    } catch(std::runtime_error &error) {
                        auto resp = HttpResponse();
                        resp.statusCode = HttpStatusCode::BadRequest;
                    }
                }

            } else if (event.filter == EVFILT_WRITE) {

            } else {
                throw std::runtime_error("Assertion error");
            }
        }

        int kq;
        int connectionSock;
    };

    struct Server final {
        explicit Server(std::string const &host, std::uint16_t port)
            : host(host), 
            port(port),
            isRunning(true) {};
        
        void start() {
            sock_fd = createNonBlockingSocket();
            int opt = 1;
            if (setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
                throw std::runtime_error("Failed to set socket options");
            }

            sockaddr_in servAddr;
            servAddr.sin_family = AF_INET;
            servAddr.sin_addr.s_addr = INADDR_ANY;
            inet_pton(AF_INET, host.c_str(), &servAddr.sin_addr.s_addr);
            servAddr.sin_port = htons(port);

            if (bind(sock_fd, reinterpret_cast<sockaddr *>(&servAddr), sizeof(servAddr)) < 0) {
                throw std::runtime_error("Failed to bind socket");
            }

            if (listen(sock_fd, BacklogSize) < 0) {
                std::ostringstream msg;
                msg << "Failed to listen on port " << port;
                throw std::runtime_error(msg.str());
            }

            socketManager = SocketManager(sock_fd);
            runLoopThread = std::thread(&Server::runLoop, this);
        }

        void runLoop() {
            while (isRunning.load()) {
                socketManager.pollEvents(clients, requestHandlers);
            }
        }

        void stop() {
            isRunning.store(false);
        }

        void registerRequestHandler(
            std::string const &path, 
            HttpMethod method,
            RequestHandler &&callback
        ) {
            requestHandlers[path].insert(
                std::make_pair(method.id(), std::move(callback))
            );
        }

        std::string host;
        std::uint16_t port;
        int sock_fd;
        SocketManager socketManager;
        std::thread runLoopThread;
        std::atomic<bool> isRunning;
        std::unordered_map<int, ClientData> clients;
        RequestHandlersMap requestHandlers;

        std::thread workerThreads[ThreadPoolSize];
        int workerEpollFd[ThreadPoolSize];
    };

}