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
#include <unistd.h>

#include "HttpMessage.hpp"
#include "Utils.hpp"

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
        MessageProcessor() : readDataBuff() {
            readDataBuff.reserve(MaxBufferSize);
        }

        std::optional<HttpRequest> processRead(char *data, size_t n) {
            std::optional<HttpRequest> ret;

            readDataBuff.append(data, n);
            size_t headersEnd = readDataBuff.find("\r\n\r\n");
            if (headersEnd == std::string::npos) {
                return ret;
            } else if (!filledHeaders) {
                std::string_view sv = std::string_view(readDataBuff).substr(0, headersEnd + 4);
                readStartLineAndHeaders(sv);
                filledHeaders = true;
            }

            if (request.headers.find("Content-Length") != request.headers.end()) {
                size_t contentLength = static_cast<size_t>(std::stoi(request.headers["Content-Length"]));
                auto contentReceived = readDataBuff.length() - (headersEnd + 4);
                if (contentReceived >= contentLength) {
                    request.body = readDataBuff.substr(headersEnd + 4, contentLength);
                    ret = request;
                    reset(headersEnd + 4 + contentLength);
                } else {
                    ret = request;
                    reset(headersEnd + 4);
                }
            } else {
                ret = request;
                reset(headersEnd + 4);
            }
            return ret;
        }

        void reset(size_t bufferPos) {
            filledHeaders = false;
            request = HttpRequest();
            readDataBuff.erase(readDataBuff.begin() + bufferPos);
        }

        void readStartLineAndHeaders(std::string_view &startLineAndHeaders) {
            size_t start = 0;
            size_t end = 0;
            bool hasReadStartLine = false;
            while ((end = startLineAndHeaders.find("\r\n", start)) != std::string_view::npos) {
                auto substr = startLineAndHeaders.substr(start, end - start);
                
                if (substr.empty()) {
                    start = end + 2;
                    continue; 
                }

                if (hasReadStartLine) {
                    readHeader(substr);
                } else {
                    readStartLine(substr);
                    hasReadStartLine = true;
                }
                start = end + 2;
            }
        }

        void readStartLine(std::string_view &startLine) {
            size_t start = 0;
            size_t end = 0;
            size_t i = 0;
            while ((end = startLine.find(" ", start)) != std::string_view::npos) {
                auto substr = startLine.substr(start, end - start);
                switch (i) {
                case 0:
                    request.method = HttpMethod::fromString(std::string(substr));
                    break;
                case 1:
                    request.url = std::string(substr);
                    break;
                default:
                    throw std::runtime_error("Wrong starting line of http request");
                }
                i++;
                start = end + 1;
            }
            request.version = httpVersionFromString(std::string(startLine.substr(start)));
        }

        void readHeader(std::string_view &header) {
            auto semicolonPos = header.find(':');
            if (semicolonPos == std::string_view::npos) {
                throw std::runtime_error("Wrong header format");
            }
            std::string key = std::string(trim(header.substr(0, semicolonPos)));
            std::string value = std::string(trim(header.substr(semicolonPos + 1)));
            request.headers.insert({key, value});
        }

        int processWrite(int sock_fd) {
            int nbyte = send(sock_fd, writeDataBuff.data(), writeDataBuff.size(), 0);
            if (nbyte < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    return 0;
                }

                throw std::runtime_error("Socket write error");
            }
            writeDataBuff.erase(nbyte);
            return nbyte;
        }

        std::string readDataBuff;
        std::string writeDataBuff;
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
        int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
        int flags = fcntl(sock_fd, F_GETFL, 0);
        fcntl(sock_fd, F_SETFL, flags | O_NONBLOCK);

        if (sock_fd < 0) {
            throw std::runtime_error("Failed to create socket");
        }
        return sock_fd;
    }

    struct SocketManager {
        SocketManager() = default;
        SocketManager(int connectionSocket) : kq(kqueue()), connectionSock(connectionSocket) {
            registerReadSock(connectionSocket);
        }

        void registerReadSock(int sock) {
            struct kevent evSet;
            EV_SET(&evSet, sock, EVFILT_READ, EV_ADD, 0, 0, nullptr);
            if (kevent(kq, &evSet, 1, nullptr, 0, nullptr) == -1) { // Adds event to kqueue
                throw std::runtime_error("Failed to register read event");
            }
        }

        void registerWriteSock(int sock) {
            struct kevent evSet;
            EV_SET(&evSet, sock, EVFILT_WRITE, EV_ADD, 0, 0, NULL);
            if (kevent(kq, &evSet, 1, nullptr, 0, nullptr) == -1) { // Adds event to kqueue
                throw std::runtime_error("Failed to register write event");
            }
        }

        void unregisterReadSock(int sock) {
            struct kevent evSet;
            EV_SET(&evSet, sock, EVFILT_READ, EV_DELETE, 0, 0, nullptr);
            if (kevent(kq, &evSet, 1, nullptr, 0, nullptr) == -1) { // Adds event to kqueue
                throw std::runtime_error("Failed to unregister read event");
            }
        }

        void unregisterWriteSock(int sock) {
            struct kevent evSet;
            EV_SET(&evSet, sock, EVFILT_WRITE, EV_DELETE, 0, 0, nullptr);
            if (kevent(kq, &evSet, 1, nullptr, 0, nullptr) == -1) { // Adds event to kqueue
                std::cerr << "Failed to unregister write event" << std::endl;
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
                    int flags = fcntl(clientSocket, F_GETFL, 0);
                    if (clientSocket < 0 || fcntl(clientSocket, F_SETFL, flags | O_NONBLOCK) < 0) {
                        std::runtime_error("Failed to accept client socket");
                    }
                    auto clientData = ClientData();
                    clientData.fd = clientSocket;
                    clients[clientSocket] = clientData;
                    registerReadSock(clientSocket);
                } else if (evList[i].flags & EV_EOF) {
                    int fd = evList[i].ident;
                    unregisterReadSock(fd);
                    unregisterWriteSock(fd);
                    clients.erase(fd);
                    close(fd);
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
                                std::stringstream ss;
                                ss << resp;
                                clientData.messageProcessor.writeDataBuff = ss.str();
                                registerWriteSock(event.ident);
                            } catch (std::out_of_range &err) {
                                std::cerr << "Can not find request handler" << std::endl;
                            }
                        }
                    } catch(std::runtime_error &error) {
                        std::cerr << error.what() << std::endl;
                        auto resp = HttpResponse();
                        resp.statusCode = HttpStatusCode::BadRequest;
                    }
                }

            } else if (event.filter == EVFILT_WRITE) {
                auto &clientData = clients.at(event.ident);
                size_t toWrite = clientData.messageProcessor.writeDataBuff.size();
                size_t nbyte = clientData.messageProcessor.processWrite(event.ident);
                unregisterWriteSock(event.ident);
                if (nbyte < toWrite) {
                    registerWriteSock(event.ident);
                    return;
                }
                close(event.ident);
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
            isRunning(false) {};
        
        void start() {
            isRunning.store(true);
            
            sock_fd = createNonBlockingSocket();
            int opt = 1;
            if (setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR /*| SO_REUSEPORT*/, &opt, sizeof(opt)) < 0) {
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