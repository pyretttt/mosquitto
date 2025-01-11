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

#include "HttpMessage.hpp"

namespace http {
    constexpr size_t MaxBufferSize = 4096;
    constexpr size_t ThreadPoolSize = 4;
    constexpr size_t BacklogSize = 1000;

    struct ClientData {
        ClientData() : fd(-1), length(0), cursor(0), buffer() {}
        int fd;
        size_t length;
        size_t cursor;
        char buffer[MaxBufferSize];
    };

    struct ReaderStream final {
        ReaderStream() : dataBuff() {
            dataBuff.reserve(MaxBufferSize);
        }

        void process(char *data, size_t n) {
            dataBuff.insert(dataBuff.end(), data, data + n);
        }

        void reset() {
            headersComplete = false;
        }

        std::vector<char> dataBuff;
        std::queue<HttpRequest> requests;

        bool headersComplete = false;
        
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
            std::unordered_map<std::string, std::unordered_map<HttpMethod, RequestHandler>> const &requestHandlers
        ) {
            struct kevent evList[BacklogSize];
            int nevent = kevent(kq, nullptr, 0, evList, BacklogSize, nullptr);
            if (nevent < 0) {
                std::runtime_error("Failed to poll events");
            }
            struct sockaddr_storage addr;
            socklen_t socklen = sizeof(addr);
            for (size_t i = 0; i < nevent; i++) {
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

                }
            }
        }

        void handleReadWrite(
            struct kevent event,
            std::unordered_map<int, ClientData> &clients,
            std::unordered_map<std::string, std::unordered_map<HttpMethod, RequestHandler>> const &requestHandlers
        ) {
            if (event.filter == EVFILT_READ) {
                auto &clientData = clients.at(event.ident);
                ssize_t nbyte = recv(event.ident, clientData.buffer, MaxBufferSize, 0);

                if (nbyte > 0) {

                }

            } else if (event.filter == EVFILT_WRITE) {

            } else {
                throw std::runtime_error("Assertion error");
            }
        }

        int kq;
        int connectionSock;
    };


    using RequestHandler = std::function<HttpResponse (HttpRequest const &)>;


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

            if (bind(sock_fd, &servAddr, sizeof(servAddr)) < 0) {
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
                socketManager.pollEvents();
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
                std::make_pair(method, std::move(callback))
            );
        }

        std::string host;
        std::uint16_t port;
        int sock_fd;
        SocketManager socketManager;
        std::thread runLoopThread;
        std::atomic<bool> isRunning;
        std::unordered_map<int, ClientData> clients;
        std::unordered_map<std::string, std::unordered_map<HttpMethod, RequestHandler>> requestHandlers;

        std::thread workerThreads[ThreadPoolSize];
        int workerEpollFd[ThreadPoolSize];
    };

}