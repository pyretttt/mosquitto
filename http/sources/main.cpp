#include <thread>
#include <chrono>
#include <unordered_map>

#include "ClientServer.hpp"
#include "HttpMessage.hpp"
#include "Utils.hpp"

int main() {
    using namespace std::chrono_literals;

    auto server = http::Server("127.0.0.1", 3000);
    http::RequestHandler homeGetHandler = [](http::HttpRequest const &req){
        std::cout << req;
        auto resp = http::HttpResponse();
        resp.body = req.body;
        resp.headers = req.headers;
        resp.statusCode = http::HttpStatusCode::Ok;
        resp.version = req.version;
        return resp;
    };
    // std::unordered_map<http::HttpMethod, http::RequestHandler> homeHandlers;
    // homeHandlers.insert({http::HttpMethod::Method::GET, homeGetHandler});
    // server.requestHandlers.insert({"/home", homeHandlers});

    while (server.isRunning) {
        std::this_thread::sleep_for(1000ms);
    }

    return 0;
}