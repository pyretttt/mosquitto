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
        resp.headers.insert({"Server", "Apache"});
        resp.headers.insert({"Host", "127.0.0.1"});
        resp.headers.insert({"Content-Type", "application/json"});
        resp.body = "{\"json\": 123}";
        resp.statusCode = http::HttpStatusCode::Ok;
        resp.version = req.version;
        return resp;
    };
    std::unordered_map<size_t, http::RequestHandler> homeHandlers;
    homeHandlers.insert({http::HttpMethod(http::HttpMethod::Method::GET).id(), homeGetHandler});
    server.requestHandlers.insert({"/home", homeHandlers});
    server.start();
    while (server.isRunning) {
        std::this_thread::sleep_for(1000ms);
    }

    return 0;
}