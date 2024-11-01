#include <iostream>
#include <functional>


struct Request {};
struct Response {};

using ResponseCallback = std::function<void(Response)>;

struct HttpClient {
    void virtual Get(Request request) const = 0;
    void virtual Post(Request request) const = 0;
    // void virtual Put(Request request) const = 0;
    // void virtual Patch(Request request) const = 0;
    virtual ~HttpClient() {};
};

struct HttpClientImpl: public virtual HttpClient {
    HttpClientImpl(std::string schema, std::string host, std::string port) : 
        HttpClient(),
        schema(schema), 
        host(host),
        port(port) {}

    void Get(Request request) const override {

    };
    void Post(Request request) const override {

    };


    ~HttpClientImpl() {
        std::cout << "deinited" << std::endl;
    }

    std::string schema;
    std::string host;
    std::string port;
private:

};

int main() {
    auto client = HttpClientImpl("123", "123", "123");
    return 0;
}