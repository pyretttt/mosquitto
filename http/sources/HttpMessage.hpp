#pragma once

#include <unordered_map>
#include <string>

namespace http {
    enum class HttpStatusCode: size_t {
        Continue = 100,
        SwitchingProtocols = 101,
        EarlyHints = 103,
        Ok = 200,
        Created = 201,
        Accepted = 202,
        NonAuthoritativeInformation = 203,
        NoContent = 204,
        ResetContent = 205,
        PartialContent = 206,
        MultipleChoices = 300,
        MovedPermanently = 301,
        Found = 302,
        NotModified = 304,
        BadRequest = 400,
        Unauthorized = 401,
        Forbidden = 403,
        NotFound = 404,
        MethodNotAllowed = 405,
        RequestTimeout = 408,
        ImATeapot = 418,
        InternalServerError = 500,
        NotImplemented = 501,
        BadGateway = 502,
        ServiceUnvailable = 503,
        GatewayTimeout = 504,
        HttpVersionNotSupported = 505
    };

    std::string to_string(HttpStatusCode status_code) {
        switch (status_code) {
        case HttpStatusCode::Continue:
            return "Continue";
        case HttpStatusCode::Ok:
            return "OK";
        case HttpStatusCode::Accepted:
            return "Accepted";
        case HttpStatusCode::MovedPermanently:
            return "Moved Permanently";
        case HttpStatusCode::Found:
            return "Found";
        case HttpStatusCode::BadRequest:
            return "Bad Request";
        case HttpStatusCode::Forbidden:
            return "Forbidden";
        case HttpStatusCode::NotFound:
            return "Not Found";
        case HttpStatusCode::MethodNotAllowed:
            return "Method Not Allowed";
        case HttpStatusCode::ImATeapot:
            return "I'm a Teapot";
        case HttpStatusCode::InternalServerError:
            return "Internal Server Error";
        case HttpStatusCode::NotImplemented:
            return "Not Implemented";
        case HttpStatusCode::BadGateway:
            return "Bad Gateway";
        default:
            return std::string();
        }
    }


    enum class HttpVersion {
        HTTP_0_9 = 9,
        HTTP_1_0 = 10,
        HTTP_1_1 = 11,
        HTTP_2_0 = 20
    };

    std::string httpVersionToString(HttpVersion const &version) {
        switch (version) {
        case HttpVersion::HTTP_0_9:
            return "HTTP/0.9";
        case HttpVersion::HTTP_1_0:
            return "HTTP/1.0";
        case HttpVersion::HTTP_1_1:
            return "HTTP/1.1";
        case HttpVersion::HTTP_2_0:
            return "HTTP/2.0";
        default:
            return std::string();
        }
    }

    HttpVersion httpVersionFromString(std::string const &version) {
        std::string version_string_uppercase;
        std::transform(version.begin(), version.end(),
                        std::back_inserter(version_string_uppercase),
                        [](char c) { return toupper(c); });

        if (version_string_uppercase == "HTTP/0.9") {
            return HttpVersion::HTTP_0_9;
        } else if (version_string_uppercase == "HTTP/1.0") {
            return HttpVersion::HTTP_1_0;
        } else if (version_string_uppercase == "HTTP/1.1") {
            return HttpVersion::HTTP_1_1;
        } else if (version_string_uppercase == "HTTP/2" ||
                    version_string_uppercase == "HTTP/2.0") {
            return HttpVersion::HTTP_2_0;
        } else {
            throw std::invalid_argument("Unexpected HTTP version");
        }
    }
    
    using Headers = std::unordered_map<std::string, std::string>;

    struct HttpMethod {
        enum class Method : size_t {
            GET,
            HEAD,
            POST,
            PUT,
            DELETE,
            CONNECT,
            OPTIONS,
            TRACE,
            PATCH
        };

        HttpMethod(Method m) : value(m) {}

        HttpMethod static fromString(std::string const &string) {
            if (string == "GET") {
                return HttpMethod(Method::GET);
            } else if (string == "HEAD") {
                return HttpMethod(Method::HEAD);
            } else if (string == "POST") {
                return HttpMethod(Method::POST);
            } else if (string == "PUT") {
                return HttpMethod(Method::PUT);
            } else if (string == "DELETE") {
                return HttpMethod(Method::DELETE);
            } else if (string == "CONNECT") {
                return HttpMethod(Method::CONNECT);
            } else if (string == "OPTIONS") {
                return HttpMethod(Method::OPTIONS);
            } else if (string == "TRACE") {
                return HttpMethod(Method::TRACE);
            } else if (string == "PATCH") {
                return HttpMethod(Method::PATCH);
            }
            throw std::invalid_argument("Unexpected http method");
        }

        std::string toString() const {
            switch (value) {
            case Method::GET:
                return "GET";
            case Method::HEAD:
                return "HEAD";
            case Method::POST:
                return "POST";
            case Method::PUT:
                return "PUT";
            case Method::DELETE:
                return "DELETE";
            case Method::CONNECT:
                return "CONNECT";
            case Method::OPTIONS:
                return "OPTIONS";
            case Method::TRACE:
                return "TRACE";
            case Method::PATCH:
                return "PATCH";
            default:
                return "";
            }
        }

    size_t id() const {
        return static_cast<size_t>(value);
    }

    private:
        Method value;
    };

    struct HttpMessage {
        HttpMessage() : version(HttpVersion::HTTP_1_1) {}

        HttpVersion version;
        Headers headers;
        std::string body;
    };

    struct HttpRequest final : public HttpMessage {
        HttpRequest() = default;
        HttpRequest(std::string const &url, HttpMethod method) 
            : method(method),
            url(url) {}

        HttpMethod method = HttpMethod(HttpMethod::Method::DELETE);
        std::string url;
    };

    struct HttpResponse final : public HttpMessage {
        HttpResponse() {}

        HttpStatusCode statusCode = HttpStatusCode::Ok;
    };
}