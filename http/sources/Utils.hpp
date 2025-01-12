#pragma once

#include <iostream>

#include "HttpMessage.hpp"
#include "ClientServer.hpp"

inline std::ostream& operator<<(std::ostream& os, http::HttpMessage const &message)
{
    for (auto const &header : message.headers) {
        os << header.first << ": " << header.second << "\r\n";
    }
    os << "\r\n";
    os << message.body;    
    return os;
}

inline std::ostream& operator<<(std::ostream& os, http::HttpRequest const &request)
{
    os << request.method.toString() << " " << request.url << " " << http::httpVersionToString(request.version) << "\r\n";
    os << static_cast<http::HttpMessage const &>(request);
    return os;
}

inline std::ostream& operator<<(std::ostream& os, http::HttpResponse const &response) {
    os << http::httpVersionToString(response.version) << " " 
        << static_cast<size_t>(response.statusCode) << " "
        << http::to_string(response.statusCode) << "\r\n";
    os << static_cast<http::HttpMessage const &>(response);
    return os;
}