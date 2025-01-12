#pragma once

#include <iostream>

#include "HttpMessage.hpp"
#include "ClientServer.hpp"

inline std::ostream& operator<<(std::ostream& os, http::HttpMessage const &message)
{
    for (auto const &header : message.headers) {
        os << header.first << " : " << header.second << '\n';
    }
    os << message.body;    
    return os;
}

inline std::ostream& operator<<(std::ostream& os, http::HttpRequest const &request)
{
    os << request.method.toString() << " " << request.url << " " << http::httpVersionToString(request.version);
    os << static_cast<http::HttpMessage const &>(request);
    return os;
}
