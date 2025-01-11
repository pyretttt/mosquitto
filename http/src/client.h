#pragma once

#include <functional>
#include <optional>

#include "common.h"

using ResponseCompletion = std::function<void(Response)>;

struct HttpClient {
    HttpClient(std::string schema, std::string host, std::optional<int> port = std::nullopt);
    void send(Request request, ResponseCompletion completion);

    std::string schema;
    std::string host;
    int port;
};