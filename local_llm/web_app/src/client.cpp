#include "client.h"

HttpClient::HttpClient(std::string schema, std::string host, std::optional<int> port) 
    : schema(schema), 
    host(host), 
    port(port.value_or(schema == "https" ? 443 : 80)) {}

void HttpClient::send(Request request, ResponseCompletion completion) {
    
}