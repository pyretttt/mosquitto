#include "hello.hpp"

#include <iostream>

#include <opencv2/core/version.hpp>

std::string cpp_hello() {
    return "Hello from C++";
}

void Check::sayHello() {
    std::cout << cpp_hello() << std::endl;
}
