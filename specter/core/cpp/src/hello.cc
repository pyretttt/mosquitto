#include "hello.h"
#include <iostream>

std::string cpp_hello() {
    return "Hello from C++";
}

void Check::sayHello() {
    std::cout << cpp_hello() << std::endl;
}
