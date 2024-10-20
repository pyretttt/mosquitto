#include <iostream>

#include "glad/gl.h"
#include "GLFW/glfw3.h"


struct T {
    static std::string name;
};

std::string T::name = "Jack";

int main() {
    GLFW_ACCUM_ALPHA_BITS;
    glfwDefaultWindowHints();

    T obj;
    obj.name;

    return 0;
}