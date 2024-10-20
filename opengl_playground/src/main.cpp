#include <iostream>

#include "glad/gl.h"
#include "GLFW/glfw3.h"


extern "C" {
void glfwDefaultWindowHints(void);
}
int main() {
    GLFW_ACCUM_ALPHA_BITS;
    glfwDefaultWindowHints();
    return 0;
}