#include <iostream>
#include <algorithm>

#define GLAD_GL_IMPLEMENTATION
#include "glad/gl.h"
#undef GLAD_GL_IMPLEMENTATION
#include "GLFW/glfw3.h"

#include "Shader.h"
#include "stb_image_impl.h"

unsigned int VBO;
unsigned int VAO;
unsigned int EBO;
float vertices[] = {
    // positions       // colors         // uv
    0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 0.f, 0.f, // bottom right
    -0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.f, 0.f, // bottom left
    0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 1.0f, 1.f, 1.f, // top left
};
unsigned int indices[] = { // note that we start from 0!
    0, 1, 2, // first triangle
    // 1, 2, 3 // second triangle
};
float texCoords[] = {
    0.0f, 0.f,
    1.f, 0.f,
    0.5f, 1.f
};

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

float mix = 0.f;

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }
    else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        mix = std::max<float>(0.f, mix - 0.01);
    } else if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        mix = std::min<float>(1.f, mix + 0.01);
    }
}

const char *vertexShaderSource = "#version 410 core\n"
"layout (location = 0) in vec3 aPos;\n"
"layout (location = 1) in vec3 color;\n"
"out vec4 vertexColor;\n"
"void main() {\n"
"vertexColor = vec4(color, 1.0);\n"
"gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"}\n";

const char *fragmentShaderSource = "#version 410 core\n"
"in vec4 vertexColor;\n"
"uniform vec4 ourColor;\n"
"out vec4 FragColor;\n"
"void main() {\n"
"FragColor = vertexColor;\n"
"}\n";

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow *window = glfwCreateWindow(800, 600, "LearnOpenGL", nullptr, nullptr);
    if (window == nullptr) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGL(glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // VBO
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), &vertices, GL_STATIC_DRAW);
    // VAO
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    // EBO
    glGenBuffers(1, &EBO); // Created and filled as VBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // ====ATTRIBUTES BINDING====
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,  8 * sizeof(float), (void *) (3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void *) (6 * sizeof(float)));
    glEnableVertexAttribArray(2);
    // ====ATTRIBUTES BINDING====
   
    // =====Load Texture=====
    stbi_set_flip_vertically_on_load(true);
    int w, h, c;
    unsigned char *data = stbi_load("resources/brick_tex.jpg", &w, &h, &c, 0);
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);

    stbi_image_free(data);

    data = stbi_load("resources/monolith.jpg", &w, &h, &c, 0);
    unsigned int texture2;
    glGenTextures(1, &texture2);
    glBindTexture(GL_TEXTURE_2D, texture2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);

    stbi_image_free(data);
    // =====Load Texture=====

    glBindVertexArray(0);             // unbind VAO
    glBindBuffer(GL_VERTEX_ARRAY, 0); // unbind VBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0); // unbind EBO
    glBindTexture(GL_TEXTURE_2D, 0); // unbind texture

    Shader shader{"shader.vs", "shader.fs"};
    shader.use();
    shader.setUniform("texture0", 0);
    shader.setUniform("texture1", 1);

    while(!glfwWindowShouldClose(window)) {
        processInput(window);
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // shader.use();
        shader.setUniform<float>("xShift", 0.5);
        shader.setUniform<float>("uMix", mix);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture2);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, sizeof(indices) / sizeof(*indices), GL_UNSIGNED_INT, 0);
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}