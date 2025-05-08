#pragma once

#include <filesystem>
#include <optional>
#include <array>
#include <span>

#include <GL/glew.h>

#include "LoadTextFile.hpp"
#include "glAttribute.hpp"

namespace gl {
template <typename Attribute>
struct RenderObject {
    using ID = unsigned int;

    RenderObject(
        std::vector<Attribute> vertexArray,
        std::filesystem::path vertexShader,
        std::filesystem::path fragmentShader,
        bool eagerInit,
        bool debug = false
    );

    void setDebug(bool);

    void prepare();
    void render();

    ~RenderObject();

    ID program = 0;
    ID vbo = 0;
    ID vao = 0;

    std::vector<Attribute> vertexArray;

    std::filesystem::path vertexShaderPath;
    std::filesystem::path fragmentShaderPath;

    bool debug;
};

template<typename Attribute>
RenderObject<Attribute>::RenderObject(
    std::vector<Attribute> vertexArray,
    std::filesystem::path vertexShader,
    std::filesystem::path fragmentShader,
    bool eagerInit,
    bool debug
) 
    : vertexShaderPath(vertexShader)
    , fragmentShaderPath(fragmentShader)
    , vertexArray(vertexArray) {
    setDebug(debug);
    if (eagerInit) {
        prepare();
    }
}

template<typename Attribute>
RenderObject<Attribute>::~RenderObject() {
    if (program) glDeleteProgram(program);
    if (vbo) glDeleteBuffers(1, &vbo);
}

template<typename Attribute>
void RenderObject<Attribute>::prepare() {
    program = glCreateProgram();

    ID vs = glCreateShader(GL_VERTEX_SHADER);
    ID fs = glCreateShader(GL_FRAGMENT_SHADER);
    auto vsStr = utils::loadTextFile(vertexShaderPath);
    auto fsStr = utils::loadTextFile(fragmentShaderPath);
    auto vsSource = vsStr.c_str();
    auto fsSource = fsStr.c_str();

    glShaderSource(vs, 1, &vsSource, nullptr);
    glCompileShader(vs);

    int success;
    char infolog[512];
    if (debug) {
        glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(vs, 512, NULL, infolog);
            std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED" << std::endl;
            std::cerr << infolog << std::endl;
            memset(infolog, 0, sizeof(infolog));
            throw std::runtime_error("ERROR::SHADER::VERTEX::COMPILATION_FAILED");
        }
    }

    glShaderSource(fs, 1, &fsSource, nullptr);
    glCompileShader(fs);

    if (debug) {
        glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(vs, 512, NULL, infolog);
            std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED" << std::endl;
            std::cerr << infolog << std::endl;
            throw std::runtime_error("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED");
        }
        memset(infolog, 0, sizeof(infolog));
    }

    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    if (debug) {
        glGetProgramInfoLog(program, 512, nullptr, infolog);
        std::cout << "SHADER_PROGRAM::LINK::RESULT" << std::endl;
        std::cout << infolog << std::endl;
        memset(infolog, 0, sizeof(infolog));
    }

    glDeleteShader(vs);
    glDeleteShader(fs);

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(
        GL_ARRAY_BUFFER, 
        vertexArray.size() * sizeof(Attribute),
        vertexArray.data(), 
        GL_STATIC_DRAW
    );
    bindAttributes<Attribute>();

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

template<typename Attribute>
void RenderObject<Attribute>::setDebug(bool debug) {
    this->debug = debug;
}

template<typename Attribute>
void RenderObject<Attribute>::render() {
    if (!program) {
        throw std::runtime_error("RENDER_OBJECT::ILLFORMED_PROGRAMM");
    }
    glUseProgram(program);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, vertexArray.size());

    glBindVertexArray(0);
    glUseProgram(0);
}
}