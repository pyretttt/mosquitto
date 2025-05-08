#pragma once

#include <filesystem>
#include <optional>
#include <array>
#include <GL/glew.h>

#include "LoadTextFile.hpp"

namespace gl {
template <size_t VboCount, typename Attribute>
struct RenderObject {
    using ID = unsigned int;

    RenderObject(
        std::array<Attribute, VboCount> vertexArray,
        std::filesystem::path vertexShader,
        std::filesystem::path fragmentShader,
        bool eagerInit,
        bool debug = false
    );

    void setDebug(bool);

    void prepare();
    void bind();
    void unbind();
    void render();

    ~RenderObject();

    ID program = 0;
    ID vbo = 0;

    std::array<Attribute, VboCount> vertexArray;

    std::filesystem::path vertexShaderPath;
    std::filesystem::path fragmentShaderPath;

    bool debug;
};


template <size_t VboCount, typename Attribute>
RenderObject<VboCount, Attribute>::RenderObject(
    std::array<Attribute, VboCount> vertexArray,
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

template <size_t VboCount, typename Attribute>
RenderObject<VboCount, Attribute>::~RenderObject() {
    if (program) glDeleteProgram(program);
}

template <size_t VboCount, typename Attribute>
void RenderObject<VboCount, Attribute>::prepare() {
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

    glGenBuffers(1, &vbo);

}

template <size_t VboCount, typename Attribute>
void RenderObject<VboCount, Attribute>::setDebug(bool debug) {
    this->debug = debug;
}

template <size_t VboCount, typename Attribute>
void RenderObject<VboCount, Attribute>::bind() {
    if (!program) {
        throw std::runtime_error("RENDER_OBJECT::ILLFORMED_PROGRAMM");
    }
    glUseProgram(program);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(
        GL_ARRAY_BUFFER, 
        vertexArray.size() * sizeof(typename decltype(vertexArray)::value_type), 
        vertexArray.data(), 
        GL_STATIC_DRAW
    );
}
}