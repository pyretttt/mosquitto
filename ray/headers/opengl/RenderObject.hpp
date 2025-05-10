#pragma once

#include <filesystem>
#include <optional>
#include <array>
#include <type_traits>

#include <GL/glew.h>

#include "LoadTextFile.hpp"
#include "glAttribute.hpp"
#include "Core.hpp"

namespace gl {

using EBO = std::vector<unsigned int>;

struct Configuration {
    struct PolygonMode {
        GLenum face = GL_FRONT;
        GLenum mode = GL_FILL;
    };

    PolygonMode polygonMode;
};

template <typename Attribute>
struct RenderObject {
    using ID = unsigned int;

    RenderObject(
        std::vector<Attribute> vertexArray,
        EBO vertexArrayIndices,
        Configuration configuration,
        std::optional<std::filesystem::path> vertexShader,
        std::optional<std::filesystem::path> fragmentShader,
        bool eagerInit,
        bool debug = false
    );

    RenderObject(RenderObject<Attribute> const &) = delete;
    RenderObject<Attribute>& operator=(RenderObject<Attribute> const&) = delete;

    RenderObject(RenderObject<Attribute> &&);
    RenderObject<Attribute>& operator=(RenderObject<Attribute>&& other);

    void setDebug(bool) noexcept;

    void setUniform(std::string const &key, attributes::BaseCases const &attr) noexcept;

    void prepare();
    void render() noexcept;

    ~RenderObject();

    ID program = 0;
    ID vbo = 0;
    ID vao = 0;
    ID ebo = 0;

    std::vector<Attribute> vertexArray;
    EBO vertexArrayIndices;

    std::optional<std::filesystem::path> vertexShaderPath;
    std::optional<std::filesystem::path> fragmentShaderPath;

    Configuration configuration;

    bool debug;
};

template<typename Attribute>
RenderObject<Attribute>::RenderObject(
    std::vector<Attribute> vertexArray,
    EBO vertexArrayIndices,
    Configuration configuration,
    std::optional<std::filesystem::path> vertexShader,
    std::optional<std::filesystem::path> fragmentShader,
    bool eagerInit,
    bool debug
) 
    : vertexShaderPath(vertexShader)
    , fragmentShaderPath(fragmentShader)
    , configuration(configuration)
    , vertexArray(std::move(vertexArray))
    , vertexArrayIndices(std::move(vertexArrayIndices)) {
    setDebug(debug);
    if (eagerInit) {
        prepare();
    }
}

template<typename Attribute>
RenderObject<Attribute>::RenderObject(RenderObject<Attribute> &&other)
    : vertexShaderPath(std::move(other.vertexShader))
    , fragmentShaderPath(std::move(other.fragmentShader))
    , vertexArray(std::move(vertexArray))
    , vertexArrayIndices(std::move(vertexArrayIndices))
    , vao(other.vao)
    , ebo(other.ebo)
    , vbo(other.vbo)
    , program(other.program)
    , debug(other.debug)
    , configuration(other.configuration)
{
    other.program = 0;
    other.vbo = 0;
    other.ebo = 0;
    other.vao = 0;
}

template<typename Attribute>
RenderObject<Attribute>& RenderObject<Attribute>::operator=(RenderObject<Attribute>&& other) {
    this->vertexShaderPath = std::move(other.vertexShader);
    this->fragmentShader = std::move(other.fragmentShader);
    this->vertexArray = std::move(other.vertexArray);
    this->vertexArrayIndices = std::move(other.vertexArrayIndices);
    this->vao = other.vao;
    this->vbo = other.vbo;
    this->ebo = other.ebo;
    this->program = other.program;
    this->debug = debug;
    this->configuration = configuration;
    other.vao = 0;
    other.ebo = 0;
    other.vbo = 0;
    other.program = 0;
}

template<typename Attribute>
RenderObject<Attribute>::~RenderObject() {
    if (program) glDeleteProgram(program);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (ebo) glDeleteBuffers(1, &ebo);
    if (vao) glDeleteVertexArrays(1, &vao);
}

template<typename Attribute>
void RenderObject<Attribute>::prepare() {
    if (vertexShaderPath && fragmentShaderPath) {
        program = glCreateProgram();

        ID vs = glCreateShader(GL_VERTEX_SHADER);
        ID fs = glCreateShader(GL_FRAGMENT_SHADER);
        auto vsStr = utils::loadTextFile(vertexShaderPath.value());
        auto fsStr = utils::loadTextFile(fragmentShaderPath.value());
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
    }

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(
        GL_ARRAY_BUFFER, 
        vertexArray.size() * sizeof(Attribute),
        vertexArray.data(), 
        GL_STATIC_DRAW
    );
    bindAttributes<Attribute>();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER, 
        sizeof(EBO::value_type) * vertexArrayIndices.size(),
        vertexArrayIndices.data(),
        GL_STATIC_DRAW
    );

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

template<typename Attribute>
void RenderObject<Attribute>::setDebug(bool debug) noexcept {
    this->debug = debug;
}

template<typename Attribute> 
void RenderObject<Attribute>::setUniform(
    std::string const &key, 
    attributes::BaseCases const &attr
) noexcept {
    if (!program) return;
    if (auto location = glGetUniformLocation(program, key.data()); location != -1) {
        glUseProgram(program);        
        using namespace attributes;
        std::visit(overload {
            [&](FloatAttr const &value) { glUniform1f(location, value.val); },
            [&](Vec2 const &value) { glUniform2f(location, value.val[0], valu∆e.val[1]); },
            [&](Vec3 const &value) { glUniform3f(location, value.val[0], value.val[1], value.val[2]); },
            [&](Vec4 const &value) { glUniform4f(location, value.val[0], value.val[1], value.val[2], value.val[3]); },
            [&](iColor const &value) { glUniform1ui(location, value.val); }
        }, attr);
    }
};

template<typename Attribute>
void RenderObject<Attribute>::render() noexcept {
    if (program) {
        glUseProgram(program);
    }
    glBindVertexArray(vao);
    glPolygonMode(configuration.polygonMode.face, configuration.polygonMode.mode);
    glDrawElements(GL_TRIANGLES, vertexArrayIndices.size(), GL_UNSIGNED_INT, 0);

    glBindVertexArray(0);
    if (program) {
        glUseProgram(0);
    }
}
}