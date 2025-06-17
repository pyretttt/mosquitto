#include <iostream>
#include <utility>

#include "GL/glew.h"

#include "opengl/Shader.hpp"
#include "LoadTextFile.hpp"
#include "Core.hpp"
#include "MathUtils.hpp"


void gl::Shader::use() {
    if (!program) setup();
    if (program) {
        glUseProgram(program);
    } else {
        throw std::runtime_error("ERROR::SHADER::USAGE::UNABLE_TO_USE");
    }
}

gl::Shader::Shader(std::filesystem::path vertex, std::filesystem::path fragment, bool eagerInit) 
    : vertex(std::move(vertex))
    , fragment(std::move(fragment)) {
    if (eagerInit) setup();
}

void gl::Shader::setup() {
    ID vs = glCreateShader(GL_VERTEX_SHADER);
    ID fs = glCreateShader(GL_FRAGMENT_SHADER);
    SCOPE_EXIT((){
        glDeleteShader(vs);
        glDeleteShader(fs);
    });

    auto vsStr = utils::loadTextFile(vertex);
    auto fsStr = utils::loadTextFile(fragment);
    auto vsSource = vsStr.c_str();
    auto fsSource = fsStr.c_str();

    glShaderSource(vs, 1, &vsSource, nullptr);
    glCompileShader(vs);

    int success;
    char infolog[512];
    glGetShaderiv(vs, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vs, 512, NULL, infolog);
        std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED" << std::endl;
        std::cerr << infolog << std::endl;
        memset(infolog, 0, sizeof(infolog));
        throw std::runtime_error("ERROR::SHADER::VERTEX::COMPILATION_FAILED");
    }


    glShaderSource(fs, 1, &fsSource, nullptr);
    glCompileShader(fs);

    glGetShaderiv(fs, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vs, 512, NULL, infolog);
        std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED" << std::endl;
        std::cerr << infolog << std::endl;
        throw std::runtime_error("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED");
    }
    memset(infolog, 0, sizeof(infolog));

    program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    glGetProgramInfoLog(program, 512, nullptr, infolog);
    std::cout << "SHADER_PROGRAM::LINK::RESULT" << std::endl;
    std::cout << infolog << std::endl;
    memset(infolog, 0, sizeof(infolog));
}

gl::Shader::~Shader() {
    if (program) glDeleteProgram(program);
}

void gl::Shader::setUniform(
    std::string const &key, 
    attributes::UniformCases const &attr
) noexcept {
    if (!program) return;
    if (auto location = glGetUniformLocation(program, key.data()); location != -1) {
        glUseProgram(program);
        using namespace attributes;
        std::visit(overload {
            [&](FloatAttr const &value) { glUniform1f(location, value.val); },
            [&](Vec2 const &value) { glUniform2f(location, value.val[0], value.val[1]); },
            [&](Vec3 const &value) { glUniform3f(location, value.val[0], value.val[1], value.val[2]); },
            [&](Vec4 const &value) { glUniform4f(location, value.val[0], value.val[1], value.val[2], value.val[3]); },
            [&](iColor const &value) { glUniform1ui(location, value.val); },
            [&](Mat4 const &value) { glUniformMatrix4fv(location, 1, GL_FALSE, ml::getPtr(value)); },
        }, attr);
        glUseProgram(0);
    }
};

void gl::Shader::setTextureSamplers(size_t max, bool unuseProgram) noexcept {
    if (!program) return;
    glUseProgram(program);
    for (size_t i = 0; i < max; i++) {
        auto key = "texture" + std::to_string(i);
        if (auto location = glGetUniformLocation(program, key.c_str()); location != -1) {
            glUniform1i(location, i);
        }
    }
    if (unuseProgram) glUseProgram(0);
}

void gl::Shader::setMaterialSamplers(gl::Material const &material, bool unuseProgram) noexcept {
    if (!program) return;
    
    glUseProgram(program);
    for (size_t i = 0; i < material.ambient.size(); i++) {
        auto key = "ambient" + std::to_string(i);
        if (auto location = glGetUniformLocation(program, key.c_str()); location != -1) {
            glUniform1i(location, i + samplerLocationOffset);
        }
    }
    for (size_t i = 0; i < material.diffuse.size(); i++) {
        auto key = "diffuse" + std::to_string(i);
        if (auto location = glGetUniformLocation(program, key.c_str()); location != -1) {
            glUniform1i(location, i + material.ambient.size() + samplerLocationOffset);
        }
    }

    for (size_t i = 0; i < material.specular.size(); i++) {
        auto key = "specular" + std::to_string(i);
        if (auto location = glGetUniformLocation(program, key.c_str()); location != -1) {
            glUniform1i(location, i + material.ambient.size() + material.specular.size() + samplerLocationOffset);
        }
    }

    for (size_t i = 0; i < material.specular.size(); i++) {
        auto key = "normals" + std::to_string(i);
        if (auto location = glGetUniformLocation(program, key.c_str()); location != -1) {
            glUniform1i(location, i + material.ambient.size() + material.specular.size() + material.diffuse.size() + samplerLocationOffset);
        }
    }

    if (unuseProgram) glUseProgram(0);
}
