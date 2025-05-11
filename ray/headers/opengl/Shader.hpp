#pragma once

#include <string>
#include <filesystem>

#include "GL/glew.h"

#include "glCommon.hpp"
#include "Attributes.hpp"
#include "Core.hpp"

namespace gl {
class Shader final {
public:
    ID program;

    Shader(std::filesystem::path vertex, std::filesystem::path fragment, bool eagerInit = false);

    void use();
    void setup();

    void setUniform(
        std::string const &key, 
        attributes::BaseCases const &attr
    ) noexcept;

    ~Shader();

    std::filesystem::path vertex;
    std::filesystem::path fragment;
};


void inline Shader::setUniform(
    std::string const &key, 
    attributes::BaseCases const &attr
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
            [&](iColor const &value) { glUniform1ui(location, value.val); }
        }, attr);
    }
};
}