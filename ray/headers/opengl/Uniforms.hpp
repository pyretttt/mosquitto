#pragma once

#include "opengl/Shader.hpp"
#include "opengl/glTexture.hpp"
#include "Attributes.hpp"
#include "Core.hpp"

template<>
void inline gl::Shader::setUniform<attributes::UniformCases>(
    std::string const &key,
    attributes::UniformCases const &attr
) const {
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
    }
}

template<>
void inline gl::Shader::setUniform<gl::Material>(
    std::string const &key,
    gl::Material const &material
) const {
    static std::string const keyPrefix = key + ".";
    if (!program) return;
    
    glUseProgram(program);
    auto shinessKey = keyPrefix + "shiness";
    if (auto location = glGetUniformLocation(program, shinessKey.c_str()); location != -1) {
        glUniform1f(location, material.shiness);
    }

    auto ambientColorKey = keyPrefix + "ambientColor";
    this->setUniform<attributes::UniformCases>(ambientColorKey, material.ambientColor);

    for (size_t i = 0; i < material.ambient.size(); i++) {
        auto key = keyPrefix + "ambient" + std::to_string(i);
        if (auto location = glGetUniformLocation(program, key.c_str()); location != -1) {
            glUniform1i(location, i + samplerLocationOffset);
        }
    }


    for (size_t i = 0; i < material.diffuse.size(); i++) {
        auto key = keyPrefix + "diffuse" + std::to_string(i);
        if (auto location = glGetUniformLocation(program, key.c_str()); location != -1) {
            glUniform1i(location, i + material.ambient.size() + samplerLocationOffset);
        }
    }

    for (size_t i = 0; i < material.specular.size(); i++) {
        auto key = keyPrefix + "specular" + std::to_string(i);
        if (auto location = glGetUniformLocation(program, key.c_str()); location != -1) {
            glUniform1i(location, i + material.ambient.size() + material.specular.size() + samplerLocationOffset);
        }
    }

    for (size_t i = 0; i < material.specular.size(); i++) {
        auto key = keyPrefix + "normals" + std::to_string(i);
        if (auto location = glGetUniformLocation(program, key.c_str()); location != -1) {
            glUniform1i(location, i + material.ambient.size() + material.specular.size() + material.diffuse.size() + samplerLocationOffset);
        }
    }
}

template<>
void inline gl::Shader::setUniform<attributes::Transforms>(
    std::string const &key,
    attributes::Transforms const &transforms
) const {
    if (!program) return;
    glUseProgram(program);

    auto const worldKey = key + "." + "worldMatrix";
    setUniform<attributes::UniformCases>(worldKey, transforms.worldMatrix);
    auto const projectionKey = key + "." + "projectionMatrix";
    setUniform<attributes::UniformCases>(projectionKey, transforms.projectionMatrix);
    auto const viewKey = key + "." + "viewMatrix";
    setUniform<attributes::UniformCases>(viewKey, transforms.viewMatrix);
}
