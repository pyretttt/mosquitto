#pragma once

#include "opengl/Shader.hpp"
#include "opengl/glMath.hpp"
#include "opengl/glTexture.hpp"
#include "Attributes.hpp"
#include "Light.hpp"
#include "Core.hpp"

template<>
void inline gl::Shader::setUniform<attributes::FloatAttr>(
    std::string const &key,
    attributes::FloatAttr const &attr
) const {
    glUseProgram(program);
    if (auto location = glGetUniformLocation(program, key.data()); location != -1) {
        glUniform1f(location, attr.val);
    }
}

template<>
void inline gl::Shader::setUniform<attributes::IntegerAttr>(
    std::string const &key,
    attributes::IntegerAttr const &attr
) const {
    glUseProgram(program);
    if (auto location = glGetUniformLocation(program, key.data()); location != -1) {
        glUniform1f(location, attr.val);
    }
}

template<>
void inline gl::Shader::setUniform<attributes::Vec2>(
    std::string const &key,
    attributes::Vec2 const &attr
) const {
    glUseProgram(program);
    if (auto location = glGetUniformLocation(program, key.data()); location != -1) {
        glUniform2f(location, attr.val[0], attr.val[1]);
    }
}

template<>
void inline gl::Shader::setUniform<attributes::Vec3>(
    std::string const &key,
    attributes::Vec3 const &attr
) const {
    glUseProgram(program);
    if (auto location = glGetUniformLocation(program, key.data()); location != -1) {
        glUniform3f(location, attr.val[0], attr.val[1], attr.val[2]);
    }
}

template<>
void inline gl::Shader::setUniform<attributes::Vec4>(
    std::string const &key,
    attributes::Vec4 const &attr
) const {
    glUseProgram(program);
    if (auto location = glGetUniformLocation(program, key.data()); location != -1) {
        glUniform4f(location, attr.val[0], attr.val[1], attr.val[2], attr.val[3]);
    }
}

template<>
void inline gl::Shader::setUniform<attributes::iColor>(
    std::string const &key,
    attributes::iColor const &attr
) const {
    glUseProgram(program);
    if (auto location = glGetUniformLocation(program, key.data()); location != -1) {
        glUniform1ui(location, attr.val);
    }
}

template<>
void inline gl::Shader::setUniform<attributes::PositionWithColor>(
    std::string const &key,
    attributes::PositionWithColor const &attr
) const {
    glUseProgram(program);
    auto const positionKey = key + "." + "position";
    setUniform<attributes::Vec3>(positionKey, attr.position);
    auto const colorKey = key + "." + "color";
    setUniform<attributes::Vec4>(colorKey, attr.color);
}

template<>
void inline gl::Shader::setUniform<attributes::PositionWithTex>(
    std::string const &key,
    attributes::PositionWithTex const &attr
) const {
    glUseProgram(program);
    auto const positionKey = key + "." + "position";
    setUniform<attributes::Vec3>(positionKey, attr.position);
    auto const colorKey = key + "." + "texture";
    setUniform<attributes::Vec2>(colorKey, attr.tex);
}

template<>
void inline gl::Shader::setUniform<attributes::Mat4>(
    std::string const &key,
    attributes::Mat4 const &attr
) const {
    glUseProgram(program);
    if (auto location = glGetUniformLocation(program, key.data()); location != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, gl::getPtr(attr));
    }
}

template<>
void inline gl::Shader::setUniform<attributes::Transforms>(
    std::string const &key,
    attributes::Transforms const &attr
) const {
    glUseProgram(program);
    auto const worldKey = key + "." + "worldMatrix";
    setUniform<attributes::Mat4>(worldKey, attr.worldMatrix);
    auto const projectionKey = key + "." + "projectionMatrix";
    setUniform<attributes::Mat4>(projectionKey, attr.projectionMatrix);
    auto const viewKey = key + "." + "viewMatrix";
    setUniform<attributes::Mat4>(viewKey, attr.viewMatrix);
}


template<>
void inline gl::Shader::setUniform<attributes::MaterialVertex>(
    std::string const &key,
    attributes::MaterialVertex const &attr
) const {
    glUseProgram(program);
    auto const positionKey = key + "." + "position";
    setUniform<attributes::Vec3>(positionKey, attr.position);
    auto const normalKey = key + "." + "normal";
    setUniform<attributes::Vec3>(normalKey, attr.normal);
    auto const texKey = key + "." + "tex";
    setUniform<attributes::Vec2>(texKey, attr.tex);
}

template<>
void inline gl::Shader::setUniform<attributes::Cases>(
    std::string const &key,
    attributes::Cases const &attr
) const {
    if (!program) return;
    if (auto location = glGetUniformLocation(program, key.data()); location != -1) {
        using namespace attributes;
        std::visit(overload {
            [&](auto const &value) { setUniform(key, value); }
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
    this->setUniform<attributes::Cases>(ambientColorKey, material.ambientColor);

    for (size_t i = 0; i < material.ambient.size(); i++) {
        auto key = keyPrefix + "ambient" + std::to_string(i);
        if (auto location = glGetUniformLocation(program, key.c_str()); location != -1) {
            glUniform1i(location, material.ambient.at(i)->unitIndex);
        }
    }


    for (size_t i = 0; i < material.diffuse.size(); i++) {
        auto key = keyPrefix + "diffuse" + std::to_string(i);
        if (auto location = glGetUniformLocation(program, key.c_str()); location != -1) {
            glUniform1i(location, i + material.diffuse.at(i)->unitIndex);
        }
    }

    for (size_t i = 0; i < material.specular.size(); i++) {
        auto key = keyPrefix + "specular" + std::to_string(i);
        if (auto location = glGetUniformLocation(program, key.c_str()); location != -1) {
            glUniform1i(location, i + material.specular.at(i)->unitIndex);
        }
    }

    for (size_t i = 0; i < material.normals.size(); i++) {
        auto key = keyPrefix + "normals" + std::to_string(i);
        if (auto location = glGetUniformLocation(program, key.c_str()); location != -1) {
            glUniform1i(location, i + material.normals.at(i)->unitIndex);
        }
    }
}


template<>
void inline gl::Shader::setUniform<LightSource>(
    std::string const &key,
    LightSource const &light
) const {
    if (!program) return;
    glUseProgram(program);

    auto const positionKey = key + "." + "position";
    setUniform<attributes::Cases>(positionKey, attributes::Vec3{ {light.position.x, light.position.y, light.position.z} });
    auto const spotDirectionKey = key + "." + "spotDirection";
    setUniform<attributes::Cases>(spotDirectionKey, attributes::Vec3{ {light.spotDirection.x, light.spotDirection.y, light.spotDirection.z} });

    auto const ambientKey = key + "." + "ambient";
    setUniform<attributes::Cases>(ambientKey, attributes::Vec3{ {light.ambient.x, light.ambient.y, light.ambient.z} });
    auto const diffuseKey = key + "." + "diffuse";
    setUniform<attributes::Cases>(diffuseKey, attributes::Vec3{ {light.diffuse.x, light.diffuse.y, light.diffuse.z} });
    auto const specularKey = key + "." + "specular";
    setUniform<attributes::Cases>(specularKey, attributes::Vec3{ {light.specular.x, light.specular.y, light.specular.z} });

    auto const cutoffRadiansKey = key + "." + "cutoff";
    setUniform<attributes::Cases>(cutoffRadiansKey, attributes::FloatAttr{ .val = light.cutoffRadians });
    auto const cutoffDecayRadiansKey = key + "." + "cutoffDecay";
    setUniform<attributes::Cases>(cutoffDecayRadiansKey, attributes::FloatAttr{ .val = light.cutoffDecayRadians });
 
    auto const attenuanceConstantKey = key + "." + "attenuanceConstant";
    setUniform<attributes::Cases>(attenuanceConstantKey, attributes::FloatAttr{ .val = light.attenuanceConstant });
    auto const attenuanceLinearKey = key + "." + "attenuanceLinear";
    setUniform<attributes::Cases>(attenuanceLinearKey, attributes::FloatAttr{ .val = light.attenuanceLinear });
    auto const attenuanceQuadraticKey = key + "." + "attenuanceQuadratic";
    setUniform<attributes::Cases>(attenuanceQuadraticKey, attributes::FloatAttr{ .val = light.attenuanceQuadratic });
}

template<>
void inline gl::Shader::setUniform<std::vector<LightSource>>(
    std::string const &key,
    std::vector<LightSource> const &lights
) const {
    if (!program) return;
    glUseProgram(program);
    for (size_t i = 0; i < lights.size(); i++) {
        setUniform<LightSource>(key + "[" + std::to_string(i) + "]", lights[i]);
    }
    std::string numLightsKey = "numLights";
    if (auto location = glGetUniformLocation(program, numLightsKey.c_str()); location != -1) {
        glUniform1i(location, lights.size());
    }
}

template<>
void inline gl::Shader::setUniform<std::array<size_t, 1>>(
    std::string const &key,
    std::array<size_t, 1> const &value
) const {
    if (!program) return;
    glUseProgram(program);
    if (auto location = glGetUniformLocation(program, key.c_str()); location != -1) {
        glUniform1i(location, value.at(0));
    }
}