#pragma once

#include <string>
#include <filesystem>

#include "GL/glew.h"

#include "opengl/glCommon.hpp"
#include "opengl/glTexture.hpp"
#include "Attributes.hpp"
#include "Core.hpp"

namespace gl {
class Shader final {
public:
    ID program = 0;

    Shader(std::filesystem::path vertex, std::filesystem::path fragment, bool eagerInit = false);

    void use();
    void setup();

    template<typename Value>
    void setUniform(
        std::string const &key,
        Value const &value
    ) const;

    ~Shader();

    std::filesystem::path vertex;
    std::filesystem::path fragment;
};

template<typename Value>
void Shader::setUniform(
    std::string const &key,
    Value const &value
) const {
    throw std::logic_error("Not implemented");
}

using ShaderPtr = std::shared_ptr<Shader>;

inline ShaderPtr materialShader = std::make_shared<gl::Shader>(gl::Shader(
    std::filesystem::path("shaders").append("vertex.vs"),
    std::filesystem::path("shaders").append("fragment.fs")       
));
inline ShaderPtr outlineShader = std::make_shared<gl::Shader>(gl::Shader(
    std::filesystem::path("shaders").append("outline.vs"),
    std::filesystem::path("shaders").append("outline.fs")       
));
}