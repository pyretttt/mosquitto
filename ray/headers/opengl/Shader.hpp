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
        attributes::UniformCases const &attr
    ) noexcept;

    void setTextureSamplers(size_t max) noexcept;

    ~Shader();

    std::filesystem::path vertex;
    std::filesystem::path fragment;
};
}