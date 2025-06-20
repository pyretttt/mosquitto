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

    void setUniform(
        std::string const &key, 
        attributes::UniformCases const &attr
    ) noexcept;

    void setTextureSamplers(size_t max, bool unuseProgram = false) noexcept;
    void setMaterialSamplers(gl::Material const &material, bool unuseProgram = false) noexcept;

    ~Shader();

    std::filesystem::path vertex;
    std::filesystem::path fragment;
};

using ShaderPtr = std::shared_ptr<Shader>;
}