#pragma once

#include "opengl/Shader.hpp"

namespace gl {
    enum class Shading {
        Material,
        Outlining,
        Texturing
    }; 

    inline auto const shaderDirectory = std::filesystem::path("shaders");

    inline ShaderPtr const materialShader = std::make_shared<gl::Shader>(gl::Shader(
        shaderDirectory / "material.vs",
        shaderDirectory / "material.fs"
    ));
    inline ShaderPtr const outlineShader = std::make_shared<gl::Shader>(gl::Shader(
        shaderDirectory / "outline.vs",
        shaderDirectory / "outline.fs"       
    ));
    inline ShaderPtr const textureShader = std::make_shared<gl::Shader>(gl::Shader(
        shaderDirectory / "texture.vs",
        shaderDirectory / "texture.fs"       
    ));

    inline auto const shaders = std::unordered_map<Shading, ShaderPtr>({
        std::make_pair(Shading::Material, materialShader),
        std::make_pair(Shading::Outlining, outlineShader),
        std::make_pair(Shading::Texturing, textureShader),
    });
}