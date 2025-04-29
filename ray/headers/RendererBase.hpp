#pragma once

#include <memory>
#include <utility>

#include "Mesh.hpp"
#include "GlobalConfig.hpp"

enum class RendererType {
    CPU,
    OpenGL
};

enum class RenderMethod {
    vertices,
    wireframe,
    fill
};

class Renderer {
public:
    using MeshData = std::vector<MeshNode>;
    virtual void prepareViewPort() = 0;
    virtual void processInput(void const *) = 0;
    virtual void update(MeshData const &data, float dt) = 0;
    virtual void render() const = 0;
    virtual ~Renderer() = 0;
};
