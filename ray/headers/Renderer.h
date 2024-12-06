#pragma once

#include <memory>
#include <utility>

#include "Mesh.h"

enum class RendererType {
    CPU,
    OpenGL
};

class Renderer {
public:
    using MeshData = std::vector<MeshNode>;
    virtual void update(MeshData const &data, float dt) = 0;
    virtual void render() const = 0;
    virtual ~Renderer() = 0;
};
