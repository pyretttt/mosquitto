#pragma once

#include <utility>
#include <memory>

#include "Mesh.h"

enum class RendererType {
    CPU,
    OpenGL
};

class Renderer {
public:
    using MeshData = std::vector<Mesh>;
    virtual void update(MeshData const &data, float dt) = 0;
    virtual void render() const = 0;
    virtual ~Renderer() = 0;
};
