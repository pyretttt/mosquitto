#pragma once

#include <memory>

#include "MathUtils.hpp"
#include "scene/Mesh.hpp"
#include "Attributes.hpp"

namespace scene {
using MeshPtr = std::shared_ptr<Mesh<attributes::AssimpVertex>>;

struct Node {

    Node(
        size_t identifier,
        std::vector<MeshPtr> meshes,
        ml::Matrix4f transform = ml::diagonal<ml::Matrix4f>(1)
    );

    ml::Matrix4f getTransform() const noexcept;

    size_t identifier;
    std::vector<MeshPtr> meshes;
    ml::Matrix4f localTransform;
    std::weak_ptr<Node> parent;
};

using NodePtr = std::shared_ptr<Node>;

}