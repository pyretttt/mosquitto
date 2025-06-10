#pragma once

#include <memory>

#include "MathUtils.hpp"
#include "scene/Mesh.hpp"
#include "scene/Identifiers.hpp"
#include "Attributes.hpp"

namespace scene {
using MaterialMeshPtr = std::shared_ptr<Mesh<attributes::AssimpVertex>>;

struct Node {

    Node(
        NodeId identifier,
        std::vector<MaterialMeshPtr> meshes,
        ml::Matrix4f transform = ml::diagonal<ml::Matrix4f>(1)
    );

    ml::Matrix4f getTransform() const noexcept;

    NodeId identifier;
    std::vector<MaterialMeshPtr> meshes;
    ml::Matrix4f localTransform;
    std::weak_ptr<Node> parent;
};
}