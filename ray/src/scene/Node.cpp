#include "scene/Node.hpp"

using namespace scene;

Node::Node(
    NodeId identifier,
    std::vector<MaterialMeshPtr> meshes,
    ml::Matrix4f transform
) 
    : identifier(identifier)
    , meshes(std::move(meshes)) 
    , localTransform(transform)
    {}

ml::Matrix4f Node::getTransform() const noexcept {
    if (auto strongPtr = parent.lock()) {
        return ml::matMul(strongPtr->getTransform(), localTransform);
    }
    return localTransform;
}