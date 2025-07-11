#include "scene/Node.hpp"

using namespace scene;

Node::Node(
    NodeId identifier,
    ml::Matrix4f transform
) 
    : identifier(identifier)
    , localTransform(transform)
    {}

ml::Matrix4f Node::getTransform() const noexcept {
    if (auto strongPtr = parent.lock()) {
        return ml::matMul(strongPtr->getTransform(), localTransform);
    }
    return localTransform;
}