#pragma once

#include <memory>
#include <numeric>
#include <unordered_map>

#include "MathUtils.hpp"

namespace scene {
template <typename Attr>
struct SceneNode {
    SceneNode(
        std::vector<Attr> vertexArray,
        std::vector<unsigned int> vertexArrayIndices,
        std::vector<size_t> textures,
        size_t identifier,
        ml::Matrix4f localTransform = ml::diagonal<ml::Matrix4f>()
    );

    ml::Matrix4f getTransform() const noexcept;

    std::vector<Attr> vertexArray;
    std::vector<unsigned int> vertexArrayIndices;
    std::vector<size_t> vertexArrayIndices;
    ml::Matrix4f localTransform;
    std::weak_ptr<SceneNode> parent;

    size_t const identifier;
};


template <typename Attr>
SceneNode<Attr>::SceneNode(
    std::vector<Attr> vertexArray,
    std::vector<unsigned int> vertexArrayIndices,
    std::vector<size_t> textures,
    size_t identifier,
    ml::Matrix4f localTransform
) 
    : vertexArray(std::move(vertexArray))
    , vertexArrayIndices(std::move(vertexArrayIndices))
    , textures(std::move(texture))
    , identifier(identifier)
    , localTransform(localTransform)  {
    if (this->vertexArrayIndices.empty()) {
        this->vertexArrayIndices = std::vector<unsigned int>(this->vertexArray.size());
        std::iota(this->vertexArrayIndices.begin(), this->vertexArrayIndices.end(), this->vertexArray.size());
    }
}

template <typename Attr>
ml::Matrix4f SceneNode<Attr>::getTransform() const noexcept {
    if (auto strongParent = parent.lock()) {
        auto parentTransform = ml::matMul(strongParent->getTransform(), localTransform);
        return parentTransform;
    } else {
        return localTransform;
    }
}
}