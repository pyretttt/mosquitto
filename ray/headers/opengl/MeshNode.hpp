#pragma once

#include <memory>
#include <numeric>

#include "glCommon.hpp"
#include "MathUtils.hpp"

namespace gl {

template <typename Attr>
struct MeshNode {
    MeshNode(
        std::vector<Attr> vertexArray,
        EBO vertexArrayIndices,
        ml::Matrix4f localTransform = ml::diagonal<ml::Matrix4f>()
    );

    ml::Matrix4f getTransform() const noexcept;

    std::vector<Attr> vertexArray;
    EBO vertexArrayIndices;
    ml::Matrix4f localTransform;
    std::weak_ptr<MeshNode> parent;
};


template <typename Attr>
MeshNode<Attr>::MeshNode(
    std::vector<Attr> vertexArray,
    EBO vertexArrayIndices,
    ml::Matrix4f localTransform
) 
    : vertexArray(std::move(vertexArray))
    , vertexArrayIndices(std::move(vertexArrayIndices))
    , localTransform(localTransform)  {
    if (this->vertexArrayIndices.empty()) {
        this->vertexArrayIndices = EBO(this->vertexArray.size());
        std::iota(this->vertexArrayIndices.begin(), this->vertexArrayIndices.end(), this->vertexArray.size());
    }
}

template <typename Attr>
ml::Matrix4f MeshNode<Attr>::getTransform() const noexcept {
    if (auto strongParent = parent.lock()) {
        auto parentTransform = ml::matMul(strongParent->getTransform(), localTransform);
        return parentTransform;
    } else {
        return localTransform;
    }
}
}