#pragma once

#include "scene/Component.hpp"

namespace scene {
    template<typename Attribute> 
    struct MeshComponent final : public Component {
        MeshComponent(std::vector<Attribute> vertexAttributes) 
        : vertexAttributes(std::move(vertexAttributes)) {}

        std::vector<Attribute> vertexAttributes;
    };
}