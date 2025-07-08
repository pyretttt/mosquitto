#pragma once

#include "scene/Component.hpp"

namespace scene {
    template <typename Attribute>
    using MeshPtr = std::shared_ptr<Mesh<Attribute>>;

    template<typename Attribute> 
    struct MeshComponent final : public Component {
        MeshComponent(
            std::vector<MeshPtr<Attribute>> meshes,
            ComponentId id
        ) 
        : Component(id)
        , meshes(std::move(meshes)) {}

        std::vector<MeshPtr<Attribute>> meshes;
    };
}