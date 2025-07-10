#pragma once

#include <unordered_map>

#include "scene/Component.hpp"
#include "Attributes.hpp"

namespace scene {
struct ShaderInfoComponent final : public Component {
    ShaderInfoComponent(
        ComponentId id, 
        std::unordered_map<std::string, attributes::Cases> uniforms
    ) 
        : Component(id)
        , uniforms(std::move(uniforms)) {}

    std::unordered_map<std::string, attributes::Cases> uniforms;

    ~ShaderInfoComponent() {}
};
}