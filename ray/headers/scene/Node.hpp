#pragma once

#include <memory>
#include <typeinfo>
#include <optional>

#include "MathUtils.hpp"
#include "scene/Mesh.hpp"
#include "scene/Identifiers.hpp"
#include "Attributes.hpp"
#include "Core.hpp"

namespace scene {
struct Node {
    Node(
        NodeId identifier,
        ml::Matrix4f transform = ml::diagonal<ml::Matrix4f>(1)
    );

    template <typename Comp, typename... Args>
    void Node::addComponent(Args... args) noexcept;

    template <typename Comp>
    std::shared_ptr<std::decay_t<Comp>> Node::getComponent() const noexcept;

    ml::Matrix4f getTransform() const noexcept;

    NodeId identifier;
    ml::Matrix4f localTransform;
    std::weak_ptr<Node> parent;

    std::unordered_map<size_t, ComponentPtr> components = {};
};

template <typename Comp, typename... Args>
void Node::addComponent(Args... args) noexcept {
    ComponentPtr component = std::make_shared<std::decay_t<Comp>>(std::forward<Args>(args)...);
    auto &typeInfo = typeid(std::decay_t<Comp>);
    components[typeInfo.hash_code()] = component;
}

template <typename Comp>
std::shared_ptr<std::decay_t<Comp>> Node::getComponent() const noexcept {
    auto &typeInfo = typeid(std::decay_t<Comp>);
    if (components.find(typeInfo.hash_code()) != components.end()) {
        return static_cast<std::shared_ptr<std::decay_t<Comp>>>(components.at(typeInfo.hash_code()));
    }
    return std::shared_ptr<std::decay_t<Comp>>();
}
}