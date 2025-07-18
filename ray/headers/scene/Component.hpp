#pragma once

#include <string>

#include "scene/Identifiers.hpp"

namespace scene {
    struct Component {
        Component(ComponentId id) : id(id) {}

        ComponentId id;

        virtual ~Component() {}
    };

    template<typename Type>
    struct ContainerComponent final : public Component {
        ContainerComponent(ComponentId id, Type value);
        
        Type value;
        ~ContainerComponent();
    };

    template<typename Type>
    ContainerComponent<Type>::ContainerComponent(ComponentId id, Type value)
        : Component(id)
        , value(std::move(value)) {}


    template<typename Type>
    ContainerComponent<Type>::~ContainerComponent() {}
}