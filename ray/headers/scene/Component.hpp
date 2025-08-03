#pragma once

#include <string>

#include "scene/Identifiers.hpp"

namespace scene {
    struct Component {
        Component(ID id) : id(id) {}

        ID id;

        virtual ~Component() {}
    };

    template<typename Type>
    struct ContainerComponent final : public Component {
        ContainerComponent(ID id, Type value);
        
        Type value;
        ~ContainerComponent() override;
    };

    template<typename Type>
    ContainerComponent<Type>::ContainerComponent(ID id, Type value)
        : Component(id)
        , value(std::move(value)) {}


    template<typename Type>
    ContainerComponent<Type>::~ContainerComponent() {}
}