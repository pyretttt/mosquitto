#pragma once

#include <string>

#include "scene/Identifiers.hpp"

namespace scene {
struct Component {
    Component(ComponentId id) : id(id) {}

    ComponentId id;

    virtual ~Component() {}
};
}