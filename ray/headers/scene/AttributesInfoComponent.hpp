#pragma once

#include <unordered_map>

#include "scene/Component.hpp"
#include "Attributes.hpp"

namespace scene {
    using AttributesInfoComponent = ContainerComponent<std::unordered_map<std::string, attributes::Cases>>;
}