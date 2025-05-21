#pragma once

#include <variant>

#include "MathUtils.hpp"

namespace sdl {
namespace light {
struct DirectionalLight {
    ml::Vector3f direction;
};

using Cases = std::variant<DirectionalLight>;
} // namespace light
}; // namespace sdl

namespace light {
    
}