#pragma once

#include <variant>

#include "MathUtils.h"

namespace sdl {
namespace light {
    struct DirectionalLight {
        ml::Vector3f direction;
    };

    using Cases = std::variant<DirectionalLight>;
}
};