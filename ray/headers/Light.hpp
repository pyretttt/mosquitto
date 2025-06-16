#pragma once

#include <variant>

#include "MathUtils.hpp"
#include "Attributes.hpp"

namespace light {

struct LightSource final {
    ml::Vector3f position;
    ml::Vector4f color;
};

struct GlobalIlluminance {
    LightSource lightSource;
    float intensity;
};

struct DirectionalIlluminance {
    LightSource lightSource;
    float intensity;
    ml::Vector3f direction;
};


using LightCase = std::variant<
    GlobalIlluminance
>;

}