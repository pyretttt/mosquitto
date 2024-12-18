#pragma once

#include "MathUtils.h"

struct Plane {
    Plane(
        ml::Vector3f point,
        ml::Vector3f normal
    ) : point(point), normal(normal) {}

    ml::Vector3f point;
    ml::Vector3f normal;
};

using ClippingPlanes = std::array<Plane, 6>;