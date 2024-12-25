#pragma once

#include <cmath>

#include "MathUtils.hpp"

struct Plane {
    Plane(
        ml::Vector3f point,
        ml::Vector3f normal
    ) : point(point), normal(normal) {}

    ml::Vector3f point;
    ml::Vector3f normal;
};

using ClippingPlanes = std::array<Plane, 6>;

ClippingPlanes inline makeClippingPlanes(
    float verticalFov, float aspectRatio
) noexcept {
    float const halfTan = tanf(verticalFov / 2);
    float const horizontalFov = atan(aspectRatio * halfTan);

    float const beta = verticalFov / 2;
    float const omega = horizontalFov / 2;

    // Maybe make it in euclidean space? NDC is tough
    auto upper(Plane({0, 0, 0}, {0, -cosf(beta), sinf(beta)}));
    auto lower(Plane({0, 0, 0}, {0, cosf(beta), sinf(beta)}));
    auto right(Plane({0, 0, 0}, {-cos(beta), 0, sin(beta)}));
    auto right(Plane({0, 0, 0}, {cos(beta), 0, sin(beta)}));
    auto near(Plane({0, 0, 0}, {0, 0, 1}))
}