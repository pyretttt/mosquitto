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

    constexpr inline float signedDistance(ml::Vector3f const &vertex) noexcept {
        ml::Vector3f const pointingVector = vertex - point;
        return ml::dotProduct(normal, pointingVector) / ml::euclideanNorm(normal);
    }
};

using ClippingPlanes = std::array<Plane, 6>;

// ClippingPlanes inline makeClippingPlanes(
//     float verticalFov, float aspectRatio
// ) noexcept {
//     float const halfTan = tanf(verticalFov / 2);
//     float const horizontalFov = atan(aspectRatio * halfTan);

//     float const beta = verticalFov / 2;
//     float const omega = horizontalFov / 2;

//     // Maybe make it in euclidean space? NDC is tough
//     auto upper(Plane({0, 0, 0}, {0, -cosf(beta), sinf(beta)}));
//     auto lower(Plane({0, 0, 0}, {0, cosf(beta), sinf(beta)}));
//     auto right(Plane({0, 0, 0}, {-cos(beta), 0, sin(beta)}));
//     auto left(Plane({0, 0, 0}, {cos(beta), 0, sin(beta)}));
//     auto near(Plane({0, 0, 0}, {0, 0, 1}));
//     auto far(Plane({0, 0, 0}, {0, 0, 1}));
// }

ClippingPlanes inline makeEuclideanClippingPlanes(
    float verticalFov, float aspectRatio, float zfar, float znear
) noexcept {
    float const halfTan = tanf(verticalFov / 2);
    float const horizontalFov = atan(aspectRatio * halfTan);

    float const beta = verticalFov / 2;
    float const omega = horizontalFov / 2;

    auto upper(Plane({0, 0, 0}, {0, -cosf(beta), -sinf(beta)}));
    auto lower(Plane({0, 0, 0}, {0, cosf(beta), -sinf(beta)}));
    auto right(Plane({0, 0, 0}, {-cos(omega), 0, -sin(omega)}));
    auto left(Plane({0, 0, 0}, {cos(omega), 0, -sin(omega)}));
    auto near(Plane({0, 0, znear}, {0, 0, -1}));
    auto far(Plane({0, 0, zfar}, {0, 0, 1}));
    return ClippingPlanes({
        upper,
        lower,
        right,
        left,
        near,
        far
    });
}