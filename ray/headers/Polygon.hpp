#pragma once

#include "MathUtils.hpp"
#include "Mesh.hpp"
#include "Plane.hpp"

constexpr static size_t maxVerticesAfterClipping = 10;

struct Triangle;
struct Plane;

struct Polygon {
    ml::Vector4f vertices[maxVerticesAfterClipping];
    size_t nVertices;
    Attributes::Cases interpolatedAttributes[maxVerticesAfterClipping];

    static Polygon fromTriangle(Triangle const &tri);

    void clip(ClippingPlanes const &planes) noexcept;
};