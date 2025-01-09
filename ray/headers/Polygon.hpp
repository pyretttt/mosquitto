#pragma once

#include <MathUtils.hpp>
#include <Mesh.hpp>

struct Polygon {
    std::vector<ml::Vector3f> vertices;
    std::vector<Attributes::Cases> interpolatedAttributes;
};