#include "Polygon.hpp"
#include "Mesh.hpp"

Polygon Polygon::fromTriangle(Triangle const &tri) {
    auto poly = Polygon {
        .nVertices = 3
    };
    poly.vertices[0] = tri.vertices[0];
    poly.interpolatedAttributes[0] = tri.attributes[0];
    poly.vertices[1] = tri.vertices[1];
    poly.interpolatedAttributes[1] = tri.attributes[1];
    poly.vertices[2] = tri.vertices[2];
    poly.interpolatedAttributes[2] = tri.attributes[2];

    return poly;
}

void Polygon::clip(ClippingPlanes const &planes) noexcept {
    for (auto const &plane : planes) {
        std::array<ml::Vector4f, maxVerticesAfterClipping> out;
        for (size_t i = 0; i < nVertices; i++) {
            
        }
    }
}

static ml::Vector4f interpolate(ml::Vector4f const &a, ml::Vector4f const &b, float t) {
    assert(t <= 1.0 && t >= 0.0);
    ml::Vector4f diff = b - a;
    ml::Vector4f const newVertex = a + ml::matrixScale(diff, 1 - t);
    return newVertex;
}