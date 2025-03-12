#include "Polygon.hpp"
#include "Mesh.hpp"


namespace {
    float signedDistanceToPlane(
        ml::Vector4f const &vertex, 
        ml::Vector3f const &planeNormal
    ) {
        // Maybe add w negation:
        // auto const w = 
        auto const vertexComponent = ml::dotProduct(planeNormal, ml::as<3, 4, float>(vertex));
        return vertex.w - vertexComponent;
    }
}

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
        std::array<ml::Vector4f, maxVerticesAfterClipping> insideVertices;
        auto currentIdx = 0;
        auto const &currentVertex = vertices[0];
        auto const &previousVertices = vertices[nVertices - 1];

        while (currentIdx != nVertices - 1) {
            auto const distanceToCurrent = signedDistanceToPlane(currentVertex, plane.normal);
            auto const distanceToPrevious = signedDistanceToPlane(previousVertices, plane.normal);
        }
    }
}


static ml::Vector4f interpolate(ml::Vector4f const &a, ml::Vector4f const &b, float t) {
    assert(t <= 1.0 && t >= 0.0);
    ml::Vector4f diff = b - a;
    ml::Vector4f const newVertex = a + ml::matrixScale(diff, 1 - t);
    return newVertex;
}