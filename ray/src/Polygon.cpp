#include <array>

#include "Polygon.hpp"
#include "Mesh.hpp"


namespace {
    float signedDistanceToPlane(
        ml::Vector4f const &vertex,
        ml::Vector3f const &plane
    ) {
        auto const w = ml::matrixScale(plane, vertex.w);
        auto const vertexComponent = ml::dotProduct(plane, ml::as<3, 4, float>(vertex));
        return (w.x + w.y + w.z) - vertexComponent;
    }

    ml::Vector4f interpolate(ml::Vector4f const &a, ml::Vector4f const &b, float t) {
        assert(t <= 1.0 && t >= 0.0);
        ml::Vector4f diff = b - a;
        ml::Vector4f const newVertex = a + ml::matrixScale(diff, 1 - t);
        return newVertex;
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

void Polygon::clip() noexcept {
    static std::vector<ml::Vector3f> planes = {
        ml::Vector3f(1, 0, 0), // Right
        ml::Vector3f(-1, 0, 0), // Left
        ml::Vector3f(0, 1, 0), // Up
        ml::Vector3f(0, -1, 0), // Down
        ml::Vector3f(0, 0, 1), // Forward
        ml::Vector3f(0, 0, -1) // Backward
    };
    for (auto const &plane : planes) {
        std::array<ml::Vector4f, maxVerticesAfterClipping> insideVertices;
        size_t currentIdx = 0;
        auto currentVertex = &vertices[currentIdx];
        auto previousVertex = &vertices[nVertices - 1];

        while (currentVertex != &vertices[nVertices - 1]) {
            auto const distanceToCurrent = signedDistanceToPlane(*currentVertex, plane);
            auto const distanceToPrevious = signedDistanceToPlane(*previousVertex, plane);

            if (distanceToCurrent * distanceToPrevious < 0) {
                auto const distance = distanceToCurrent - distanceToPrevious;
                auto const t = distanceToCurrent / (distanceToCurrent - distanceToPrevious);
                auto const intersection = ml::matrixScale(
                    ml::matrixScale(*previousVertex, distanceToCurrent) 
                        - ml::matrixScale(*currentVertex, distanceToPrevious), 
                    distance
                );
                insideVertices[currentIdx++] = intersection;
            } else {
                insideVertices[currentIdx++] = *currentVertex;
            }

            previousVertex = currentVertex++;
        }

        for (size_t i = 0; i <= currentIdx; i++) {
            vertices[i] = insideVertices[i];
        }
        nVertices = currentIdx + 1;
    }
}

std::array<Triangle, maxTrianglesAfterClipping> Polygon::getTriangles() noexcept {
    std::array<Triangle, maxTrianglesAfterClipping> res = {};
    for (size_t i = 0; i < numTriangles(); i++) {
        res[i] = Triangle(
            std::array<ml::Vector4f, 3>({vertices[0], vertices[i + 2], vertices[i + 1]}),
            std::array<Attributes::Cases, 3>({interpolatedAttributes[0], interpolatedAttributes[1], interpolatedAttributes[2]})
        );
    }

    return res;
}

size_t Polygon::numTriangles() noexcept {
    return nVertices - 2;
}