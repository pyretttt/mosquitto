#include <array>
#include <iostream>
#include <algorithm>

#include "Polygon.hpp"
#include "Mesh.hpp"
#include "Output.hpp"


namespace {
    float signedDistanceToPlane(
        ml::Vector4f const &vertex,
        Plane plane
    ) {
        plane.point = ml::matrixScale(plane.point, vertex.w);
        return plane.signedDistance(ml::as<3, 4, float>(vertex));
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
    static std::vector<Plane> planes = {
        Plane(ml::Vector3f(1, 0, 0), ml::Vector3f(-1, 0, 0)), // Right
        Plane(ml::Vector3f(-1, 0, 0), ml::Vector3f(1, 0, 0)), // Left
        Plane(ml::Vector3f(0, 1, 0), ml::Vector3f(0, -1, 0)), // Up
        Plane(ml::Vector3f(0, -1, 0), ml::Vector3f(0, 1, 0)), // Down
        // Forward and backward planes here are 180 degrees rotated around Y by origin point.
        // In righthand system we're interested in vertices with negative `z`. After perspective projection matrix negative `z` becames positive.
        // So to acknowledge it we flip forward and backward planes.
        Plane(ml::Vector3f(0, 0, 1), ml::Vector3f(0, 0, -1)), // Forward
        Plane(ml::Vector3f(0, 0, 0), ml::Vector3f(0, 0, 1)), // Backward
    };
    for (auto const &plane : planes) {
        std::array<ml::Vector4f, maxVerticesAfterClipping> insideVertices;
        std::array<Attributes::Cases, maxVerticesAfterClipping> insideAttributes;
        size_t insideCount = 0;
        auto currentVertex = &vertices[0];
        auto previousVertex = &vertices[nVertices - 1];
        auto currentAttributes = &insideAttributes[0];
        auto previousAttributes = &insideAttributes[nVertices - 1];

        while (currentVertex != &vertices[nVertices]) {
            auto const distanceToCurrent = signedDistanceToPlane(*currentVertex, plane);
            auto const distanceToPrevious = signedDistanceToPlane(*previousVertex, plane);

            if (distanceToCurrent * distanceToPrevious < 0) {
                auto const distance = distanceToPrevious - distanceToCurrent;
                auto const t = distanceToPrevious / distance;
                auto const intersection = ml::matrixScale(
                    ml::matrixScale(*currentVertex, distanceToPrevious) 
                        - ml::matrixScale(*previousVertex, distanceToCurrent), 
                        distance
                );
                insideAttributes[insideCount] = Attributes::intepolate(*currentAttributes, *previousAttributes, t);
                insideVertices[insideCount++] = intersection;
            }
            if (distanceToCurrent > 0) {
                insideAttributes[insideCount] = *currentAttributes;
                insideVertices[insideCount++] = *currentVertex;
            }

            previousVertex = currentVertex;
            previousAttributes = currentAttributes;
            currentVertex++;
            currentAttributes++;
        }

        for (size_t i = 0; i < insideCount; i++) {
            vertices[i] = insideVertices[i];
            interpolatedAttributes[i] = insideAttributes[i];
        }
        nVertices = insideCount;
    }
}

std::array<Triangle, maxTrianglesAfterClipping> Polygon::getTriangles() noexcept {
    std::array<Triangle, maxTrianglesAfterClipping> res = {};
    for (size_t i = 0; i < numTriangles(); i++) {
        res[i] = Triangle(
            std::array<ml::Vector4f, 3>({vertices[0], vertices[i + 2], vertices[i + 1]}),
            std::array<Attributes::Cases, 3>({
                interpolatedAttributes[0],
                interpolatedAttributes[i + 2], 
                interpolatedAttributes[i + 1]
            })
        );
    }

    return res;
}

size_t Polygon::numTriangles() noexcept {
    return std::max(0, static_cast<int>(nVertices - 2));
}