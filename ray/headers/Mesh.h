#pragma once

#include <array>
#include <variant>

#include "MathUtils.h"

struct Attributes {
    struct Color {
        std::array<uint32_t, 3> color;
    };
    struct Texture {
        std::array<Vector2f, 3> uv;
    };

    using Cases = std::variant<Color, Texture>;
};

struct Face {
    Face(int a, int b, int c, Attributes::Cases attributes);
    int a, b, c;
    Attributes::Cases attributes;
};

struct MeshBuffer {
    MeshBuffer(std::vector<Vector3f> vertices, std::vector<Face> faces);
    std::vector<Vector3f> vertices;
    std::vector<Face> faces;
    // TODO: Add texture
};

struct MeshNode {
    MeshNode() = delete;
    MeshNode(MeshBuffer meshBuffer);
    MeshBuffer meshBuffer;
    Matrix3f transform = eye<3>();
    std::weak_ptr<MeshNode> parent;
    std::vector<MeshNode> children;

    Matrix3f getTransform() const noexcept;
};

struct Triangle {
    Triangle(
        std::array<Vector4f, 3> vertices,
        Attributes::Cases attributes
    );
    std::array<Vector4f, 3> vertices;
    Attributes::Cases attributes;
};