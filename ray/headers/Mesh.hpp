#pragma once

#include <array>
#include <variant>

#include "MathUtils.hpp"

struct Attributes {
    struct Color {
        std::array<uint32_t, 3> color;
    };
    struct Texture {
        std::array<ml::Vector2f, 3> uv;
    };

    using Cases = std::variant<Color, Texture>;
};

struct Face {
    Face(int a, int b, int c, Attributes::Cases attributes);
    int a, b, c;
    Attributes::Cases attributes;
};

struct MeshBuffer {
    MeshBuffer(std::vector<ml::Vector3f> const &vertices, std::vector<Face> const &faces);
    std::vector<ml::Vector3f> vertices;
    std::vector<Face> faces;
    // TODO: Add texture
};

struct MeshNode {
    MeshNode() = delete;
    explicit MeshNode(MeshBuffer const &meshBuffer);
    MeshBuffer meshBuffer;
    ml::Matrix4f transform = ml::diagonal<ml::Matrix4f>();
    std::weak_ptr<MeshNode> parent;
    std::vector<MeshNode> children;

    ml::Matrix4f getTransform() const noexcept;
};

struct Triangle {
    Triangle(
        std::array<ml::Vector4f, 3> vertices,
        Attributes::Cases attributes
    );
    std::array<ml::Vector4f, 3> vertices;
    Attributes::Cases attributes;
};