#pragma once

#include <array>
#include <variant>

#include "MathUtils.hpp"

struct Attributes {
    using Color = uint32_t;
    using TextureUV = ml::Vector2f;
    using Value = float;

    using Cases = std::variant<Color, TextureUV, Value>;
};

struct Face {
    Face(int a, int b, int c);
    int a, b, c;
};

struct MeshBuffer {
    MeshBuffer(
        std::vector<ml::Vector3f> const &vertices, 
        std::vector<Face> const &faces, 
        std::vector<Attributes::Cases> const &attributes
    );
    std::vector<ml::Vector3f> vertices;
    std::vector<Face> faces;
    std::vector<Attributes::Cases> attributes;
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
        std::array<Attributes::Cases, 3> attributes
    );
    std::array<ml::Vector4f, 3> vertices;
    std::array<Attributes::Cases, 3> attributes;
};