#pragma once

#include <array>
#include <cmath>
#include <variant>
#include <utility>

#include "Core.hpp"
#include "MathUtils.hpp"
#include "Utility.hpp"

struct Attributes {
    struct Color {
        uint32_t val;
    };
    using TextureUV = ml::Vector2f;
    struct Value {
        float val;
    };

    using Cases = std::variant<Color, TextureUV, Value>;

    static Attributes::Cases intepolate(
        Attributes::Cases const &a,
        Attributes::Cases const &b,
        float t
    ) {
        return std::visit(overload {
            [t](Attributes::Color const &a, Attributes::Color const &b) -> Attributes::Cases { 
                return Color { .val = interpolateRGBAColor(a.val, b.val, 1.0) }; 
            },
            [t](Attributes::Value const &a, Attributes::Value const &b) -> Attributes::Cases{ 
                return Value { .val = std::lerp(a.val, b.val, t) };
             },
            [t](Attributes::TextureUV const &a, Attributes::TextureUV const &b) -> Attributes::Cases { 
                return ml::lerp(a, b, t);
             },
            [](auto const &a, auto const &b) -> decltype(auto) {
                if constexpr (!std::is_same_v<std::decay_t<decltype(a)>, std::decay_t<decltype(b)>>) {
                    std::cerr << "Mismatched types in interpolate: " << typeid(a).name() << " vs " << typeid(b).name() << "\n";
                }
                return Attributes::Cases();
            }
        }, a, b);
    }
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
    Triangle() = default;
    Triangle(
        std::array<ml::Vector4f, 3> vertices,
        std::array<Attributes::Cases, 3> attributes
    );
    std::array<ml::Vector4f, 3> vertices;
    std::array<Attributes::Cases, 3> attributes;
};