#pragma once

#include <array>
#include <variant>

#include "Eigen/Dense"

struct Attributes {
    struct Color {
        std::array<uint32_t, 3> color;
    };
    struct Texture {
        std::array<Eigen::Vector2f, 3> uv;
    };

    using Cases = std::variant<Color, Texture>;
};

struct Face {
    Face(int a, int b, int c, Attributes::Cases attributes);
    int a, b, c;
    Attributes::Cases attributes;
};

struct Mesh {
    Mesh(std::vector<Eigen::Vector3f> vertices, std::vector<Face> faces);
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Face> faces;
    // TODO: Add texture
};


struct Triangle {
    Triangle(
        std::array<Eigen::Vector4f, 3> vertices,
        Attributes::Cases attributes
    );
    std::array<Eigen::Vector4f, 3> vertices;
    Attributes::Cases attributes;
};