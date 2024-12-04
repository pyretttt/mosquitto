#pragma once

#include <array>

#include "Eigen/Dense"

struct Face {
    Face(int a, int b, int c, std::array<Eigen::Vector2f, 3> uv);
    int a, b, c;
    std::array<Eigen::Vector2f, 3> uv;
};

struct Mesh {
    Mesh(std::vector<Eigen::Vector3f> vertices, std::vector<Face> faces);
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Face> faces;
    // TODO: Add texture
};

struct Attributes {
    struct Color { uint32_t color; };
    struct Texture { Eigen::Vector2f uv; };
    
    using Cases = std::variant<Color, Texture>;
};

struct Triangle {
    Triangle(
        std::array<Eigen::Vector4f, 3> vertices,
        std::array<Eigen::Vector2f, 3> uv
    );
    std::array<Eigen::Vector4f, 3> vertices;
    std::array<Eigen::Vector2f, 3> uv;
};