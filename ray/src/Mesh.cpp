#include "Mesh.h"

Face::Face(int a, int b, int c, std::array<Eigen::Vector2f, 3> uv)
    : a(a),
      b(b),
      c(c),
      uv(uv) {}

Mesh::Mesh(std::vector<Eigen::Vector3f> vertices, std::vector<Face> faces)
    : vertices(vertices),
      faces(faces) {}

Triangle::Triangle(
    std::array<Eigen::Vector4f, 3> vertices,
    std::array<Eigen::Vector2f, 3> uv
)
    : vertices(vertices),
      uv(uv) {}