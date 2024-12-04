#include "Mesh.h"

Face::Face(int a, int b, int c, Attributes::Cases attributes)
    : a(a),
      b(b),
      c(c),
      attributes(attributes) {}

MeshBuffer::MeshBuffer(std::vector<Eigen::Vector3f> vertices, std::vector<Face> faces)
    : vertices(vertices),
      faces(faces) {}

Triangle::Triangle(
    std::array<Eigen::Vector4f, 3> vertices,
    Attributes::Cases attributes
)
    : vertices(vertices),
      attributes(attributes) {}