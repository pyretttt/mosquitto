#include "Mesh.hpp"

Face::Face(
    int a, int b, int c
)
    : a(a),
      b(b),
      c(c) {}

MeshBuffer::MeshBuffer(
    std::vector<ml::Vector3f> const &vertices, 
    std::vector<Face> const &faces,
    std::vector<Attributes::Cases> const &attributes
)
    : vertices(vertices),
      faces(faces),
      attributes(attributes) {}

MeshNode::MeshNode(
    MeshBuffer const &meshBuffer
) : meshBuffer(meshBuffer) {}

ml::Matrix4f MeshNode::getTransform() const noexcept {
    if (auto par = parent.lock()) {
        return ml::matMul(par->getTransform(), transform);
    }
    return transform;
}

Triangle::Triangle(
    std::array<ml::Vector4f, 3> vertices,
    std::array<Attributes::Cases, 3> attributes
)
    : vertices(vertices),
      attributes(attributes) {}
