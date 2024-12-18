#include "Mesh.h"

Face::Face(int a, int b, int c, Attributes::Cases attributes)
    : a(a),
      b(b),
      c(c),
      attributes(attributes) {}

MeshBuffer::MeshBuffer(std::vector<ml::Vector3f> const &vertices, std::vector<Face> const &faces)
    : vertices(vertices),
      faces(faces) {}

MeshNode::MeshNode(MeshBuffer const &meshBuffer) : meshBuffer(meshBuffer) {}

ml::Matrix4f MeshNode::getTransform() const noexcept {
    if (auto par = parent.lock()) {
        return ml::matMul(par->getTransform(), transform);
    }
    return transform;
}

Triangle::Triangle(
    std::array<ml::Vector4f, 3> vertices,
    Attributes::Cases attributes
)
    : vertices(vertices),
      attributes(attributes) {}
