#include "Mesh.h"

Face::Face(int a, int b, int c, Attributes::Cases attributes)
    : a(a),
      b(b),
      c(c),
      attributes(attributes) {}

MeshBuffer::MeshBuffer(std::vector<Vector3f> vertices, std::vector<Face> faces)
    : vertices(vertices),
      faces(faces) {}

MeshNode::MeshNode(MeshBuffer meshBuffer) : meshBuffer(meshBuffer) {}

Matrix3f MeshNode::getTransform() const noexcept {
    if (auto par = parent.lock()) {
        return matMul(par->getTransform(), transform);
    }
    return transform;
}

Triangle::Triangle(
    std::array<Vector4f, 3> vertices,
    Attributes::Cases attributes
)
    : vertices(vertices),
      attributes(attributes) {}
