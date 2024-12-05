#pragma once

#include <type_traits>

#include "Eigen/Dense"

using Vector4f = Eigen::Vector4f;
using Vector3f = Eigen::Vector3f;
using Vector2f = Eigen::Vector2f;
using Vector4i = Eigen::Vector4i;
using Vector3i = Eigen::Vector3i;
using Vector2i = Eigen::Vector2i;
using Matrix4f = Eigen::Matrix4f;
using Matrix3f = Eigen::Matrix3f;


template <typename Vector>
inline constexpr std::decay<Vector>::type projection(
    Vector &&a, Vector &&on
) {
    auto const onNorm = on.normalized();
    return (onNorm * a.dot(onNorm)).eval();
}

template <typename Vector>
inline constexpr std::decay<Vector>::type rejection(
    Vector &&a, Vector &&on
) {
    return (a - projection(a, on)).eval();
}

Vector4f inline asVec4(Vector3f v, float fillValue = 0.f) {
    return {
        v.x(),
        v.y(),
        v.z(),
        fillValue
    };
}

Matrix4f inline perspectiveProjectionMatrix(
    float fov,
    float aspectRatio,
    bool keepHeight,
    float far,
    float near
) {
    Matrix4f mat = Matrix4f::Zero();
    auto angleMeasure = tanf(fov / 2);
    mat(0, 0) = 1.f / (aspectRatio * angleMeasure);
    mat(1, 1) = 1.f / angleMeasure;
    mat(2, 2) = far / (far - near);
    mat(2, 3) = -far * near / (far - near);
    mat(3, 2) = 1.f;
    return mat;
}

Matrix4f inline screenSpaceProjection(
    int width,
    int height
) {
    Matrix4f mat = Matrix4f::Zero();
    mat(0, 0) = width / 2.f;
    mat(1, 1) = -height / 2.f;
    mat(2, 2) = 1.f;
    mat(3, 3) = 1.f;
    mat(0, 3) = width / 2.f;
    mat(1, 3) = height / 2.f;

    return mat;
}