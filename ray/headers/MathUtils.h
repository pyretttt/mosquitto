#pragma once

#include <cmath>

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
) noexcept {
    auto const onNorm = on.normalized();
    return (onNorm * a.dot(onNorm)).eval();
}

template <typename Vector>
inline constexpr std::decay<Vector>::type rejection(
    Vector &&a, Vector &&on
) noexcept {
    return (a - projection(a, on)).eval();
}

Vector4f inline asVec4(Vector3f v, float fillValue = 0.f) noexcept {
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
) noexcept {
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
) noexcept {
    Matrix4f mat = Matrix4f::Zero();
    mat(0, 0) = width / 2.f;
    mat(1, 1) = -height / 2.f;
    mat(2, 2) = 1.f;
    mat(3, 3) = 1.f;
    mat(0, 3) = width / 2.f;
    mat(1, 3) = height / 2.f;

    return mat;
}

template <typename Mat>
Mat inline matrixScale(Mat &&matrix, float scalar) noexcept {
    matrix *= scalar;
    return matrix;
}

template <typename M1, typename M2>
decltype(auto) inline matMul(M1 &&lhs, M2 &&rhs) noexcept {
    return (lhs * rhs).eval();
}

template <typename Mat>
Mat inline ones() noexcept {
    return Mat::Ones().eval();
}

template <typename Mat>
Mat inline zeros() noexcept {
    return Mat::Zero();
}

template <size_t size>
decltype(auto) inline eye() noexcept {
    return Eigen::Matrix<float, size, 1>::Ones().asDiagonal();
}

decltype(auto) inline rodriguezRotationMatrix(
    Vector3f axis,
    float angle
) noexcept {
    auto rotationMatrix = Metrix3f::Zero();
    auto cosValue = cosf(angle);
    auto sinValue = sinf(angle);
    rotationMatrix(0, 0) = cosValue + powf(axis(0, 0), 2.f) * (1 - cosValue);
    rotationMatrix(1, 0) = cosValue + powf(axis(0, 0), 2.f) * (1 - cosValue);
    rotationMatrix(2, 0) = cosValue + powf(axis(0, 0), 2.f) * (1 - cosValue);
    rotationMatrix(0, 1) = cosValue + powf(axis(0, 0), 2.f) * (1 - cosValue);
    rotationMatrix(1, 1) = cosValue + powf(axis(1, 0), 2.f) * (1 - cosValue);
    rotationMatrix(2, 1) = cosValue + powf(axis(0, 0), 2.f) * (1 - cosValue);
    rotationMatrix(0, 2) = cosValue + powf(axis(0, 0), 2.f) * (1 - cosValue);
    rotationMatrix(1, 2) = cosValue + powf(axis(0, 0), 2.f) * (1 - cosValue);
    rotationMatrix(2, 2) = cosValue + powf(axis(2, 0), 2.f) * (1 - cosValue);
    Eigen::Rotation3D
}

decltype(auto) rotateAround(Matrix3f matrix, Vector3f axis, float angle) {
    auto rotationMatrix = Metrix3f::Zero();
    rotationMatrix
}