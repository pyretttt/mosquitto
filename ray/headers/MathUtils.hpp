#pragma once

#include <cmath>

#include <type_traits>

#include "Eigen/Dense"

namespace ml {

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
    mat(3, 2) = -1.f;
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
Mat inline matrixScale(
    Mat matrix, float scalar
) noexcept {
    matrix *= scalar;
    return matrix;
}

template <typename M1, typename M2>
decltype(auto) inline matMul(
    M1 &&lhs, M2 &&rhs
) noexcept {
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
Eigen::Matrix<float, size, size> inline eye() noexcept {
    return Eigen::Matrix<float, size, size>::Identity();
}

decltype(auto) inline rodriguezRotationMatrix(
    Vector3f axis, float angle
) noexcept {
    axis.normalize();
    Matrix4f rotationMatrix = Matrix4f::Zero();
    auto cosValue = cosf(angle);
    auto sinValue = sinf(angle);

    float x = axis(0, 0);
    float y = axis(1, 0);
    float z = axis(2, 0);

    rotationMatrix(0, 0) = cosValue + x * x * (1 - cosValue);
    rotationMatrix(0, 1) = x * y * (1 - cosValue) - z * sinValue;
    rotationMatrix(0, 2) = x * z * (1 - cosValue) + y * sinValue;

    rotationMatrix(1, 0) = y * x * (1 - cosValue) + z * sinValue;
    rotationMatrix(1, 1) = cosValue + y * y * (1 - cosValue);
    rotationMatrix(1, 2) = y * z * (1 - cosValue) - x * sinValue;

    rotationMatrix(2, 0) = z * x * (1 - cosValue) - y * sinValue;
    rotationMatrix(2, 1) = z * y * (1 - cosValue) + x * sinValue;
    rotationMatrix(2, 2) = cosValue + z * z * (1 - cosValue);

    rotationMatrix(3, 3) = 1.0f;

    return rotationMatrix;
}

Matrix4f inline scaleMatrix(
    float x, float y, float z = 1, float w = 1
) {
    Vector4f m({x, y, z, w});
    return m.asDiagonal();
}

decltype(auto) inline translationMatrix(
    float tx, float ty, float tz
) {
    Matrix4f res = Matrix4f::Identity();
    res(0, 3) = tx;
    res(1, 3) = ty;
    res(2, 3) = tz;
    return res;
}

Matrix4f inline rotateAroundPoint(
    Vector3f supportPoint, Vector3f rotationAxis, float angle
) {
    Matrix4f translation = Matrix4f::Identity();
    translation(0, 3) = -supportPoint(0, 0);
    translation(1, 3) = -supportPoint(1, 0);
    translation(2, 3) = -supportPoint(2, 0);
    Matrix4f rotation = rodriguezRotationMatrix(rotationAxis, angle);
    Matrix4f reverseTranslation = Matrix4f::Identity();
    reverseTranslation(0, 3) = supportPoint(0, 0);
    reverseTranslation(1, 3) = supportPoint(1, 0);
    reverseTranslation(2, 3) = supportPoint(2, 0);

    Matrix4f result = reverseTranslation * (rotation * translation);
    return result;
}

Vector3f inline crossProduct(
    Vector3f const &u, Vector3f const &v
) {
    return u.cross(v).eval();
}

template <typename Vector>
float inline dotProduct(
    Vector const &u, Vector const &v
) {
    return u.dot(v);
}

template <typename Vector>
float inline cosineSimilarity(
    Vector const &u, Vector const &v
) {
    float uNorm = u.template lpNorm<2>();
    float vNorm = v.template lpNorm<2>();
    float res = u.dot(v) / (uNorm * vNorm);
    return res;
}

template <size_t to, size_t from, typename Scalar>
Eigen::Matrix<Scalar, to, 1> inline as(
    Eigen::Matrix<Scalar, from, 1> vector, float fillValue = 0
) {
    Eigen::Matrix<Scalar, to, 1> res = Eigen::Matrix<Scalar, to, 1>::Constant(fillValue);
    auto vecSize = vector.size();
    if (to < vecSize) {
        vecSize = to;
    }
    for (int i = 0; i < vecSize; i++) {
        res(i, 0) = vector(i, 0);
    }
    return res;
}

std::tuple<float, float, float> inline barycentricWeights(
    Vector2f u,
    Vector2f v,
    Vector2f w,
    Vector2f point
) {
    Vector3f uu = {u.x(), u.y(), 0};
    Vector3f vv = {v.x(), v.y(), 0};
    Vector3f ww = {w.x(), w.y(), 0};
    Vector3f pp = {point.x(), point.y(), 0};
    auto doubledArea = crossProduct(
                           (vv - uu).eval(),
                           (ww - uu).eval()
    )
                           .lpNorm<2>();

    Vector3f up = uu - pp;
    Vector3f vp = vv - pp;
    Vector3f wp = ww - pp;
    auto alpha = std::abs(
        crossProduct(vp, wp).lpNorm<2>() / doubledArea
    );
    auto beta = std::abs(
        crossProduct(up, wp).lpNorm<2>() / doubledArea
    );
    auto gamma = std::abs(
        crossProduct(up, vp).lpNorm<2>() / doubledArea
    );

    return std::make_tuple(alpha, beta, gamma);
}

template <typename Value>
Value perspectiveInterpolate(
    Value const &A,
    Value const &B,
    Value const &C,
    std::tuple<float, float, float> const &weights
) noexcept {
    Value aRecip = 1 / A;
    Value bRecip = 1 / B;
    Value cRecip = 1 / C;

    Value recipInterpolated = std::get<0>(weights) * aRecip
        + std::get<1>(weights) * bRecip 
        + std::get<2>(weights) * cRecip;
    return 1 / recipInterpolated;
}

ml::Matrix4f inline cameraMatrix(
    Vector3f position, Vector3f cameraTarget, Vector3f up = {0, 1, 0}
) {
    Vector3f lookAt = (position - cameraTarget).normalized();
    Vector3f right = crossProduct(up, lookAt).normalized();
    // Already normalized, but let's keep for robust
    Vector3f upper = crossProduct(lookAt, right).normalized();
    Matrix4f result;
    result(0, 0) = right(0, 0);
    result(0, 1) = right(1, 0);
    result(0, 2) = right(2, 0);
    result(1, 0) = upper(0, 0);
    result(1, 1) = upper(1, 0);
    result(1, 2) = upper(2, 0);
    result(2, 0) = lookAt(0, 0);
    result(2, 1) = lookAt(1, 0);
    result(2, 2) = lookAt(2, 0);
    result(0, 3) = -dotProduct(position, right);
    result(1, 3) = -dotProduct(position, upper);
    result(2, 3) = -dotProduct(position, lookAt);
    result(3, 3) = 1.f;
    return result;
}

template <typename Vector>
Vector lerp(
    Vector const &a,
    Vector const &b,
    float alpha
) {
    Vector res;
    for (size_t i = 0; i < res.size(); i++) {
        res(i, 0) = std::lerp(a(i, 0), b(i, 0), alpha);
    }
    return res;
}
} // namespace ml