#pragma once

#include <cmath>

#include <type_traits>

#include "Eigen/Dense"
#include "glm/glm.hpp"

namespace ml {

using Vector4f = glm::vec4;
using Vector3f = glm::vec3;
using Vector2f = glm::vec2;
using Vector4i = glm::ivec4;
using Vector3i = glm::ivec3;
using Vector2i = glm::ivec2;
using Matrix4f = glm::mat4;
using Matrix3f = glm::mat3;

template <typename Mat>
float inline euclideanNorm(
    Mat const &matrix
) noexcept {
    return glm::length(matrix);
}

template <typename Mat>
Mat inline matrixScale(
    Mat matrix, float scalar
) noexcept {
    matrix *= scalar;
    return matrix;
}

template <typename Vector>
Vector inline normalized(
    Vector const &u
) {
    return glm::normalize(u);
}

Vector3f inline crossProduct(
    Vector3f const &u, Vector3f const &v
) {
    return glm::cross(u, v);
}

template <typename Vector>
float inline dotProduct(
    Vector const &u, Vector const &v
) {
    return glm::dot(u, v);
}

template <typename Matrix>
Matrix inline uniform(
    float value = 0.f
) noexcept {
    return Matrix(0.f) + value;
}

template <typename Matrix>
Matrix inline diagonal(
    float value = 1.f
) noexcept {
    return Matrix(value);
}

template <typename Vector>
inline constexpr std::decay<Vector>::type projection(
    Vector &&a, Vector &&on
) noexcept {
    auto const onNorm = normalized(on);
    return matrixScale(std::move(onNorm), dotProduct(a, onNorm));
}

template <typename Vector>
inline constexpr std::decay<Vector>::type rejection(
    Vector &&a, Vector &&on
) noexcept {
    return a - projection(a, on);
}

Matrix4f inline perspectiveProjectionMatrix(
    float fov,
    float aspectRatio,
    bool keepHeight,
    float far,
    float near
) noexcept {
    auto mat = uniform<Matrix4f>(0.f);
    auto angleMeasure = tanf(fov / 2);
    mat[0][0] = 1.f / (aspectRatio * angleMeasure);
    mat[1][1] = 1.f / angleMeasure;
    mat[2][2] = far / (far - near);
    mat[2][3] = -far * near / (far - near);
    mat[3][2] = -1.f;
    return glm::transpose(mat);
}

Matrix4f inline screenSpaceProjection(
    int width,
    int height
) noexcept {
    auto mat = uniform<Matrix4f>(0.f);
    mat[0][0] = width / 2.f;
    mat[1][1] = -height / 2.f;
    mat[2][2] = 1.f;
    mat[3][3] = 1.f;
    mat[0][3] = width / 2.f;
    mat[1][3] = height / 2.f;

    return glm::transpose(mat);
}

template <typename M1, typename M2>
decltype(auto) inline matMul(
    M1 &&lhs, M2 &&rhs
) noexcept {
    return lhs * rhs;
}

decltype(auto) inline rodriguezRotationMatrix(
    Vector3f const &axis, float angle
) noexcept {
    auto axis_ = normalized(axis);
    Matrix4f rotationMatrix = uniform<Matrix4f>(0.f);
    auto cosValue = cosf(angle);
    auto sinValue = sinf(angle);

    float x = axis[0];
    float y = axis[1];
    float z = axis[2];

    rotationMatrix[0][0] = cosValue + x * x * (1 - cosValue);
    rotationMatrix[0][1] = x * y * (1 - cosValue) - z * sinValue;
    rotationMatrix[0][2] = x * z * (1 - cosValue) + y * sinValue;

    rotationMatrix[1][0] = y * x * (1 - cosValue) + z * sinValue;
    rotationMatrix[1][1] = cosValue + y * y * (1 - cosValue);
    rotationMatrix[1][2] = y * z * (1 - cosValue) - x * sinValue;

    rotationMatrix[2][0] = z * x * (1 - cosValue) - y * sinValue;
    rotationMatrix[2][1] = z * y * (1 - cosValue) + x * sinValue;
    rotationMatrix[2][2] = cosValue + z * z * (1 - cosValue);

    rotationMatrix[3][3] = 1.0f;

    return glm::transpose(rotationMatrix);
}

Matrix4f inline scaleMatrix(
    float x, float y, float z = 1, float w = 1
) {
    auto m = uniform<Matrix4f>(0.f);
    m[0][0] = x;
    m[1][1] = y;
    m[2][2] = z;
    m[3][3] = w;
    return m;
}

decltype(auto) inline translationMatrix(
    float tx, float ty, float tz
) {
    Matrix4f res = diagonal<Matrix4f>(1.f);
    res[0][3] = tx;
    res[1][3] = ty;
    res[2][3] = tz;
    return glm::transpose(res);
}

Matrix4f inline rotateAroundPoint(
    Vector3f const &supportPoint, Vector3f const &rotationAxis, float angle
) {
    Matrix4f translation = diagonal<Matrix4f>(1.f);
    translation[0][3] = -supportPoint[0];
    translation[1][3] = -supportPoint[1];
    translation[2][3] = -supportPoint[2];
    Matrix4f rotation = rodriguezRotationMatrix(rotationAxis, angle);
    auto reverseTranslation = diagonal<Matrix4f>(1.f);
    reverseTranslation[0][3] = supportPoint[0];
    reverseTranslation[1][3] = supportPoint[1];
    reverseTranslation[2][3] = supportPoint[2];

    Matrix4f result = reverseTranslation * (rotation * translation);
    return glm::transpose(result);
}

template <typename Vector>
float inline cosineSimilarity(
    Vector const &u, Vector const &v
) {
    float uNorm = euclideanNorm(u);
    float vNorm = euclideanNorm(v);
    float res = dotProduct(u, v) / (uNorm * vNorm);
    return res;
}

template <size_t to, size_t from, typename Scalar>
glm::vec<to, float, glm::defaultp> inline as(
    glm::vec<from, float, glm::defaultp> const &vector, float fillValue = 0
) {
    glm::vec<to, float, glm::defaultp> res = glm::vec<to, float, glm::defaultp>(0) + fillValue;
    auto vecSize = from;
    if (to < vecSize) {
        vecSize = to;
    }
    for (int i = 0; i < vecSize; i++) {
        res[i] = vector[i];
    }
    return res;
}

std::tuple<float, float, float> inline barycentricWeights(
    Vector2f const &u,
    Vector2f const &v,
    Vector2f const &w,
    Vector2f const &point
) {
    Vector3f uu = {u.x, u.y, 0};
    Vector3f vv = {v.x, v.y, 0};
    Vector3f ww = {w.x, w.y, 0};
    Vector3f pp = {point.x, point.y, 0};
    auto doubledArea = euclideanNorm(cross(vv - uu, ww - uu));

    Vector3f up = uu - pp;
    Vector3f vp = vv - pp;
    Vector3f wp = ww - pp;
    auto alpha = std::abs(
        euclideanNorm(cross(vp, wp)) / doubledArea
    );
    auto beta = std::abs(
        euclideanNorm(cross(up, wp)) / doubledArea
    );
    auto gamma = std::abs(
        euclideanNorm(cross(up, vp)) / doubledArea
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

    Value recipInterpolated = std::get<0>(weights) * aRecip + std::get<1>(weights) * bRecip + std::get<2>(weights) * cRecip;
    return 1 / recipInterpolated;
}

ml::Matrix4f inline cameraMatrix(
    Vector3f position, Vector3f cameraTarget, Vector3f up = {0, 1, 0}
) {
    Vector3f lookAt = normalized(position - cameraTarget);
    Vector3f right = normalized(crossProduct(up, lookAt));
    // Already normalized, but let's keep for robust
    Vector3f upper = normalized(crossProduct(lookAt, right));
    Matrix4f result = uniform<Matrix4f>(0.f);
    result[0][0] = right[0];
    result[0][1] = right[1];
    result[0][2] = right[2];
    result[1][0] = upper[0];
    result[1][1] = upper[1];
    result[1][2] = upper[2];
    result[2][0] = lookAt[0];
    result[2][1] = lookAt[1];
    result[2][2] = lookAt[2];
    result[0][3] = -dotProduct(position, right);
    result[1][3] = -dotProduct(position, upper);
    result[2][3] = -dotProduct(position, lookAt);
    result[3][3] = 1.f;
    return glm::transpose(result);
}

template <typename Vector>
Vector lerp(
    Vector const &a,
    Vector const &b,
    float alpha
) {
    Vector res;
    for (size_t i = 0; i < res.size(); i++) {
        res[i] = std::lerp(a[i], b[i], alpha);
    }
    return res;
}
} // namespace ml