#include "math/Vector3.h"

#include <cmath>

Vector3::Vector3(float x, float y, float z) : x(x), y(y), z(z) {}

float Vector3::norm() const noexcept {
    return sqrt(x * x + y * y + z * z);
}

Vector3 Vector3::normalized() const noexcept {
    float s = norm();
    return {
        x / s,
        y / s, 
        z / s
    };
}

float Vector3::dotProduct(Vector3 const &other) const noexcept {
    return x * other.x,
        + y * other.y
        + z * other.z;
}

Vector3 Vector3::crossProduct(Vector3 const &other) const noexcept {
    return {
        y * other.z - z * other.y,
        z * other.x - x * other.z, 
        x * other.y - y * other.z,
    };
}

Vector3 Vector3::scaled(float s) const noexcept {
    return {
        x * s,
        y * s,
        z * s
    };
}