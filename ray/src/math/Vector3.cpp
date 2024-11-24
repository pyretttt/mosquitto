#include "math/Vector3.h"

#include <cmath>

Vector3::Vector3(float x, float y, float z) : x(x), y(y), z(z) {}

Vector3::Vector3(Vector3 const &other) : x(other.x), y(other.y), z(other.z) {}

void Vector3::operator=(Vector3 const &rhs) {
    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
}

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

Vector3 Vector3::projection(Vector3 const &rhs) const noexcept {
    auto norm_ = rhs.norm();
    return rhs * (dotProduct(rhs) / norm_ * norm_);
}

Vector3 Vector3::rejection(Vector3 const &rhs) const noexcept {
    return *this - projection(rhs);
}

Vector3 Vector3::operator-() const noexcept {
    return {-x, -y, -z};
}

Vector3 Vector3::operator+(Vector3 const &rhs) const noexcept {
    return { x + rhs.x, y + rhs.y, z + rhs.z };
}

Vector3 Vector3::operator-(Vector3 const &rhs) const noexcept {
    return this->operator+(-rhs);
}

Vector3 Vector3::operator*(float s) const noexcept {
    return {x * s, y * s, z * s};
}
