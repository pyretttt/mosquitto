#pragma once

struct Vector3 {
    Vector3() = default;
    Vector3(float x, float y, float z);
    Vector3(Vector3 const &other);
    void operator=(Vector3 const &rhs);

    Vector3 crossProduct(Vector3 const &other) const noexcept;
    float dotProduct(Vector3 const &other) const noexcept;
    Vector3 normalized() const noexcept;
    float norm() const noexcept;
    Vector3 operator-() const noexcept;
    Vector3 projection(Vector3 const &rhs) const noexcept;
    Vector3 rejection(Vector3 const &rhs) const noexcept;

    Vector3 operator+(Vector3 const &rhs) const noexcept;
    Vector3 operator-(Vector3 const &rhs) const noexcept;
    Vector3 operator*(float s) const noexcept;

    float x{0.f}, y{0.f}, z{0.f};
};