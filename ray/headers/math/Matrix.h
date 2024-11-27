#pragma once

#include <cstdio>
#include <array>


// template<size_t n, size_t m>
// struct Matrix {
//     struct Indices;
    
//     Matrix() = default;
//     Matrix(std::array<std::array<float, m>, n> elements);

//     float &operator[](Indices &&indices);

//     std::array<std::array<float, m>, n> elements;
// private:
//     struct Indices
//     {
//         Indices(size_t nn, size_t mm) : nn(nn), mm(mm) {}
//         size_t nn, size_t mm;
//     }
// };

// template <size_t n, size_t m>
// Matrix<n, m>::Matrix(std::array<std::array<float, m>, n> elements) : elements(elements) {}

// template <size_t n, size_t m>
// float & Matrix<n, m>::operator[](Indices &&indices)
// {
//     return elements[indices.nn, indices.mm];
// }

// template <size_t n1, size_t m1, size_t n2, size_t m2>
// Matrix<n1, m2> operator*(Matrix<n1, m1> a, Matrix<n2, m2>) {
    
// }

struct Matrix4 {
    Matrix4() = default;
    Matrix4(std::array<std::array<float, 4>, 4>);

    std::array<std::array<float, 4>, 4> elements;
};

Matrix4 operator*(Matrix4& lhs, Matrix4& rhs);